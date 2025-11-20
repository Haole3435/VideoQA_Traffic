import json
import torch
import logging

logger = logging.getLogger(__name__)

def build_optimizer_parameters(config, model):
    # Lấy Learning Rate gốc từ config
    base_lr = float(config.OPTIMIZER.lr)
    
    # Hệ số tăng tốc cho các lớp mới (Vision Proj, Classifier...)
    # Các lớp này cần học nhanh hơn Backbone (Swin/PhoBERT)
    HEAD_LR_MULT = 10.0 

    # Danh sách các từ khóa để nhận diện lớp mới (cần học nhanh)
    # 'vision_proj': lớp cầu nối bạn mới thêm
    # 'head', 'classifier', 'cls': các lớp đầu ra
    # 'adapter': nếu có dùng adapter
    head_keywords = ['vision_proj', 'classifier', 'head', 'cls', 'adapter']

    # Danh sách các tham số không áp dụng Weight Decay (chuẩn chung)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'pos_embed', 'relative_position_bias_table']

    if "weight_decay" in config.TRAINING.keys():
        weight_decay = config.TRAINING["weight_decay"]
    else:
        weight_decay = 0.01

    # Chia tham số thành 4 nhóm:
    # 1. Backbone (Swin/PhoBERT) - Có Decay
    # 2. Backbone (Swin/PhoBERT) - Không Decay (bias, layernorm...)
    # 3. New Layers (VisionProj/Head) - Có Decay (Học nhanh gấp 10 lần)
    # 4. New Layers (VisionProj/Head) - Không Decay (Học nhanh gấp 10 lần)
    
    backbone_decay = []
    backbone_no_decay = []
    head_decay = []
    head_no_decay = []

    # Duyệt qua toàn bộ tham số của mô hình
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Bỏ qua pooler của BERT nếu không cần thiết (như code cũ của bạn)
        if 'pooler' in name:
            continue

        # Kiểm tra xem tham số này thuộc nhóm HEAD (Lớp mới) hay BACKBONE (Cũ)
        is_head = any(k in name for k in head_keywords)
        
        # Kiểm tra xem có áp dụng Weight Decay không
        is_no_decay = any(nd in name for nd in no_decay)

        if is_head:
            if is_no_decay:
                head_no_decay.append(param)
            else:
                head_decay.append(param)
        else:
            if is_no_decay:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)

    # Tạo danh sách optimizer grouped parameters
    optimizer_grouped_parameters = [
        # Nhóm 1: Backbone (LR chuẩn)
        {
            'params': backbone_decay,
            'weight_decay': weight_decay,
            'lr': base_lr
        },
        {
            'params': backbone_no_decay,
            'weight_decay': 0.0,
            'lr': base_lr
        },
        # Nhóm 2: New Layers (LR nhân lên 10 lần)
        {
            'params': head_decay,
            'weight_decay': weight_decay,
            'lr': base_lr * HEAD_LR_MULT
        },
        {
            'params': head_no_decay,
            'weight_decay': 0.0,
            'lr': base_lr * HEAD_LR_MULT
        }
    ]
    
    # In thông tin để kiểm tra (Debug)
    print(f"🔥 OPTIMIZER SETUP:")
    print(f"   - Base LR (Backbone): {base_lr}")
    print(f"   - Head LR (New Layers): {base_lr * HEAD_LR_MULT}")
    print(f"   - Backbone params: {len(backbone_decay) + len(backbone_no_decay)}")
    print(f"   - Head params (High LR): {len(head_decay) + len(head_no_decay)}")

    return optimizer_grouped_parameters