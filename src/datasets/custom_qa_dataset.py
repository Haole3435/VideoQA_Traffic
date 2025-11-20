import torch
from torch.utils.data import Dataset
import json
import os
from decord import VideoReader, cpu
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import numpy as np
from src.utils.traffic_law_knowledge import get_law_context

class CustomOCRDataset(Dataset):
    """
    Optimized VideoQA Dataset cho GPU 3090 24GB
    - Giảm số frames cho video dài
    - Tích hợp luật giao thông Việt Nam
    - Loại bỏ OCR processing
    """
    def __init__(self, 
                 dataset_path, 
                 video_dir_path,
                 tokenizer,
                 num_frames=8,  # GIẢM từ 16→8 frames cho video dài
                 max_text_len=256,  # TĂNG để chứa context luật
                 num_choices=4,
                 is_train=True):
        
        print(f"Loading dataset from: {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)['data']
            if is_train:
                self.data = [item for item in all_data if 'answer' in item and item['answer']]
                print(f"✅ Train: {len(self.data)}/{len(all_data)} samples")
            else:
                self.data = all_data
                print(f"✅ Val: {len(self.data)} samples")
            
        self.video_dir_path = video_dir_path
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.max_text_len = max_text_len
        self.num_choices = num_choices
        self.is_train = is_train
        
        self.answer_mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
        
        # Transform cho video
        self.video_transform = Compose([
            Resize(224),
            CenterCrop(224),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        question = item['question']
        # video_path = os.path.join(self.video_dir_path, item['video_path'])
        # video_path = item['video_path']
        # if not os.path.isabs(video_path):  # Nếu là relative path
        #     video_path = os.path.join(video_path)
        video_filename = os.path.basename(item['video_path'])
        video_path = os.path.join(self.video_dir_path, video_filename)
        
        # === 1. ĐỌC VIDEO (Optimized cho video dài) ===
        model_frames = torch.zeros(3, self.num_frames, 224, 224)
        
        if not os.path.exists(video_path):
            print(f"⚠️ Video not found: {video_path}")
        else:
            try:
                vr = VideoReader(video_path, ctx=cpu(0))
                total_frames = len(vr)
                
                # Smart sampling cho video dài
                if total_frames > 0:
                    # Lấy frames đều từ đầu đến cuối video
                    indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
                    frames = vr.get_batch(indices).float() / 255.0
                    frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
                    model_frames = self.video_transform(frames)
                    model_frames = model_frames.permute(1, 0, 2, 3)  # [C, T, H, W]
                    
            except Exception as e:
                print(f"❌ Lỗi đọc video {video_path}: {e}")
        
        # === 2. TÍCH HỢP LUẬT GIAO THÔNG ===
        # Lấy context luật liên quan đến câu hỏi
        law_context = get_law_context(question)
        
        # Format câu hỏi với context luật (QUAN TRỌNG cho hiểu tiếng Việt)
        enhanced_question = f"{question}\n[Luật liên quan: {law_context}]"
        
        # === 3. XỬ LÝ TEXT ===
        choices = []
        for choice in item['choices']:
            # Loại bỏ prefix A. B. C. D.
            if ". " in choice and choice[0] in ['A', 'B', 'C', 'D']:
                choices.append(choice.split(". ", 1)[1])
            else:
                choices.append(choice)
        
        # Tokenize từng choice với question + law context
        all_token_ids = []
        all_attention_masks = []
        
        for i in range(min(len(choices), self.num_choices)):
            choice_text = choices[i]
            # Format: Question + Law Context [SEP] Choice
            text_to_tokenize = f"{enhanced_question} </s> {choice_text}"
            
            encoded = self.tokenizer(
                text_to_tokenize,
                choice_text,
                max_length=self.max_text_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            all_token_ids.append(encoded['input_ids'].squeeze(0))
            all_attention_masks.append(encoded['attention_mask'].squeeze(0))
        
        # Pad nếu thiếu choices
        while len(all_token_ids) < self.num_choices:
            all_token_ids.append(torch.zeros(self.max_text_len, dtype=torch.long))
            all_attention_masks.append(torch.zeros(self.max_text_len, dtype=torch.long))
        
        input_ids = torch.stack(all_token_ids[:self.num_choices], dim=0).unsqueeze(1)
        attention_mask = torch.stack(all_attention_masks[:self.num_choices], dim=0).unsqueeze(1)
        
        # === 4. XỬ LÝ LABEL ===
        answer_label = 0
        if 'answer' in item and item['answer']:
            answer_text = item['answer']
            try:
                # Loại bỏ prefix nếu có
                if ". " in answer_text:
                    answer_text_clean = answer_text.split(". ", 1)[1]
                else:
                    answer_text_clean = answer_text
                
                answer_label = choices.index(answer_text_clean)
            except ValueError:
                # Fallback: dùng letter mapping
                answer_letter = answer_text.split(".")[0].strip()
                answer_label = self.answer_mapping.get(answer_letter, 0)
            except Exception:
                answer_label = 0
        
        return {
            "video_frames": model_frames,  # [C, T, H, W]
            "text_ids": input_ids,  # [num_choices, seq_len]
            "attention_mask": attention_mask,
            "label": torch.tensor(answer_label, dtype=torch.long),
            "temporal_labels": torch.zeros(self.num_frames, dtype=torch.long), 
            "temporal_label_weights": torch.ones(self.num_frames, dtype=torch.float),
            "span_label_weights": torch.ones(self.num_frames, dtype=torch.float)
        }