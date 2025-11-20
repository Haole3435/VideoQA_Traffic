import easyocr
import json
import os
import cv2
from decord import VideoReader, cpu
import numpy as np
from tqdm import tqdm

def extract_text_from_video(video_path, reader, num_ocr_frames=10):
    """
    Trích xuất văn bản từ một file video sử dụng EasyOCR trên GPU.
    """
    detected_texts = set() 
    try:
        if not os.path.exists(video_path):
            return ["Video File Not Found"]
            
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        if total_frames == 0:
            return ["Video Load Error or Empty Video"]

        ocr_frame_indices = np.linspace(0, total_frames - 1, num_ocr_frames, dtype=int)
        frames_rgb = vr.get_batch(ocr_frame_indices).asnumpy()
        
        for frame_img_rgb in frames_rgb:
            results = reader.readtext(frame_img_rgb)
            for (bbox, text, prob) in results:
                if prob > 0.3: 
                    detected_texts.add(text)
                    
    except Exception as e:
        print(f"Lỗi khi xử lý video {video_path}: {e}")
        return [f"Video Processing Error: {e}"]
    
    return list(detected_texts)

def process_json(input_json_path, output_json_path, video_dir_path, reader):
    """
    Đọc file JSON đầu vào, thêm trường 'ocr_text' và lưu file mới.
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_data = {'__count__': data['__count__'], 'data': []}
    
    for item in tqdm(data['data'], desc=f"Đang xử lý {os.path.basename(input_json_path)}"):
        video_path = os.path.join(video_dir_path, item['video_path'])
        ocr_texts = extract_text_from_video(video_path, reader, num_ocr_frames=10)
        item['ocr_text'] = ocr_texts
        output_data['data'].append(item)
        
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"Đã lưu file xử lý xong vào: {output_json_path}")

def main():
    # --- KHỞI TẠO EASYOCR TRÊN GPU ---
    print("Đang tải mô hình EasyOCR (chỉ lần đầu)...")
    reader = easyocr.Reader(['vi', 'en'], gpu=True) 
    print("Tải EasyOCR thành công (đang chạy trên GPU).")
    # -----------------------------------

    # --- ĐỊNH NGHĨA ĐƯỜNG DẪN ---
    # !! THAY ĐỔI CÁC ĐƯỜNG DẪN NÀY !!
    base_video_dir = "/home/user/RoadBuddy/data/" # Nơi chứa thư mục 'videos'
    
    # File Input
    train_json_in = "/home/user/RoadBuddy/data/train.json"
    test_json_in = "/home/user/RoadBuddy/data/public_test.json"
    
    # File Output
    train_json_out = "/home/user/RoadBuddy/ocr_data/train_ocr.json"
    test_json_out = "/home/user/RoadBuddy/ocr_data/public_test_ocr.json"
    # -----------------------------------

    print("Bắt đầu xử lý file train...")
    process_json(train_json_in, train_json_out, base_video_dir, reader)
    
    print("\nBắt đầu xử lý file public_test...")
    process_json(test_json_in, test_json_out, base_video_dir, reader)
    
    print("\nHoàn tất tiền xử lý OCR!")

if __name__ == "__main__":
    main()