import re

TRAFFIC_LAWS = {
    "SPEED": "Quy định tốc độ (Thông tư 31/2019): Khu đông dân cư tối đa 60km/h (đường đôi), 50km/h (2 chiều). Ngoài khu dân cư: 90km/h (đường đôi), 80km/h (2 chiều). Cao tốc tuân thủ biển báo (thường max 120, min 60).",
    "OVERTAKE": "Luật vượt xe (Điều 14): Chỉ vượt bên trái. Cấm vượt nơi tầm nhìn hạn chế, đường vòng, đầu dốc, nơi có biển Cấm vượt, trên cầu hẹp, nơi giao nhau.",
    "PRIORITY": "Thứ tự ưu tiên: 1. Xe đã vào giao lộ. 2. Xe ưu tiên (Chữa cháy > Quân sự > Công an > Cứu thương). 3. Đường ưu tiên. 4. Bên phải không vướng. 5. Rẽ phải > Đi thẳng > Rẽ trái.",
    "PARKING": "Cấm dừng đỗ: Bên trái đường 1 chiều, trên miệng cống, nơi giao nhau, trên cầu, nơi che khuất biển báo, nơi có biển Cấm dừng đỗ (P.130, P.131).",
    "TURN": "Chuyển hướng: Phải giảm tốc độ, có tín hiệu báo hướng (xi nhan). Cấm quay đầu ở phần đường người đi bộ, trên cầu, trong hầm, đường cao tốc.",
    "DEFAULT": "Tuân thủ biển báo và vạch kẻ đường."
}

SIGN_DEFINITIONS = {
    "P.102": "Biển Cấm đi ngược chiều.", "P.123": "Biển Cấm rẽ.", 
    "P.125": "Biển Cấm vượt.", "P.127": "Biển Tốc độ tối đa.", 
    "W.207": "Giao nhau với đường không ưu tiên (Được đi trước).",
    "R.420": "Khu đông dân cư."
}

def get_law_context(question, ocr_texts=None):
    q_lower = question.lower()
    combined_text = q_lower + " " + (" ".join(ocr_texts).lower() if ocr_texts else "")
    context = []

    # 1. Tra biển báo từ mã (Ví dụ: P.102)
    found_codes = re.findall(r'[pPwWrR][\.\s]?[0-9]{3}[a-z]?', combined_text)
    for code in found_codes:
        code_clean = code.replace(" ", "").upper()
        for k, v in SIGN_DEFINITIONS.items():
            if code_clean in k: context.append(v)

    # 2. Tra luật theo từ khóa
    if any(w in q_lower for w in ["tốc độ", "km/h", "nhanh"]): context.append(TRAFFIC_LAWS["SPEED"])
    if any(w in q_lower for w in ["vượt", "lấn làn"]): context.append(TRAFFIC_LAWS["OVERTAKE"])
    if any(w in q_lower for w in ["ưu tiên", "nhường", "ngã tư"]): context.append(TRAFFIC_LAWS["PRIORITY"])
    if any(w in q_lower for w in ["dừng", "đỗ"]): context.append(TRAFFIC_LAWS["PARKING"])
    if any(w in q_lower for w in ["quay đầu", "rẽ"]): context.append(TRAFFIC_LAWS["TURN"])

    if not context: return TRAFFIC_LAWS["DEFAULT"]
    return " ".join(list(set(context)))