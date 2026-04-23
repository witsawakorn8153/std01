# กำหนดรูปแบบคำศัพท์
import re
VIOLATION_KEYWORD = ["ละเมิด", "ผิดกฎหมาย", "ผิดศีลธรรม", "ไม่เหมาะสม", "อันตราย", "รุนแรง", "ก้าวร้าว", "เหยียดหยาม", "ดูถูก", "ข่มขู่", "คุกคาม", "ล่วงละเมิด", "ละเมิดสิทธิ์", "ละเมิดความเป็นส่วนตัว  "]
PATENT_KEYWORD = ["สิทธิบัตร", "เครื่องหมายการค้า", "ลิขสิทธิ์", "ทรัพย์สินทางปัญญา", "การละเมิดสิทธิบัตร", "การละเมิดเครื่องหมายการค้า", "การละเมิดลิขสิทธิ์", "การละเมิดทรัพย์สินทางปัญญา"]
COPYRIGHT_KEYWORD = ["ลิขสิทธิ์", "การละเมิดลิขสิทธิ์", "การละเมิดทรัพย์สินทางปัญญา"]
TRADEMARK_KEYWORD = ["เครื่องหมายการค้า", "การละเมิดเครื่องหมายการค้า", "การละเมิดทรัพย์สินทางปัญญา"]
def detect_category(text):
    # Regex pattern for detecting keywords
    is_violation = any(re.search(k, text) for k in VIOLATION_KEYWORD)
    is_patent = any(re.search(k, text) for k in PATENT_KEYWORD)
    is_copyright = any(re.search(k, text) for k in COPYRIGHT_KEYWORD)
    is_trademark = any(re.search(k, text) for k in TRADEMARK_KEYWORD)
    if is_violation:
        if is_violation: return 1
        if is_patent: return 2
        if is_copyright: return 3
        if is_trademark: return 4
    return 0
# Example usage
sample = "การละเมิดสิทธิบัตรเป็นเรื่องที่ต้องระวัง"
print(f"text: {sample}")
print(f"Predicted Class: {detect_category(sample)}")

#Context-aware confidence scores
def detect_category_with_confidence(text, predicted_class):
    base_confidence = 0.70
    signal = []
    if "มาตรา" in text or "พ.ร.บ." in text:
        base_confidence += 0.15
        signal.append("statotury_reference")
    if "คำพิพากษา" in text:
        base_confidence += 0.10
        signal.append("precedent_reference")
    return min(base_confidence, 0.99), signal
# Example usage
text_with_context = "การละเมิดลิขสิทธิ์ตามมาตรา 10 แห่ง พ.ร.บ.ลิขสิทธิ์ เป็นเรื่องที่ต้องระวัง"
predicted_class = detect_category(text_with_context)
confidence, signals = detect_category_with_confidence(text_with_context, predicted_class)
print(f"text: {text_with_context}")
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence}")
print(f"Signals: {signals}")
