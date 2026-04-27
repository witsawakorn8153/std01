# 1. การตัดคำ Tokenization + Custom Dictionary
import re
from pythainlp import word_tokenize
from pythainlp.tokenize import word_tokenize

LEGAL_KEYWORD = ["ละเมิดสิทธิบัตร", "เครื่องหมายการค้า", "คำพิพากษา", "มาตรา", "พ.ร.บ.", "กฎหมาย", "ศาล", "คดี", "ข้อหา", "บทลงโทษ", "ความรับผิด", "การฟ้องร้อง", "การดำเนินคดี", "การละเมิดลิขสิทธิ์", "การละเมิดเครื่องหมายการค้า", "การละเมิดสิทธิ์", "การละเมิดกฎหมาย", "การละเมิดข้อหา", "การละเมิดบทลงโทษ", "การละเมิดความรับผิด", "การละเมิดการฟ้องร้อง", "การละเมิดการดำเนินคดี"]


def legal_tokenizer(text): # ตัดคำ Baseline
    sorted_keywords = sorted(LEGAL_KEYWORD, key=len, reverse=True)
    placeholder = {}
    protected = text
    for i, keyword in enumerate(sorted_keywords):
        ph = f"__KW__{i}__"
        if keyword in protected:
            placeholder[ph] = keyword
            protected = protected.replace(keyword, ph)
            
# 2. tokenize the protected text pythaiNLP
    tokens = word_tokenize(protected, engine="newmm", keep_whitespace=False)
    
# 3. restore placeholder to original keywords
    return [placeholder.get(token, token) for token in tokens]

test_text = "จำเลยกระทำความผิดฐานละเมิดสิทธิบัตรและเครื่องหมายการค้า ตามมาตรา 10 แห่ง พ.ร.บ.ลิขสิทธิ์ และถูกฟ้องร้องในศาล"
tokens = legal_tokenizer(test_text)
# print(f"Input: {test_text}")
print(f"\n{'-'*30}Legal Tokenizer Output{'-'*30}")
print(f"Input: {test_text}")
print(f"Output: {tokens}\n")
print(f"{'-'*80}\n")

# การวัด ความกำกวม Ambiguity เทียบระหว่าง Dict-based+Regex กับ WangchanBERT (27 เมษายน 2024 - อัพเดตข้อมูลล่าสุด)
def calculate_baseline_ambiguity(text):
    matches = []
    for word in LEGAL_KEYWORD:
        for m in re.finditer(re.escape(word), text):
            matches.append((m.start(), m.end(), word))

    # ตรวจสอบความทับซ้อน (Overlap) ของคำที่พบ
    overlaps = 0
    for i in range(len(matches)):
        for j in range(i+1, len(matches)):
            # ถ้าตำแหน่งเริ่ม/สิ้นสุดของคำที่ ทับซ้อนกัน แสดงว่ามีความกำกวม
            if matches[i][0] < matches[j][1] and matches[i][1] > matches[j][0]: #ตรวจสอบว่ามีการทับซ้อนกันหรือไม่
                overlaps += 1
    return overlaps/len(matches) if matches else 0 
        
# รันแสดงผล Beaseline Ambiguity

sample_text = "คดีการละเมิดสิทธิบัตรและเครื่องหมายการค้า"        
baseline_tokens = legal_tokenizer(sample_text)
baseline_rate = calculate_baseline_ambiguity(sample_text)
print(f"\n{'-'*30}W1 Baseline Results---{'-'*30}")
print(f"Tokens: {baseline_tokens}")
print(f"Baseline Ambiguity Rate : {baseline_rate : .3f}")
print(f"{'-'*80}\n")




# WangchanBERT Pre-trained Model (อัพเดตข้อมูลล่าสุด 27 เมษายน 2024) Lab3
from transformers import AutoTokenizer, AutoModel
# 1. โหลด WangchanBERT
model_name = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def berta_tokenizer(text):
    tokens = tokenizer.tokenize(text)
    return [t.replace(" ", "") for t in tokens if t.replace(" ", "")] #ลบสัญลักษณ์ที่ใช้ในการตัดคำของ WangchanBERT

# รันแสดงผล WangchanBERT Tokenization ทดสอบวัดความกำกวม Ambiguity เน้นด้านกฎหมาย
def analyze_refined_ambiguity(text, legal_keywords):
    tokens = berta_tokenizer(text)
    frag_scores = []
    for keyword in legal_keywords:
        if keyword in text:
            keyword_tokens = berta_tokenizer(keyword)
            fragment_ratio = len(keyword_tokens)/1   #สมมติว่า Fragmentation Score = จำนวน Token ของคำที่ถูกตัด / 1 (ค่ามาตรฐาน)
            frag_scores.append(fragment_ratio)
        # Ambiguity = Average Fragmentation Score -1 แต่ถ้าตัดได้พอดี เท่ากับ 0
    
    avg_frag_= sum(frag_scores) / len(frag_scores) -1 if frag_scores else 0
    return min(avg_frag_, 1.0) #จำกัดค่า Ambiguity สูงสุดที่ 1.0

# รันแสดงผล WangchanBERT Ambiguity เปรียบเทียบ
refined_tokrns = berta_tokenizer(sample_text)
refined_rate = analyze_refined_ambiguity(sample_text, LEGAL_KEYWORD)
print(f"\n{'-'*30}W1 : Refined With WangchanBERT---{'-'*30}")
print(f"Tokens: {refined_tokrns}")
print(f"New Ambiguity Fragmentation Rate : {refined_rate : .3f}")
print(f"{'-'*80}\n")

            
            
            
            
            
            

# overlaps = 0
# for i in range(len(matches)):
#     for j in range(i+1, len(matches)):
#         if matches[i][0] < matches[j][1] and matches[i][1] > matches[j][0]:
#             overlaps += 1

#     return overlaps / len(matches) if matches else 0


# 2. Context-Aware Tokenization
# ในบางกรณี การตัดคำแบบธรรมดาอาจไม่เพียงพอ เนื่องจากคำบางคำอาจมีความหมายที่แตกต่างกันขึ้นอยู่กับบริบท เช่น คำว่า "คดี" อาจหมายถึง "คดีความ" หรือ "คดีศึกษา" ขึ้นอยู่กับบริบทของประโยค ในกรณีนี้ เราอาจต้องใช้เทคนิคการตัดคำที่มีความสามารถในการเข้าใจบริบท เช่น การใช้โมเดล Deep Learning ที่ได้รับการฝึกฝนมาแล้วสำหรับภาษาไทย ซึ่งสามารถเรียนรู้บริบทของคำและตัดคำได้อย่างแม่นยำมากขึ้น โดยเฉพาะในกรณีที่มีคำที่มีความหมายหลายอย่างหรือคำที่ไม่ค่อยพบในภาษาไทยทั่วไป 
def extract_legal_entities(text):
    entities = []
    if "สิทธิบัตร" in text:
        entities.append({"type":"IP_TYPE", "value":"PATENT","confidence":0.95})
    if "ละเมิด" in text:
        entities.append({"type":"ACTION", "value":"INFRINGEMENT","confidence":0.85})
    return entities

sample = "มีการละเมิดสิทธิบัตรในคดีนี้เพราะจำเลยได้ผลิตสินค้าที่มีลักษณะคล้ายกับสิทธิบัตรที่ถูกจดทะเบียนไว้"
found = extract_legal_entities(sample)
print(f"\n{'-'*30}Entity Extraction{'-'*30}")
for e in found:
    print(f"Type: {e['type']}, Value: {e['value']}, Confidence: {e['confidence']}")
    print(f"{'-'*80}\n")
    
    
# 3. Feature Engineering (IF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ["จำเลยกระทำความผิดฐานละเมิดสิทธิบัตรและเครื่องหมายการค้าจึงถูกฟ้องร้องในศาล",
          "คดีนี้เกี่ยวข้องกับการละเมิดสิทธิบัตรและการฟ้องร้องในศาล",
          "การละเมิดเครื่องหมายการค้าเป็นปัญหาที่พบได้บ่อยในคดีความเกี่ยวกับทรัพย์สินทางปัญญา", 
          "การกระทำความผิด ลิขสิทธิ์"
]

#สร้างตัง Vectorizer โดยใช้ Tokenizer ที่กำหนดเอง
vectorizer = TfidfVectorizer(tokenizer=legal_tokenizer, token_pattern=None) #ใช้ฟังก์ชัน legal_tokenizer ที่เราสร้างขึ้นมาเป็นตัวตัดคำ   
tfudf_matrix = vectorizer.fit_transform(corpus)
print(f"\n{'-'*30}TF-IDF Vector (Shape: {tfudf_matrix.shape})---{'-'*30}")
print(f"Vocabulary: {vectorizer.get_feature_names_out()}")
print(f"Voctor Sample (Doc 0):\n {tfudf_matrix[0].toarray()}")
print(f"{'-'*80}\n")


# 4. Physical Gate Weight 
def calculate_physical_gate_weight(entities):
    base_weight = 5.0
    for entity in entities:
        if entity['value'] == "PATENT": base_weight += 2.0 #เพิ่มน้ำหนักสำหรับสิทธิบัตร
        if entity['value'] == "INFRINGEMENT": base_weight += 1.5 #เพิ่มน้ำหนักสำหรับการละเมิด
    return (base_weight, 10.0) #คืนค่าน้ำหนักและน้ำหนักสูงสุด

#ทดสอบการคำนวณน้ำหนักจาก เอนทิตีที่ถูกสกัดออกมา
weight = calculate_physical_gate_weight(found)
print(f"\n{'-'*30}Physical Gate Bridge Weight{'-'*30}")
print(f"--Legal Context Weight: {weight[0]:.2f}/ 10.0--")
print(f"--Status: {'High Alert - Trigger Sensor' if weight[0] > 7.0 else 'Normal Monitoring'}--")
print(f"{'-'*80}\n")


   
   
    
# def legal_tokenizer(text):
#     compound = "|".join(map(re.escape,sorted(LEGAL_KEYWORD, key=len, reverse=True)))
#     pattern = (compound + r"|[\u0E00-\u0E7F]+" + r"|\[a-zA-Z0-9]+")
#     return re.findall(pattern, text)

# ตัวอย่างการใช้งาน
# test_text = "จำเลยกระทำความผิดฐานละเมิดสิทธิบัตรและเครื่องหมายการค้า ตามมาตรา 10 แห่ง พ.ร.บ.ลิขสิทธิ์ และถูกฟ้องร้องในศาล"
# tokens = legal_tokenizer(test_text)
# print(f"Input: {test_text}")
# print(f"Tokens: {tokens}")

# 2. tokenization ด้วย PyThaiNLP
# PyThaiNLP เป็นไลบรารีที่มีฟังก์ชันสำหรับ
# tokens_row = word_tokenize(protectext, engine="newmm", keep_whitespace=False)


# 3. restore original text
# ในบางกรณี เราอาจต้องการคืนค่าข้อความต้นฉบับหลังจากการตัดคำ เพื่อให้สามารถนำไปใช้ในขั้นตอนต่อไปได้อย่างถูกต้อง โดยการใช้ฟังก์ชันที่สามารถรวมคำที่ถูกตัดออกมาเป็นข้อความเดิมได้ เช่น การใช้ join() ใน Python เพื่อรวมคำที่ถูกตัดออกมาเป็นประโยคเดิม
# original_text = " ".join(tokens_row) 


# ตัดด้วย Deep Learning Model
# สำหรับการตัดคำด้วยโมเดล Deep Learning เราสามารถใช้โมเดลที่ได้รับการฝึกฝนมาแล้ว เช่น BERT หรือ LSTM ที่ถูกปรับแต่งสำหรับภาษาไทย โดยโมเดลเหล่านี้สามารถเรียนรู้บริบทของคำและตัดคำได้อย่างแม่นยำมากขึ้น โดยเฉพาะในกรณีที่มีคำที่มีความหมายหลายอย่างหรือคำที่ไม่ค่อยพบในภาษาไทยทั่วไป
# import deepcut
# print(f"Output DeepCut: {deepcut.tokenize(test_text)}")