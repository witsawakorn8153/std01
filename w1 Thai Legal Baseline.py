# 1. การตัดคำ Tokenization + Custom Dictionary

import re
from pythainlp import word_tokenize
from pythainlp.tokenize import word_tokenize

LEGAL_KEYWORD = ["ละเมิดสิทธิบัตร", "เครื่องหมายการค้า", "คำพิพากษา", "มาตรา", "พ.ร.บ.", "กฎหมาย", "ศาล", "คดี", "ข้อหา", "บทลงโทษ", "ความรับผิด", "การฟ้องร้อง", "การดำเนินคดี", "การละเมิดลิขสิทธิ์", "การละเมิดเครื่องหมายการค้า", "การละเมิดสิทธิ์", "การละเมิดกฎหมาย", "การละเมิดข้อหา", "การละเมิดบทลงโทษ", "การละเมิดความรับผิด", "การละเมิดการฟ้องร้อง", "การละเมิดการดำเนินคดี"]


def legal_tokenizer(text):
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
print(f"Input: {test_text}")
print(f"Output: {tokens}")

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
print(f"---Entity Extraction---")
for e in found:
    print(f"Type: {e['type']}, Value: {e['value']}, Confidence: {e['confidence']}")
    
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
print(f"---TF-IDF Vector (Shape: {tfudf_matrix.shape})---") 
print(f"Vocabulary: {vectorizer.get_feature_names_out()}")
print(f"Voctor Sample (Doc 0):\n {tfudf_matrix[0].toarray()}")


# 4. Physical Gate Weight 
def calculate_physical_gate_weight(entities):
    base_weight = 5.0
    for entity in entities:
        if entity['value'] == "PATENT": base_weight += 2.0 #เพิ่มน้ำหนักสำหรับสิทธิบัตร
        if entity['value'] == "INFRINGEMENT": base_weight += 1.5 #เพิ่มน้ำหนักสำหรับการละเมิด
    return (base_weight, 10.0) #คืนค่าน้ำหนักและน้ำหนักสูงสุด

#ทดสอบการคำนวณน้ำหนักจาก เอนทิตีที่ถูกสกัดออกมา
weight = calculate_physical_gate_weight(found)
print(f"--Physical Gate Bridge Weight--")
print(f"--Legal Context Weight: {weight[0]:.2f}/ 10.0--")
print(f"--Status: {'High Alert - Trigger Sensor' if weight[0] > 7.0 else 'Normal Monitoring'}--")


   
   
    
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
# 