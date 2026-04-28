# W2: Thai IP Legal NLP - Data Augmentation, SMOTE with Fallback, BiLSTM
# 1. คำพ้องและคำที่มีความหมายใกล้เคียงกัน (Synonyms and Near-Synonyms) ทำพจนานุกรม

import random

LEGAL_SYNONYMS = {
 "ละเมิด": ["กระทำผิด", "ฝ่าฝืน", "ล่วงสิทธิ์"],
 "จำหน่าย": ["ขาย", "เผยแพร่", "กระจายสินค้า"],
 "ปลอมแปลง": ["ปลอม", "เลียนแบบ", "ทำเลทียม"]   
}

def augment_legal_text(text):
    word = text.split()
    new_words = word.copy()
    for i, word in enumerate(word):
        if word in LEGAL_SYNONYMS:
            new_words[i] = random.choice(LEGAL_SYNONYMS[word])
    return " ".join(new_words)

# ทดสอบตัวอย่างใช้งาน
original_= "จำเลย ละเมิด และ จำหน่าย สินค้าที่มีลักษณะคล้ายกับสิทธิบัตรที่ถูกจดทะเบียนไว้"
augmented = augment_legal_text(original_)
print(f"---Data Augmentation---")
print(f"Original: {original_}")
print(f"Augmented: {augmented}")

# SMOTE with Failback แก้ปัญหาการลำเอียง 
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter
def smote_with_fallback(X, y):
    Counts = Counter(y)
    print(f"Original Class Distribution: {Counts}")
    
    # เช็คว่ามีคลาสที่มีน้อยที่สุดมีกี่ตัว
    min_samples = min(Counts.values())
    if min_samples > 1:
        sampler = SMOTE(k_neighbors=min(5, min_samples-1),random_state=1)
    else:
        sampler = RandomOverSampler(random_state=1)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    print(f"Balanced Distribution: {Counter(y_resampled)}")
    return X_resampled, y_resampled

# จำลองข้อมูล Imbalanced (Class 0 = 10 ตัว, Class 1 = 2 ตัว)
X_mock = np.random.rand(12, 5)  # 12 ตัวอย่าง, 5 ฟีเจอร์
y_mock = np.array([0]*10 + [1]*2)  # คลาสที่ไม่สมดุล
X_resampled, y_resampled = smote_with_fallback(X_mock, y_mock)

# 3. BiLSTM
import torch
import torch.nn as nn

class LegalBiLSTM(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, output_dim=3):
        super(LegalBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)  # *2 เพราะ BiLSTM มี 2 ทิศทาง
        nn.init.xavier_uniform_(self.fc.weight)  # Xavier Initialization
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # ใช้เอาต์พุตจาก timestep สุดท้าย
        # Mean Pooling
        pooled = torch.mean(lstm_out, dim=1)
        return self.fc(pooled)
    
# ทดสอบโมเดลด้วยข้อมูลจำลอง
model = LegalBiLSTM()
sample_input = torch.rand(4, 10, 16)  # batch_size=4, seq_len=10, input_dim=16 # 1 Doc 5 Tokens 16 Dimensions
output = model(sample_input)
print(f"---BiLSTM Output---")
print(f"Logits: {output.detach().numpy()}")
 
        
    
        
        
    
