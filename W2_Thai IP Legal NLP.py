# W2: Thai IP Legal NLP - Data Augmentation, SMOTE with Fallback, BiLSTM
# 1. คำพ้องและคำที่มีความหมายใกล้เคียงกัน (Synonyms and Near-Synonyms) ทำพจนานุกรม

import random
import torch
import torch.nn as nn
# Confusion Matrix Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


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
original_= "จำเลย ละเมิด และ จำหน่าย สินค้า"
augmented = augment_legal_text(original_)
print(f"\n{'🟢'*50}")
print(f"---Data Augmentation---")
print(f"Original: {original_}")
print(f"Augmented: {augmented}")
print(f"{'='*100}\n")

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
    print(f"{'='*100}\n")
    return X_resampled, y_resampled

# จำลองข้อมูล Imbalanced (Class 0 = 10 ตัว, Class 1 = 2 ตัว)
X_mock = np.random.rand(12, 5)  # 12 ตัวอย่าง, 5 ฟีเจอร์
y_mock = np.array([0]*10 + [1]*2)  # คลาสที่ไม่สมดุล
X_resampled, y_resampled = smote_with_fallback(X_mock, y_mock)

X_res_tensor = torch.tensor(X_resampled, dtype=torch.float32)
X_res_3d = X_res_tensor.unsqueeze(1)
y_res_tensor = torch.tensor(y_resampled, dtype=torch.float32)

print(f"Shape หลังจากทำ SMOTE (2D) : {X_resampled}")
print(f"Shape สำหรับนำเข้า (3D) : {X_res_3d.shape}")

X_train = X_res_3d

# 3.1 BiLSTM
class LegalBiLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=3):
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
sample_input = torch.rand(20, 1, 5)  # 
output = model(sample_input)
print(f"\n{'🟢'*50}")
print(f"---BiLSTM Output---")
print(f"Logits: {output.detach().numpy()}")
print(f"{'='*100}\n")


# ปรับมิติให้รองรับ LSTM , BiLSTM โดยการเพิ่มมิติตรงกลาง (Seq =1)

# print(f"X_res Shape : {X_resampled.shape}")
# X_train = X_res_3d






# 3.2 LSTM with Attention ทิศทางเดียว
class LegalLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=3):
        super(LegalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.fc.weight) # การกำหนดค่าเริ่มต้นของน้ำหนักด้วย Xavier Initialization เลือกค่าที่เหมาะสมในการเทรนรอบแรกเพื่อช่วยให้โมเดลเรียนรู้ได้ดีขึ้นและเร็วขึ้น
        
        # กำหนดฟังก์ใช้ในการเทรนโมเดล ไปข้างหน้าเราจะใช้ CrossEntropyLoss ซึ่งเหมาะสำหรับงานจำแนกประเภทหลายคลาส
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1)  # Mean Pooling เพื่อรวมข้อมูลจากทุก timestep
        return self.fc(pooled)
        
        
# 3.3 การเทรนโมเดลและการประเมินผลระหว่าง LSTM และ BiLSTM
def train_and_evaluate(model_class, name, X, y, Class_names):
    print(f"\n{'🟢'*50}")
    print(f"---Training : {name}---")
    # Convert NumPy arrays to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    model = model_class()
    # Cost-Sensitive Weights เพื่อจัดการกับความไม่สมดุลของคลาส (FN = False Negative)
    # 0 ไม่ได้ทำผิด, 1 ละเมิดสิทธิบัตร, 2 ละเมิดลิขสิทธิ์
    weights = torch.tensor([1.0, 2.0, 2.0])  # ตัวอย่างน้ำหนักสำหรับแต่ละคลาส (ปรับตามความสำคัญของแต่ละคลาส)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # การปรับค่า learning rate เป็น 0.001 เพื่อช่วยให้โมเดลเรียนรู้ได้ดีขึ้นในช่วงเริ่มต้นของการเทรน
    
    # Sample Training Loop (จำลองการเทรนด้วยข้อมูลจำลอง)
    for epoch in range(50):  # เทรน 50 รอบ
        model.train() # ตั้งโมเดลให้อยู่ในโหมดการเทรน
        optimizer.zero_grad() # เคลียร์กราฟของ gradients ก่อนการคำนวณใหม่ในแต่ละรอบ
        outputs = model(X) # เอาต์พุตของโมเดลสำหรับข้อมูลอินพุต X
        loss = criterion(outputs, y) # คำนวณค่า loss ระหว่างเอาต์พุตของโมเดลและป้ายกำกับจริง
        loss.backward() # คำนวณ gradients ของโมเดล
        optimizer.step() # อัปเดตน้ำหนักของโมเดลตาม gradients ที่คำนวณได้
    # Evaluate With Confusion Matrix (จำลองการประเมินผลด้วยข้อมูลจำลอง) CM
    model.eval() # ตั้งโมเดลให้อยู่ในโหมดการประเมินผล
    with torch.no_grad(): # ปิดการคำนวณ gradients ในช่วงการประเมินผลเพื่อประหยัดหน่วยความจำและเพิ่มความเร็ว
        y_pred = torch.argmax(model(X), dim=1) # ทำนายคลาสที่มีความน่าจะเป็นสูงสุดสำหรับแต่ละตัวอย่างใน X
    # สร้าง Confusion Matrix
    cm = confusion_matrix(y.numpy(), y_pred.numpy()) # สร้าง confusion matrix โดยใช้ป้ายกำกับจริงและป้ายกำกับที่ทำนาย

    # Heapmap Visualization
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples'if 'Bi' in name else 'Greens', xticklabels=Class_names, yticklabels=Class_names) # สร้าง heatmap จาก confusion matrix โดยแสดงค่าตัวเลขในแต่ละช่องและใช้สีฟ้าเพื่อแสดงความเข้มของค่าต่างๆ
    plt.title(f'Confusion Matrix - {name}') # ตั้งชื่อกราฟด้วยชื่อของโมเดลที่กำลังประเมินผล
    plt.xlabel('Predicted') # ตั้งชื่อแกน x เป็น 'Predicted' เพื่อแสดงว่าค่าบนแกนนี้เป็นค่าที่โมเดลทำนาย
    plt.ylabel('Actual') # ตั้งชื่อแกน y เป็น 'Actual' เพื่อแสดงว่าค่าบนแกนนี้เป็นค่าจริงจากข้อมูล
    plt.show() # แสดงกราฟ confusion matrix

# ต้องการรันเปรียบเทียบระหว่าง Training & Evaluation ของ LSTM และ BiLSTM ด้วยข้อมูลจำลอง
Class_list = ['NO-INF', 'Patent-INF', 'Copyright-INF']
    # X_train, y_train = กรณียังไม่ผ่าน MODEL TRAINING
    # X_resampled, y_resampled = smote_with_fallback(X_train, y_train) # ใช้ SMOTE กับ Fallback เพื่อปรับสมดุลของข้อมูลก่อนการเทรน
report_lstm = train_and_evaluate(LegalLSTM, "Unidirectional LSTM", X_train, y_resampled, Class_list) # เทรนและประเมินผลโมเดล LSTM 1 ทิศทาง
report_bilstm = train_and_evaluate(LegalBiLSTM, "Bidirectional LSTM", X_train, y_resampled, Class_list) # เทรนและประเมินผลโมเดล BiLSTM 2 ทิศทาง


# ตรวจสอบ input Shape ของโมเดล
print(f"\n{'🟢'*50}")
print(f"---Input Shape Check---")
print(f"Sample X_resampled Shape : {X_resampled.shape}") # ตรวจสอบรูปแบบของข้อมูลที่ถูกปรับสมดุลแล้ว
print(f"{'='*100}\n")

