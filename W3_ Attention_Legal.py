# Sinusoidal Positional Embeddings การเข้ารหัสตำแหน่งแบบไซนูซอยดัล เพื่อให้โมเดลเข้าใจลำดับของคำในประโยค
import numpy as np
class SinusoidalPositionalEmbedding: # การจัดลำดับของคำ
    def __init__(self, max_seq_len = 10, d_model = 16):
        pe = np.zeros((max_seq_len, d_model))
        pos = np.arange(max_seq_len).reshape(-1, 1)
        div = np.power(10000.0, np.arange(0, d_model, 2) / d_model)
        pe[:, 0::2] = np.sin(pos / div)
        pe[:, 1::2] = np.cos(pos / div)
        self.positional_encoding = pe
        
    def show(self, seq_len=5):
        print(f"{'-'*30}Sinusoidal Positional Encoding (First {seq_len}) tokens{'-'*30}")
        for p in range(seq_len):
            print(f"Position : {p}"+" ".join(f"{v:.2f} " for v in self.positional_encoding[p, :4])+"...")
            
    def show_with_words(self, words):
        print(f"{'Index':<7} | {'Word':<10} | {'Positional Encoding (First 4 dims)':<40}")
        for i, word in enumerate(words):
            if i >= len(self.positional_encoding):
                break
            vec = self.positional_encoding[i, :10]  # แสดงแค่ 10 มิติแรกเพื่อความกระชับ
            vec_str = " ".join(f"{v:.3f}" for v in vec)
            print(f"Pos_{i:<3} | {word : <10} | {vec_str : <40}")

        
# การใช้งาน Sinusoidal Positional Embeddings ในการประมวลผลข้อความทางกฎหมาย
word_list = ["I", "love", "learning", "AI", "Techonology"]      
print(f"\n{'🟢'*50}")            
pe = SinusoidalPositionalEmbedding(max_seq_len=10, d_model=16)
pe.show_with_words(word_list)
print(f"{'='*100}\n")


# 2. Scale Dot-Product Attention and Padding Mask ให้ความยาวของเอกสารเท่ากัน (อัพเดตข้อมูลล่าสุด 27 เมษายน 2024)
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    # scores = (Q @ K.T) / np.sqrt(d_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)  # batch matrix multiplication
    if mask is not None:
        # เราจะกำหนดให้ตำแหน่งที่เป็น 0 ใน mask มีค่า score เป็น -inf เพื่อปิดกั้นตำแหน่งนั้น
        scores = np.where(mask == 0, -1e9, scores)
    weights = np.exp(scores - np.max(scores))  # ใช้เทคนิคการลบ max เพื่อป้องกัน overflow
    weights /= weights.sum(axis=-1, keepdims=True)  # Normalize weights
    return weights @ V, weights

# จำลองข้อมูล 1 ประโยค 3 Tokens, Vector 4 มิติ ทดสอบการทำงานของ Attention
q = k = v = np.random.rand(1, 3, 4)  # batch_size=1, seq_len=3, d_model=4
output, weights = scaled_dot_product_attention(q, k, v)
print(f"\n{'🟢'*50}")
print(f"\n{'-'*35}Attention Weights 3x3 Matrix{'-'*35}")
print(weights[0].round(2))
print(f"{'='*100}\n")

# 27 เมษายน 2024 - อัพเดตข้อมูลล่าสุดเกี่ยวกับการทำงานของ Attention และกาารใช้ Masking ในการประมวลผลข้อความทางกฎหมา เพื่อให้โมเดลสามารถจัดการกับความยาวของเอกสารที่แตกต่างกันได้อย่างมีประสิทธิภาพมากขึ้น

# 29 เมษายน 2024 - Update
# 3.1 Multi-Head Attention (การมองหลายมุมมอง) ช่วยให้เข้าใจได้พร้อมความสัมพันธ์หลายรูปแบบพร้อมกัน
class MultiHeadAttentionsSimple:
    def __init__(self, d_model=16, n_heads=4):
        # d_k = q.shape[-1]
        self.n_heads = n_heads
        self.d_k = int(d_model // n_heads)
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.shape
        return x.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0,2,1,3)

# จำลอง Input ขนาด 16 มิติ (d) แบ่งเป็น 4 Head (Head ละ 4 dim)
input_data = np.random.randn(1,5,16)
mha = MultiHeadAttentionsSimple()
heads = mha.split_heads(input_data)
print(f"\n{'🟢'*50}\n")
print(f"Origin Shape : {input_data.shape}")
print(f"Heads Shape : {heads.shape} (Batch, Heads, Seq_len, Depth)")
print(f"{'='*100}\n")
        
# 3.2 