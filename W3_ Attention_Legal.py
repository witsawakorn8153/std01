# Sinusoidal Positional Embeddings การเข้ารหัสตำแหน่งแบบไซนูซอยดัล เพื่อให้โมเดลเข้าใจลำดับของคำในประโยค
import numpy as np

class SinusodalPositionalEmbedding: # การจัดลำดับของคำ
    def __init__(self, max_seq_len = 10, d_model = 16):
        pe = np.zeros((max_seq_len, d_model))
        pos = np.arange(max_seq_len).reshape(-1, 1)
        div = np.power(10000.0, np.arange(0, d_model, 2) / d_model)
        pe[:,0::2] = np.sin(pos/div)
        pe[:, 1::2] = np.cos(pos/div)
        self.pe = pe
        
    def show(self, seq_len=5):
        print(f"{'-'*30}Sinusodal Positional Encoding (First {seq_len}) tokens{'-'*30}")
        for p in range(seq_len):
            print(f"Position : {p}"+" ".join(f"{v:.2f} " for v in self.pe[p, :4])+"...")
            
    def show_with_words(self, words):
        print(f"{'Index':<7} | {'Word':<10} | {'Positional Encoding (First 4 dims)':<40}")
        for i, word in enumerate(words):
            if i >= len(self.pe):
                break
            vec = self.pe[i, :10]  # แสดงแค่ 10 มิติแรกเพื่อความกระชับ
            vec_str = " ".join(f"{v:.3f}" for v in vec)
            print(f"Pos_{i:<3} | {word : <10} | {vec_str : <40}")

        
# การใช้งาน Sinusoidal Positional Embeddings ในการประมวลผลข้อความทางกฎหมาย
word_list = ["I", "love", "learning", "AI", "Techonology"]      
print(f"\n{'🟢'*50}")            
pe = SinusodalPositionalEmbedding(max_seq_len=10, d_model=16)
pe.show_with_words(word_list)
pe = SinusodalPositionalEmbedding()
pe.show()
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
q = k = v = np.random.rand(1, 3, 4)  # batch=1, seq_len=3, d_model=4
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
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
        
        # d_k = q.shape[-1]
        self.n_heads = n_heads
        self.d_k = int(d_model // n_heads)
    def split_heads(self, x):
        batch, seq_len, d_model = x.shape
        x = x.reshape(batch, seq_len,self.n_heads,self.d_k)
        return x.transpose(0,2,1,3)
    
    def combine_heads(self, x): # การรวม Head กลับ (Batch, Heads, seq, d_k)
        batch,heads,seq_len,self.d_k = x.shape
        x = x.transpose(0,2,1,3)
        return x.reshape(batch, seq_len, heads*self.d_k)
    
    def forward(self, x):
        # 1. Linear Projection ก่อนแยก Head
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        # 2. แยกออกเป็น 4 หัว
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        # 3. Attention แต่ละ Head-Reshape เพื่อให้ batch รวมกัน
        batch = x.shape[0]
        Q_r= Q.reshape(batch * self.n_heads, x.shape[1], self.d_k)
        K_r = K.reshape(batch * self.n_heads, x.shape[1], self.d_k)
        V_r = V.reshape(batch * self.n_heads, x.shape[1], self.d_k)
        attn_out,attn_weigths = scaled_dot_product_attention(Q_r,K_r,V_r)
    # Reshape Attention Weight --> (batch, heads, seq)  
        attn_weigths = attn_weigths.reshape(batch,self.n_heads, x.shape[1], x.shape[1])
        # Step 4 : รวม Heads กลับ
        attn_out = attn_out.reshape(batch,self.n_heads, x.shape[1], self.d_k)
        concat = self.combine_heads(attn_out)
        # 5. การทำ Output Projection
        output = np.matmul(concat,self.W_o)
        return output , attn_weigths       
        # return x.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0,2,1,3)

# จำลอง Input ขนาด 16 มิติ (d) แบ่งเป็น 4 Head (Head ละ 4 dim)
input_data = np.random.randn(1,5,16)
mha = MultiHeadAttentionsSimple()
heads = mha.split_heads(input_data)
print(f"\n{'🟢'*50}\n")
print(f"Origin Shape : {input_data.shape}")
print(f"Heads Shape : {heads.shape} (Batch, Heads, Seq_len, Depth)")
print(f"{'='*100}\n")
        
# Positional
# def positional_encoding(seq_len)        
        
# 3.2 Postional Encoding Multi-Head Attention
d_model = 16
n_heads = 4
seq_len = 5
batch = 1
# Step 1 : Random Input Embeding
np.random.seed(1)
token_embeddings = np.random.randn(batch, seq_len, d_model)
# Step 2 : บวก Positional Encoding
pe_encoder = SinusodalPositionalEmbedding(max_seq_len=10, d_model=d_model)
pe_encoder.show(seq_len)
x = token_embeddings + pe_encoder.pe[:seq_len]
# Step 3 : Multi Head Att
mha = MultiHeadAttentionsSimple(d_model=d_model, n_heads=n_heads)
output, attn_weights = mha.forward(x)

# แสดงผล
print(f"\n{'🟢'*50}")
print(f"--MultiHead Attention (4 Head)")
print(f"Input Shape : {x.shape}")
print(f"Output Shape : {output.shape}")
print(f"Weight Shape : {attn_weights.shape}")
for h in range(n_heads):
    print(f"\nHead {h+1} Attention Weight :")
    for row in attn_weights[0,h]:
        print(" " , " ".join(f"{v: .3f}" for v in row))
