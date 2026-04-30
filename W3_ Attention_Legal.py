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
    def __init__(self, d_model=16, n_heads=4, seed=1):
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
    
    
    # การทำ Pre-Norm Multi-Head Attention แก้ปัญหา Vanishing Gradient รองรับเกิน 100 Lyers
    def layer_norm(self, x):
        return (x - x.mean())/(x.std()+1e-6)
    
    
    
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
        
# 4 Transformer Encoder (BERT) and Decode (GPT)
def _print_heatmap_(W,T,width=4):
    print("  " + "".join(f" t{j}" for j in range(T)))
    for i in range(T):
        row = f" t{i} "
        for j in range(T):
            row += " " + ("█" * int(W[i, j] * width)).ljust(width)
        print(row)
    

    # 4.1 Feed Forward Network (FFN)
class FeedFordward:
    def __init__(self, d_model, d_ff=None, seed = 1):
        rag = np.random.RandomState(seed)
        d_ff = d_ff or d_model * 4
        s = np.sqrt(2.0/d_model)


# Code From AJ
# ============================================================
# 1. Sinusoidal Positional Encoding
# ============================================================
class SinusodalPositionEncoding:
    def __init__(self, max_seq_len=10, d_model=16):
        pe = np.zeros((max_seq_len, d_model))
        pos = np.arange(max_seq_len).reshape(-1, 1)
        div = np.power(10000.0, np.arange(0, d_model, 2) / d_model)
        pe[:, 0::2] = np.sin(pos / div)
        pe[:, 1::2] = np.cos(pos / div)
        self.pe = pe

    def encode(self, X):                          # เพิ่ม method นี้
        return X + self.pe[:X.shape[0], :]

    def show(self, seq_len=5):
        print(f"Position Encoding (First {seq_len} tokens)")
        for p in range(seq_len):
            print(f"Pos : {p} " + " ".join(f"{v:.2f}" for v in self.pe[p, :10]) + "...")

    def show_with_words(self, words):
        print(f"{'index':<7} | {'word':<10} | {'Positional Encoding (First 4 dims)':<40}")
        for i, word in enumerate(words):
            if i >= len(self.pe):
                break
            vec = self.pe[i, :10]
            vec_str = " ".join(f"{v:+.3f}" for v in vec)
            print(f"Pos_{i:<3} | {word:<10}| [{vec_str}...]")


# ============================================================
# 2. Scaled Dot-Product Attention + Causal/Padding Mask
# ============================================================
def scale_dot_product_attention(Q, K, V, mask=None, causal_mask=None):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    if causal_mask is not None:                   # บัง token อนาคต (GPT)
        scores = np.where(causal_mask, -1e9, scores)

    if mask is not None:                          # padding mask (BERT)
        scores = np.where(mask == 0, -1e9, scores)

    exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_s / exp_s.sum(axis=-1, keepdims=True)
    return weights @ V, weights


# ============================================================
# 3. Multi-Head Attention — แก้ไข forward() ให้สมบูรณ์
# ============================================================
class MultiHeadAttentionSimple:
    def __init__(self, d_model=16, n_heads=4, seed=42):
        rng = np.random.RandomState(seed)
        self.W_q = rng.randn(d_model, d_model) * 0.1
        self.W_k = rng.randn(d_model, d_model) * 0.1
        self.W_v = rng.randn(d_model, d_model) * 0.1
        self.W_o = rng.randn(d_model, d_model) * 0.1
        self.n_heads = n_heads
        self.d_k = int(d_model // n_heads)

    def split_heads(self, x):
        batch, seq_len, d_model = x.shape
        return x.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        batch, heads, seq_len, d_k = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch, seq_len, heads * d_k)

    def layer_norm(self, x):
        return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)

    def forward(self, x, padding_mask=None, causal_mask=None, return_weights=False):
        # รับ shape (seq, d_model) หรือ (batch, seq, d_model)
        if x.ndim == 2:
            x = x[np.newaxis, :]          # เพิ่ม batch dim → (1, seq, d_model)

        # [Pre-Norm] normalize ก่อนเข้า attention
        x_norm = self.layer_norm(x)

        batch, seq, _ = x.shape

        # Linear projection → แยก heads
        Q = self.split_heads(np.matmul(x_norm, self.W_q))  # (batch,heads,seq,d_k)
        K = self.split_heads(np.matmul(x_norm, self.W_k))
        V = self.split_heads(np.matmul(x_norm, self.W_v))

        # reshape เพื่อคำนวณ attention พร้อมกันทุก head
        Q_r = Q.reshape(batch * self.n_heads, seq, self.d_k)
        K_r = K.reshape(batch * self.n_heads, seq, self.d_k)
        V_r = V.reshape(batch * self.n_heads, seq, self.d_k)

        # ขยาย causal_mask ให้ครอบทุก head
        cm = None
        if causal_mask is not None:
            cm = np.tile(causal_mask[np.newaxis], (batch * self.n_heads, 1, 1))

        attn_out, attn_weights = scale_dot_product_attention(
            Q_r, K_r, V_r, mask=padding_mask, causal_mask=cm
        )

        # reshape กลับ → combine heads → output projection
        attn_weights = attn_weights.reshape(batch, self.n_heads, seq, seq)
        attn_out = self.combine_heads(
            attn_out.reshape(batch, self.n_heads, seq, self.d_k)
        )
        output = np.matmul(attn_out, self.W_o)

        # residual connection (บวก x เดิม ไม่ใช่ x_norm)
        output = output + x

        # คืน shape (seq, d_model) ถ้า input เดิมเป็น 2D
        output = output.squeeze(0)
        attn_weights = attn_weights.squeeze(0)

        return (output, attn_weights) if return_weights else (output, attn_weights)


# ============================================================
# 4. Helpers
# ============================================================
def layer_norm(X):
    return (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)

def _print_heatmap(W, T, width=4):
    print("   " + "".join(f"  t{j}" for j in range(T)))
    for i in range(T):
        row = f"  t{i}  "
        for j in range(T):
            row += " " + ("█" * int(W[i, j] * width)).ljust(width)
        print(row)


# ============================================================
# 5. Feed-Forward Network
# ============================================================
class FeedForward:
    def __init__(self, d_model, d_ff=None, seed=42):
        rng = np.random.RandomState(seed)
        d_ff = d_ff or d_model * 4
        s = np.sqrt(2.0 / d_model)
        self.W1 = rng.randn(d_ff, d_model) * s
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.randn(d_model, d_ff) * s
        self.b2 = np.zeros(d_model)

    def forward(self, X):
        return np.maximum(0, X @ self.W1.T + self.b1) @ self.W2.T + self.b2


# ============================================================
# 6. Encoder Block (BERT)
# ============================================================
class TransformerEncoderBlock:
    def __init__(self, d_model=32, n_heads=4, seed=42):
        self.mha = MultiHeadAttentionSimple(d_model, n_heads, seed=seed)
        self.ffn = FeedForward(d_model, seed=seed + 1)

    def forward(self, X, padding_mask=None, return_weights=False):
        attn_out, hw = self.mha.forward(X, padding_mask=padding_mask,
                                         causal_mask=None,
                                         return_weights=return_weights)
        x1 = X + attn_out
        x2 = x1 + self.ffn.forward(layer_norm(x1))
        return x2, hw


# ============================================================
# 7. Decoder Block (GPT)
# ============================================================
class TransformerDecoderBlock:
    def __init__(self, d_model=32, n_heads=4, seed=42):
        self.mha = MultiHeadAttentionSimple(d_model, n_heads, seed=seed)
        self.ffn = FeedForward(d_model, seed=seed + 1)

    @staticmethod
    def _causal_mask(T):
        return np.triu(np.ones((T, T), dtype=bool), k=1)

    def forward(self, X, padding_mask=None, return_weights=False):
        T = X.shape[0]
        attn_out, hw = self.mha.forward(X, causal_mask=self._causal_mask(T),
                                         return_weights=return_weights)
        x1 = X + attn_out
        x2 = x1 + self.ffn.forward(layer_norm(x1))
        return x2, hw


# ============================================================
# 8. MiniBERT
# ============================================================
class MiniBERT:
    def __init__(self, input_size, d_model=32, n_heads=4, n_layers=2, n_classes=3, seed=42):
        rng = np.random.RandomState(seed)
        s = np.sqrt(2.0 / input_size)
        self.W_proj = rng.randn(d_model, input_size) * s
        self.pe = SinusodalPositionEncoding(512, d_model)
        self.layers = [TransformerEncoderBlock(d_model, n_heads, seed=seed + i)
                       for i in range(n_layers)]
        self.W_out = rng.randn(n_classes, d_model) * s
        self.b_out = np.zeros((n_classes, 1))

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def forward(self, x_seq, padding_mask=None, return_weights=False):
        X = self.pe.encode(x_seq @ self.W_proj.T)
        all_weights = []
        for layer in self.layers:
            X, hw = layer.forward(X, padding_mask=padding_mask,
                                   return_weights=return_weights)
            if return_weights:
                all_weights.append(hw)
        ctx = X.mean(axis=0)
        probs = self._softmax((self.W_out @ ctx.reshape(-1, 1) + self.b_out).flatten())
        return probs, all_weights


# ============================================================
# 9. MiniGPT
# ============================================================
class MiniGPT:
    def __init__(self, input_size, d_model=32, n_heads=4, n_layers=2, seed=42):
        rng = np.random.RandomState(seed)
        s = np.sqrt(2.0 / input_size)
        self.W_proj = rng.randn(d_model, input_size) * s
        self.pe = SinusodalPositionEncoding(512, d_model)
        self.layers = [TransformerDecoderBlock(d_model, n_heads, seed=seed + i)
                       for i in range(n_layers)]
        self.W_lm = rng.randn(input_size, d_model) * s

    def forward(self, x_seq, return_weights=False):
        X = self.pe.encode(x_seq @ self.W_proj.T)
        all_weights = []
        for layer in self.layers:
            X, hw = layer.forward(X, return_weights=return_weights)
            if return_weights:
                all_weights.append(hw)
        return self.W_lm @ X[-1], all_weights


# ============================================================
# MAIN DEMO
# ============================================================
def print_section(title, char="="):
    print(f"\n{char*58}\n  {title}\n{char*58}")


def run_demo():
    d_model   = 32
    n_heads   = 4
    seq_len   = 4
    input_dim = 8

    rng = np.random.RandomState(0)
    X_sample = rng.randn(seq_len, input_dim) * 0.1   # (4, 8)

    # Project + PE ก่อนใช้ใน STEP 2-3
    W_proj = rng.randn(d_model, input_dim) * 0.1
    pe = SinusodalPositionEncoding(512, d_model)
    X_emb = pe.encode(X_sample @ W_proj.T)            # (4, 32)

    # ── STEP 1 ──────────────────────────────────────────────
    print_section("STEP 1: Encoder Block — สร้างและแสดง config")
    enc_block = TransformerEncoderBlock(d_model=d_model, n_heads=n_heads, seed=42)
    print(f"  d_model={d_model}, n_heads={n_heads}, d_k={enc_block.mha.d_k}")

    # ── STEP 2 ──────────────────────────────────────────────
    print_section("STEP 2: Encoder Block (BERT) — Full Attention")
    enc_out, enc_ws = enc_block.forward(X_emb, return_weights=True)
    print(f"  Input  shape : {X_emb.shape}")
    print(f"  Output shape : {enc_out.shape}")
    print(f"\n  Head 1 — Full Attention:")
    _print_heatmap(enc_ws[0], seq_len)

    # ── STEP 3 ──────────────────────────────────────────────
    print_section("STEP 3: Decoder Block (GPT) — Causal Mask")
    dec_block = TransformerDecoderBlock(d_model=d_model, n_heads=n_heads, seed=42)
    dec_out, dec_ws = dec_block.forward(X_emb, return_weights=True)
    print(f"\n  Head 1 — Causal Masked:")
    _print_heatmap(dec_ws[0], seq_len)
    print(f"\n  Causal Mask pattern:")
    mask = TransformerDecoderBlock._causal_mask(seq_len)
    print("  pos  " + "".join(f"  t{j}" for j in range(seq_len)))
    for i in range(seq_len):
        row = "".join("  ✓" if not mask[i, j] else "  ✗" for j in range(seq_len))
        print(f"   t{i}  {row}")

    # ── STEP 4 ──────────────────────────────────────────────
    print_section("STEP 4: MiniBERT — 2-Layer Encoder")
    bert = MiniBERT(input_size=input_dim, d_model=d_model,
                    n_heads=n_heads, n_layers=2, n_classes=3, seed=42)
    probs, bert_ws = bert.forward(X_sample, return_weights=True)
    print(f"  Class probabilities : {probs.round(4)}")
    print(f"  Predicted class     : {np.argmax(probs)}")
    print(f"\n  Layer 1 — Head 1:")
    _print_heatmap(bert_ws[0][0], seq_len)
    print(f"\n  Layer 2 — Head 1:")
    _print_heatmap(bert_ws[1][0], seq_len)

    # ── STEP 5 ──────────────────────────────────────────────
    print_section("STEP 5: MiniGPT — 2-Layer Decoder")
    gpt = MiniGPT(input_size=input_dim, d_model=d_model,
                  n_heads=n_heads, n_layers=2, seed=42)
    logits, gpt_ws = gpt.forward(X_sample, return_weights=True)
    print(f"  Next-token logits : {logits.round(4)}")
    print(f"\n  Layer 1 — Head 1 (Causal):")
    _print_heatmap(gpt_ws[0][0], seq_len)
    print(f"\n  Layer 2 — Head 1 (Causal):")
    _print_heatmap(gpt_ws[1][0], seq_len)

    # ── STEP 6 ──────────────────────────────────────────────
    print_section("STEP 6: Encoder vs Decoder — สรุป")
    print("""
  ┌─────────────────────┬──────────────────────┬──────────────────────┐
  │                     │  Encoder (BERT)       │  Decoder (GPT)       │
  ├─────────────────────┼──────────────────────┼──────────────────────┤
  │ Attention Mask      │ Full (ไม่มี mask)     │ Causal (บังอนาคต)   │
  │ token t มองเห็น     │ ทุก token             │ t0 .. t เท่านั้น    │
  │ Output              │ Mean pool → classify  │ Last token → predict │
  │ งานหลัก             │ Classification, NER   │ Text generation      │
  │ ตัวอย่าง            │ BERT, RoBERTa         │ GPT-2, GPT-4         │
  └─────────────────────┴──────────────────────┴──────────────────────┘
    """)


if __name__ == "__main__":
    run_demo()
    
    
# 30 มิ.ย. 69
# XAI Analysis การตีความค่า Weight Attention <-- อธิบายได้ (Explainable AI) เพื่อวิเคราะห์ว่าคำไหนมีความสำคัญในการตัดสินคดี
# ============================================================
# XAI Analysis
# ============================================================
def explainable_attention(tokens, weight):
    # ค่าเฉลี่ย Head 0
    avg_weight = weight[0].mean(axis=0)
    # print(f"\n{'🟢'*50}")
    print(f"--- XAI : Legal Importance Analysis ---")
    for i , token in enumerate(tokens):
        importance = avg_weight[i]
        bar = "|" *int(importance*20)
        print(f"{token:<15} | {bar} ({importance:.2f})")
        
# ============================================================
# Pre-Norm ทดสอบตรงๆ
# ============================================================
def test_prenorm():
    print("\n--------- Pre-Norm ---------")
    mha = MultiHeadAttentionSimple(d_model=16, n_heads=4)
    sample_input = np.random.randn(4, 16)
    output, _ = mha.forward(sample_input)
    print(f"Input Shape : {sample_input.shape} -> Output Shape {output.shape} (Success)")
    # print(f"\n{'🟢'*50}")


if __name__ == "__main__":
    # XAI demo
    tokens = ["จำเลย", "ละเมิด", "สิทธิบัตร", "การประดิษฐ์"]

    mock_weight = np.array([[[0.1, 0.4, 0.4, 0.1]]])
    explainable_attention(tokens, mock_weight)

    mock_3heads = np.array([[[0.1, 0.5, 0.3, 0.1],
                              [0.2, 0.2, 0.5, 0.1],
                              [0.4, 0.2, 0.2, 0.2]]])
    print(f"--- 1 Head  Shape (Batch,Head,Seq) --- : {mock_weight.shape}")
    print(f"--- 3 Heads Shape (Batch,Head,Seq) --- : {mock_3heads.shape}")
    print(f"--- 3 Heads Average ---")
    explainable_attention(tokens, mock_3heads)


# Pre-Norm test
test_prenorm()

# Full demo
run_demo()