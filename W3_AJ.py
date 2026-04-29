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
# 3. Multi-Head Attention (รับ causal_mask และ padding_mask)
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

    def forward(self, x, padding_mask=None, causal_mask=None, return_weights=False):
        # รับ x shape: (seq, d_model) หรือ (batch, seq, d_model)
        if x.ndim == 2:
            x = x[np.newaxis, :]               # เพิ่ม batch dim

        batch, seq, _ = x.shape
        Q = self.split_heads(np.matmul(x, self.W_q))
        K = self.split_heads(np.matmul(x, self.W_k))
        V = self.split_heads(np.matmul(x, self.W_v))

        Q_r = Q.reshape(batch * self.n_heads, seq, self.d_k)
        K_r = K.reshape(batch * self.n_heads, seq, self.d_k)
        V_r = V.reshape(batch * self.n_heads, seq, self.d_k)

        # ขยาย causal_mask ให้ครอบทุก head
        cm = None
        if causal_mask is not None:
            cm = np.tile(causal_mask[np.newaxis], (batch * self.n_heads, 1, 1))

        attn_out, attn_weights = scale_dot_product_attention(Q_r, K_r, V_r,
                                                              mask=padding_mask,
                                                              causal_mask=cm)
        attn_weights = attn_weights.reshape(batch, self.n_heads, seq, seq)
        attn_out = self.combine_heads(attn_out.reshape(batch, self.n_heads, seq, self.d_k))
        output = np.matmul(attn_out, self.W_o)

        # คืน shape (seq, d_model) ถ้า input เป็น 2D
        return output.squeeze(0), attn_weights.squeeze(0)


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