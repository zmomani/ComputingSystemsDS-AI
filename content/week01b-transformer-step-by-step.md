---
week: 1
title: "Transformer — Complete Step-by-Step Guide with One Example"
source: https://albanna-tutorials.com/transformer-step-by-step.html
---
# Transformer — Complete Step-by-Step Guide with One Example

*Intermediate · ~30 minutes · AI Applications*

*Material by Dr. Abdulkarim Albanna (Dr. Abedal-Kareem Al-Banna), Assistant Professor, Data Science & AI, University of Petra — ilmy / albanna-tutorials.com. Reproduced here for teaching use with the explanatory prose kept as close to the original; the worked-example arithmetic has been checked and recomputed where necessary.*

In this guide we follow **one example sentence** through the whole Transformer, computing every stage by hand with small example numbers:

> **Input (English):** I love learning
> **Output (Arabic):** أنا أحب التعلم

Real models use large vectors (d_model = 512). To keep the arithmetic readable, the example uses vectors of size 4 (embedding) and size 3 (attention).

---

## 1. Input Embedding

Each word is first converted into a vector of numbers (its **embedding**). These values are learned during training, so words with similar meanings end up with similar vectors.

In the example each word becomes a vector of **4 dimensions** (real models use 512):

```text
"I"        → [0.2, 0.5, 0.1, 0.8]
"love"     → [0.9, 0.3, 0.7, 0.2]
"learning" → [0.4, 0.6, 0.5, 0.9]
```

---

## 2. Positional Encoding

The Transformer processes all words **in parallel**, so it has no built-in notion of word order. To tell the model *where* each word is, a position vector is **added** to each embedding.

The position vectors are built with sine and cosine functions:

```text
PE(pos, 2i)   = sin( pos / 10000^(2i/d) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d) )
```

- `pos` = position of the word in the sentence (0, 1, 2, …)
- `i` = index of the dimension
- `d` = model dimension

Early dimensions change quickly from one position to the next and later dimensions change slowly — like the hands of a clock — so every position gets a unique pattern.

Positional encoding vectors for our 3 positions (d = 4):

```text
PE₀ ("I",        pos 0) → [0.00,  1.00, 0.00, 1.00]
PE₁ ("love",     pos 1) → [0.84,  0.54, 0.01, 1.00]
PE₂ ("learning", pos 2) → [0.91, -0.42, 0.02, 1.00]
```

Embedding + Positional Encoding:

```text
"I"        → [0.2, 0.5, 0.1, 0.8] + [0.00,  1.00, 0.00, 1.00] = [0.20, 1.50, 0.10, 1.80]
"love"     → [0.9, 0.3, 0.7, 0.2] + [0.84,  0.54, 0.01, 1.00] = [1.74, 0.84, 0.71, 1.20]
"learning" → [0.4, 0.6, 0.5, 0.9] + [0.91, -0.42, 0.02, 1.00] = [1.31, 0.18, 0.52, 1.90]
```

These three vectors are the input **X** to the encoder.

---

## 3. Self-Attention

Self-attention lets each word ask: *"which other words in the sentence matter for my meaning?"*

### Step 3.1 — Create Query, Key and Value

Each input vector is multiplied by three learned weight matrices:

```text
Q = X × Wq      (Query  — what the word is looking for)
K = X × Wk      (Key    — what the word offers to be matched against)
V = X × Wv      (Value  — the information the word passes on)
```

For readability the example uses simplified vectors of size 3 (d_k = 3):

```text
            Q            K            V
"I"      [1, 0, 1]    [0, 1, 1]    [1, 0, 0]
"love"   [0, 1, 0]    [1, 1, 0]    [0, 1, 0]
"learn"  [1, 1, 0]    [0, 0, 1]    [0, 0, 1]
```

### Step 3.2 — Attention scores: Q × Kᵀ

Each Query is dot-multiplied with every Key:

```text
Scores = Q × Kᵀ

           I   love  learn
"I"      [ 1,   1,    1 ]
"love"   [ 1,   1,    0 ]
"learn"  [ 1,   2,    0 ]
```

### Step 3.3 — Scale by √d_k

Dividing by √d_k (√3 ≈ 1.73) keeps the scores from becoming too large, which would push softmax into saturation (outputs near 0 or 1) and slow learning:

```text
Scaled = Scores / √3

"I"      [0.58, 0.58, 0.58]
"love"   [0.58, 0.58, 0.00]
"learn"  [0.58, 1.15, 0.00]
```

### Step 3.4 — Softmax

Softmax converts each row into probabilities that sum to 1:

```text
Attention weights

"I"      [0.33, 0.33, 0.33]
"love"   [0.39, 0.39, 0.22]
"learn"  [0.30, 0.53, 0.17]
```

Notice that **"learning" pays the most attention to "love" (0.53)** — the model has linked the two semantically related words.

### Step 3.5 — Multiply by V

Each output is the weighted sum of the Value vectors:

```text
Output = softmax(Q × Kᵀ / √d_k) × V

"I"      → [0.33, 0.33, 0.33]
"love"   → [0.39, 0.39, 0.22]
"learn"  → [0.30, 0.53, 0.17]
```

The complete formula:

```text
Attention(Q, K, V) = softmax( Q × Kᵀ / √d_k ) × V
```

---

## 4. Multi-Head Attention

Instead of running attention once, the Transformer runs it **several times in parallel**. Each run is a **head** with its own Wq, Wk, Wv, and each head learns a different kind of relationship — e.g. grammar, nearby words, or meaning. The original model uses **8 heads**.

```text
head₁ = Attention(X×Wq¹, X×Wk¹, X×Wv¹)
head₂ = Attention(X×Wq², X×Wk², X×Wv²)
...
head₈ = Attention(X×Wq⁸, X×Wk⁸, X×Wv⁸)

MultiHead(Q, K, V) = Concat(head₁, …, head₈) × Wᴼ
```

The example uses **2 heads** for the word "learn":

```text
Head 1 ("learn") → [0.30, 0.53, 0.17]
Head 2 ("learn") → [0.45, 0.10, 0.45]

Concat             → [0.30, 0.53, 0.17, 0.45, 0.10, 0.45]
× Wᴼ (project back) → [0.30, 0.40, 0.30]
```

---

## 5. Residual Connection + Layer Normalization

### Residual (skip) connection

The input of each sub-layer is **added back** to its output. This preserves the original information and helps gradients flow through deep networks (prevents vanishing gradients):

```text
output = x + Sublayer(x)
```

### Layer Normalization

The result is then normalized to zero mean and unit variance, followed by learnable scale (γ) and shift (β):

```text
LayerNorm(x) = (x − μ) / √(σ² + ε) × γ + β
```

Worked example for "learn" (first 3 dimensions):

```text
x (input)          = [1.31, 0.18, 0.52]
Attention output   = [0.30, 0.53, 0.17]
After residual     = [1.61, 0.71, 0.69]

μ (mean)           = 1.00
σ² (variance)      = 0.18
√(σ² + ε)          ≈ 0.43

LayerNorm output   = [1.41, -0.68, -0.73]
```

---

## 6. Feed-Forward Network (FFN)

Each position is passed **independently** through a small two-layer network. The first layer expands the dimension (512 → 2048 in the real model), a ReLU sets negative values to zero, and the second layer projects back to the original size:

```text
FFN(x) = max(0, x × W₁ + b₁) × W₂ + b₂
```

Worked example:

```text
Input              = [1.41, -0.68, -0.73]
After W₁ + b₁      = [2.1, -0.3, 1.5, -1.2]
After ReLU         = [2.1,  0.0, 1.5,  0.0]
After W₂ + b₂      = [0.8,  0.5, 0.3]
```

A second **residual connection + LayerNorm** follows the FFN. The result is the **encoder output** — a contextualized representation of every input word. The whole encoder block (sections 3–6) is repeated **6 times**.

---

## 7. Decoder

The decoder generates the translation **one word at a time**. Suppose it has already produced "أنا" and "أحب" and is now generating the third word (shown as `???`):

```text
Decoder input so far: ["أنا", "أحب", ???]
```

Each decoder layer has **three** sub-layers.

### 7.1 Masked Self-Attention

The decoder must not look at **future** words (they have not been generated yet). Future positions are masked by setting their scores to −∞ **before** softmax, so they get a weight of exactly 0.

```text
Scores before mask            After mask                    After softmax
          أنا  أحب  ???              أنا   أحب   ???                 أنا   أحب   ???
"أنا"   [0.8, 0.5, 0.3]      "أنا"  [0.8,  -∞,   -∞ ]      "أنا"  [1.00, 0.00, 0.00]
"أحب"   [0.6, 0.9, 0.4]      "أحب"  [0.6,  0.9,  -∞ ]      "أحب"  [0.43, 0.57, 0.00]
"???"   [0.3, 0.7, 0.8]      "???"  [0.3,  0.7,  0.8]      "???"  [0.24, 0.36, 0.40]
```

Each word attends only to itself and the words before it.

### 7.2 Cross-Attention (Encoder–Decoder Attention)

Here the decoder looks at the **input sentence**:

- **Q** comes from the decoder (the Arabic words so far)
- **K** and **V** come from the **encoder output** (the English words)

```text
Q for "???"              = [0.5, 0.8, 0.3]

Scores against encoder keys:
   "I"        → 0.53
   "love"     → 0.90
   "learning" → 0.83

Softmax → [0.26, 0.38, 0.36]
```

The third Arabic word attends most strongly to "love" and "learning" — exactly the English words it needs to translate.

### 7.3 Feed-Forward + Residual + LayerNorm

As in the encoder. The decoder block is also repeated **6 times**.

---

## 8. Linear + Softmax

The final decoder vector is projected by a **linear layer** onto the whole vocabulary (which can be ~50,000 words), then **softmax** turns the scores into probabilities:

```text
Decoder output for "???" = [0.25, 0.60, 0.15]

Linear → Softmax over the vocabulary:
   "أنا"     → 0.02
   "أحب"     → 0.05
   "التعلم"  → 0.85   ← highest
   "كتاب"    → 0.03
   "بيت"     → 0.05
```

The word with the highest probability, **"التعلم"**, is selected.

**Final output: أنا أحب التعلم** ✅

---

## 9. Full Pipeline Overview

```text
                 ENCODER (×6 layers)                          DECODER (×6 layers)
 ┌────────────────────────────────────────┐      ┌────────────────────────────────────────────┐
 │  "I love learning"                     │      │  Output so far: ["أنا", "أحب", ...]        │
 │        ↓                               │      │        ↓                                   │
 │  Input Embedding                       │      │  Output Embedding                          │
 │        ↓                               │      │        ↓                                   │
 │  + Positional Encoding                 │      │  + Positional Encoding                     │
 │        ↓                               │      │        ↓                                   │
 │  Multi-Head Self-Attention             │      │  Masked Multi-Head Self-Attention          │
 │        ↓                               │      │        ↓                                   │
 │  Add & LayerNorm                       │      │  Add & LayerNorm                           │
 │        ↓                               │      │        ↓                                   │
 │  Feed-Forward Network                  │ ───► │  Cross-Attention (Q: decoder, K,V: encoder)│
 │        ↓                               │      │        ↓                                   │
 │  Add & LayerNorm                       │      │  Add & LayerNorm                           │
 │        ↓                               │      │        ↓                                   │
 │  Encoder output (contextual vectors)   │      │  Feed-Forward Network                      │
 └────────────────────────────────────────┘      │        ↓                                   │
                                                 │  Add & LayerNorm                           │
                                                 │        ↓                                   │
                                                 │  Linear → Softmax                          │
                                                 │        ↓                                   │
                                                 │  Next word: "التعلم"                       │
                                                 └────────────────────────────────────────────┘

 Final output: أنا أحب التعلم
```

---

## 10. Key Facts (original paper)

| Item | Value |
|---|---|
| Paper | "Attention Is All You Need" (2017), Vaswani et al., Google |
| Encoder layers | 6 |
| Decoder layers | 6 |
| d_model (embedding size) | 512 |
| Attention heads (h) | 8 |
| d_k per head | 512 / 8 = 64 |
| Feed-forward inner dimension (d_ff) | 2048 |
| Optimizer | Adam with warmup and decay |
| Loss function | Cross-entropy |
| Decoder training technique | Teacher forcing (feed the correct previous token) |

---

## 11. Three Types of Transformer Models

| Type | Examples | How it works | Typical uses |
|---|---|---|---|
| **Encoder-only** | BERT, RoBERTa, DistilBERT | Bidirectional — sees the whole input at once | Classification, Named-Entity Recognition, understanding tasks |
| **Decoder-only** | GPT family (GPT-2, GPT-3, GPT-4, …) | Autoregressive — predicts left to right | Text generation, chat |
| **Encoder–Decoder** | Original Transformer, BART, T5 | Full sequence-to-sequence | Translation, summarization |

---

*Prepared by Dr. Abdulkarim Albanna — AI Applications Course.*
*Related pages on the original site: [Transformer Overview](https://albanna-tutorials.com/transformer.html) · [Transformer Q&A](https://albanna-tutorials.com/transformer-qa.html) · [الشرح بالعربي](https://albanna-tutorials.com/transformer-step-by-step-ar.html)*
