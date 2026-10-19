---
week: 1
title: "Transformer: Questions & Answers"
source: https://albanna-tutorials.com/transformer-qa.html
---
# Transformer: Questions & Answers

*21 questions covering every component of the Transformer architecture.*
*Intermediate · 21 Questions · Covers All Components · Exam Ready*

*Material by Dr. Abdulkarim Albanna (Dr. Abedal-Kareem Al-Banna), Assistant Professor, Data Science & AI, University of Petra — ilmy / albanna-tutorials.com. Reproduced here for teaching use; formulas, code blocks, tables and examples are as on the original page, with the prose kept as close to the original as the extraction allowed.*

---

## Q1. What is a Transformer?

A **Transformer** is a deep learning architecture introduced in the 2017 paper **"Attention Is All You Need"** by Vaswani et al. (Google).

- It processes an **entire sequence in parallel** using **self-attention**, with **no recurrence and no convolution**.
- It consists of an **Encoder** (understands the input) and a **Decoder** (generates the output).
- It is the foundation of modern NLP models: BERT, GPT, T5, BART, and today's large language models.

---

## Q2. How does the Transformer solve the problems of RNNs / LSTMs?

| Problem with RNN / LSTM | Transformer solution |
|---|---|
| **Sequential processing** — words must be processed one after another, which is slow | Processes **all words in parallel** |
| **Vanishing gradients** over long sequences | **Residual connections + Layer Normalization** keep gradients flowing |
| **Long-range dependencies** are hard to capture (information fades over many steps) | **Self-attention** relates every word to every other word directly, in one step |
| **Limited scalability** | Parallelism makes full use of **GPUs**, so models scale to billions of parameters |

---

## Q3. What are the main components of the Transformer?

1. **Input Embedding** — converts each token to a vector.
2. **Positional Encoding** — adds word-order information.
3. **Multi-Head Self-Attention** — each word attends to all other words.
4. **Feed-Forward Network (FFN)** — a two-layer network applied to each position.
5. **Residual Connections + Layer Normalization** — around every sub-layer.
6. **Linear + Softmax** output layer — in the decoder, predicts the next word.

The encoder stacks (3)(4)(5); the decoder adds **masked** self-attention and **cross-attention** before its FFN, then ends with (6).

---

## Q4. What is self-attention, and how does it work? (with example)

**Self-attention** lets each word look at **all other words** in the sentence and decide how much each one matters for its own meaning.

Example: *"The cat sat on the mat because it was soft."* — to understand **"it"**, the model attends to all words and gives the highest attention score to **"mat"** (not "cat"), because "soft" fits a mat.

Steps:

1. Create **Query (Q)**, **Key (K)** and **Value (V)** vectors for every word.
2. Compute the scores: `Score = Q × Kᵀ`
3. **Scale by √dₖ**
4. Apply **softmax** to turn scores into probabilities (weights that sum to 1).
5. Multiply the weights by **V** to get the weighted output for each word.

---

## Q5. What are Query, Key and Value? Why three vectors?

| Vector | Meaning |
|---|---|
| **Query (Q)** | What the current word is **looking for** |
| **Key (K)** | What each word **offers** to be matched against |
| **Value (V)** | The actual **information** a word passes on once it is attended to |

**Library analogy:** your **Query** is what you type into the search; the **Keys** are the index entries of the books; the **Values** are the contents of the books you then read.

Why three separate vectors? Because keeping them separate lets the model learn **different representations** for the three roles: how a word *searches*, how it *gets found*, and what it *contributes*. Using one vector for all three would force a single representation to do all three jobs.

---

## Q6. What is the self-attention formula, and what does each part do?

```text
Attention(Q, K, V) = softmax( Q × Kᵀ / √dₖ ) × V
```

| Part | Role |
|---|---|
| `Q × Kᵀ` | Dot product of each Query with every Key → a **relevance score** between every pair of words |
| `/ √dₖ` | **Scaling.** Large dot products would push softmax into near-0 / near-1 outputs (saturation) and slow down learning; dividing by √dₖ keeps the scores in a healthy range |
| `softmax(…)` | Turns each row of scores into **weights that sum to 1** |
| `× V` | **Weighted sum** of the Value vectors — the context-aware output of each word |

---

## Q7. What is multi-head attention, and why use multiple heads?

Instead of one attention computation, the Transformer runs **h heads in parallel**, each with its own learned projection matrices. Their outputs are **concatenated** and projected once more:

```text
MultiHead(Q, K, V) = Concat(head₁, …, headₕ) × Wᴼ

headᵢ = Attention( Q × WᵢQ,  K × WᵢK,  V × WᵢV )
```

Why multiple heads?

- Each head can **specialize** in a different kind of relationship — e.g. one head learns **grammar** (subject–verb), one learns **nearby words**, one learns **meaning** (semantic similarity).
- Several "views" of the sentence are combined into a richer representation.
- Original Transformer: **8 heads**, each with dimension **64** (512 / 8).

---

## Q8. Why does the Transformer need positional encoding?

Self-attention treats the input as a **set**: it processes all tokens at once and has **no built-in notion of order**. Without position information, these two sentences look identical to the model:

```text
"The cat ate the fish"  =  "The fish ate the cat"
```

Positional encoding **adds a position-specific vector** to every token embedding so the model knows *where* each word is.

---

## Q9. What is the sin/cos positional-encoding formula, and why sine and cosine?

```text
PE(pos, 2i)   = sin( pos / 10000^(2i/d) )
PE(pos, 2i+1) = cos( pos / 10000^(2i/d) )
```

- `pos` — position of the token; `i` — dimension index; `d` — model dimension.
- **Even** dimensions use **sine**, **odd** dimensions use **cosine**.

Why sin and cos?

- Together they give every position a **unique pattern**.
- **Relative positions** can be expressed as a **linear function** of the encodings, so the model can easily learn "k steps away".
- Values are **bounded** (−1 to 1) and the scheme **works for any sequence length**, even longer than those seen in training.
- The constant **10000** spreads the wavelengths: low dimensions change **quickly** with position, high dimensions change **slowly** — like the hands of a clock.

---

## Q10. What is masked self-attention, and why is it needed?

In the **decoder**, each word may attend only to **itself and earlier words**. Future positions are **masked** by setting their scores to **−∞ before softmax**, so they receive a weight of exactly **0**.

```text
When generating "love" in "I love learning":
   Can see:    "I"         ✅
   Can see:    "love"      ✅ (itself)
   Cannot see: "learning"  ❌ (future — not generated yet)
```

Why? During training the whole target sentence is available at once. Without the mask the decoder could simply **copy the next word** instead of learning to predict it. Masking forces it to predict from earlier tokens only — the same situation it faces at inference time.

---

## Q11. How does cross-attention differ from self-attention?

| | Self-attention | Cross-attention |
|---|---|---|
| **Q comes from** | the same sequence | the **decoder** |
| **K, V come from** | the same sequence | the **encoder output** |
| **Where** | Encoder **and** decoder | **Decoder only** |
| **Purpose** | Relate words *within* a sentence | Connect the output being generated to the **input sentence** |

Example: when the decoder produces the Arabic word for "learning", its Query attends to the encoder's representation of **"learning"** (and "love") in "I love learning".

---

## Q12. What are residual (skip) connections, and why do they help?

```text
output = x + Sublayer(x)
```

The input of each sub-layer is **added back** to its output.

Benefits:

- **Gradient flow** — the identity path lets gradients pass straight through, reducing vanishing gradients.
- **Preserves the original information** — the sub-layer only has to learn a *correction*.
- Makes **deep stacks** (6, 12, 96 layers…) trainable.

---

## Q13. What is layer normalization?

```text
LayerNorm(x) = γ × (x − μ) / √(σ² + ε) + β
```

| Symbol | Meaning |
|---|---|
| μ | mean of the vector's values |
| σ² | variance of the vector's values |
| ε | tiny constant to avoid division by zero |
| γ | learnable **scale** |
| β | learnable **shift** |

Purpose: every position's vector is re-centred to **zero mean and unit variance**, which **stabilizes training**, allows **higher learning rates**, and reduces sensitivity to initialization. Unlike batch normalization, it normalizes across the **features of one token**, so it does not depend on the batch.

---

## Q14. What is the feed-forward network, and why is it needed?

```text
FFN(x) = max(0, x·W₁ + b₁) · W₂ + b₂
```

- Two linear layers with a **ReLU** in between.
- The hidden layer **expands to 4×** the model dimension: **512 → 2048 → 512**.
- Applied to **each position independently** (same weights for every position).

Why needed? Attention is essentially a **weighted average** of value vectors — a (mostly) linear operation. The FFN adds **non-linearity** and extra capacity so the model can transform and combine the features each word has gathered.

---

## Q15. What are the three types of Transformer models?

| Type | Examples | Attention direction | Typical tasks |
|---|---|---|---|
| **Encoder-only** | BERT, RoBERTa | **Bidirectional** — sees the whole input | Classification, NER, sentiment, understanding tasks |
| **Decoder-only** | GPT, GPT-4 | **Left-to-right** (masked, autoregressive) | Text generation, chat |
| **Encoder–Decoder** | BART, T5, original Transformer | Encoder bidirectional + masked decoder | **Sequence-to-sequence**: translation, summarization |

---

## Q16. What are the key hyperparameters of the original Transformer?

| Hyperparameter | Value |
|---|---|
| Encoder layers (N) | 6 |
| Decoder layers (N) | 6 |
| d_model (embedding size) | 512 |
| Attention heads (h) | 8 |
| d_k (per head) | 512 / 8 = 64 |
| FFN inner dimension (d_ff) | 2048 |
| Optimizer | Adam with warmup schedule |
| Training technique | Teacher forcing |

---

## Q17. What is teacher forcing?

During training, at each step the decoder is fed the **correct previous word from the target sentence**, not its own (possibly wrong) prediction.

```text
Step 1: Input = [START]              → predict "أنا"
Step 2: Input = [START, أنا]         → predict "أحب"    (correct "أنا", not model's guess)
Step 3: Input = [START, أنا, أحب]    → predict "التعلم"
```

Benefits:

- **Faster training** — all positions can be computed in parallel because the whole target is known.
- **No error cascade** — one wrong early prediction cannot derail every later step.

---

## Q18. How does the full translation pipeline work?

Example: **"I love learning" → "أنا أحب التعلم"**

**Encoder**

1. Embed each English word as a vector.
2. Add positional encoding.
3. Pass through **6 encoder layers** (multi-head self-attention → Add & Norm → FFN → Add & Norm).
4. Output: contextualized representations of "I", "love", "learning".

**Decoder** (one word at a time)

1. Start with the `[START]` token.
2. Embed the tokens generated so far and add positional encoding.
3. **Masked self-attention** — the decoder cannot see future words.
4. **Cross-attention** to the encoder output (Q from decoder, K/V from encoder).
5. FFN → **Linear → Softmax** → predict the next word (e.g. "أنا").
6. Append the predicted word and **repeat** (→ "أحب" → "التعلم") until the model predicts `[END]`.

---

## Q19. How do encoder attention and decoder attention differ?

| Aspect | Encoder | Decoder |
|---|---|---|
| Self-attention | **Unmasked** — every word sees all words | **Masked** — cannot see future words |
| Cross-attention | None | **Yes** — Q from decoder, K and V from encoder |
| Sub-layers per layer | **2** (self-attention, FFN) | **3** (masked self-attention, cross-attention, FFN) |

---

## Q20. Why is the Transformer better than a CNN for NLP?

| Aspect | CNN | Transformer |
|---|---|---|
| Context window | **Limited** by kernel size | **Global** — the whole sequence at once |
| Long-range dependencies | Need **many stacked layers** | Captured in **a single attention layer** |
| Parallelization | Good | **Excellent** |
| Position awareness | Built in (local) | Requires **positional encoding** |
| Dominant in NLP | No | **Yes**, since 2018 |

---

## Q21 (Bonus). If Transformers process in parallel, why is text generation sequential?

- **Training is parallel.** The full target sentence is known, so with **teacher forcing + masking** every position is computed at the same time in one pass.
- **Inference is sequential.** The next word is not known until it is predicted, so the decoder must generate **one token at a time**, feeding each output back in as input.
- This is why large language models still produce text **word by word** — even though, at each step, they process all previous tokens in parallel.

---

*Prepared by Dr. Abdulkarim Albanna — AI Applications Course.*
*Related pages on the original site: [Transformer Overview](https://albanna-tutorials.com/transformer.html) · [Step-by-Step Guide](https://albanna-tutorials.com/transformer-step-by-step.html) · [الشرح بالعربي](https://albanna-tutorials.com/transformer-step-by-step-ar.html)*
