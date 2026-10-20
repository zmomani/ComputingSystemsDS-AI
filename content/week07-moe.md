---
week: 7
title: Mixture of Experts in LLMs
source: https://albanna-tutorials.com/moe.html
---
# Mixture of Experts in LLMs

> Based on Dr. Abdulkarim Albanna's lesson "Mixture of Experts in LLMs" (albanna-tutorials.com; an Arabic version exists at moe_ar.html).

## What is Mixture of Experts (MoE)?

Mixture of Experts is a neural network architecture that replaces a single dense model with **several specialized sub-models called experts**. In LLMs, MoE swaps the dense feedforward (FFN) layers of the transformer for sets of expert networks, so the *total* parameter count can grow enormously while the compute spent *per token* stays manageable.

A **router** (gating network) decides which experts process each token, so only a small subset of the parameters is active at any time. Typical configurations use 8–64 experts per layer with top-2 routing, leaving roughly 20% of parameters active per token.

> "Mixture of Experts enable models to be pretrained with far less compute, which means you can dramatically scale up the model or dataset size with the same compute budget as a dense model." — Hugging Face blog


## How MoE Works in LLMs

1. **Input processing:** the token embedding enters the MoE layer inside a transformer block.
2. **Router evaluation:** a learned linear layer followed by softmax scores every expert for the token.
3. **Expert selection:** the top-k experts (typically k = 2) are chosen for each token.
4. **Parallel processing:** the chosen experts process the token independently.
5. **Output aggregation:** their outputs are merged as a **weighted sum** using the router's probabilities.
6. **Forward propagation:** the combined output moves on to the next transformer layer.


## Key Components

### Experts
- Individual feedforward sub-networks that tend to specialize in patterns such as math, code, creative writing or multilingual tasks.
- Experts usually share the **same architecture** and differ only in their **learned weights**.
- Specialization is not programmed in; it **emerges during training**.

### Router / Gate Network
- Computes a probability distribution over experts for each token and selects the top-k.
- Implemented as a **linear layer followed by softmax**.
- Routing strategies:
  - **Top-1** — Switch Transformer.
  - **Top-2** — Mixtral.
  - **Expert choice** — experts pick tokens rather than tokens picking experts.

### Load Balancing
Prevents **expert collapse**, where the router sends most tokens to one or two experts. Techniques:
- An **auxiliary loss** that penalizes uneven routing.
- **Expert capacity** limits on how many tokens an expert can process per batch.
- A **router z-loss** that prevents extreme routing logits (introduced in Google's ST-MoE).


## Advantages of MoE

1. **Improved scalability:** models can reach hundreds of billions of parameters while activating only a fraction per token.
2. **Computational efficiency:** DeepSeek-V3 (671B total) runs at roughly the cost of a 37B dense model per token — about **18x** parameter efficiency.
3. **Specialized knowledge:** experts develop niches (math, code, specific languages), giving both depth and breadth.
4. **Training efficiency:** Mixtral 8x7B matches or beats Llama 2 70B quality while using about 5x fewer active parameters per token (12.9B vs. 70B); DeepSeek-V3 was trained in 2.788M H800 GPU-hours.
5. **Flexible resource allocation:** compute is directed where each token needs it, unlike dense models that apply uniform computation.

> "Mixtral outperforms Llama 2 70B on most benchmarks with 6x faster inference." — Mistral AI (2023)

### Dense vs. MoE comparison

| Aspect | Dense model | MoE model |
|---|---|---|
| Active parameters per token | All parameters | Roughly 10–25% of total |
| Inference memory | Scales with model size | High — all experts must be loaded |
| Inference compute (FLOPs) per token | Grows with model size | Low relative to total parameter count |
| Training compute for equal quality | Higher | Lower |
| Specialization | Spread evenly across the network | Concentrated in specific experts |
| Fine-tuning stability | More stable | Needs more careful tuning |

## Real-World Implementations

### Mixtral 8x7B / 8x22B (Mistral AI)
- 8x7B: **46.7B total, 12.9B active** parameters; outperforms Llama 2 70B.
- 8x22B (April 2024): **141B total, ~39B active**; 64K context; rivals GPT-4-class models on coding and reasoning.
- Open weights under **Apache 2.0**, token-level routing, **32K context** for 8x7B.
- Technically: every FFN layer of a Mistral-7B-style transformer is replaced by 8 experts; the router picks the **top 2** and combines their outputs with normalized softmax weights.

### DeepSeek-V3 (2024)
- **671B total, 37B active** parameters per token; **256 routed experts per layer with top-8 routing, plus 1 shared expert** (fine-grained experts).
- **Multi-Head Latent Attention (MLA)** for KV-cache efficiency.
- Trained on 14.8T tokens for about $5.6M (2.788M H800 GPU-hours); matches or exceeds GPT-4o on coding (HumanEval-Mul 82.6% vs. 80.5%).
- Technically: **auxiliary-loss-free load balancing** using learned bias terms added to routing logits; FP8 mixed-precision training; a multi-token prediction auxiliary objective.

### Switch Transformer (Google, 2021)
- Routes each token to **exactly one expert (top-1)**, keeping the design simple and reducing communication overhead.
- Scales to **1.6 trillion parameters**; reports up to a 7x pre-training speedup over T5-Base (and 4x over T5-XXL) at the same compute budget.
- Key ideas: expert **capacity factor**, **selective precision**, and an **auxiliary load-balancing loss**.

### MoE model timeline (2020–2025)

| Model | Organization | Year | Total params | Active params | Experts / routing |
|---|---|---|---|---|---|
| GShard | Google | 2020 | — | — | Top-2, alternating MoE layers |
| Switch Transformer | Google | 2021 | 1.6T | — | Top-1 |
| GLaM | Google | 2021 | 1.2T | — | 64 experts per layer |
| NLLB-MoE | Meta AI | 2022 | 54.5B | — | Translation model, top-2 |
| Mixtral 8x7B | Mistral AI | 2023 | 46.7B | 12.9B | 8 experts, top-2 |
| Grok-1 | xAI | 2024 | 314B | — | 8 experts, top-2 |
| Mixtral 8x22B | Mistral AI | 2024 | 141B | 39B | 8 experts, top-2 |
| Qwen1.5-MoE | Alibaba | 2024 | 14.3B | 2.7B | — |
| DeepSeek-V2 | DeepSeek | 2024 | 236B | 21B | Fine-grained MoE |
| DeepSeek-V3 | DeepSeek | 2024 | 671B | 37B | 256 routed experts + 1 shared, top-8 |

## Code Examples: Building an MoE Layer in PyTorch

### Step 1: The Expert Network — a feedforward sub-model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# A single Expert = a 2-layer feedforward network (same as a transformer FFN)
class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)   # expand
        self.fc2 = nn.Linear(d_ff, d_model)   # project back

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [num_tokens, d_model]
        return self.fc2(F.relu(self.fc1(x)))

# Try it out
expert = Expert(d_model=512, d_ff=2048)
tokens = torch.randn(4, 512)   # batch of 4 tokens
output = expert(tokens)
print(f"Input:  {tokens.shape}")   # [4, 512]
print(f"Output: {output.shape}")   # [4, 512] — same shape!
```

### Step 2: The Router — deciding which experts process each token

```python
class TopKRouter(nn.Module):
    """Routes each token to the top-k most relevant experts."""
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        # Learned: maps token embedding → score per expert
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        # x: [num_tokens, d_model]
        # Step A: raw routing scores for every expert
        logits = self.gate(x)                          # [num_tokens, num_experts]
        # Step B: softmax → probability distribution
        probs = F.softmax(logits, dim=-1)              # [num_tokens, num_experts]
        # Step C: pick the top-k experts per token
        top_k_probs, top_k_idx = torch.topk(probs, self.top_k, dim=-1)
        # top_k_probs: [num_tokens, k] — how much weight per expert
        # top_k_idx:   [num_tokens, k] — which expert indices
        # Step D: renormalize so the k weights sum to 1
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        return top_k_probs, top_k_idx, probs   # probs used for load-balancing loss

# Example: 4 tokens, 8 experts, select top-2
router = TopKRouter(d_model=512, num_experts=8, top_k=2)
tokens = torch.randn(4, 512)
weights, indices, all_probs = router(tokens)

print("Expert indices per token:", indices)
# e.g. tensor([[3, 6], [1, 7], [0, 2], [5, 3]])
print("Expert weights per token: ", weights.round(decimals=2))
# e.g. tensor([[0.62, 0.38], [0.71, 0.29], ...])
```

### Step 3: The MoE Layer — combining router + experts

```python
class MoELayer(nn.Module):
    """
    Sparse Mixture of Experts Layer.
    Replaces a single FFN in a Transformer with N expert FFNs.
    Only k out of N experts activate per token (k << N).
    """
    def __init__(self, d_model: int, d_ff: int,
                 num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        # N independent expert networks
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_experts)
        ])
        # Router selects which experts process each token
        self.router = TopKRouter(d_model, num_experts, top_k)

    def forward(self, x: torch.Tensor):
        # x: [batch, seq_len, d_model]
        B, S, D = x.shape
        # Flatten batch and sequence dims — treat all tokens equally
        x_flat = x.reshape(-1, D)                   # [B*S, d_model]
        N = x_flat.shape[0]                         # total tokens

        # Get routing decisions
        weights, indices, all_probs = self.router(x_flat)
        # weights: [N, top_k] — how much each expert contributes
        # indices: [N, top_k] — which expert to use

        output = torch.zeros_like(x_flat)           # [N, d_model]

        # For each expert slot (0 to top_k-1)
        for k in range(self.top_k):
            expert_ids  = indices[:, k]             # [N] which expert for slot k
            slot_weight = weights[:, k:k+1]         # [N, 1] contribution weight

            # Group tokens by their assigned expert and run each expert once
            for e in range(self.num_experts):
                mask = (expert_ids == e)            # which tokens go to expert e
                if not mask.any():
                    continue                        # skip unused experts (saves compute!)
                expert_out = self.experts[e](x_flat[mask])   # [n_e, D]
                output[mask] += slot_weight[mask] * expert_out  # weighted add

        return output.reshape(B, S, D), all_probs   # restore original shape

# Test the full MoE layer
moe = MoELayer(d_model=512, d_ff=2048, num_experts=8, top_k=2)
x = torch.randn(2, 10, 512)                         # batch=2, seq_len=10
out, probs = moe(x)
print(f"Input:  {x.shape}")                         # torch.Size([2, 10, 512])
print(f"Output: {out.shape}")                       # torch.Size([2, 10, 512]) ✓

# Count parameters: 8 experts × (512×2048 + 2048 + 2048×512 + 512) = ~16.8M expert params
# (plus 8×512 = 4,096 router weights). Only 2/8 experts activate per token = 25% active
total = sum(p.numel() for p in moe.parameters())
active = total * (moe.top_k / moe.num_experts)
print(f"Total params: {total:,} | Active per token: {int(active):,}")
```

### Step 4: Load Balancing Loss — preventing expert collapse

```python
def auxiliary_load_balancing_loss(
    router_probs: torch.Tensor,   # [N, num_experts] — softmax output
    expert_indices: torch.Tensor, # [N, top_k] — selected expert IDs
    num_experts: int
) -> torch.Tensor:
    """
    Switch Transformer / Mixtral style auxiliary load-balancing loss.
    Encourages all experts to receive roughly equal token counts.

    Loss = num_experts * sum_i(f_i * P_i)
      f_i = fraction of tokens routed to expert i
      P_i = mean router probability for expert i
    Both terms should be ~1/num_experts for uniform distribution.
    """
    N = router_probs.shape[0]

    # f_i: fraction of tokens dispatched to each expert
    one_hot = F.one_hot(expert_indices, num_experts).float()   # [N, k, E]
    tokens_per_expert = one_hot.sum(dim=[0, 1]) / (N * expert_indices.shape[1])
    # Shape: [num_experts] — ideally all equal to 1/num_experts

    # P_i: mean router probability assigned to each expert
    mean_probs = router_probs.mean(dim=0)   # Shape: [num_experts] — ideally all equal to 1/num_experts

    # Dot product × num_experts: 1.0 when perfectly balanced
    loss = num_experts * (tokens_per_expert * mean_probs).sum()
    return loss

# ------ How to use it in your training loop ------
moe = MoELayer(d_model=512, d_ff=2048, num_experts=8, top_k=2)
x = torch.randn(2, 10, 512)

out, all_probs = moe(x)
_, indices, _ = moe.router(x.reshape(-1, 512))

# Main task loss (e.g. cross-entropy for language modelling)
task_loss = torch.tensor(2.5)   # placeholder

# Add a small load-balancing penalty (coefficient α = 0.01)
balance_loss = auxiliary_load_balancing_loss(all_probs, indices, num_experts=8)
total_loss = task_loss + 0.01 * balance_loss

print(f"Task loss:    {task_loss.item():.4f}")
print(f"Balance loss: {balance_loss.item():.4f}")   # ideally close to 1.0
print(f"Total loss:   {total_loss.item():.4f}")
```

### Step 5: Using Mixtral / DeepSeek via HuggingFace Transformers

```python
# Install: pip install transformers accelerate bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# ── Option A: Mixtral 8x7B (open weights, Apache-2.0) ──────────────────────
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load with 4-bit quantization: reduces VRAM from ~90 GB → ~24 GB
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",        # spread across available GPUs/CPU
    torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),   # requires bitsandbytes
)

# Prepare a chat-formatted prompt
messages = [
    {"role": "user", "content": "Explain Mixture of Experts in three sentences."}
]
input_ids = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
).to(model.device)

# Generate a response
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
    )

response = tokenizer.decode(
    output_ids[0][input_ids.shape[1]:],
    skip_special_tokens=True
)
print(response)

# ── Option B: Inspect which experts were activated ─────────────────────────
# Access the router logits for interpretability
outputs = model(input_ids, output_router_logits=True)
router_logits = outputs.router_logits   # tuple of tensors per MoE layer
for layer_idx, logits in enumerate(router_logits):
    top2 = logits.topk(2, dim=-1).indices   # [seq, 2]
    print(f"Layer {layer_idx}: top-2 experts used → {top2[0].tolist()}")
```

## Formulas (summary)

- Router logits: `logits = W_gate · x`, with `W_gate ∈ R^{num_experts × d_model}` (no bias).
- Router probabilities: `p = softmax(logits)` over experts.
- Top-k renormalization: `w_j = p_j / Σ_{i∈TopK} p_i` so the k weights sum to 1.
- Expert output: `E(x) = W_2 · ReLU(W_1 · x + b_1) + b_2`.
- Layer output: `y = Σ_{j∈TopK} w_j · E_j(x)`.
- Load-balancing loss: `L_aux = N · Σ_i f_i · P_i`, where `f_i` = fraction of token assignments sent to expert i and `P_i` = mean router probability for expert i; equals 1.0 under perfectly uniform routing.
- Total training loss: `L = L_task + 0.01 · L_aux`.
- Active parameters ≈ `total_params × (top_k / num_experts)`.

## Technical Challenges

- **High memory requirements:** all experts must sit in VRAM even though only a few are active per token. *Mitigation:* quantization (e.g., 4-bit loading).
- **Training instability:** discrete top-k routing is non-differentiable and can destabilize gradients. *Mitigations:* auxiliary losses, router z-loss, careful initialization, keeping the router in FP32.
- **Fine-tuning difficulty:** MoE models overfit faster during supervised fine-tuning. *Mitigation:* applying LoRA only to router or non-expert weights.
- **Load imbalance / expert collapse:** the router can collapse onto a few experts, wasting capacity. *Mitigations:* auxiliary balancing losses and capacity factors.
- **Communication overhead:** distributed training needs all-to-all token transfers between devices, which can reduce throughput by roughly **30–50%**. *Mitigations:* expert parallelism and optimized serving stacks (vLLM, TensorRT-LLM).
- **Expert capacity planning / token drops:** tokens beyond an expert's capacity are dropped, hurting quality. *Mitigation:* careful tuning of the capacity factor.

## Future Directions

- **Hardware co-design:** accelerators built for sparse expert dispatch with less communication overhead; faster interconnects; paging experts between CPU and GPU memory.
- **Advanced routing:** routing informed by semantic context, task type or language; expert-choice routing, soft routing, auxiliary-loss-free balancing.
- **Multimodal MoE:** dedicated experts for vision, audio and text plus cross-modal experts; combination with retrieval for grounded generation.
- **Expert merging and pruning:** combining experts after training to produce compact dense models.
- **Long-context MoE:** pairing MoE with sliding-window or latent attention to reach 128K+ context at manageable memory cost.
- **Reasoning models:** combining MoE with reinforcement learning, as in DeepSeek-R1.
- **On-device MoE:** small sparse models bringing large-capacity behaviour to mobile hardware.

## Key Terms

- **Mixture of Experts (MoE):** replaces a single dense network with several sub-models (experts), only a subset of which runs per input.
- **Experts:** feedforward sub-networks with shared architecture and different learned weights.
- **Router / gate network:** learned layer that scores experts per token and selects which to use.
- **Top-k routing:** selecting the k highest-scoring experts (top-1: Switch; top-2: Mixtral; top-8: DeepSeek-V3).
- **Expert choice:** experts pick tokens instead of tokens picking experts.
- **Load balancing / expert collapse / auxiliary loss / expert capacity / router z-loss / token drop:** see Key Components and Challenges above.
- **Active parameters:** parameters actually used for a given token, vs. total parameters.
- **All-to-all communication:** cross-device exchange needed to send tokens to experts on other devices.
- **Expert parallelism:** distributing different experts across different GPUs.
- **Multi-Head Latent Attention (MLA):** DeepSeek's attention variant that compresses the KV cache.
- **Quantization (4-bit / 8-bit):** lower numeric precision to reduce memory use.

## Resources
- Hugging Face MoE blog: https://huggingface.co/blog/moe
- Switch Transformers paper: https://arxiv.org/abs/2101.03961
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- Mixtral of Experts (paper and weights): https://mistral.ai/news/mixtral-of-experts/
- Grok-1 open weights: https://github.com/xai-org/grok-1
- DeepSeek-V3 model: https://huggingface.co/deepseek-ai/DeepSeek-V3


## Key concepts worth simulating

- **MoE router: gating softmax picks top-k experts** — show a token vector, the 8 gate logits, the softmax bars, the top-2 selection and the renormalization `w_j = p_j / Σ p_topk`; let students drag the logits and watch the selected experts change.
- **Weighted-sum aggregation** — animate two expert FFN outputs scaled by 0.62 and 0.38 being summed into one output vector; toggle k = 1, 2, 4 to see the mix.
- **Active vs. total parameters** — sliders for `num_experts`, `top_k`, `d_model`, `d_ff` driving a live count of total vs. active parameters and FLOPs per token (Mixtral 46.7B → 12.9B; DeepSeek-V3 671B → 37B).
- **Expert collapse and the auxiliary loss** — a heatmap of tokens-per-expert over training steps; with α = 0 the router collapses to one or two experts; with α = 0.01 the distribution flattens and `L_aux → 1.0`. Plot `f_i`, `P_i` and `N·Σ f_i P_i`.
- **Expert capacity and token drops** — a bucket per expert with a capacity-factor slider; overflow tokens turn red and are dropped.
- **Token-choice vs. expert-choice routing** — side-by-side bipartite graphs: tokens picking experts vs. experts picking their top tokens.
- **Emergent specialization** — colour tokens by type (code, math, Arabic text, English prose) and show the routing histogram per expert after a toy training run.
- **All-to-all communication in expert parallelism** — tokens moving between 4 GPUs each holding 2 experts, with a throughput meter showing the 30–50% overhead.

## Test code

```python
# 1. The core routing step on its own: softmax → top-k → renormalize
import torch, torch.nn.functional as F
torch.manual_seed(0)
num_experts, top_k = 8, 2
gate = torch.nn.Linear(512, num_experts, bias=False)
x = torch.randn(4, 512)                       # 4 tokens
probs = F.softmax(gate(x), dim=-1)
w, idx = torch.topk(probs, top_k, dim=-1)
w = w / w.sum(dim=-1, keepdim=True)
print("experts chosen per token:\n", idx)
print("renormalized weights (rows sum to 1):\n", w.round(decimals=2), w.sum(-1))
```

```python
# 2. Show that an MoE layer preserves shape and uses only k/N of its expert parameters
import torch, torch.nn as nn, torch.nn.functional as F

class Expert(nn.Module):
    def __init__(s, d, f): super().__init__(); s.fc1, s.fc2 = nn.Linear(d, f), nn.Linear(f, d)
    def forward(s, x): return s.fc2(F.relu(s.fc1(x)))

class MoE(nn.Module):
    def __init__(s, d=64, f=256, E=8, k=2):
        super().__init__(); s.k, s.E = k, E
        s.experts = nn.ModuleList(Expert(d, f) for _ in range(E)); s.gate = nn.Linear(d, E, bias=False)
    def forward(s, x):
        B, S, D = x.shape; xf = x.reshape(-1, D)
        p = F.softmax(s.gate(xf), -1); w, idx = torch.topk(p, s.k, -1); w = w / w.sum(-1, keepdim=True)
        out = torch.zeros_like(xf)
        for slot in range(s.k):
            for e in range(s.E):
                m = idx[:, slot] == e
                if m.any(): out[m] += w[m, slot:slot+1] * s.experts[e](xf[m])
        return out.reshape(B, S, D), p, idx

moe = MoE(); y, p, idx = moe(torch.randn(2, 10, 64))
total = sum(q.numel() for q in moe.experts.parameters())
print("output shape:", tuple(y.shape), "| expert params:", total, "| active per token:", total * moe.k // moe.E)
```

```python
# 3. Load-balancing loss: balanced vs. collapsed routing
import torch, torch.nn.functional as F
def aux_loss(probs, idx, E):
    f = F.one_hot(idx, E).float().sum(dim=[0, 1]) / (probs.shape[0] * idx.shape[1])
    P = probs.mean(0)
    return E * (f * P).sum()
E, N = 8, 1000
uniform_p = torch.full((N, E), 1 / E); uniform_idx = torch.stack([torch.arange(N) % E, (torch.arange(N) + 1) % E], 1)
collapsed_p = F.one_hot(torch.zeros(N, dtype=torch.long), E).float() * 0.9 + 0.1 / E
collapsed_idx = torch.zeros(N, 2, dtype=torch.long)
print("balanced  :", aux_loss(uniform_p, uniform_idx, E).item())    # ≈ 1.0
print("collapsed :", aux_loss(collapsed_p, collapsed_idx, E).item()) # >> 1.0
```

```python
# 4. Watch the auxiliary loss fix collapse during a toy training loop
import torch, torch.nn.functional as F
torch.manual_seed(1)
E, k, d = 8, 2, 32
gate = torch.nn.Linear(d, E, bias=False)
with torch.no_grad(): gate.weight[0] += 3.0          # bias the router toward expert 0 (collapse)
opt = torch.optim.Adam(gate.parameters(), lr=0.05)
for step in range(201):
    x = torch.randn(256, d)
    p = F.softmax(gate(x), -1); _, idx = torch.topk(p, k, -1)
    f = F.one_hot(idx, E).float().sum([0, 1]) / (256 * k)
    loss = E * (f * p.mean(0)).sum()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        print(f"step {step:3d}  L_aux={loss.item():.3f}  tokens/expert={(f*100).round().int().tolist()}%")
```
