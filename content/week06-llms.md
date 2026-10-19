---
week: 6
title: Large Language Models — Meta's LLAMA4 Herd
source: https://albanna-tutorials.com/llama4.html
---
# Large Language Models — Meta's LLAMA4 Herd

> Based on Dr. Abdulkarim Albanna's lesson "LLAMA4 — Meta's Next Generation of AI" (albanna-tutorials.com). Prose sections were extracted via a fetcher that paraphrases long passages, so wording is close to but not identical to the original; all figures, tables and code are as published. The author's "Example System Prompt" block could not be retrieved verbatim and is summarized below.

## Introduction

LLAMA4 is Meta's most advanced family of large language models to date: **natively multimodal, open-weight** models that process text and images in a single backbone and use a **mixture-of-experts (MoE)** design for efficiency. The lesson is written as an approximately 30-minute read.

## Why LLAMA4 Matters

- The first open-weight, natively multimodal models with an industry-leading context window.
- A mixture-of-experts architecture that activates only part of the model for each token, improving efficiency.
- LLAMA4 Scout offers a context window of **up to 10 million tokens**.
- Claimed to outperform GPT-4o and Gemini 2.0 on a range of benchmarks.
- Pre-trained on **200 languages**, with 12 major languages officially supported.

## What Is LLAMA4?

- Released **April 5, 2025** as the successor to Llama 3 and the **first Llama family to use mixture-of-experts**.
- The "herd" consists of three models: **Scout**, **Maverick**, and **Behemoth** (still in training).
- Uses **early fusion** to combine text and vision tokens in one model backbone from the start, rather than bolting a vision module onto a text model.
- Compared with Llama 3: context of up to 10M tokens (vs. 128K) and training on about 40T tokens (vs. about 15T).
- Available under the **Llama 4 Community License** for commercial and research use.

## The Models: Scout, Maverick and Behemoth

| Model | Active Parameters | Total Parameters | Experts | Context Window | Training Tokens | Hardware | Status |
|---|---|---|---|---|---|---|---|
| LLAMA4 Scout | 17 billion | 109 billion | 16 | 10 million tokens | ~40 trillion | Single H100 GPU with int4 quantization | Available |
| LLAMA4 Maverick | 17 billion | 400 billion | 128 | 1 million tokens | ~22 trillion | Single H100 DGX host with FP8 quantization | Available |
| LLAMA4 Behemoth | 288 billion | ~2 trillion | 16 | — | — | — | Still in training |

### LLAMA4 Scout
The efficiency-focused model: 17B active / 109B total parameters, 16 experts, 10M-token context, ~40T training tokens; runs on a single H100 GPU with int4 quantization.

**Best for:** extremely long context, hardware-constrained deployments, large documents or codebases, and a balance of performance and efficiency.

### LLAMA4 Maverick
The performance-focused model: 17B active / 400B total parameters, 128 experts, 1M-token context, ~22T training tokens; runs on a single H100 DGX host with FP8 quantization. In Maverick, each token is routed to **one shared expert plus one of the 128 routed experts**.

**Best for:** top multimodal performance, advanced reasoning and coding, production-grade assistants, and deployments with stronger infrastructure.

### LLAMA4 Behemoth
A very large model with 288B active and roughly 2T total parameters across 16 experts. Still in training; it serves as a **teacher model** that the smaller models are distilled from. Meta claims it beats GPT-4.5, Claude Sonnet 3.7 and Gemini 2.0 Pro on STEM benchmarks.

## Key Features

- **Mixture-of-experts:** each token activates only a subset of parameters, improving compute efficiency.
- **Native multimodality:** early fusion integrates text and vision tokens without specialized connector modules.
- **Long context:** Scout's 10M-token window is roughly 80x Llama 3's 128K.
- **Multilingual support:** pre-trained on 200 languages with about 10x more multilingual tokens than Llama 3.
- **Code generation:** stronger coding ability, with claimed gains over GPT-4o on many coding benchmarks.
- **Image grounding:** strong visual question answering and object localization in images.
- **Efficient inference:** quantization (int4 / FP8) allows deployment on consumer-grade hardware with little performance loss.
- **Safety:** built-in safety protections, reduced political bias and improved refusal handling.
- **iRoPE architecture:** interleaved attention layers *without* positional embeddings enable very long context.
- **Inference-time temperature scaling of attention:** helps the model generalize to inputs far longer than those seen in training.

## Technical Specifications

| Specification | LLAMA4 Scout | LLAMA4 Maverick |
|---|---|---|
| Model Architecture | Auto-regressive with MoE, early-fusion multimodal | Auto-regressive with MoE, early-fusion multimodal |
| Active Parameters | 17 billion | 17 billion |
| Total Parameters | 109 billion | 400 billion |
| Expert Structure | 16 experts | 128 experts |
| Context Window | 10 million tokens | 1 million tokens |
| Pretraining Tokens | ~40 trillion | ~22 trillion |
| Supported Languages | Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai, Vietnamese | same |
| Input Modalities | Multilingual text and images | Multilingual text and images |
| Output Modalities | Multilingual text and code | Multilingual text and code |
| Multi-image Support | Up to 8 images tested, 48 in training | Up to 8 images tested, 48 in training |
| Knowledge Cutoff | August 2024 | August 2024 |
| Hardware Requirements | Single H100 GPU with int4 quantization | Single H100 DGX host with FP8 quantization |
| License | Llama 4 Community License | Llama 4 Community License |

### Training compute and emissions

| Item | Value |
|---|---|
| Scout training compute | 5.0M GPU hours on H100-80GB (TDP 700W) |
| Maverick training compute | 2.38M GPU hours on H100-80GB (TDP 700W) |
| Training precision | FP8 |
| Location-based emissions | 1,999 tons CO2eq |
| Market-based emissions | 0 tons CO2eq (100% renewable energy) |

- The vision encoder is based on **MetaCLIP**, trained separately in conjunction with a frozen Llama model.
- Weights: BF16 for Scout; BF16 and FP8 for Maverick, with on-the-fly int4 quantization code provided.

### Example System Prompt for LLAMA4

The author includes Meta's sample system prompt (summarized here; see the original page for the exact text). It sets up an "expert conversationalist who responds to the best of your ability" and asks the model to:

- match tone to the user's mood (listening, humor, emotional support vs. detailed help);
- avoid templated, preachy or moralizing language and not lecture the user;
- engage with political and social topics rather than refusing;
- identify itself as Llama 4 with a knowledge cutoff of August 2024;
- list the 12 supported languages and respond in the user's language by default.

## Benchmarks

### Pre-trained model benchmarks

| Benchmark | Llama 3.1 70B | Llama 4 Scout | Llama 4 Maverick |
|---|---|---|---|
| MMLU | 79.3 | 79.6 | 85.5 |
| MMLU-Pro | 53.8 | 58.2 | 62.9 |
| MATH | 41.6 | 50.3 | 61.2 |
| MBPP (Code) | 66.4 | 67.8 | 77.6 |
| TydiQA | 29.9 | 31.5 | 31.7 |

### Instruction-tuned model benchmarks

| Benchmark | Llama 3.3 70B | Llama 4 Scout | Llama 4 Maverick |
|---|---|---|---|
| MMLU Pro | 68.9 | 74.3 | 80.5 |
| GPQA Diamond | 50.5 | 57.2 | 69.8 |
| LiveCodeBench | 33.3 | 32.8 | 43.4 |
| MGSM | 91.1 | 90.6 | 92.3 |

### Multimodal benchmarks

| Benchmark | Llama 4 Scout | Llama 4 Maverick |
|---|---|---|
| MMMU | 69.4 | 73.4 |
| MMMU Pro | 52.2 | 59.6 |
| MathVista | 70.7 | 73.7 |
| ChartQA | 88.8 | 90.0 |
| DocVQA | 94.4 | 94.4 |

### Long-context benchmarks (MTOB — Machine Translation from One Book)

| Benchmark | Llama 3.3 70B | Llama 4 Scout | Llama 4 Maverick |
|---|---|---|---|
| MTOB (half book) eng→kgv | context window is 128K | 54.0 | 42.2 |
| MTOB (half book) kgv→eng | context window is 128K | 46.4 | 36.6 |
| MTOB (full book) eng→kgv | context window is 128K | 50.8 | 39.7 |
| MTOB (full book) kgv→eng | context window is 128K | 46.7 | 36.3 |

### Competitive claims
- Maverick exceeds GPT-4o and Gemini 2.0 Flash on several categories, reached **1417 ELO** on LMArena (experimental chat version), and competes with DeepSeek v3.1 on reasoning and coding with fewer active parameters.
- Scout outperforms Gemma 3, Gemini 2.0 Flash-Lite and Mistral 3.1 across many benchmarks.
- Behemoth beats GPT-4.5, Claude Sonnet 3.7 and Gemini 2.0 Pro on STEM tasks.

These are the publisher's (Meta's) own claims; verify against the official model cards.

## How to Use LLAMA4

### Deployment options
- **Cloud platforms:** AWS, Azure, Google Cloud, Hugging Face, Together AI.
- **Direct download:** from llama.com, Hugging Face or GitHub to run on your own infrastructure (typically an H100 or equivalent).
- **Pre-built apps:** Meta AI, WhatsApp, Messenger and Instagram Direct, with no setup needed.

### Getting-started steps
1. **Choose a deployment method** — cloud, local installation or a pre-built app, depending on needs and technical capacity.
2. **Select a model** — Scout for long context and efficiency, Maverick for stronger performance and reasoning.
3. **Set up the environment** — for self-hosting, confirm hardware (H100 or equivalent) and follow installation instructions.
4. **Optimize for your use case** — configure system prompts, adjust generation parameters, add safety measures.
5. **Integrate and deploy** — connect through the provided APIs/SDKs and roll out to users or internal systems.

### Using with Hugging Face

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Create prompt
prompt = """
<|system|>
You are an expert assistant that helps users with information about technology.
</|system|>

<|user|>
What are the key features of LLAMA4?
</|user|>

<|assistant|>
"""

# Tokenize prompt
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
)

# Decode and print response
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

### Using with vLLM

```python
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
)

# Create sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

# Create prompt
prompt = """
<|system|>
You are an expert assistant that helps users with information about technology.
</|system|>

<|user|>
What are the key features of LLAMA4?
</|user|>

<|assistant|>
"""

# Generate response
outputs = llm.generate([prompt], sampling_params)

# Print response
print(outputs[0].outputs[0].text)
```

### Using LLAMA4 with Images

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image
import requests

# Load model and processor
model = LlavaForConditionalGeneration.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")
processor = AutoProcessor.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")

# Load image
url = "https://example.com/image.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Create prompt
prompt = "What can you see in this image?"

# Process inputs
inputs = processor(prompt, image, return_tensors="pt").to(model.device)

# Generate response
outputs = model.generate(**inputs, max_new_tokens=100)

# Decode and print response
print(processor.decode(outputs[0], skip_special_tokens=True))
```

> Note: the image example uses the `LlavaForConditionalGeneration` class; the official Transformers class for Llama 4 is `Llama4ForConditionalGeneration`. Check the current Hugging Face model card before running.

## Use Cases

- **Software development:** code generation, debugging, documentation, modernizing legacy code, understanding large codebases.
- **Data analysis:** interpreting charts, summarizing data, identifying trends, analyzing financial reports.
- **Education and research:** personalized tutoring, research synthesis, educational content, literature analysis.
- **Healthcare:** medical literature review, image analysis assistance, record summarization, trial documentation.
- **Content creation:** long-form writing, image and video analysis, marketing material, multilingual adaptation.
- **Scout-specific (long context):** analyzing whole books or many papers at once, comparing lengthy documents, understanding full codebases, maintaining very long conversation history.
- **Enterprise:** multimodal customer support, contract review, knowledge-base enrichment, business intelligence.
- **Creative:** screenplay development, visual inspiration, interactive storytelling, art or music critique.
- **Personal:** note-taking, photo management, multi-subject learning, travel planning with visual input.

## FAQ

**What makes LLAMA4 different from earlier Llama models?** It is the first Llama to use mixture-of-experts, it is natively multimodal, it supports up to 10M-token context (vs. 128K), and it was trained on up to 40T tokens.

**How do I choose between Scout and Maverick?** Choose Scout for very long context or limited hardware. Choose Maverick for top multimodal performance, advanced reasoning and production deployments with more capable infrastructure.

**What hardware is needed?** Scout runs on a single H100 with int4 quantization. Maverick needs a single H100 DGX host with FP8 quantization. Cloud providers are an alternative to self-hosting.

**Which languages are supported?** Twelve officially (Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai, Vietnamese), though pre-training covered 200 languages.

**Is it available for commercial use?** Yes, under the Llama 4 Community License, subject to its terms and Meta's acceptable use policy.

**When will Behemoth be released?** No date has been announced. It is still in training and serves as a teacher model for Scout and Maverick.

## Figures on the original page
1. Meta's LLAMA4 Models — The Next Generation of AI
2. LLAMA4 Mixture-of-Experts Architecture
3. LLAMA4 Benchmark Comparison with Competing Models
4. LLAMA4 Scout's Long Context Capability Visualization
5. LLAMA4 Behemoth Architecture Overview
6. LLAMA4 Performance on Image Understanding Benchmarks
7. LLAMA4 Setting New Standards in AI Performance
8. LLAMA4 Powering Various Real-World Applications
9. Meta AI interface powered by LLAMA4

## Resources
- Official site and downloads: https://www.llama.com — https://www.llama.com/llama-downloads/
- Documentation: https://www.llama.com/docs/
- Technical blog: https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- GitHub: https://github.com/meta-llama
- Hugging Face: https://huggingface.co/meta-llama
- Meta AI: https://meta.ai/
- Community: https://discord.gg/meta-llama

## Key concepts worth simulating

- **Active vs. total parameters in MoE:** slider over `num_experts` and `top_k` showing active = total × k/N; compare Scout (16 experts, 109B → 17B) and Maverick (128 experts, 400B → 17B) and plot per-token FLOPs vs. memory.
- **Shared expert + one routed expert (Maverick):** animate a token flowing through the always-on shared FFN and a softmax gate that picks one of 128 routed FFNs; show how the two outputs are summed.
- **Early fusion multimodality:** show image patches tokenized by the MetaCLIP-based encoder being interleaved with text tokens into a single sequence entering the same transformer backbone.
- **Context window scale:** interactive bar chart on a log axis — 128K (Llama 3) vs. 1M (Maverick) vs. 10M (Scout) — translated into "pages of text" or "lines of code".
- **iRoPE / positional-embedding-free attention:** toggle RoPE on/off in an attention layer and show how attention scores change as sequence length grows beyond training length.
- **Inference-time temperature scaling of attention:** slider scaling attention logits and plotting the resulting softmax sharpness for long sequences.
- **Autoregressive decoding with temperature and top-p:** visualize next-token probability distribution, nucleus truncation at p = 0.9, and sampling at T = 0.7.
- **Quantization (BF16 → FP8 → int4):** histogram of weights before/after quantization and resulting memory footprint for 109B parameters.

## Test code

```python
# 1. Active-parameter math for MoE models (no GPU needed)
models = {
    "Scout":    dict(total=109e9, experts=16,  active=17e9),
    "Maverick": dict(total=400e9, experts=128, active=17e9),
    "Behemoth": dict(total=2e12,  experts=16,  active=288e9),
}
for name, m in models.items():
    frac = m["active"] / m["total"]
    print(f"{name:9s} active/total = {frac:6.1%}  "
          f"(~{m['total']*2/1e9:.0f} GB in BF16, ~{m['total']*0.5/1e9:.0f} GB in int4)")
```

```python
# 2. Softmax sampling with temperature and top-p (nucleus), as used in the generate() calls
import numpy as np

def sample_next(logits, temperature=0.7, top_p=0.9, rng=np.random.default_rng(0)):
    z = np.asarray(logits) / temperature
    p = np.exp(z - z.max()); p /= p.sum()
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    keep = order[: np.searchsorted(cum, top_p) + 1]
    q = p[keep] / p[keep].sum()
    return rng.choice(keep, p=q), dict(zip(keep.tolist(), q.round(3).tolist()))

tok, kept = sample_next([2.0, 1.5, 1.0, 0.2, -1.0])
print("sampled token id:", tok, "| nucleus:", kept)
```

```python
# 3. Context-window scale: how many tokens is a document? (needs: pip install tiktoken)
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
text = open(__file__).read()  # any file you like
n = len(enc.encode(text))
for model, ctx in [("Llama 3", 128_000), ("Maverick", 1_000_000), ("Scout", 10_000_000)]:
    print(f"{model:9s} window holds ~{ctx // n:,} copies of this file ({n} tokens)")
```

```python
# 4. Smoke test the Hugging Face loading pattern on a tiny model (CPU-friendly stand-in for Llama 4)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
name = "sshleifer/tiny-gpt2"   # swap for "meta-llama/Llama-4-Scout-17B-16E-Instruct" on an H100
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
inputs = tok("What are the key features of LLAMA4?", return_tensors="pt")
out = model.generate(inputs.input_ids, max_new_tokens=20, do_sample=True, temperature=0.7, top_p=0.9)
print(tok.decode(out[0], skip_special_tokens=True))
```
