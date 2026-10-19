---
week: 0
title: Course Syllabus — 606363 Computing Systems for Data Science and AI
source: https://albanna-tutorials.com/cs/syllabus.html
author: Dr. Abed Alkarim Banna (Dr. Abdulkarim Albanna), University of Petra
---
# Course Syllabus — 606363 Computing Systems for Data Science and AI

> Syllabus by Dr. Abed Alkarim Banna (Dr. Abdulkarim Albanna), Faculty of Information Technology, University of Petra. Content below is a close paraphrase of the original page (read through a paraphrasing fetch tool); all numbers, weights, weeks and references are preserved as stated. The original page contains no code.

## Course Information

| Item | Detail |
|---|---|
| Course number | 606363 |
| Course title | Computing Systems for Data Science and AI |
| Semester | Second semester, 2025–2026 |
| Faculty | Faculty of Information Technology, University of Petra |
| Credit hours | 3 |
| Prerequisite | 606362 |
| Course coordinator | Dr. Abed Alkarim Banna |

## Course Description

The course covers the full stack for building modern AI systems: from transformer models and NLP (text classification, named entity recognition) through large language models, mixture-of-experts architectures and agent frameworks (LangChain, LangGraph), to AI agents, multi-agent systems, the Model Context Protocol (MCP), large action models and diffusion models. Students learn through hands-on, competition-based projects that integrate every topic into one working system.

## Course Objectives

1. Understand transformer architectures and their use in NLP and named entity recognition.
2. Study large language models, mixture-of-experts models and prompt engineering.
3. Build applications with LangChain (retrieval-augmented generation) and LangGraph (agent orchestration).
4. Design agents using the ReAct (reasoning and acting) framework.
5. Understand multi-agent systems, design patterns and communication protocols.
6. Explore large action models that act autonomously.
7. Understand generative AI through diffusion models.

## Intended Learning Outcomes (ILOs)

| Code | Domain | Outcome | Assessed by |
|---|---|---|---|
| K1, K2 | Knowledge | Explain transformers, LLMs, mixture-of-experts, agent architectures, ReAct and multi-agent systems | Final exam |
| I1 | Intellectual | Design LangGraph agent workflows with reasoning, tool use and collaboration | Mid exam |
| P1, P2 | Practical | Apply fine-tuned BERT and GPT models to NER and classification; build RAG applications; build agents and multi-agent systems | Project rubric |
| T1 | Transferable | Apply systematic thinking to designing and deploying AI applications | Mid exam |

## Weekly Schedule

| Week | Topic | Subtopics / details | ILO | Readings | Assignments / project deliverables |
|---|---|---|---|---|---|
| 1 | Transformer Architecture | Encoder–decoder framework; evolution of transformers; Hugging Face ecosystem (models, datasets, tokenizers, pipelines) | K1 | Ref 1 Ch.1; Ref 7 | Form teams, create GitHub repo, Transformer Analysis document |
| 2 | Text Classification | Transformer-based classification; sentiment and multi-class tasks; feature extraction vs. fine-tuning; evaluation metrics | K1, P1 | Ref 1 Ch.2 | Classification Agent: categorize sources by type and relevance |
| 3 | Named Entity Recognition (NER) | Token classification with BERT; BIO/IOB tagging; multilingual NER; fine-tuning BERT for custom NER | P1, T1 | Ref 1 Ch.4 | NER Agent: extract entities (people, orgs, tech, dates) from sources |
| 4 | LangChain Fundamentals | Chains, prompt templates, memory, output parsers; document loaders, text splitters, vector stores; RAG | K1, P1 | Ref 4 | Agent prompts, document loaders, ChromaDB vector store setup |
| 5 | LangGraph | Agent orchestration as graphs; nodes, edges, state, conditional routing; stateful cyclical graphs; checkpointing; human-in-the-loop | I1, P2 | Ref 4 (LangGraph) | LangGraph StateGraph skeleton with all agent nodes connected |
| 6 | Large Language Models (LLMs) | GPT-4, LLaMA, Gemini, Claude, Mistral, DeepSeek; open vs. closed models; prompt engineering; local models with Ollama | K1, K2 | Ref 3; Lectures | LLM config per agent, benchmark 2+ models, token cost tracking |
| 7 | Mixture of Experts (MoE) | Sparse activation; router/gating network; expert networks; Top-K routing; Switch Transformer, Mixtral, DeepSeek-MoE | K1, T1 | Ref 5; Papers | **CHECKPOINT 1:** demo NLP pipeline + MoE analysis report |
| 8 | Midterm Exam (Weeks 1–7) | — | — | — | — |
| 9 | AI Agents & ReAct | Agent architecture (LLM, tools, memory, planning); ReAct (Thought → Action → Observation); tool use; agent memory | K2, P2 | Ref 2; Ref 4; Ref 6 | Research Agent with ReAct loop, Tavily search, web scraping |
| 10 | Multi-Agent Systems | Agentic design patterns; sequential, parallel, hierarchical workflows; LangGraph, CrewAI, AutoGen; A2A protocols | K2, I1, P2 | Ref 3; Ref 4; Ref 6 | Full pipeline: Analyzer + Writer + Critic agents, revision loop |
| 11 | Model Context Protocol (MCP) | MCP architecture (hosts, clients, servers); resources, tools, prompts, sampling; building MCP servers | K2, P2 | Lectures; MCP docs | **CHECKPOINT 2:** full multi-agent pipeline + 2 MCP tool servers |
| 12 | Large Action Models (LAMs) | LLM → agent → LAM; examples: Rabbit R1, Anthropic Computer Use, OpenAI Operator; function calling; browser automation | K2, P2, T1 | Ref 6; Lectures | Browser Agent (LAM): web navigation, form filling, data extraction |
| 13 | Diffusion Models | Diffusion fundamentals; forward/reverse process; denoising architectures; Stable Diffusion, DALL-E; practical generation | K1, P1 | Ref 8; Lectures | Illustration Agent (diffusion-based figures) + Streamlit UI |
| 14 | Project Development | Final development, end-to-end testing, bug fixes, documentation, instructor consultation | P1, P2 | — | 5 test reports, README, technical docs, demo video, slides |
| 15 | Competition Day | Live head-to-head competition: 3 unseen topics, 30 min/topic; judging panel; 10-min presentations; peer review | P1, P2, T1 | — | **COMPETITION:** live run + presentation + peer review |
| 15–16 | Final Exam | — | K1, K2, I1, T1 | — | — |

## Project: Competition-Based Learning

Teams of 2–3 students build a **Multi-Agent Research Assistant** that integrates every course topic (see the project page, `project.md`). Each team builds its own system and the systems compete head-to-head.

### Checkpoints

| Week | Checkpoint | What to demo | Type |
|---|---|---|---|
| 7 | Checkpoint 1: NLP Pipeline | Classification Agent, NER Agent, Research Agent with search; transformer analysis document; MoE comparison report | Instructor feedback |
| 11 | Checkpoint 2: Full Pipeline + MCP | Complete pipeline (Research → Classify → NER → Analyze → Write → Critic) with MCP tool servers and revision loop | Instructor + peer feedback |
| 15 | Competition Day | 3 unseen topics revealed live; all systems run simultaneously (30 min/topic); anonymized reports scored by a judging panel; 10-minute team presentation | Graded (judging panel) |

### Competition scoring rubric (100 points per report)

| Criterion | Points | Description |
|---|---|---|
| Report quality | 20 | Coherence, depth, clarity, structure |
| Source coverage | 15 | Number and diversity of sources (minimum 10), proper citations |
| Entity extraction (NER) | 10 | Accuracy of extracted people, organizations, technologies |
| Source classification | 10 | Correct categorization by type, domain, relevance |
| Illustrations | 10 | Relevance and quality of diffusion-generated figures |
| Speed | 10 | Pipeline completion time (faster scores higher, within 30 min) |
| Cost efficiency | 5 | Total API token cost per report (lower scores higher) |
| Error handling | 5 | Handling of API errors, bad sources, timeouts |
| Innovation | 15 | MoE routing, advanced MCP tools, LAM actions |

### Competition awards (bonus points)

| Award | Bonus | Criteria |
|---|---|---|
| 1st place (Gold) | +5 | Highest overall competition score |
| 2nd place (Silver) | +3 | Second-highest score |
| 3rd place (Bronze) | +2 | Third-highest score |
| Best Innovation | +2 | Most creative technical features (stackable with placement) |
| Best Report Quality | +2 | Highest report-quality score (stackable with placement) |
| Fastest System | +1 | Fastest average pipeline completion time |

## Assessment

| Component | Weight |
|---|---|
| **Project (competition-based learning)** | **30%** |
| — Weekly deliverables | 10% |
| — Competition score (average across 3 unseen topics) | 8% |
| — Architecture & topic coverage | 4% |
| — Code quality & documentation | 3% |
| — Live presentation | 3% |
| — Peer review (feedback on 2 other teams' reports) | 2% |
| **Mid exam (Weeks 1–7)** | **30%** |
| **Final exam** | **40%** |
| **Total** | **100%** |

## Course Policies

- **Attendance:** University regulations apply.
- **Academic honesty:** Academic dishonesty is not tolerated at the University of Petra. Violations are disciplined under University rules and regulations.
- **Make-up exams:** Only students with valid excuses may take make-up exams. Excuses must be signed by the Faculty Dean, and the student must arrange an exam date with the instructor before the next regular exam.
- **Late work:** Assignments are due on the specified date. Marks are deducted for late work, which is accepted for no more than 3 days after the deadline.
- **Quizzes:** No make-up quizzes under any circumstances.

## References

| Ref | Authors | Title | Year |
|---|---|---|---|
| 1 | Lewis Tunstall, Leandro von Werra, Thomas Wolf | *Natural Language Processing with Transformers*, Revised Edition (O'Reilly) | 2022 |
| 2 | Shunyu Yao et al. | *ReAct: Synergizing Reasoning and Acting in Language Models* (paper) | 2023 |
| 3 | Chip Huyen | *AI Engineering* (O'Reilly) | 2025 |
| 4 | — | LangChain & LangGraph documentation — https://langchain.com | — |
| 5 | Hugging Face | "Mixture of Experts Explained" (blog) and related papers | — |
| 6 | Maarten Grootendorst | *A Visual Guide to LLM Agents* | 2025 |
| 7 | Ashish Vaswani et al. | *Attention Is All You Need* (paper) | 2017 |
| 8 | Calvin Luo | *Understanding Diffusion Models: A Unified Perspective* (paper) | 2022 |

## Key concepts worth simulating

- **Course roadmap timeline:** 16 weeks on a horizontal track with topic nodes, checkpoint markers (W7, W11), midterm (W8), competition (W15) and final exam; clicking a week shows topic, readings and deliverable.
- **Grade calculator:** sliders for project sub-components (10/8/4/3/3/2), mid (30) and final (40) producing the weighted total; show how competition rubric points and bonus awards flow into the 8% competition component.
- **Competition rubric radar/stacked bar:** the nine criteria (20/15/10/10/10/10/5/5/15) for several hypothetical teams, with bonus-award stacking.
- **Project pipeline as the course map:** each syllabus week lights up the agent it produces (Week 2 → Classification Agent, Week 3 → NER Agent, … Week 13 → Illustration Agent).
- **ReAct loop preview (Week 9):** Thought → Action → Observation cycle animation feeding the Research Agent.

## Test code

```python
# 1. Final-grade calculator following the syllabus weights
weights = {
    "weekly_deliverables": 10, "competition_score": 8, "architecture_coverage": 4,
    "code_quality_docs": 3, "live_presentation": 3, "peer_review": 2,
    "mid_exam": 30, "final_exam": 40,
}
assert sum(weights.values()) == 100

def final_grade(scores_pct):
    """scores_pct: component -> score as a percentage (0-100)."""
    return sum(weights[k] * scores_pct[k] / 100 for k in weights)

print(final_grade({k: 85 for k in weights}))   # 85.0
```

```python
# 2. Competition report score from the 100-point rubric, plus bonus awards
rubric = {"report_quality": 20, "source_coverage": 15, "entity_extraction": 10,
          "source_classification": 10, "illustrations": 10, "speed": 10,
          "cost_efficiency": 5, "error_handling": 5, "innovation": 15}
bonus = {"gold": 5, "silver": 3, "bronze": 2, "innovation": 2, "report_quality": 2, "fastest": 1}

def report_score(fractions):
    """fractions: criterion -> fraction of points earned (0-1)."""
    return sum(rubric[c] * fractions.get(c, 0) for c in rubric)

team = {c: 0.8 for c in rubric}
topics = [report_score(team) for _ in range(3)]          # three unseen topics
avg = sum(topics) / 3
print(f"average competition score: {avg:.1f}/100, with Gold + Innovation bonus: {avg + bonus['gold'] + bonus['innovation']:.1f}")
```

```python
# 3. Weekly schedule as structured data (for a timeline widget)
schedule = [
    (1, "Transformer Architecture", "Form teams, GitHub repo, Transformer Analysis document"),
    (2, "Text Classification", "Classification Agent"),
    (3, "Named Entity Recognition", "NER Agent"),
    (4, "LangChain Fundamentals", "Prompts, loaders, ChromaDB"),
    (5, "LangGraph", "StateGraph skeleton"),
    (6, "Large Language Models", "LLM config, benchmark 2+ models, token cost tracking"),
    (7, "Mixture of Experts", "CHECKPOINT 1"),
    (8, "Midterm Exam", ""),
    (9, "AI Agents & ReAct", "Research Agent"),
    (10, "Multi-Agent Systems", "Analyzer + Writer + Critic"),
    (11, "Model Context Protocol", "CHECKPOINT 2: 2 MCP servers"),
    (12, "Large Action Models", "Browser Agent"),
    (13, "Diffusion Models", "Illustration Agent + Streamlit UI"),
    (14, "Project Development", "5 test reports, docs, video, slides"),
    (15, "Competition Day", "Live run + presentation + peer review"),
]
for week, topic, deliverable in schedule:
    print(f"Week {week:>2}: {topic:<28} {deliverable}")
```
