---
week: 0
title: Final Project — Multi-Agent Research Assistant
source: https://albanna-tutorials.com/cs/project_research_assistant.html
author: Dr. Abdulkarim Albanna, University of Petra
---
# Final Project — Multi-Agent Research Assistant

> *Automated Research, Writing & Review System.* Course 606363, Computing Systems for Data Science and AI, 2nd semester 2025–2026. Instructor: Dr. Abdulkarim Albanna, Faculty of Information Technology, Data Science and AI, University of Petra.
>
> **Team size:** 2–3 students · **Difficulty:** Advanced · **Deadline:** Week 14 · **Competition Day:** Week 15

## 1. Project Overview

Teams build a **multi-agent system** in which specialized agents collaborate to produce a reviewed, cited research report on any topic. The input is a research question; the output is a Markdown/PDF report with **named-entity extraction**, **source classification**, **browser-based data collection** and **AI-generated illustrations**.

**The problem.** Real research involves searching, reading, classifying sources, extracting entities, organizing findings, writing, illustrating and reviewing. The project automates this pipeline, with each agent handling one stage under a central orchestrator.

**Deliverables.** A working application, source code, documentation, a transformer analysis report, a presentation and a demo video.

## 2. Mapping to Course Topics

| Week | Topic | Project component |
|---|---|---|
| 1 | Transformers | Document the architecture of the chosen LLM backbone |
| 2 | Text classification | Classification Agent: source categorization |
| 3 | NER | NER Agent: entity extraction |
| 4 | LangChain | Chains, prompts, output parsers, document loaders |
| 5 | LangGraph | Orchestration (StateGraph) |
| 6 | LLMs | Model configuration and tuning per agent |
| 7 | Mixture of Experts | MoE analysis; optional Mixtral integration |
| 8 | Midterm | (no project deliverable) |
| 9 | ReAct | Research Agent |
| 10 | Multi-agent systems | Full integration, revision loop |
| 11 | MCP | Tool protocol: MCP servers and clients |
| 12 | LAMs | Browser Agent (browser automation) |
| 13 | Diffusion models | Illustration Agent |
| 14 | — | Testing, documentation, presentation |
| 15 | — | Competition Day |

## 3. System Architecture

```text
                    ┌──────────────────────────────┐
   User input ───►  │  ORCHESTRATOR (LangGraph      │
  (research topic)  │  StateGraph, conditional edges)│
                    └──────────────┬───────────────┘
          ┌──────────────┬─────────┼─────────┬──────────────┐
          ▼              ▼         ▼         ▼              ▼
   ┌────────────┐ ┌────────────┐ ┌─────┐ ┌─────────┐ ┌──────────┐
   │  Research  │ │Classifica- │ │ NER │ │ Browser │ │ Analyzer │
   │ Agent      │ │tion Agent  │ │Agent│ │ Agent   │ │ Agent    │
   │ (ReAct+MCP)│ │            │ │     │ │ (LAM)   │ │ (CoT)    │
   └─────┬──────┘ └─────┬──────┘ └──┬──┘ └────┬────┘ └────┬─────┘
         └──────────────┴───────────┴─────────┴───────────┘
                                    ▼
                           ┌─────────────────┐
                           │  Writer Agent   │◄──────────────┐
                           └────────┬────────┘               │
                                    ▼                        │ REVISE
                        ┌──────────────────────┐             │ (score < 7,
                        │  Illustration Agent  │             │  max 3 cycles)
                        │  (Stable Diffusion / │             │
                        │   DALL-E)            │             │
                        └──────────┬───────────┘             │
                                   ▼                         │
                           ┌─────────────────┐               │
                           │  Critic Agent   │───────────────┘
                           │  (Reflection,   │
                           │   score 1–10)   │
                           └────────┬────────┘
                                    │ APPROVE (score ≥ 7)
                                    ▼
                          Final Report (Markdown / PDF)
```

Flow: the user's topic enters the **Orchestrator** (a LangGraph `StateGraph`). It fans out to the **Research**, **Classification**, **NER**, **Browser** and **Analyzer** agents, whose outputs converge on the **Writer**. The draft passes to the **Illustration Agent**, then to the **Critic**, which either approves (producing the final report) or sends the draft back to the Writer for revision.

## 4. Technology Stack

| Component | Options |
|---|---|
| Transformer backbone | Transformer LLMs such as GPT-4, Claude or Mixtral; students must document the architecture |
| Agent orchestration | LangGraph (stateful graph, conditional edges) |
| Chain framework | LangChain (chains, prompts, output parsers, document loaders) |
| LLM | OpenAI GPT-4 or Claude API; Ollama for local models; optional Mixtral (MoE) |
| Text classification | LLM-based classifier, or fine-tuned transformer (BERT) |
| NER | spaCy NER pipeline, or LLM-based structured extraction |
| Web search | Tavily Search API or Brave Search API |
| Web scraping | BeautifulSoup or Trafilatura |
| Browser automation (LAM) | Playwright or Selenium: form filling, navigation, structured extraction |
| Tool protocol | Model Context Protocol (MCP) |
| Image generation | Stable Diffusion API or DALL-E |
| Vector store | ChromaDB |
| Output format | Markdown with embedded images → PDF via markdown2 + WeasyPrint |
| UI | Streamlit or Gradio |
| Language | Python 3.12+ |

## 5. Agent Specifications

### 5.1 Orchestrator (Manager)

- **Input:** research topic from the user.
- **Output:** routing decisions, revision loops, finalization.
- **Tools:** LangGraph `StateGraph` with conditional edges.
- **Responsibilities:** define research scope; route tasks to agents; manage shared state; decide whether to revise or finalize; implement human-in-the-loop checkpoints.

### 5.2 Research Agent (ReAct)

- **Input:** `topic`, `sub_questions`.
- **Output:** `raw_sources` entries `{url, title, content, score}`.
- **Tools (via MCP server):** Web Search (Tavily or Brave), Web Scraper (BeautifulSoup or Trafilatura), URL Reader.
- **Responsibilities:** generate search queries; search multiple sources; scrape pages; score relevance 1–10 with an LLM; keep URLs for citation. Runs a Thought → Action → Observation loop until enough sources are gathered.

### 5.3 Classification Agent

- **Input:** `raw_sources`.
- **Output:** `classified_sources` `{url, title, category, relevance}` plus a classification summary.
- **Tools:** LLM-based classifier or fine-tuned BERT.
- **Responsibilities:** assign topic labels (e.g. technical, review, tutorial, opinion, dataset); rate relevance high / medium / low; filter off-topic sources.

### 5.4 NER Agent (Entity Extractor)

- **Input:** collected and classified sources.
- **Output:** `entities` `{text, label, source_url, frequency}` and `entity_relationships` `{entity1, entity2, relation, context}`.
- **Tools:** spaCy NER pipeline or LLM extraction with structured output.
- **Responsibilities:** extract people, organizations, technologies, dates and locations; build an entity registry; track co-occurrences; report frequencies.

### 5.5 Browser Agent (LAM)

- **Input:** target pages and academic databases.
- **Output:** `browser_results` `{url, action, extracted_data}`.
- **Tools:** Playwright or Selenium with LLM planning.
- **Responsibilities:** render JavaScript-heavy pages; fill search forms (Google Scholar, Semantic Scholar); extract table data; download PDFs and datasets; handle authentication when credentials are provided.

### 5.6 Analyzer Agent

- **Input:** classified sources and entities.
- **Output:** `organized_findings` `{theme, findings, sources, entities}` and an `outline`.
- **Pattern:** Chain-of-Thought.
- **Responsibilities:** identify 4–6 themes; group findings by topic; detect contradictions between sources; rank findings; build a report outline that includes entity relationships.

### 5.7 Writer Agent

- **Input:** `outline`, `organized_findings`.
- **Output:** `draft` in Markdown, with image placeholders and prompts for the Illustration Agent.
- **Tools:** structured output and citation formatting.
- **Responsibilities:** write introduction, body, conclusion and executive summary with inline citations; weave in extracted entities.

### 5.8 Illustration Agent

- **Input:** image prompts from the Writer.
- **Output:** `illustrations` `{section, prompt, image_path}`.
- **Tools:** Stable Diffusion API or DALL-E.
- **Responsibilities:** generate diagrams, architecture illustrations and data visualizations; optimize prompts for an academic style; embed images in the report.

### 5.9 Critic Agent

- **Input:** draft, sources, entities, classification labels.
- **Output:** `critic_feedback`, `critic_score`, decision.
- **Pattern:** Reflection.
- **Responsibilities:** check accuracy against sources; find coverage gaps; verify entity and label use; score 1–10 on accuracy, completeness, clarity, citations and visual quality. **Score ≥ 7 → APPROVE; otherwise → REVISE** with specific feedback, returning the draft to the Writer. Maximum **3 revision cycles**.

## 6. Shared State Schema

```python
from typing import List, Optional
from typing_extensions import TypedDict


class ResearchState(TypedDict):
    # Input
    topic: str
    sub_questions: List[str]

    # Research Agent
    raw_sources: List[dict]            # {url, title, content, score}

    # Classification Agent
    classified_sources: List[dict]     # {url, title, category, relevance}

    # NER Agent
    entities: List[dict]               # {text, label, source_url, frequency}
    entity_relationships: List[dict]   # {entity1, entity2, relation, context}

    # Browser Agent (LAM)
    browser_results: List[dict]        # {url, action, extracted_data}

    # Analyzer Agent
    organized_findings: List[dict]     # {theme, findings, sources, entities}
    outline: str

    # Writer Agent
    draft: str

    # Illustration Agent
    illustrations: List[dict]          # {section, prompt, image_path}

    # Critic Agent
    critic_feedback: Optional[str]
    critic_score: Optional[int]
    revision_count: int
    final_report: Optional[str]

    # Course-topic artefacts
    transformer_config: dict
    moe_analysis: Optional[str]
    status: str
```

## 7. Competition Format (Competition-Based Learning)

Each team builds its own system; teams compete head-to-head on identical tasks.

- **Checkpoint 1 (Week 7):** demo of the NLP pipeline (classification, NER, research agent). Instructor feedback only, unscored.
- **Checkpoint 2 (Week 11):** demo of the full multi-agent pipeline with MCP. Peer-feedback round.
- **Competition Day (Week 15):**
  1. The instructor reveals **3 unseen research topics**, the same for all teams.
  2. All teams run their systems simultaneously.
  3. Each system produces 3 reports, **30 minutes per topic**.
  4. Reports are anonymized and given to a judging panel: the instructor plus 2 external judges or peer teams.
  5. Each team gives a **10-minute live presentation** of its architecture and a demo.
  6. Judges score reports and presentations independently; scores are averaged across judges and topics.

### Scoring rubric (100 points per report)

| Criterion | Points | What is judged |
|---|---|---|
| Report quality | 20 | Coherence, depth, clarity, structure |
| Source coverage | 15 | At least 10 relevant, diverse sources; proper citations |
| Entity extraction (NER) | 10 | Accuracy of people, organizations, technologies |
| Source classification | 10 | Correct type / domain / relevance labels |
| Illustrations | 10 | Relevance and quality of generated figures |
| Speed | 10 | Completion time (faster is better, within 30 min) |
| Cost efficiency | 5 | Total API token cost (lower is better) |
| Error handling | 5 | API errors, bad sources, timeouts |
| Innovation | 15 | MoE routing, advanced MCP tools, LAM actions, multi-model strategies |

### Prizes and bonus points

| Award | Bonus |
|---|---|
| 1st place (Gold) | +5 |
| 2nd place (Silver) | +3 |
| 3rd place (Bronze) | +2 |
| Innovation Award (stackable) | +2 |
| Best Report Quality Award (stackable) | +2 |
| Speed Award | +1 |

### Academic integrity

Code must be the team's original work. Open-source libraries and LLM APIs are allowed; copying other teams' agent logic or prompts is prohibited. Violations lead to disqualification and a failing grade.

## 8. Phase-by-Phase Timeline

### Phase 1 — Transformer foundations and text classification (Weeks 1–2)

- 1.1 Written analysis of the transformer architecture of the chosen backbone
- 1.2 Project repository with `requirements.txt`
- 1.3 `.env` with credentials and documented model configuration
- 1.4 Working development environment
- 1.5 Working source classifier
- 1.6 Classification accuracy report

### Phase 2 — NER and LangChain setup (Weeks 3–4)

- 2.1 NER pipeline for people, organizations, technologies and dates
- 2.2 Structured entity database with co-occurrence tracking
- 2.3 Reusable LangChain chain components
- 2.4 Multi-format document ingestion pipeline
- 2.5 Complete `ResearchState` schema

### Phase 3 — LangGraph orchestration and LLM configuration (Weeks 5–6)

- 3.1 Connected graph with conditional edges
- 3.2 Working orchestration flow
- 3.3 Documented LLM configuration per agent
- 3.4 LLM benchmark comparison table (2+ models, token cost tracking)
- 3.5 Searchable ChromaDB vector store

### Phase 4 — MoE analysis and Research Agent (Week 7, then Week 9)

- 4.1 Written MoE analysis: dense vs. sparse (Week 7)
- 4.2 MoE integration or comparative benchmark, optional (Week 7)
- 4.3 ReAct research agent (Week 9)
- 4.4 Working search with result parsing (Week 9)
- 4.5 Full-text extraction from URLs (Week 9)
- 4.6 Autonomous iterative search (Week 9)

*Week 7: Checkpoint 1. Week 8: Midterm exam.*

### Phase 5 — Multi-agent integration and MCP (Weeks 10–11)

- 5.1 End-to-end multi-agent pipeline
- 5.2 Organized findings with entity-enriched themes
- 5.3 Complete draft with inline citations
- 5.4 Review loop with a maximum of 3 revision cycles
- 5.5 MCP-compliant tool definitions for search, scrape, classify and NER
- 5.6 MCP client integration so agents call tools through a standard protocol

*Week 11: Checkpoint 2.*

### Phase 6 — Browser Agent (LAM) and diffusion models (Weeks 12–13)

- 6.1 Playwright agent that navigates, fills forms and extracts data
- 6.2 Automated paper search and metadata extraction
- 6.3 Illustration agent producing diagrams and figures
- 6.4 Prompt optimization for academic illustrations
- 6.5 Reports with embedded generated images
- 6.6 Streamlit web UI with real-time progress indicators showing agent activity
- 6.7 Expandable panels showing agent reasoning (transparent decision log)

### Phase 7 — Testing, documentation and presentation (Week 14)

- 7.1 Five complete reports on diverse topics, with illustrations
- 7.2 Benchmarks: time, token cost, MoE vs. dense
- 7.3 Component metrics: NER, classification, illustration quality
- 7.4 README, technical documentation, transformer analysis
- 7.5 Slides and a 5-minute video walkthrough

*Week 15: Competition Day.*

## 9. Submission Requirements

- [ ] GitHub repository with clean commit history and README
- [ ] Transformer analysis of the chosen LLM backbone
- [ ] MoE vs. dense comparison report with benchmarks
- [ ] Technical report: architecture, design decisions, NER and classification evaluation, challenges, results
- [ ] Demo video (5 minutes) showing NER, classification, browser actions and illustrations
- [ ] Five sample research reports on different topics, with entity summaries, classification reports and illustrations
- [ ] Live presentation (10 minutes plus Q&A)
- [ ] Week 7 Checkpoint 1 demo (NLP pipeline)
- [ ] Week 11 Checkpoint 2 demo (full pipeline with MCP) and peer-feedback round
- [ ] Peer review of two other teams' reports
- [ ] **Deadline: Week 14**

## 10. Project Grading

| Component | Weight | Notes |
|---|---|---|
| Competition score | 40% | Average across the 3 unseen topics |
| Architecture & topic coverage | 20% | All 12 course topics (Weeks 1–7 and 9–13) represented |
| Code quality & documentation | 15% | |
| Live presentation | 15% | 10-minute demo plus Q&A |
| Peer review | 10% | Each team reviews two other teams; graded on review quality |

(These are the weights *within* the project's competition and final-submission component, which is 20% of the course grade — 8% competition score, 4% architecture, 3% code quality, 3% presentation, 2% peer review. Weekly deliverables add a further 10%, so the project as a whole is 30% of the course grade — see the syllabus.)

## 11. Expected Output Example

```markdown
# Research Report: <topic>

## Executive Summary
...

## Named Entities Identified
| Entity | Type | Frequency | Sources |
|---|---|---|---|

## Source Classification Summary
| Category | Count | Avg. relevance |
|---|---|---|

## 1. Introduction
## 2. Background
## 3. Key Themes
## 4. Comparative Analysis
[Figure 1: MoE vs Dense Model Efficiency Comparison -- generated illustration]
## 5. Contradictions and Open Questions
## 6. Discussion
## 7. Conclusion

## References
[1] ...
```

## 12. Useful Links

- spaCy: https://spacy.io
- LangChain: https://python.langchain.com
- LangGraph: https://langchain-ai.github.io/langgraph/
- Model Context Protocol: https://modelcontextprotocol.io
- Playwright: https://playwright.dev
- Stability AI: https://stability.ai
- Tavily: https://tavily.com
- Course home: https://albanna-tutorials.com/cs/ · Syllabus: https://albanna-tutorials.com/cs/syllabus.html
- Instructor: abanna@uop.edu.jo · GitHub https://github.com/abedbanna · Hugging Face https://huggingface.co/abedbanna

## Key concepts worth simulating

- **LangGraph StateGraph execution:** animate the `ResearchState` dict flowing node to node (Research → Classify → NER → Analyze → Write → Illustrate → Critic), highlighting which fields each node fills in.
- **Critic revision loop:** conditional edge with score threshold 7 and `revision_count` capped at 3; a slider for the critic score shows APPROVE vs REVISE routing and the loop terminating.
- **ReAct loop of the Research Agent:** Thought → Action (search / scrape tool call via MCP) → Observation, accumulating `raw_sources` with 1–10 relevance scores.
- **MCP host/client/server:** agents (clients) discovering and invoking `search`, `scrape`, `classify`, `ner` tools from an MCP server with JSON-schema'd arguments.
- **Entity co-occurrence graph:** entities as nodes, `entity_relationships` as edges, node size = frequency; built live as sources are processed.
- **Source classification funnel:** raw sources → category labels (technical / review / tutorial / opinion / dataset) → relevance high/medium/low → filtered set; show the minimum-10-sources rubric line.
- **Competition scoring:** 9 rubric criteria summing to 100, averaged over 3 topics and judges, plus stackable bonuses.
- **Parallel fan-out / fan-in:** the five mid-tier agents running concurrently vs. sequentially and the effect on the 30-minute time budget and token cost.

## Test code

```python
# 1. Minimal LangGraph skeleton with the Critic revision loop
#    pip install langgraph typing_extensions
from typing import List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class ResearchState(TypedDict, total=False):
    topic: str
    draft: str
    critic_score: Optional[int]
    critic_feedback: Optional[str]
    revision_count: int
    final_report: Optional[str]

def writer(state):
    n = state.get("revision_count", 0)
    return {"draft": f"Draft v{n+1} on '{state['topic']}'"}

def critic(state):
    score = 5 + state.get("revision_count", 0)          # toy critic: improves each round
    return {"critic_score": score, "critic_feedback": "add citations" if score < 7 else "ok",
            "revision_count": state.get("revision_count", 0) + 1}

def route(state):
    if state["critic_score"] >= 7 or state["revision_count"] >= 3:
        return "finalize"
    return "writer"

def finalize(state):
    return {"final_report": state["draft"]}

g = StateGraph(ResearchState)
g.add_node("writer", writer); g.add_node("critic", critic); g.add_node("finalize", finalize)
g.add_edge(START, "writer"); g.add_edge("writer", "critic")
g.add_conditional_edges("critic", route, {"writer": "writer", "finalize": "finalize"})
g.add_edge("finalize", END)
app = g.compile()

print(app.invoke({"topic": "Mixture of Experts", "revision_count": 0}))
```

```python
# 2. NER Agent prototype with spaCy  (pip install spacy && python -m spacy download en_core_web_sm)
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")
text = ("In 2017 Vaswani and colleagues at Google introduced the Transformer. "
        "OpenAI later released GPT-4, while Mistral AI built Mixtral, a mixture-of-experts model.")
doc = nlp(text)

entities = [{"text": e.text, "label": e.label_} for e in doc.ents]
print(entities)
print(Counter(e["label"] for e in entities))
```

```python
# 3. Classification Agent prototype using a zero-shot transformer
from transformers import pipeline

clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
categories = ["technical", "review", "tutorial", "opinion", "dataset"]
source = "This step-by-step guide shows how to fine-tune DistilBERT for emotion classification."
out = clf(source, candidate_labels=categories)
print(dict(zip(out["labels"], [round(s, 3) for s in out["scores"]])))
```

```python
# 4. Competition score: average of 3 topics over the 100-point rubric
rubric = {"report_quality": 20, "source_coverage": 15, "entity_extraction": 10,
          "source_classification": 10, "illustrations": 10, "speed": 10,
          "cost_efficiency": 5, "error_handling": 5, "innovation": 15}

def score_report(earned_fraction: dict) -> float:
    return sum(rubric[k] * earned_fraction.get(k, 0.0) for k in rubric)

topic_scores = [score_report({k: f for k in rubric}) for f in (0.75, 0.8, 0.9)]
print("per topic:", topic_scores, "average:", round(sum(topic_scores) / 3, 1))
```
