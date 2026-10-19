---
week: 10
title: "A Visual Guide to LLM Agents — Multi-Agent Systems"
source: https://albanna-tutorials.com/llm_agents.html
---
# A Visual Guide to LLM Agents — Multi-Agent Systems

> Explore the fascinating world of Large Language Model Agents. Level: Intermediate · ~30 min.
>
> Course page by Dr. Abedal-Kareem Al-Banna (Al-banna Tutorials), based on the tutorial "A Visual Guide to LLM Agents" by Maarten Grootendorst (https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents). Images and original content are © Maarten Grootendorst; the diagrams referenced below (44 figures) live on the source page and are not reproduced here.

## 1. Introduction to LLMs

- Traditionally, a Large Language Model does nothing more than **predict the next token**. By sampling many tokens in a row, the model can mimic a conversation and give longer answers.
- Two main limitations motivate agents:
  - **No memory between conversations.** An LLM does not remember what was said earlier; a follow-up question cannot be connected to the previous exchange (figure: "LLM Memory Limitation").
  - **Weak basic arithmetic.** LLMs often fail at basic multiplication and division (figure: "LLM Math Limitation").
- This does not make LLMs bad. They do not need to do everything, because **external tools, memory, and retrieval systems** can compensate for these gaps.

## 2. Augmented LLMs & Agents

- Anthropic calls an LLM extended with external systems (tools, memory, retrieval) an **Augmented LLM**. For example, the model can call a calculator rather than do multiplication itself (figure: "Tool Use Example").
- The page uses the Russell & Norvig definition (*AI: A Modern Approach*, 2016): an agent is *anything that can be viewed as perceiving its environment through sensors and acting upon that environment through actuators*.

| Component | Description |
|---|---|
| Environment | The world the agent interacts with |
| Sensors | Used to observe the environment |
| Actuators | Tools used to interact with the environment |
| Effectors | The "brain" or rules that decide how to go from observations to actions |

- Because LLMs are textual models, an LLM agent **observes the environment through textual input** and **acts through tools** (e.g. searching the web). The Augmented LLM is a generalization of this framework: the LLM handles understanding and planning, tools handle actions, and memory tracks what has been done (figures: "Agent Framework", "Augmented LLM Framework", "Agent Components").
- **Autonomy levels.** Agents vary in their degree of autonomy. The more the LLM decides how the system behaves, the more "agentic" the system is (figure: "Agent Autonomy Levels"). Tool use can follow a fixed order set by the framework, or the LLM can choose which tool to use and when.

## 3. Memory Systems

- **Core problem:** LLMs are forgetful; more precisely, they perform no memorization between interactions.

### Short-term memory (working memory)
- A buffer for the (near-)immediate context, including the agent's recent actions.
- The most straightforward implementation is the **context window** — the number of tokens the LLM can process. The LLM is effectively *told* what the conversation was rather than memorizing it (figures: "Context Window", "Context Window Memory").
- **Summarization:** for smaller context windows or large histories, another LLM summarizes the conversation so far (figure: "Memory Summarization").

### Long-term memory
- Covers dozens or hundreds of steps the agent may need to retain.
- Common technique: store all previous interactions, actions, and conversations in an **external vector database**; retrieve relevant pieces when needed. This is **Retrieval-Augmented Generation (RAG)** (figures: "Vector Database Creation", "Vector Database Retrieval").

### Memory types (CoALA)
The paper *Cognitive Architectures for Language Agents* couples psychological memory types to LLM agents. Two are defined on the page:

| Memory type | Definition |
|---|---|
| Semantic | Facts about the world |
| Working | Current and recent circumstances |

Different information can live in different stores — e.g. semantic memory kept separate from working memory — which helps when designing agent frameworks (figure: "Memory Types").

## 4. Tools & Interaction

- Tools let an LLM interact with an external environment (a database) or run external applications (custom code). Two categories (figure: "Tool Types"):
  - **Fetching data** — retrieve up-to-date information.
  - **Taking action** — e.g. scheduling a meeting or ordering food.
- To use a tool, the LLM generates text matching the tool's API, usually **JSON**, so a code interpreter can execute it (figure: "JSON Tool Call"). Illustrative shape of such a call:

```json
{"name": "multiply", "arguments": {"a": 123, "b": 456}}
```

- **Function calling:** developers define custom functions (e.g. a basic multiplication function) that the LLM can call with parameters. Most current LLMs can perform tool use; some need careful prompting.
- **Fixed tool order:** the agentic framework predefines the order of tool use. **Autonomous tool selection:** the LLM decides which tool to use and when. Agents are essentially sequences of LLM calls with autonomous selection of actions and tools.
- Research moved from *prompting* for tool use to *training* for it:
  - **Toolformer** — one of the first approaches to train a model to decide which APIs to call and how (three-step figure series).
  - **ToolLLM** — LLMs using thousands of tools.
  - **Gorilla** — LLMs easily retrieving the most relevant tools.
- **Model Context Protocol (MCP):** Anthropic developed MCP to make tools easier to implement across agentic frameworks; it standardizes API access for services such as weather apps and GitHub (figure: "MCP Components"). See Week 11.

## 5. Planning & Reasoning

- **Planning** breaks a task into actionable steps and lets the model reflect iteratively on past behavior, updating the plan when necessary.
- Planning requires complex **reasoning**; the LLM should show this reasoning before taking the next step. Reasoning is enabled by fine-tuning or by prompt engineering:
  - **Chain-of-Thought prompting** — provide examples of step-by-step reasoning so the model explains its thinking before answering.
  - **Zero-shot Chain-of-Thought** — add the phrase "Let's think step-by-step" without examples.

### ReAct (Reason and Act)
Achieved through prompt engineering rather than training. The prompt describes three steps:

| Step | Description |
|---|---|
| Thought | A reasoning step about the current situation |
| Action | A set of actions to execute (e.g., tools) |
| Observation | A reasoning step about the result of the action |

The cycle Thought → Action → Observation repeats until an action specifies that the result should be returned. By iterating, the LLM can plan actions, observe their output, and adjust.

### Reflexion
- Adds what ReAct lacks: **reflection on failure**. Uses verbal reinforcement so agents learn from prior failures.
- Memory modules track actions (short-term) and self-reflections (long-term); together they help the agent identify improved actions.

### SELF-REFINE
- A loop in which **refining the output** and **generating feedback** are repeated.

## 6. Multi-Agent Systems

### Why one agent is not enough
- Too many tools complicate tool selection.
- Context becomes too complex.
- The task may require specialization.

### Structure
- Multi-agent systems are frameworks in which multiple agents — each with tools, memory, and planning — interact with each other and with their environments.
- Typically **specialized agents**, each with its own toolset, overseen by a **supervisor** that manages communication and assigns tasks (figures: "Multi-Agent System", "Multi-Agent Architecture").
- Two core design components:
  - **Agent initialization** — how individual specialized agents are created.
  - **Agent orchestration** — how all agents are coordinated.

### Generative Agents
*Generative Agents: Interactive Simulacra of Human Behavior* (Park et al.) created software agents that simulate believable human behavior. The **Memory** module is the most vital component; it stores planning and reflection behaviors and all events so far.

Components common to multi-agent frameworks: **Profile**, **Perception** of the environment, **Memory**, **Planning**, **Available actions**.

### Frameworks
| Framework | Notes |
|---|---|
| AutoGen | Collaborative; inter-agent communication differs from the others |
| MetaGPT | Collaborative; inter-agent communication differs from the others |
| CAMEL | Role-playing: the user defines an **AI User** (represents the human and guides the process) and an **AI Assistant**; the two collaborate by interacting with each other to resolve the query |

The growth of these frameworks has been explosive (figure: "Framework Growth").

## 7. Conclusion

LLM Agents combine the language understanding of LLMs with the ability to interact with an environment through tools, keep information in memory, and plan complex actions. Areas of ongoing progress:

- More advanced planning and reasoning
- Better memory systems for context retention
- Improved tool integration and execution
- Stronger coordination among multiple agents
- Self-reflection and self-improvement mechanisms

> "This concludes our journey of LLM Agents! Hopefully, this guide gives a better understanding of how LLM Agents are built." — Maarten Grootendorst

## Key concepts worth simulating

- **Next-token sampling**: an autoregressive loop where each sampled token is appended and fed back; show how temperature changes the distribution and the resulting text.
- **Context window as working memory**: a sliding buffer of N tokens; animate tokens falling off the front, then show the summarization fix (an LLM compresses old turns into a short summary that stays in the window).
- **RAG long-term memory**: embed past interactions into vectors, store them, embed the new query, retrieve top-k by cosine similarity, and inject into the prompt.
- **Function calling**: LLM emits a JSON tool call → program validates against a schema → executes → returns observation; contrast fixed tool order vs autonomous selection.
- **ReAct loop**: state machine Thought → Action → Observation → … → Final answer, with a live trace on a small task (e.g. "what is 17 × 23 then search X").
- **Reflexion**: on task failure, write a verbal self-reflection into long-term memory and re-run; show success rate rising over trials.
- **Supervisor multi-agent orchestration**: a supervisor node routes subtasks to specialized agents (each with its own tools), collects results, and merges them; CAMEL's AI User ↔ AI Assistant dialogue as a two-agent variant.

## Test code

```python
# 1. Minimal ReAct loop with a toy tool set (no API key needed: a scripted "LLM")
import re, json

TOOLS = {
    "multiply": lambda a, b: a * b,
    "search": lambda q: f"Stub result for '{q}'",
}

def fake_llm(trace):
    # A scripted policy standing in for the model; replace with a real LLM call.
    if "Observation" not in trace:
        return 'Thought: I need 17*23.\nAction: {"name":"multiply","arguments":{"a":17,"b":23}}'
    return "Thought: I have the answer.\nFinal Answer: 391"

trace = "Question: What is 17 times 23?"
for step in range(5):
    out = fake_llm(trace)
    trace += "\n" + out
    if "Final Answer:" in out:
        break
    call = json.loads(re.search(r"Action: (\{.*\})", out).group(1))
    obs = TOOLS[call["name"]](**call["arguments"])
    trace += f"\nObservation: {obs}"
print(trace)
```

```python
# 2. Short-term memory: a fixed-size context window with summarization fallback
from collections import deque

WINDOW = 6  # tokens for illustration
history = deque(maxlen=WINDOW)
summary = ""

def add_turn(text):
    global summary
    for tok in text.split():
        if len(history) == WINDOW:
            summary = (summary + " " + history[0])[-40:]  # crude "summarizer"
        history.append(tok)

for turn in ["my name is Zaid", "I teach big data", "what is my name"]:
    add_turn(turn)
print("window :", list(history))
print("summary:", summary.strip())
```

```python
# 3. Long-term memory via RAG with a tiny vector store (numpy only)
import numpy as np

docs = ["The meeting is on Tuesday at 10am.",
        "The project deadline is 30 November.",
        "Zaid prefers Python for data science labs."]

def embed(text):  # bag-of-characters embedding, enough to demonstrate retrieval
    v = np.zeros(128)
    for ch in text.lower():
        v[ord(ch) % 128] += 1
    return v / (np.linalg.norm(v) + 1e-9)

store = np.stack([embed(d) for d in docs])
query = "when is the deadline?"
scores = store @ embed(query)
print("retrieved:", docs[int(scores.argmax())])
```

```python
# 4. Supervisor routing in a multi-agent system
AGENTS = {
    "math":   lambda task: f"[math agent] {eval(task)}",
    "writer": lambda task: f"[writer agent] Draft: {task.title()}",
}

def supervisor(task):
    agent = "math" if any(c in task for c in "+-*/") else "writer"
    return AGENTS[agent](task)

for t in ["12*12", "introduce multi-agent systems"]:
    print(supervisor(t))
```
