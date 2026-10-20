---
week: 5
title: LangGraph
source: https://albanna-tutorials.com/langgraph.html
---
# LangGraph


## Step 1: Introduction to LangGraph

LangGraph is a framework for building stateful, multi-actor applications powered by large language models. It models agent workflows as **graphs** instead of linear chains, which allows cycles, branching, and complex orchestration.

**Why LangGraph exists.** Imagine a research assistant that searches the web, evaluates the results, decides whether to search again or answer, and optionally asks a human for approval. That needs loops, branches, and state that persists between steps. Linear chains struggle to express this.

**LangGraph vs LangChain.** LangGraph does not replace LangChain; the two complement each other.

| Feature | LangChain | LangGraph |
|---|---|---|
| Architecture | Linear chains and pipelines | Directed graphs (with cycles) |
| State | Passed along the chain | Central, persistent state object |
| Loops | Not native | First-class |
| Best for | Simple pipelines, RAG | Agents, multi-step workflows |
| Human-in-the-loop | Manual | Built in |
| Persistence | Not built in | Checkpointing built in |

**Core model.** Every LangGraph workflow is built from three primitives:

- **State** — the shared data that flows through the graph.
- **Nodes** — Python functions that read the state and return updates.
- **Edges** — connections that control the flow, including conditional routes and cycles.

> **Key insight:** the edge from the *tools* node back to the *agent* node is what sets LangGraph apart. The agent can call tools repeatedly until it has enough information. This is the ReAct (Reason + Act) pattern.

**When to use LangGraph:** ReAct-style tool-using agents, multi-agent systems, workflows with human approval steps, state that persists across conversations, and pipelines with branching and error handling.

**Where the tutorial ends:** a multi-agent research assistant with tools, conditional routing, memory, and human-in-the-loop support.

## Step 2: Installation & Setup

LangGraph works alongside the LangChain ecosystem. The tutorial uses OpenAI models, but any LangChain-compatible model can be substituted.

### Installing Required Packages

```bash
# Install core LangGraph and LangChain packages
pip install langgraph langchain-openai langchain-core

# Optional: additional tools and utilities
pip install langchain-community tavily-python

# Verify installation
python -c "from importlib.metadata import version; print(f'LangGraph version: {version(\"langgraph\")}')"
```

### Setting Up API Keys

```python
import os

# Option 1: Set environment variables directly
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"

# Option 2 (recommended): Use a .env file with python-dotenv
# pip install python-dotenv
from dotenv import load_dotenv
load_dotenv()  # reads from .env file in project root

# Option 3: Tavily for web search (used in later steps)
os.environ["TAVILY_API_KEY"] = "tvly-your-key-here"
```

> **Security warning:** never hardcode API keys in source code. Keep them in environment variables or a `.env` file that is excluded from version control.

### Project Structure

```text
langgraph-tutorial/
├── .env                  # API keys (add to .gitignore!)
├── requirements.txt      # Package dependencies
├── simple_chatbot.py     # Step 6: First graph
├── tool_agent.py         # Step 7: Tool integration
├── react_agent.py        # Step 8: ReAct pattern
├── checkpoint_agent.py   # Step 9: Memory & human-in-the-loop
└── multi_agent.py        # Step 10: Multi-agent system
```

### Basic Imports

```python
# --- Graph construction ---
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState

# --- Prebuilt components ---
from langgraph.prebuilt import ToolNode, tools_condition

# --- Checkpointing ---
from langgraph.checkpoint.memory import MemorySaver

# --- LLM provider ---
from langchain_openai import ChatOpenAI

# --- Message types ---
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

# --- Tool decorator ---
from langchain_core.tools import tool

# --- Python standard library ---
from typing import TypedDict, Annotated, Literal
import operator
```

### Verify Everything Works

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Test that the LLM is accessible
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm.invoke([HumanMessage(content="Say 'LangGraph is ready!' in one sentence.")])
print(response.content)
# Expected output: LangGraph is ready!
```

> **Checkpoint:** if you get a valid response from the LLM, your environment is ready.

## Step 3: Core Concepts – State

**State** is the central data structure of a LangGraph application. Every node receives the full state and returns only the parts it changed; LangGraph merges those partial updates.

### Defining State with TypedDict

State is usually a Python `TypedDict` with typed keys. A node that returns a key replaces only that key.

```python
from typing import TypedDict

class AgentState(TypedDict):
    """State that flows through the entire graph."""
    query: str          # The user's original question
    context: str        # Retrieved information
    answer: str         # The generated answer
    iteration: int      # How many times we've looped
```

### MessagesState (Built-in)

A built-in state type with a single `messages` key. New messages are **appended** rather than replacing the list, which is exactly what a chatbot needs. This behaviour comes from a *reducer* function.

```python
from langgraph.graph.message import MessagesState

# MessagesState is equivalent to:
# class MessagesState(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]
#
# The add_messages reducer automatically appends new messages
# to the existing list instead of replacing it.

# Usage: your nodes just return {"messages": [new_message]}
# and it gets appended to the conversation history.
```

> **Why MessagesState?** Chat history should grow by appending, never by overwriting.

### Custom State with Annotated Types and Reducers

`Annotated` types attach a **reducer** that controls how updates merge. A reducer takes the current value and the update and returns the new value.

```python
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
import operator

class ResearchState(TypedDict):
    """Custom state with multiple reducer strategies."""

    # Messages use add_messages reducer: new messages are APPENDED
    messages: Annotated[list[AnyMessage], add_messages]

    # Sources use operator.add: new lists are CONCATENATED
    sources: Annotated[list[str], operator.add]

    # Simple fields (no reducer): values are REPLACED
    query: str
    final_answer: str

    # Counter with a custom reducer: values are ADDED
    search_count: Annotated[int, operator.add]
```

| Reducer | Behaviour | Example |
|---|---|---|
| None (default) | Replaces the old value | `"a"` → `"b"` keeps only `"b"` |
| `add_messages` | Appends messages intelligently | New message added to the end of the list |
| `operator.add` | Concatenates lists / adds numbers | `[1] + [2] → [1, 2]`, `3 + 1 → 4` |

### Complete State Example

```python
from typing import TypedDict, Annotated
import operator

class CounterState(TypedDict):
    value: int                                    # replaced each time
    history: Annotated[list[str], operator.add]   # appended each time

def double_it(state: CounterState) -> dict:
    """Node that doubles the value."""
    new_val = state["value"] * 2
    return {
        "value": new_val,
        "history": [f"doubled to {new_val}"]
    }

def add_ten(state: CounterState) -> dict:
    """Node that adds ten."""
    new_val = state["value"] + 10
    return {
        "value": new_val,
        "history": [f"added 10 to get {new_val}"]
    }

# Initial state: {"value": 5, "history": []}
# After double_it: {"value": 10, "history": ["doubled to 10"]}
# After add_ten:   {"value": 20, "history": ["doubled to 10", "added 10 to get 20"]}
```

> **Key takeaway:** state design is the most important decision in a LangGraph application; it determines how data moves through the whole system.

## Step 4: Core Concepts – Nodes

**Nodes** are plain Python functions. Each takes the current state, performs an operation, and returns a partial update.

### Node Signature

The function receives the **full** state and returns **only** the keys it modifies. Keys that are not returned are left unchanged.

```python
def my_node(state: MyState) -> dict:
    """
    Args:
        state: The FULL current state of the graph.

    Returns:
        A dict with ONLY the keys you want to update.
        Keys not included remain unchanged.
    """
    # Read from state
    current_value = state["some_key"]

    # Do something (call LLM, run tool, transform data)
    new_value = process(current_value)

    # Return partial state update
    return {"some_key": new_value}
```

### Chatbot Node

Calls an LLM with the conversation history and returns the reply as a message.

```python
from langgraph.graph.message import MessagesState
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def chatbot_node(state: MessagesState) -> dict:
    """Call the LLM with the full conversation history."""
    # state["messages"] contains all messages so far
    response = llm.invoke(state["messages"])

    # Return the AI response; add_messages reducer appends it
    return {"messages": [response]}
```

### Tool Execution Node

Runs the tools the LLM requested and wraps each result in a `ToolMessage`. (In practice the prebuilt `ToolNode` from Step 7 does this for you.)

```python
from langchain_core.messages import ToolMessage

# tool_registry maps each tool's name to the tool object,
# e.g. tool_registry = {t.name: t for t in tools}

def tool_executor_node(state: MessagesState) -> dict:
    """Execute tools that the LLM requested."""
    last_message = state["messages"][-1]
    results = []

    for tool_call in last_message.tool_calls:
        # Look up the tool by name and invoke it
        tool_fn = tool_registry[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])

        # Wrap the result in a ToolMessage
        results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": results}
```

### Data Processing Node

Performs plain computation (here, text analysis) with or without an LLM call.

```python
from typing import TypedDict, Annotated
import operator
from langchain_openai import ChatOpenAI

class AnalysisState(TypedDict):
    raw_text: str
    word_count: int
    keywords: Annotated[list[str], operator.add]
    summary: str

def analyze_text(state: AnalysisState) -> dict:
    """Node that analyzes text without calling an LLM."""
    text = state["raw_text"]

    # Pure Python processing
    words = text.split()
    common_words = {"the", "a", "an", "is", "are", "was", "were"}
    keywords = [w.lower() for w in words if w.lower() not in common_words and len(w) > 3]

    return {
        "word_count": len(words),
        "keywords": keywords[:10],  # top 10 keywords
    }

def summarize_text(state: AnalysisState) -> dict:
    """Node that uses an LLM to summarize."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke(
        f"Summarize this in one sentence: {state['raw_text']}"
    )
    return {"summary": response.content}
```

### Adding Nodes to a Graph

Nodes are registered with a unique name string, which edges use to refer to them.

```python
from langgraph.graph import StateGraph

# Create the graph with your state type
graph = StateGraph(AnalysisState)

# Add nodes - the string is the node's name in the graph
graph.add_node("analyze", analyze_text)
graph.add_node("summarize", summarize_text)

# Node names are used when defining edges (next step!)
```

> **Common mistake:** node names must be unique. Adding a second node with the same name raises a `ValueError` ("Node `analyze` already present").

> **Key takeaway:** a node can do almost anything — call an LLM, run a tool, hit an API. It only has to accept state and return a partial state.

## Step 5: Core Concepts – Edges

**Edges** decide which node runs next. LangGraph supports normal edges, conditional edges, and the special entry/finish points.

### Normal Edges

A fixed connection: when node A finishes, node B always runs next.

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(MyState)

graph.add_node("fetch_data", fetch_data)
graph.add_node("process_data", process_data)
graph.add_node("generate_report", generate_report)

# Normal edges: linear flow
graph.add_edge(START, "fetch_data")          # Entry point
graph.add_edge("fetch_data", "process_data") # fetch -> process
graph.add_edge("process_data", "generate_report")  # process -> report
graph.add_edge("generate_report", END)       # Exit point
```

**START and END** are special markers imported as constants, not real nodes. `START` marks the entry point; `END` marks where the graph stops and returns its final state.

### Conditional Edges

The most powerful edge type. A **routing function** inspects state and returns a string; a **path map** turns that string into the next node. This enables branching and loops.

```python
from typing import Literal
from langgraph.graph.message import MessagesState

def route_after_agent(state: MessagesState) -> Literal["tools", "end"]:
    """Decide what happens after the agent node runs.

    This function inspects the state and returns a STRING
    that maps to the next node name.
    """
    last_message = state["messages"][-1]

    # If the LLM made tool calls, go to the tools node
    if last_message.tool_calls:
        return "tools"

    # Otherwise, we're done
    return "end"

# Register the conditional edge
graph.add_conditional_edges(
    source="agent",           # After this node finishes...
    path=route_after_agent,   # ...run this function to decide next node
    path_map={                # Map return values to actual node names
        "tools": "tool_node",
        "end": END,
    },
)
```

### Creating Cycles with Conditional Edges

Routing back to an earlier node creates a cycle. The classic agent loop:

```text
START → agent ──(tool_calls?)──► tool_node ──► agent ──► ... ──(no tool_calls)──► END
```

```python
# This creates the classic ReAct agent loop:
#   START -> agent -> (tools -> agent -> tools -> ...) -> END

graph.add_edge(START, "agent")

# Conditional: agent decides if we need tools or are done
graph.add_conditional_edges("agent", route_after_agent, {
    "tools": "tool_node",
    "end": END,
})

# After tools run, ALWAYS go back to the agent
graph.add_edge("tool_node", "agent")   # This creates the cycle!
```

| Edge type | Method | Purpose |
|---|---|---|
| Normal | `add_edge("A", "B")` | Fixed, sequential flow |
| Conditional | `add_conditional_edges("A", router, path_map)` | Dynamic routing and loops |
| Entry | `add_edge(START, "A")` | Sets the starting node |
| Finish | `add_edge("A", END)` | Ends the graph |

## Step 6: Building Your First Graph

**What we are building:** a minimal chatbot that (1) takes a user message, (2) sends it to GPT-4o-mini, (3) returns the response.

### Complete Working Code

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# ── Step 1: Initialize the LLM ─────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Step 2: Define the node function ───────────────────────
def chatbot(state: MessagesState) -> dict:
    """The only node in our graph: call the LLM."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# ── Step 3: Build the graph ────────────────────────────────
# Create a StateGraph using the built-in MessagesState
graph_builder = StateGraph(MessagesState)

# Add our chatbot node
graph_builder.add_node("chatbot", chatbot)

# Define the flow: START -> chatbot -> END
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# ── Step 4: Compile the graph ──────────────────────────────
# Compiling validates the graph and returns a runnable object
graph = graph_builder.compile()

# ── Step 5: Run the graph ──────────────────────────────────
result = graph.invoke({
    "messages": [HumanMessage(content="What is LangGraph?")]
})

# The result is the final state
print(result["messages"][-1].content)
```

**Understanding each step**

- **StateGraph construction** — creates the graph builder; its argument defines the state shape all nodes must follow.
- **Adding nodes** — registers a function under a string name that edges use to refer to it.
- **Adding edges** — connects `START` → `chatbot` → `END`, a linear flow.
- **Compiling** — validates the graph structure and returns a runnable object.
- **Invoking** — runs the graph on an initial state and returns the final state.

> **That's it!** The five-part pattern — LLM, node, graph, compile, invoke — applies to every LangGraph project.

### Interactive Chat Loop

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def chatbot(state: MessagesState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# ── Interactive loop ───────────────────────────────────────
print("LangGraph Chatbot (type 'quit' to exit)")
print("-" * 45)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    result = graph.invoke({
        "messages": [HumanMessage(content=user_input)]
    })
    ai_message = result["messages"][-1]
    print(f"\nBot: {ai_message.content}")
```

> **No memory yet:** the chatbot forgets earlier messages between `invoke()` calls because each call starts fresh. Step 9 adds memory through checkpointing.

### Streaming Output

```python
# Stream events from the graph
for event in graph.stream(
    {"messages": [HumanMessage(content="Explain LangGraph in 3 sentences.")]},
    stream_mode="values",
):
    # Each event contains the state after a node executes
    last_msg = event["messages"][-1]
    last_msg.pretty_print()
```

## Step 7: Tool Integration

Tools turn ordinary Python functions into capabilities the LLM can request. The **docstring** is what tells the LLM when to use each tool.

### Defining Tools with @tool

```python
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the web for current information about a topic.

    Use this when you need up-to-date information that may not
    be in your training data.

    Args:
        query: The search query string.
    """
    # In production, use Tavily, SerpAPI, or similar
    # For this tutorial, we'll simulate a search
    return f"Search results for '{query}': LangGraph is a framework for building stateful AI agents using graph-based orchestration. Latest version is 0.2+."

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Use this for any calculations the user asks about.

    Args:
        expression: A valid Python math expression (e.g., '2 + 2', '100 * 0.15').
    """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"

# Collect tools into a list
tools = [web_search, calculator]
```

### Binding Tools to the LLM

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Bind tools to the LLM - now it knows about web_search and calculator
llm_with_tools = llm.bind_tools(tools)
```

### ToolNode: Automatic Tool Execution

```python
from langgraph.prebuilt import ToolNode

# Create a ToolNode from our list of tools
tool_node = ToolNode(tools)

# That's it! This node will:
# 1. Read the last AI message from state["messages"]
# 2. Extract any tool_calls from the message
# 3. Execute each tool with the given arguments
# 4. Return ToolMessage results appended to messages
```

### tools_condition: Conditional Routing

```python
from langgraph.prebuilt import tools_condition

# tools_condition returns:
#   "tools"  if the last AI message has tool_calls
#   "__end__" if the last AI message has no tool_calls (just text)

# Use it as a conditional edge:
graph.add_conditional_edges("chatbot", tools_condition)
```

### Complete Example: Chatbot with Tools

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# ── Define tools ───────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """Search the web for current information about a topic."""
    return f"Search results for '{query}': LangGraph v0.2 supports cycles, persistence, and human-in-the-loop patterns."

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return f"Result: {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"

tools = [web_search, calculator]

# ── Set up the LLM with tools ─────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ── Define the chatbot node ───────────────────────────────
def chatbot(state: MessagesState) -> dict:
    """Call the LLM (which may decide to use tools)."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ── Build the graph ───────────────────────────────────────
graph_builder = StateGraph(MessagesState)

# Add nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))

# Add edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")  # Loop back after tools

# Compile and run
graph = graph_builder.compile()

# ── Test it ───────────────────────────────────────────────
result = graph.invoke({
    "messages": [HumanMessage(content="Search for the latest LangGraph features, then calculate 42 * 17")]
})

for msg in result["messages"]:
    print(f"{msg.type}: {msg.content[:200]}")
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"  Tool calls: {[tc['name'] for tc in msg.tool_calls]}")
```

> **How it works:** the LLM reads the user message and chooses which tools to call. `ToolNode` runs them, the results go back to the LLM, and the LLM writes a final answer that combines them.

## Step 8: Conditional Routing & Cycles

The **ReAct** pattern (Reason + Act) is a loop between an agent node and a tools node. The agent:

1. **Reasons** about the next step based on the current state.
2. **Acts** by calling a tool.
3. **Observes** the tool result.
4. **Repeats** until the task is complete.

### The should_continue Pattern

```python
from typing import Literal

def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    """Decide whether to continue the agent loop or stop.

    This function is called after the agent node runs.
    It checks if the LLM wants to call any tools.
    """
    last_message = state["messages"][-1]

    # If the LLM made tool calls, continue to the tools node
    if last_message.tool_calls:
        return "tools"

    # Otherwise, the LLM is done -- go to END
    return "__end__"
```

> **Note:** `"__end__"` is the internal name for the END node. You can use the `END` constant in a path map instead, and the prebuilt `tools_condition` does the same job as this custom `should_continue`.

### Building a ReAct Agent with Cycles

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from typing import Literal

# ── Tools ──────────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for information."""
    knowledge = {
        "langgraph": "LangGraph is a library for building stateful agents with LLMs using graph-based orchestration.",
        "react": "ReAct is a pattern where agents alternate between reasoning and acting.",
        "state": "State in LangGraph is a TypedDict that flows through every node.",
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return "No relevant information found."

@tool
def get_current_date() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [search_knowledge_base, get_current_date]

# ── LLM with tools ────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

# ── Nodes ──────────────────────────────────────────────────
def agent(state: MessagesState) -> dict:
    """The reasoning node: calls the LLM to decide what to do."""
    system = SystemMessage(content=(
        "You are a helpful research assistant. "
        "Use tools to look up information when needed. "
        "When you have enough information, provide a final answer."
    ))
    response = llm.invoke([system] + state["messages"])
    return {"messages": [response]}

# ── Routing function ──────────────────────────────────────
def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    """Check if the agent wants to use tools or is done."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"

# ── Build the graph ───────────────────────────────────────
graph_builder = StateGraph(MessagesState)

# Add nodes
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", ToolNode(tools))

# Define the flow with a cycle
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue)
graph_builder.add_edge("tools", "agent")   # CYCLE: tools -> agent

# Compile
react_agent = graph_builder.compile()

# ── Run ───────────────────────────────────────────────────
result = react_agent.invoke({
    "messages": [HumanMessage(
        content="What is LangGraph? Also, what is today's date?"
    )]
})

# Print the full conversation
for msg in result["messages"]:
    role = msg.type.upper()
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"\n{role}: [Calling tools: {[tc['name'] for tc in msg.tool_calls]}]")
    elif msg.type == "tool":
        print(f"TOOL ({msg.name}): {msg.content}")
    else:
        print(f"\n{role}: {msg.content}")
```

**Execution flow walkthrough**

| Stage | What happens |
|---|---|
| Iteration 1 — Agent | Decides it needs to search the knowledge base *and* check the date |
| Action | Calls `search_knowledge_base` and `get_current_date` |
| Tool execution | Results are appended to `messages`; flow returns to Agent |
| Iteration 2 — Agent | Decides it has enough information |
| Decision | No tool calls, so `should_continue` returns `"__end__"` |
| Result | A final answer that combines both tool outputs |

### Adding Iteration Limits

```python
MAX_ITERATIONS = 5

def should_continue_with_limit(state: MessagesState) -> Literal["tools", "__end__"]:
    """Continue the loop, but enforce a maximum iteration count."""
    last_message = state["messages"][-1]

    # Count how many tool call rounds we've done
    tool_call_count = sum(
        1 for msg in state["messages"]
        if hasattr(msg, "tool_calls") and msg.tool_calls
    )

    if tool_call_count >= MAX_ITERATIONS:
        return "__end__"  # Force stop after MAX_ITERATIONS

    if last_message.tool_calls:
        return "tools"

    return "__end__"
```

> **Safety tip:** always set iteration limits to prevent runaway agents. You can also pass `{"recursion_limit": N}` in the config when invoking the graph.

## Step 9: Checkpointing & Human-in-the-Loop

A **checkpointer** saves the state after every node. `MemorySaver` enables:

- Conversation memory across multiple invocations
- Resuming after an interruption or crash
- "Time travel": replaying or inspecting earlier states

### MemorySaver: State Persistence

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def chatbot(state: MessagesState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# ── Compile WITH checkpointer ─────────────────────────────
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```

### Thread-Based Conversations

```python
# ── Conversation with memory ──────────────────────────────
config = {"configurable": {"thread_id": "user-123"}}

# First message
result1 = graph.invoke(
    {"messages": [HumanMessage(content="My name is Alice.")]},
    config=config,
)
print(result1["messages"][-1].content)
# Output: "Nice to meet you, Alice! How can I help you today?"

# Second message -- same thread, so the agent remembers!
result2 = graph.invoke(
    {"messages": [HumanMessage(content="What is my name?")]},
    config=config,
)
print(result2["messages"][-1].content)
# Output: "Your name is Alice!"

# Different thread -- no memory of Alice
config_new = {"configurable": {"thread_id": "user-456"}}
result3 = graph.invoke(
    {"messages": [HumanMessage(content="What is my name?")]},
    config=config_new,
)
print(result3["messages"][-1].content)
# Output: "I don't know your name. Could you tell me?"
```

> **Key insight:** `thread_id` works like a session ID. Invocations that share a thread ID share a conversation history; different thread IDs are fully isolated.

> **Production usage:** `MemorySaver` keeps state in RAM, so it is lost on restart. For persistence use `SqliteSaver` or `PostgresSaver` from the `langgraph-checkpoint-sqlite` / `langgraph-checkpoint-postgres` packages.

### Human-in-the-Loop: Interrupt Before/After

| Parameter | When it pauses | Use case |
|---|---|---|
| `interrupt_before=["tools"]` | Before the tools node runs | Approve tool calls before they execute |
| `interrupt_after=["agent"]` | After the agent node runs | Review the agent's reasoning |

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the specified recipient."""
    return f"Email sent to {to} with subject '{subject}'."

tools = [send_email]
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def agent(state: MessagesState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")

# ── Compile with interrupt BEFORE the tools node ──────────
memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"],  # Pause before executing tools
)

# ── Run -- the graph will pause before tools ──────────────
config = {"configurable": {"thread_id": "approval-thread"}}

result = graph.invoke(
    {"messages": [HumanMessage(
        content="Send an email to bob@example.com about the meeting tomorrow"
    )]},
    config=config,
)

# The graph is now PAUSED before the tools node
# Check what tool the agent wants to call:
last_msg = result["messages"][-1]
print("Agent wants to call:")
for tc in last_msg.tool_calls:
    print(f"  {tc['name']}({tc['args']})")

# ── Human reviews and approves ────────────────────────────
# Option A: Approve -- just resume with None input
approved = input("Approve? (y/n): ")
if approved.lower() == "y":
    # Resume execution from where it paused
    final_result = graph.invoke(None, config=config)
    print(final_result["messages"][-1].content)
else:
    print("Action cancelled by human.")
```

## Step 10: Complete Project – Multi-Agent Task System

**Architecture (Supervisor pattern).** A central **Supervisor** (orchestrator) receives the task from `START` and routes it to one of three specialised workers; every worker reports back to the Supervisor, which decides who runs next or finishes.

```text
            START
              │
              ▼
        ┌────────────┐
   ┌───►│ Supervisor │────────────► END
   │    └─────┬──────┘
   │    ┌─────┼──────────┐
   │    ▼     ▼          ▼
   │ Researcher Analyst Writer
   │ (web search)(analysis)(report)
   └────┴─────┴──────────┘   (all return to Supervisor)
```

### Full Working Code

```python
"""
Multi-Agent Research Assistant with Supervisor Pattern
=====================================================
A complete LangGraph application with:
- Supervisor agent (orchestrator)
- Researcher agent (web search)
- Analyst agent (data analysis)
- Writer agent (report generation)
- Memory persistence
- Human-in-the-loop approval
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.tools import tool
import operator

# ══════════════════════════════════════════════════════════
# 1. DEFINE THE STATE
# ══════════════════════════════════════════════════════════

class MultiAgentState(TypedDict):
    """Shared state for the multi-agent system."""
    messages: Annotated[list, add_messages]   # Conversation history
    task: str                                  # Current task description
    research_data: str                         # Data from researcher
    analysis: str                              # Analysis from analyst
    report: str                                # Final report from writer
    next_agent: str                            # Who should run next
    iteration_count: Annotated[int, operator.add]  # Safety counter

# ══════════════════════════════════════════════════════════
# 2. DEFINE TOOLS
# ══════════════════════════════════════════════════════════

@tool
def web_search(query: str) -> str:
    """Search the web for information on a topic."""
    # Simulated search results for the tutorial
    results = {
        "AI agents": "AI agents are autonomous systems that perceive their environment and take actions to achieve goals. Key frameworks include LangGraph, CrewAI, and AutoGen.",
        "LangGraph": "LangGraph v0.2+ supports stateful agents, cycles, human-in-the-loop, and multi-agent architectures.",
        "market trends": "The AI agent market is projected to reach $65B by 2030, with enterprise adoption growing 40% annually.",
    }
    for key, value in results.items():
        if key.lower() in query.lower():
            return value
    return f"Found general information about: {query}"

@tool
def analyze_data(data: str) -> str:
    """Analyze research data and extract key insights."""
    word_count = len(data.split())
    return (
        f"Analysis of {word_count}-word dataset:\n"
        f"- Key themes identified: AI agents, market growth, framework adoption\n"
        f"- Sentiment: Positive (high growth trajectory)\n"
        f"- Confidence: High (multiple corroborating sources)"
    )

# ══════════════════════════════════════════════════════════
# 3. DEFINE AGENT NODES
# ══════════════════════════════════════════════════════════

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def supervisor_node(state: MultiAgentState) -> dict:
    """The supervisor decides which agent should work next."""
    system = SystemMessage(content="""You are a project supervisor managing a research team.
Based on the current state of the project, decide who should work next:
- "researcher": if you need more information gathered
- "analyst": if you have research data that needs analysis
- "writer": if analysis is done and you need a final report
- "FINISH": if the report is complete

Respond with ONLY one of: researcher, analyst, writer, FINISH""")

    context_msg = HumanMessage(content=f"""
Current project state:
- Task: {state.get('task', 'Not set')}
- Research data: {'Available' if state.get('research_data') else 'Not yet gathered'}
- Analysis: {'Available' if state.get('analysis') else 'Not yet done'}
- Report: {'Available' if state.get('report') else 'Not yet written'}
- Iterations so far: {state.get('iteration_count', 0)}

Who should work next?""")

    response = llm.invoke([system, context_msg])
    next_agent = response.content.strip().lower()

    # Validate the response
    valid_agents = {"researcher", "analyst", "writer", "finish"}
    if next_agent not in valid_agents:
        next_agent = "finish"  # Default to finish if unclear

    return {
        "next_agent": next_agent,
        "messages": [AIMessage(content=f"Supervisor: Routing to {next_agent}")],
        "iteration_count": 1,
    }

def researcher_node(state: MultiAgentState) -> dict:
    """The researcher gathers information using web search."""
    system = SystemMessage(content=(
        "You are a research specialist. Use the web_search tool to "
        "gather comprehensive information about the given task. "
        "Compile all findings into a clear summary."
    ))

    task_msg = HumanMessage(content=f"Research this topic: {state['task']}")
    llm_with_search = llm.bind_tools([web_search])
    response = llm_with_search.invoke([system, task_msg])

    # Execute tool calls if any
    research_results = []
    if response.tool_calls:
        for tc in response.tool_calls:
            result = web_search.invoke(tc["args"])
            research_results.append(result)

    research_data = "\n".join(research_results) if research_results else response.content

    return {
        "research_data": research_data,
        "messages": [AIMessage(content=f"Researcher: {research_data[:200]}...")],
    }

def analyst_node(state: MultiAgentState) -> dict:
    """The analyst processes research data and extracts insights."""
    system = SystemMessage(content=(
        "You are a data analyst. Analyze the provided research data "
        "and produce structured insights with key findings."
    ))

    data_msg = HumanMessage(
        content=f"Analyze this research data:\n{state['research_data']}"
    )
    response = llm.invoke([system, data_msg])

    return {
        "analysis": response.content,
        "messages": [AIMessage(content=f"Analyst: {response.content[:200]}...")],
    }

def writer_node(state: MultiAgentState) -> dict:
    """The writer produces the final report."""
    system = SystemMessage(content=(
        "You are a technical writer. Using the research data and analysis "
        "provided, write a concise, professional report."
    ))

    report_msg = HumanMessage(content=f"""
Write a report based on:

RESEARCH DATA:
{state['research_data']}

ANALYSIS:
{state['analysis']}

Original task: {state['task']}
""")
    response = llm.invoke([system, report_msg])

    return {
        "report": response.content,
        "messages": [AIMessage(content=f"Writer: Report complete.")],
    }

# ══════════════════════════════════════════════════════════
# 4. ROUTING FUNCTION
# ══════════════════════════════════════════════════════════

def route_supervisor(
    state: MultiAgentState,
) -> Literal["researcher", "analyst", "writer", "__end__"]:
    """Route to the next agent based on supervisor's decision."""
    next_agent = state.get("next_agent", "finish")

    # Safety: stop after too many iterations
    if state.get("iteration_count", 0) > 10:
        return "__end__"

    if next_agent == "researcher":
        return "researcher"
    elif next_agent == "analyst":
        return "analyst"
    elif next_agent == "writer":
        return "writer"
    else:
        return "__end__"

# ══════════════════════════════════════════════════════════
# 5. BUILD THE GRAPH
# ══════════════════════════════════════════════════════════

graph_builder = StateGraph(MultiAgentState)

# Add all nodes
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("researcher", researcher_node)
graph_builder.add_node("analyst", analyst_node)
graph_builder.add_node("writer", writer_node)

# Entry point
graph_builder.add_edge(START, "supervisor")

# Supervisor routes to workers (or END)
graph_builder.add_conditional_edges("supervisor", route_supervisor)

# All workers report back to the supervisor
graph_builder.add_edge("researcher", "supervisor")
graph_builder.add_edge("analyst", "supervisor")
graph_builder.add_edge("writer", "supervisor")

# Compile with memory
memory = MemorySaver()
multi_agent = graph_builder.compile(checkpointer=memory)

# ══════════════════════════════════════════════════════════
# 6. RUN THE SYSTEM
# ══════════════════════════════════════════════════════════

config = {"configurable": {"thread_id": "research-project-1"}}

result = multi_agent.invoke(
    {
        "messages": [HumanMessage(content="Start the research project")],
        "task": "Analyze the current state of AI agent frameworks and market trends",
        "research_data": "",
        "analysis": "",
        "report": "",
        "next_agent": "",
        "iteration_count": 0,
    },
    config=config,
)

# Print the final report
print("=" * 60)
print("FINAL REPORT")
print("=" * 60)
print(result.get("report", "No report generated"))
print("\n" + "=" * 60)
print("CONVERSATION LOG")
print("=" * 60)
for msg in result["messages"]:
    print(f"  {msg.content[:100]}")
```

> **Congratulations!** You have built a multi-agent system with a supervisor, specialised workers, tools, state management, and memory persistence.

**Ways to extend this system**

- Add more agents: create new node functions and register them in the supervisor's routing logic.
- Replace the simulated tools with real ones (Tavily for web search, database queries, API calls).
- Add human approval: `interrupt_before=["writer"]` so a person reviews before the report is finalised.
- Swap `MemorySaver` for `PostgresSaver` for production persistence.
- Stream intermediate results to the user.

> **Pro tip:** use LangSmith to trace and debug LangGraph apps in production — which nodes ran, what state was passed, and how long each step took.

## Step 11: Django Integration

Wrap the LangGraph agent in a Django web application: a module-level agent shared across requests, a per-session `thread_id` for memory, a JSON API endpoint, and a browser chat UI.

### Project Structure

```text
langgraph_django/
├── manage.py
├── requirements.txt
├── .env
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── chat/
│   ├── __init__.py
│   ├── urls.py
│   ├── views.py
│   ├── agent.py              # LangGraph agent logic
│   └── templates/
│       └── chat/
│           └── index.html     # Chat UI
└── static/
    └── chat/
        └── style.css
```

### Step 1: Create Django Project & Install Dependencies

```bash
# Create project directory
mkdir langgraph_django && cd langgraph_django

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install django langgraph langchain-openai langchain-core langchain-community python-dotenv

# Create Django project and app
django-admin startproject config .
python manage.py startapp chat
```

### Step 2: requirements.txt

```text
django>=5.0
langgraph>=0.2.0
langchain-openai>=0.2.0
langchain-core>=0.3.0
langchain-community>=0.3.0
python-dotenv>=1.0.0
```

### Step 3: Environment Variables (.env)

```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
DJANGO_SECRET_KEY=your-django-secret-key
DEBUG=True
```

> **Security:** never commit `.env` to version control; add it to `.gitignore`.

### Step 4: Django Settings (config/settings.py)

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "change-me-in-production")
DEBUG = os.getenv("DEBUG", "False") == "True"
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "chat",  # Our chat app
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
```

### Step 5: LangGraph Agent (chat/agent.py)

```python
"""
LangGraph Agent for Django Integration
---------------------------------------
A conversational AI agent with tool-calling capabilities,
built with LangGraph and integrated into Django.
"""

import os
from typing import Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ── Define Tools ──────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use this for any math calculations.
    Examples: '2 + 2', '15 * 3.5', '100 / 7', 'pow(2, 10)'
    """
    try:
        # Safe evaluation of math expressions
        allowed_names = {"pow": pow, "abs": abs, "round": round, "min": min, "max": max}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def search_knowledge(query: str) -> str:
    """Search for information about a topic. Use this when the user asks
    about facts, concepts, or anything you need to look up.
    """
    # In production, connect to a real search API (Tavily, Google, etc.)
    # For this demo, return a helpful response
    return (
        f"Search results for '{query}': This is a demo search tool. "
        f"In production, connect this to Tavily, Google Search API, "
        f"or your own knowledge base for real results."
    )

# ── Build the LangGraph Agent ────────────────────────────

# Initialize LLM with tools
tools = [calculator, get_current_time, search_knowledge]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
llm_with_tools = llm.bind_tools(tools)

def chatbot_node(state: MessagesState) -> dict:
    """The main chatbot node that processes messages with the LLM."""
    system_msg = SystemMessage(content=(
        "You are a helpful AI assistant integrated into a Django web application. "
        "You can perform calculations, tell the current time, and search for information. "
        "Be concise and helpful. Use the available tools when appropriate."
    ))
    response = llm_with_tools.invoke([system_msg] + state["messages"])
    return {"messages": [response]}

# Build the graph
def create_agent_graph():
    """Create and compile the LangGraph agent."""
    graph = StateGraph(MessagesState)

    # Add nodes
    graph.add_node("chatbot", chatbot_node)
    graph.add_node("tools", ToolNode(tools=tools))

    # Add edges
    graph.add_edge(START, "chatbot")
    graph.add_conditional_edges("chatbot", tools_condition)
    graph.add_edge("tools", "chatbot")

    # Compile with memory checkpointing
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

# ── Singleton Agent Instance ──────────────────────────────

# Create one agent instance shared across requests
agent = create_agent_graph()

def chat_with_agent(user_message: str, thread_id: str = "default") -> str:
    """
    Send a message to the LangGraph agent and get a response.

    Args:
        user_message: The user's input message
        thread_id: Unique thread ID for conversation memory

    Returns:
        The agent's response as a string
    """
    config = {"configurable": {"thread_id": thread_id}}
    input_msg = {"messages": [HumanMessage(content=user_message)]}

    result = agent.invoke(input_msg, config=config)

    # Extract the last AI message
    ai_message = result["messages"][-1]
    return ai_message.content
```

> **Key design decisions:**
> - The agent is built once as a module-level singleton so the graph is not rebuilt on every request.
> - `MemorySaver` with thread IDs gives each user session its own conversation memory.
> - `tools_condition` routes to tools when the LLM requests them, then loops back to the chatbot.

### Step 6: Django URLs (config/urls.py)

```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("chat.urls")),
]
```

### Step 6 (continued): App URLs (chat/urls.py)

```python
from django.urls import path
from . import views

urlpatterns = [
    path("", views.chat_page, name="chat_page"),
    path("api/chat/", views.chat_api, name="chat_api"),
    path("api/reset/", views.reset_chat, name="reset_chat"),
]
```

### Step 7: Django Views (chat/views.py)

```python
import json
import uuid

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .agent import chat_with_agent

def chat_page(request):
    """Render the chat UI page."""
    # Assign a unique thread_id per session for conversation memory
    if "thread_id" not in request.session:
        request.session["thread_id"] = str(uuid.uuid4())
    return render(request, "chat/index.html")

@csrf_exempt
@require_http_methods(["POST"])
def chat_api(request):
    """API endpoint: receive user message, return agent response."""
    try:
        data = json.loads(request.body)
        user_message = data.get("message", "").strip()

        if not user_message:
            return JsonResponse({"error": "Message cannot be empty"}, status=400)

        # Get or create thread_id from session
        if "thread_id" not in request.session:
            request.session["thread_id"] = str(uuid.uuid4())

        thread_id = request.session["thread_id"]

        # Call the LangGraph agent
        response = chat_with_agent(user_message, thread_id=thread_id)

        return JsonResponse({
            "response": response,
            "thread_id": thread_id,
        })

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Agent error: {str(e)}"}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def reset_chat(request):
    """Reset the conversation by assigning a new thread_id."""
    request.session["thread_id"] = str(uuid.uuid4())
    return JsonResponse({"status": "ok", "message": "Conversation reset."})
```

### Step 8: Chat UI Template (chat/templates/chat/index.html)

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LangGraph Chat</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0b0f19; color: #f1f5f9;
            height: 100vh; display: flex; flex-direction: column;
        }
        .header {
            background: #111827; border-bottom: 1px solid rgba(255,255,255,0.08);
            padding: 16px 24px; display: flex; justify-content: space-between; align-items: center;
        }
        .header h1 { font-size: 1.2rem; font-weight: 700; }
        .header h1 span { color: #3b82f6; }
        .reset-btn {
            background: #1e293b; border: 1px solid rgba(255,255,255,0.08);
            color: #94a3b8; padding: 8px 16px; border-radius: 8px;
            cursor: pointer; font-size: 0.85rem; transition: all 0.2s;
        }
        .reset-btn:hover { background: #ef4444; color: #fff; border-color: #ef4444; }

        .chat-container {
            flex: 1; overflow-y: auto; padding: 24px;
            display: flex; flex-direction: column; gap: 16px;
        }
        .message { max-width: 75%; padding: 14px 18px; border-radius: 16px; line-height: 1.6; font-size: 0.95rem; }
        .message.user {
            align-self: flex-end; background: #3b82f6; color: #fff;
            border-bottom-right-radius: 4px;
        }
        .message.assistant {
            align-self: flex-start; background: #1e293b;
            border: 1px solid rgba(255,255,255,0.06);
            border-bottom-left-radius: 4px;
        }
        .message.assistant pre {
            background: #0d1117; padding: 12px; border-radius: 8px;
            margin: 8px 0; overflow-x: auto; font-size: 0.85rem;
        }
        .message.assistant code { font-family: 'JetBrains Mono', monospace; }
        .typing { opacity: 0.5; font-style: italic; }

        .input-area {
            background: #111827; border-top: 1px solid rgba(255,255,255,0.08);
            padding: 16px 24px; display: flex; gap: 12px;
        }
        .input-area input {
            flex: 1; background: #1e293b; border: 1px solid rgba(255,255,255,0.08);
            color: #f1f5f9; padding: 14px 18px; border-radius: 12px;
            font-size: 1rem; outline: none; transition: border-color 0.2s;
        }
        .input-area input:focus { border-color: #3b82f6; }
        .input-area input::placeholder { color: #475569; }
        .send-btn {
            background: #3b82f6; color: #fff; border: none;
            padding: 14px 24px; border-radius: 12px; cursor: pointer;
            font-size: 1rem; font-weight: 600; transition: background 0.2s;
        }
        .send-btn:hover { background: #2563eb; }
        .send-btn:disabled { background: #1e293b; color: #475569; cursor: not-allowed; }

        .welcome { text-align: center; margin: auto; color: #94a3b8; }
        .welcome h2 { font-size: 1.5rem; color: #f1f5f9; margin-bottom: 8px; }
        .welcome p { font-size: 0.95rem; }
        .tools-list { display: flex; gap: 8px; justify-content: center; margin-top: 16px; flex-wrap: wrap; }
        .tools-list span {
            background: #1e293b; border: 1px solid rgba(255,255,255,0.08);
            padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; color: #3b82f6;
        }
    </style>
</head>
<body>

<div class="header">
    <h1>🤖 <span>LangGraph</span> Chat</h1>
    <button class="reset-btn" onclick="resetChat()">🗑 New Chat</button>
</div>

<div class="chat-container" id="chatContainer">
    <div class="welcome">
        <h2>Welcome to LangGraph Chat</h2>
        <p>Ask me anything! I can calculate, tell the time, and search for info.</p>
        <div class="tools-list">
            <span>📊 Calculator</span>
            <span>🕒 Current Time</span>
            <span>🔍 Search</span>
        </div>
    </div>
</div>

<div class="input-area">
    <input type="text" id="messageInput" placeholder="Type your message..."
           onkeydown="if(event.key==='Enter') sendMessage()">
    <button class="send-btn" id="sendBtn" onclick="sendMessage()">Send</button>
</div>

<script>
const chatContainer = document.getElementById('chatContainer');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
let firstMessage = true;

function addMessage(content, role) {
    if (firstMessage) {
        const welcome = chatContainer.querySelector('.welcome');
        if (welcome) welcome.remove();
        firstMessage = false;
    }
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.textContent = content;
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return div;
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    addMessage(message, 'user');
    messageInput.value = '';
    sendBtn.disabled = true;

    const typing = addMessage('Thinking...', 'assistant typing');

    try {
        const res = await fetch('/api/chat/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        const data = await res.json();
        typing.remove();

        if (data.error) {
            addMessage('Error: ' + data.error, 'assistant');
        } else {
            addMessage(data.response, 'assistant');
        }
    } catch (err) {
        typing.remove();
        addMessage('Connection error. Please try again.', 'assistant');
    }

    sendBtn.disabled = false;
    messageInput.focus();
}

async function resetChat() {
    await fetch('/api/reset/', { method: 'POST' });
    chatContainer.innerHTML = `
        <div class="welcome">
            <h2>Welcome to LangGraph Chat</h2>
            <p>Ask me anything! I can calculate, tell the time, and search for info.</p>
            <div class="tools-list">
                <span>📊 Calculator</span>
                <span>🕒 Current Time</span>
                <span>🔍 Search</span>
            </div>
        </div>`;
    firstMessage = true;
}
</script>
</body>
</html>
```

### Step 9: Run the Application

```bash
# Apply migrations
python manage.py migrate

# Run the development server
python manage.py runserver

# Open in browser: http://127.0.0.1:8000/
```

**Try these prompts**

| Prompt | What it exercises |
|---|---|
| "What is 1024 * 768?" | Calculator tool |
| "What time is it?" | Current-time tool |
| "Search for LangGraph architecture" | Search tool |
| "Explain what LangGraph is" | Direct LLM response, no tools |

**Extension ideas**

- Add streaming responses with Django's `StreamingHttpResponse` and the agent's `stream()` method.
- Connect a real search API (Tavily, SerpAPI).
- Add user authentication and per-user thread management.
- Store conversations in a database instead of `MemorySaver`.
- Add file upload for RAG over user documents.
- Deploy with Gunicorn and Nginx.

## Tutorial Complete — Next Steps & Resources

| Resource | URL |
|---|---|
| LangGraph Documentation | https://langchain-ai.github.io/langgraph/ |
| LangGraph GitHub | https://github.com/langchain-ai/langgraph |
| LangGraph Tutorials | https://langchain-ai.github.io/langgraph/tutorials/ |
| LangSmith (Tracing) | https://smith.langchain.com/ |

## Key concepts worth simulating

- **State object flowing through the graph**: a JSON-like state card that travels from START through each node; each node highlights the keys it reads and the partial dict it returns, and the merged result is shown after the update.
- **Reducers side by side**: the same update `{"history": ["x"]}` applied to a field with no reducer (replace), `operator.add` (concatenate) and `add_messages` (append with ID de-duplication); use the `CounterState` example with `value` vs `history`.
- **Normal vs conditional edges and the ReAct cycle**: an interactive graph where the student writes/toggles the `should_continue` return value and sees the flow go to `tools` → back to `agent`, or to `END`; a counter shows tool-call rounds and trips the `MAX_ITERATIONS` / `recursion_limit` guard.
- **Tool calling round-trip**: AIMessage with `tool_calls` → `ToolNode` executes each call → `ToolMessage(tool_call_id=...)` appended → LLM reads the results and writes a final answer; animate the `messages` list growing.
- **Thread-based checkpointing**: two parallel timelines (`user-123`, `user-456`) with a MemorySaver "database" in the middle; show a checkpoint written after every node and how a new thread starts empty while the old one remembers "Alice".
- **Human-in-the-loop interrupt**: the graph pausing at `interrupt_before=["tools"]`, the pending tool call displayed for review, and `invoke(None, config)` resuming from the checkpoint (or cancelling).
- **Supervisor multi-agent routing**: supervisor reads task/research_data/analysis/report flags and emits `researcher` / `analyst` / `writer` / `FINISH`; each worker fills its slot in the shared state and hands back; the `iteration_count` reducer ticks up as a safety counter.
- **Request lifecycle in the Django app**: browser `fetch('/api/chat/')` → `chat_api` view → session `thread_id` → singleton compiled graph → JSON response → DOM update, with a `reset` that mints a new thread ID.

## Test code

```python
# 1. State + reducers with NO LLM: run the CounterState graph and watch the state merge
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

class CounterState(TypedDict):
    value: int
    history: Annotated[list[str], operator.add]

def double_it(state):
    v = state["value"] * 2
    return {"value": v, "history": [f"doubled to {v}"]}

def add_ten(state):
    v = state["value"] + 10
    return {"value": v, "history": [f"added 10 to get {v}"]}

g = StateGraph(CounterState)
g.add_node("double", double_it)
g.add_node("add_ten", add_ten)
g.add_edge(START, "double")
g.add_edge("double", "add_ten")
g.add_edge("add_ten", END)
graph = g.compile()

for event in graph.stream({"value": 5, "history": []}, stream_mode="values"):
    print(event)
# {'value': 5, 'history': []}
# {'value': 10, 'history': ['doubled to 10']}
# {'value': 20, 'history': ['doubled to 10', 'added 10 to get 20']}
```

```python
# 2. Conditional edges and a cycle with NO LLM: loop until a counter reaches 3
from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, START, END

class LoopState(TypedDict):
    n: int
    log: Annotated[list[str], operator.add]

def work(state):
    return {"n": state["n"] + 1, "log": [f"iteration {state['n'] + 1}"]}

def should_continue(state) -> Literal["work", "__end__"]:
    return "work" if state["n"] < 3 else "__end__"

g = StateGraph(LoopState)
g.add_node("work", work)
g.add_edge(START, "work")
g.add_conditional_edges("work", should_continue)
graph = g.compile()

print(graph.invoke({"n": 0, "log": []}))
print(graph.get_graph().draw_mermaid())   # Mermaid source; paste into https://mermaid.live to view
# (draw_ascii() needs `pip install grandalf` and cannot lay out the work -> work self-loop)
```

```python
# 3. ReAct agent with tools (needs OPENAI_API_KEY)
from langgraph.graph import StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression such as '42 * 17'."""
    return str(eval(expression, {"__builtins__": {}}, {}))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools([calculator])

def agent(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

g = StateGraph(MessagesState)
g.add_node("agent", agent)
g.add_node("tools", ToolNode([calculator]))
g.add_edge(START, "agent")
g.add_conditional_edges("agent", tools_condition)
g.add_edge("tools", "agent")
graph = g.compile()

result = graph.invoke({"messages": [HumanMessage(content="What is 42 * 17? Use the tool.")]},
                      config={"recursion_limit": 10})
for m in result["messages"]:
    m.pretty_print()
```

```python
# 4. Memory with MemorySaver and thread IDs (needs OPENAI_API_KEY)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
g = StateGraph(MessagesState)
g.add_node("chatbot", lambda s: {"messages": [llm.invoke(s["messages"])]})
g.add_edge(START, "chatbot")
g.add_edge("chatbot", END)
graph = g.compile(checkpointer=MemorySaver())

a = {"configurable": {"thread_id": "A"}}
b = {"configurable": {"thread_id": "B"}}
graph.invoke({"messages": [HumanMessage(content="My name is Alice.")]}, config=a)
print("A:", graph.invoke({"messages": [HumanMessage(content="What is my name?")]}, config=a)["messages"][-1].content)
print("B:", graph.invoke({"messages": [HumanMessage(content="What is my name?")]}, config=b)["messages"][-1].content)
print("Checkpoints on thread A:", len(list(graph.get_state_history(a))))
```
