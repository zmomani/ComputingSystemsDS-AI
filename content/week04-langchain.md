---
week: 4
title: LangChain Fundamentals
source: https://albanna-tutorials.com/langchain.html
---
# LangChain Fundamentals

> Material by Dr. Abedal-Kareem Al-Banna (ilmy / albanna-tutorials.com). Original page: "LangChain Python Tutorial" — Beginner level, about 10 minutes. The source page is an overview/orientation page with no code blocks; the test code at the end of this file was written for the course and is consistent with the libraries the page describes.

## Welcome to LangChain Python

A framework for developing applications powered by large language models (LLMs).

Official documentation: https://python.langchain.com/

## The LangChain Framework

LangChain simplifies every stage of the LLM application lifecycle, from development to productionization to deployment.

![LangChain Framework Diagram](https://python.langchain.com/svg/langchain_stack_062024.svg)

*Figure: the LangChain stack — open-source libraries (langchain-core, langchain, langchain-community, LangGraph) at the base, LangSmith for productionization, and LangGraph Cloud / LangServe for deployment.*

## Development

Build your applications using LangChain's open-source building blocks, components, and third-party integrations.

### Building Blocks (LCEL)

Create composable chains with LangChain Expression Language, providing flexibility and reusability.

Reference: https://python.langchain.com/docs/concepts/#langchain-expression-language-lcel

### LangGraph

Build stateful agents with first-class streaming and human-in-the-loop support for complex applications.

Reference: https://python.langchain.com/docs/concepts/#langgraph

## Productionization

Use LangSmith to inspect, monitor and evaluate your chains, so that you can continuously optimize and deploy with confidence.

### LangSmith

A developer platform that lets you debug, test, evaluate, and monitor LLM applications.

Reference: https://docs.smith.langchain.com/

## Deployment

Turn your LangGraph applications into production-ready APIs and Assistants with LangGraph Cloud.

### LangGraph Cloud

Deploy your LangGraph applications as production-ready APIs and Assistants.

Reference: https://langchain-ai.github.io/langgraph/cloud/

## LangChain Libraries

The framework consists of the following open-source libraries:

| Library | Role |
|---|---|
| **langchain-core** | Base abstractions and LangChain Expression Language that form the foundation of LangChain. |
| **langchain-community** | Third party integrations enabling connections to external services and tools. |
| **langchain** | Chains, agents, and retrieval strategies that make up an application's cognitive architecture. |
| **LangGraph** | Build robust and stateful multi-actor applications with LLMs by modeling steps as edges and nodes in a graph. |
| **LangServe** | Deploy LangChain chains as REST APIs for easy integration with other services. |
| **LangSmith** | A developer platform that lets you debug, test, evaluate, and monitor LLM applications. |

> Note: these docs focus on the Python LangChain library. For the JavaScript version see https://js.langchain.com/.

## Tutorials

If you're looking to build something specific or are more of a hands-on learner, check out our tutorials section.

### Build a Simple LLM Application

Learn how to create a basic application powered by large language models.
https://python.langchain.com/docs/tutorials/llm_chain/

### Build a Chatbot

Create your own conversational AI assistant using LangChain's components.
https://python.langchain.com/docs/tutorials/chatbot/

### Build an Agent

Develop intelligent agents that can perform tasks and make decisions autonomously.
https://python.langchain.com/docs/tutorials/agents/

### Introduction to LangGraph

Learn how to use LangGraph to build stateful, multi-actor applications with LLMs.
https://langchain-ai.github.io/langgraph/tutorials/introduction/

All tutorials: https://python.langchain.com/docs/tutorials/

## Guides & Reference

### How-to guides

Short answers to "How do I….?" types of questions. These guides help you quickly accomplish common tasks.
https://python.langchain.com/docs/how_to/ — LangGraph how-tos: https://langchain-ai.github.io/langgraph/how-tos/

### Conceptual guide

Introductions to all the key parts of LangChain you'll need to know! High level explanations of all LangChain concepts.
https://python.langchain.com/docs/concepts/ — LangGraph concepts: https://langchain-ai.github.io/langgraph/concepts/

### API reference

Full documentation of all classes and methods in the LangChain Python packages.
https://python.langchain.com/api_reference/

## Additional Resources

- **LangSmith** — Trace and evaluate your language model applications and intelligent agents to help you move from prototype to production. https://docs.smith.langchain.com/
- **LangGraph** — Build stateful, multi-actor applications with LLMs. Integrates smoothly with LangChain, but can be used without it. https://langchain-ai.github.io/langgraph
- **Versions** — See what changed in v0.3, learn how to migrate legacy code, read up on versioning policies, and more. https://python.langchain.com/docs/versions/v0_3/
- **Security** — Read up on security best practices to make sure you're developing safely with LangChain. https://python.langchain.com/docs/security/
- **Integrations** — LangChain is part of a rich ecosystem of tools that integrate with our framework and build on top of it. https://python.langchain.com/docs/integrations/providers/
- **Contributing** — Check out the developer's guide for guidelines on contributing and help getting your dev environment set up. https://python.langchain.com/docs/contributing/

## Quick Links

- LangChain docs: https://python.langchain.com/
- Tutorials: https://python.langchain.com/docs/tutorials/
- How-to guides: https://python.langchain.com/docs/how_to/
- Concepts: https://python.langchain.com/docs/concepts/
- API reference: https://python.langchain.com/api_reference/
- GitHub: https://github.com/langchain-ai/langchain

## Key concepts worth simulating

- **The LLM application lifecycle as three stages**: Development (building blocks) → Productionization (LangSmith traces/evals) → Deployment (LangGraph Cloud / LangServe API). Animate an app moving through the stages, with the relevant library lighting up at each stage.
- **The library stack diagram**: langchain-core at the base; langchain and langchain-community on top; LangGraph beside them; LangSmith and LangServe/LangGraph Cloud as the outer layers. Let the student click each box to see what it provides and which imports come from it (`from langchain_core...`, `from langchain_community...`).
- **LCEL pipe composition**: prompt template → chat model → output parser as a pipeline built with `|`; show a user dict entering on the left, the rendered prompt, the raw AIMessage, then the parsed string exiting on the right.
- **Chains vs agents**: a linear chain where the path is fixed, versus an agent where the LLM chooses a tool at run time; a toggle that re-routes the same question through both.
- **Prompt variable substitution**: a template with `{topic}` placeholders filling in live as the student types into input fields.
- **Tracing in LangSmith**: a nested run tree (chain → prompt → llm → parser) with latency and token counts per span, to motivate "productionization".

## Test code

```python
# 1. Verify installation and call a chat model directly
# pip install langchain langchain-core langchain-openai
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

os.environ["OPENAI_API_KEY"] = "your-api-key-here"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print(llm.invoke([HumanMessage(content="In one sentence, what is LangChain?")]).content)
```

```python
# 2. A minimal LCEL chain: prompt template -> LLM -> output parser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise tutor for a data-science course."),
    ("human", "Explain {topic} in two sentences."),
])
chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()

print(chain.invoke({"topic": "LangChain Expression Language"}))
```

```python
# 3. Inspect the pieces of a chain without an API key
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Summarize {text} for a {audience}."),
])
messages = prompt.format_messages(text="the LangChain stack", audience="first-year student")
for m in messages:
    print(f"{m.type}: {m.content}")
print("Input variables:", prompt.input_variables)
```

```python
# 4. Stream tokens from a chain (streaming is first-class in LCEL)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

chain = (
    ChatPromptTemplate.from_template("List three uses of {tool} as bullet points.")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)
for chunk in chain.stream({"tool": "LangSmith"}):
    print(chunk, end="", flush=True)
print()
```
