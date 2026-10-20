---
week: 11
title: "Model Context Protocol (MCP)"
source: https://albanna-tutorials.com/mcp.html
---
# Model Context Protocol (MCP)

> Visual Guide to MCP. Level: Intermediate · ~25 min.
>

## Introduction

The Model Context Protocol is changing how AI assistants connect to data sources and tools. **Think of it as a USB-C for AI**: just as USB-C gives devices one standard way to connect, MCP gives AI systems one standard way to connect to data — a universal connector that standardizes integration.

### Why MCP matters
- Standardizes how AI models connect to data sources.
- Reduces fragmentation in AI tool integrations.
- Lets AI access data securely within existing infrastructure.
- Simplifies building AI applications with added capabilities.
- Gives flexibility to switch between LLM providers.

## What is MCP?

MCP is an open standard introduced by Anthropic in November 2024 for connecting AI applications with external tools, data sources, and systems. Messages are JSON-RPC 2.0, carried over **stdio** (the client launches the server as a subprocess) or **Streamable HTTP** (the earlier HTTP+SSE transport is legacy). It addresses a core problem: AI models are isolated from data trapped in silos and legacy systems, and every new integration needs custom work.

**Key characteristics**
- **Secure** — access stays within your own security boundaries.
- **Open** — a community-driven, open-source standard anyone can contribute to.
- **Interoperable** — works across different LLM providers.
- **Extensible** — adapts to many kinds of data sources.

## MCP Architecture

MCP uses a **client–server** design with three components.

| Component | Role |
|---|---|
| **MCP Host** | Applications such as Claude Desktop, IDEs, or other AI-powered tools that want data through MCP. The host is where the user interacts. |
| **MCP Client** | Maintains a **1:1 connection** with each server and manages the protocol communication between host and server. |
| **MCP Server** | Lightweight programs that each expose specific capabilities — **tools** (functions the model can call), **resources** (data the host can read) and **prompts** (reusable templates) — and connect to the actual data sources and tools. |

**Data sources**
- **Local data sources** — files, databases, and services on your computer. MCP servers can access them without sending the data to external services.
- **Remote services** — external systems reached over the internet through APIs, e.g. GitHub, Slack, weather services.

The design is modular, and sensitive data stays within your security boundaries.

## How MCP Works


1. **User query** — the user asks an AI assistant a question through an MCP host such as Claude Desktop.
2. **AI determines context needs** — the model recognizes it needs more information and identifies which servers may have it.
3. **Host queries servers** — the host sends requests to the appropriate servers through the MCP client.
4. **Server processes request** — each server accesses its data source, processes the request, and returns the relevant information.
5. **Response aggregation** — the client collects and formats the responses from all servers and passes them to the model.
6. **AI response generation** — the model uses the added context to produce a more informed, accurate answer.

**Key protocol features**
- **Server discovery** — hosts can find which servers are available and what they can do.
- **Standardized schema** — all exchanged data follows a consistent format models can parse.
- **Request/response pattern** — requests use structured parameters and return structured values.
- **Access controls** — built-in authentication and authorization protect data sources.

## Implementation Examples

### Example 1: Accessing local files
*Scenario:* a user asks Claude to analyze a spreadsheet on their local machine.
1. Claude recognizes it needs local files.
2. The host queries the Local Files MCP server.
3. The server opens the file **with the user's permission**.
4. Claude receives the file content in a standard format.
5. Claude analyzes the data and responds.

### Example 2: Integrating with GitHub
*Scenario:* a developer asks an AI coding assistant to explain recent changes in a GitHub repository.
1. The AI recognizes it needs GitHub data.
2. The host connects to the GitHub MCP server.
3. The server authenticates with GitHub and retrieves the commit history.
4. The AI receives the repository data in a formatted structure.
5. The AI analyzes the changes and explains them.

### Sample MCP server implementation (Python)
Using the official `mcp` Python SDK (`pip install "mcp[cli]"`). The tool's input schema is generated from the function signature and docstring; the **Test code** section adds a matching client.

```python
from mcp.server import MCPServer   # mcp 1.x: from mcp.server.fastmcp import FastMCP as MCPServer

# Create a simple MCP server
mcp = MCPServer("FileReader", instructions="Reads files from the local filesystem")

# Define and implement a tool (the JSON schema is derived from the type hints)
@mcp.tool()
def read_file(path: str) -> str:
    """Reads content from a file."""
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"error: {e}"

# Start the server (stdio transport by default)
if __name__ == "__main__":
    mcp.run()
```

### Available servers

| Server | What it connects to |
|---|---|
| GitHub | Repositories — e.g. retrieving commit history |
| Slack | Workspace messages and channels (remote service) |
| Google Drive | Documents and files in Drive |
| PostgreSQL | Relational databases |
| Git | Local repositories |
| Local Files | Files on your machine, with your permission |
| Puppeteer | Browser automation |
| Many more | Community-built servers |

## Benefits of MCP


**For developers**
- Simplified integration — one protocol works with multiple AI models.
- Modularity — components can be built and maintained independently.
- Interoperability — switch between LLM providers.
- Reusable components — existing MCP servers can be reused.
- Built-in security patterns — security practices are part of the design.

**For organizations**
- Data security — sensitive data stays within your security boundaries.
- Productivity — AI tools can reach internal systems.
- Control — governance over data access is maintained.
- Future-proofing — adopt new AI models without rewiring integrations.
- Cost efficiency — lower integration-maintenance costs.

**For end users**
- Smarter responses — the AI fetches the information it needs, making answers more accurate and relevant.
- Enhanced privacy — sensitive data can stay on local devices while remaining available to the assistant.
- More capabilities — the AI handles more complex tasks by retrieving the right information at the right time.

**Long-term vision**
- A universal layer connecting AI models to many kinds of data sources.
- Marketplaces of specialized MCP servers for different use cases.
- Advanced AI workflows that span multiple systems.
- Context-awareness that carries across different applications.

## Use Cases


| Domain | Applications |
|---|---|
| Software development | Code analysis using repository context; bug fixing with access to logs; generating and reviewing pull requests |
| Data analysis | Querying data from various sources; generating visualizations and reports; trend analysis and forecasting |
| Healthcare | Research assistance using medical data (privacy-compliant); summarizing patient history; analyzing treatment options |
| Education | Personalized tutoring using course materials; help with research papers; curriculum development using historical data |
| Legal | Legal research using case databases; contract analysis and drafting; comparing legal precedents |

## Success Stories

> "Open technologies like the Model Context Protocol are the bridges that connect AI to real-world applications, ensuring innovation is accessible, transparent, and rooted in collaboration."
> — Dhanji R. Prasanna, Chief Technology Officer, Block

- **Apollo** has integrated MCP into its systems to strengthen its platform and offer better AI-powered features.
- **Zed, Replit, Codeium, and Sourcegraph** are working with MCP to improve their platforms so AI agents can retrieve relevant information more effectively.

## Getting Started
- **Explore:** GitHub repository — https://github.com/modelcontextprotocol · Quickstart — https://modelcontextprotocol.io/docs/develop/build-server · Docs — https://modelcontextprotocol.io/docs/getting-started/intro · Examples — https://modelcontextprotocol.io/examples
- **Contribute:** https://modelcontextprotocol.io/community/contributing · Discussions — https://github.com/orgs/modelcontextprotocol/discussions
- **Try it:** Download Claude Desktop — https://claude.ai/download
- **Announcement:** https://www.anthropic.com/news/model-context-protocol

## Key concepts worth simulating

- **Host ↔ Client ↔ Server topology**: one host holding several clients, each pinned 1:1 to a server; highlight that adding a server adds a client, not a new integration in the model.
- **Six-step request flow**: user query → model decides it needs context → client sends `tools/call` → server reads data source → response aggregated → model answers; animate the JSON-RPC messages on the wire.
- **Initialization handshake**: `initialize` request/response with protocol version and capability negotiation, then `notifications/initialized`.
- **Discovery**: `tools/list`, `resources/list`, `prompts/list` responses populating the model's available-tool menu at runtime.
- **Local vs remote boundary**: a local file server returning content over stdio (data never leaves the machine) vs a remote GitHub server over HTTP with auth.
- **Permission gate**: the "with the user's permission" step in Example 1 as an explicit approve/deny interrupt in the flow.
- **USB-C analogy**: N models × M data sources as N·M custom adapters vs N + M MCP connections.

## Test code

```python
# 1. A real MCP server with the official SDK (pip install "mcp[cli]")
#    Save as server.py and run: python server.py   (speaks JSON-RPC over stdio)
from mcp.server import MCPServer   # mcp 1.x: from mcp.server.fastmcp import FastMCP as MCPServer

mcp = MCPServer("FileReader")

@mcp.tool()
def read_file(path: str) -> str:
    """Reads content from a file."""
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"error: {e}"

if __name__ == "__main__":
    mcp.run()  # stdio transport by default
```

```python
# 2. An MCP client that launches the server above, lists tools, and calls one
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command="python", args=["server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()                 # handshake + capability negotiation
            tools = await session.list_tools()         # discovery
            print([t.name for t in tools.tools])
            result = await session.call_tool("read_file", {"path": "server.py"})
            print(result.content[0].text[:80])

asyncio.run(main())
```

```python
# 3. Hand-build the JSON-RPC messages MCP exchanges, to see the wire format
import json

initialize = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-06-18",
                         "capabilities": {},
                         "clientInfo": {"name": "demo-host", "version": "0.1"}}}
list_tools = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
call_tool  = {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
              "params": {"name": "read_file", "arguments": {"path": "notes.txt"}}}

for msg in (initialize, list_tools, call_tool):
    print(json.dumps(msg))
```

```python
# 4. The USB-C argument in numbers: custom adapters vs MCP connections
models, sources = 4, 6
print("custom integrations :", models * sources)   # every model x every source
print("with MCP            :", models + sources)   # each side implements the protocol once
```
