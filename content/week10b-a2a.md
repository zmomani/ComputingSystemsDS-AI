---
week: 10
title: "Agent-to-Agent Protocols: A2A and MCP"
source: https://albanna-tutorials.com/a2a.html
---
# Agent-to-Agent Protocols: A2A and MCP

> Standardizing communication between AI agents and data sources. Level: Intermediate · AI Agents · ~25 min.
>

Google's Agent2Agent (A2A) protocol and Anthropic's Model Context Protocol (MCP) are two complementary protocols that are changing how AI systems interact.

## A2A (Agent2Agent Protocol)

A2A is an open protocol announced by Google in April 2025. It is designed to let AI agents communicate and interoperate regardless of the framework or vendor they are built on. More than 50 technology and service partners support it, including Atlassian, Box, Cohere, Intuit, LangChain, MongoDB, PayPal, Salesforce, SAP, and ServiceNow.

### Design principles
- **Embrace agentic capabilities** — supports natural, collaborative workflows between agents.
- **Build on existing standards** — HTTP, SSE (Server-Sent Events), and JSON-RPC.
- **Secure by default** — enterprise-grade authentication and authorization.
- **Support long-running tasks** — real-time feedback and state updates.
- **Modality agnostic** — multiple content formats, including audio and video.

### Key components
- **Agent Card** — a public metadata file (JSON, served at `/.well-known/agent-card.json`) describing an agent's capabilities, skills, endpoint URL, and authentication requirements. Clients use it for discovery.
- **Task-oriented communication** — work is organized as tasks with a defined lifecycle. A task may complete immediately or remain active for an extended period.
- **Messages, Parts, Artifacts** — agents exchange messages whose content is carried as *parts* (text, files, or structured data); a finished task may return *artifacts*.
- **UX / capability negotiation** — agents negotiate the right content format, explicitly including user-interface capabilities.

### Agent Card fields (from the sample)
| Field | Meaning |
|---|---|
| `protocolVersion` | A2A protocol version the agent speaks |
| `name`, `description`, `version` | Display name, what the agent does, version string |
| `url`, `preferredTransport` | The agent's A2A endpoint and its transport (`JSONRPC`, `GRPC` or `HTTP+JSON`) |
| `securitySchemes` | Named auth schemes (sample: an `oauth2` authorization-code flow with `authorizationUrl`, `tokenUrl` and `scopes`) |
| `security` | Which scheme(s) and scopes a client must present |
| `capabilities.streaming` | Whether SSE streaming is supported |
| `capabilities.pushNotifications` | Whether the agent can push task updates to a client webhook |
| `defaultInputModes`, `defaultOutputModes` | Media types the agent accepts and produces (e.g. `text/plain`, `application/json`) |
| `skills[]` | Each with `id`, `name`, `description`, `tags`, optional `examples`, `inputModes`, `outputModes` |

### A2A workflow (step by step)
1. **Agent Discovery** — the client agent reads the remote agent's Agent Card to learn its capabilities and endpoints.
2. **Task Initialization** — the client sends a task request with an initial message and task parameters.
3. **Task Processing** — the remote agent completes the task immediately or enters a *working* state for longer operations.
4. **Status Updates** — for long-running tasks, the remote agent streams status updates over SSE.
5. **Message Exchange** — agents exchange text, files, or structured data as parts to collaborate.
6. **Task Completion** — the task reaches a terminal state (*completed*, *failed*, *canceled*, or *rejected*) and may return artifacts. Along the way it may pause in *input-required* or *auth-required* until the client responds.

### Sample A2A Agent Card (JSON)

```json
{
  "protocolVersion": "0.3.0",
  "name": "Recruitment Assistant",
  "description": "An agent that helps find and evaluate job candidates",
  "version": "1.0.0",
  "url": "https://recruitment-agent.example.com/a2a",
  "preferredTransport": "JSONRPC",
  "securitySchemes": {
    "oauth": {
      "type": "oauth2",
      "flows": {
        "authorizationCode": {
          "authorizationUrl": "https://auth.example.com/authorize",
          "tokenUrl": "https://auth.example.com/token",
          "scopes": {
            "read": "Read candidate data",
            "write": "Create and update candidate records"
          }
        }
      }
    }
  },
  "security": [{ "oauth": ["read", "write"] }],
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "defaultInputModes": ["text/plain", "application/json"],
  "defaultOutputModes": ["text/plain", "application/json"],
  "skills": [
    {
      "id": "find-candidates",
      "name": "findCandidates",
      "description": "Find candidates matching a job title, required skills and preferred location",
      "tags": ["recruiting", "search"],
      "examples": ["Find Python developers in Amman with Spark experience"],
      "inputModes": ["application/json"],
      "outputModes": ["application/json"]
    }
  ]
}
```

### Frameworks and implementations
- **Frameworks:** Agent Development Kit (ADK), CrewAI, LangGraph, Genkit.
- **Samples:** client/server examples in Python and JavaScript, multi-agent web applications, and command-line interfaces.

### A2A scenarios
**Candidate sourcing** — a recruiting manager needs candidates for a software engineering position.
1. The manager's agent connects to specialized recruiting agents.
2. Each agent handles one part: sourcing, screening, or scheduling.
3. The agents collaborate over A2A to find suitable candidates.
4. The manager receives a unified view of the results.
5. After selection, the manager's agent uses A2A to coordinate interview scheduling.

**Multi-agent research** — a data scientist needs comprehensive research on a complex topic.
1. The primary agent divides the research tasks among specialized agents.
2. A market-research agent gathers industry trends.
3. A data-analysis agent processes relevant statistics.
4. A visualization agent creates charts from the processed data.
5. The primary agent synthesizes everything into a cohesive report.

## MCP (Model Context Protocol)

MCP is an open standard from Anthropic (announced November 2024) that standardizes how AI applications connect to external tools, data sources, and systems, so assistants can reach data where it lives and reduce information silos. (Full treatment in Week 11.)

### Architecture
- **Host** — an application such as Claude Desktop or an IDE that wants data through MCP; the entry point for user interaction.
- **Client** — maintains a 1:1 connection with each server and manages protocol details.
- **Server** — a lightweight program exposing specific capabilities and connecting to the underlying data sources and tools.

### Key characteristics
- **Secure** — operates within your security boundaries.
- **Open** — community-driven, open-source standard.
- **Interoperable** — works across LLM providers.
- **Extensible** — adaptable to various data sources.

### MCP workflow
1. **Connection Initialization** — the MCP client in the host connects to a server and exchanges capabilities and protocol versions.
2. **User Query** — the user interacts with the host; the model identifies a need for external data.
3. **Resource Request** — the client requests the needed data or tool from the appropriate server.
4. **Server Processing** — the server accesses the resource (file, database, API) and formats the data.
5. **Response Delivery** — the server returns the formatted data to the client via the standard protocol.
6. **AI Response Generation** — the model uses the retrieved data to produce a more informed response.

### Sample MCP server implementation (Python)
Using the official `mcp` Python SDK (`pip install "mcp[cli]"`). The tool's JSON schema is generated from the type hints and docstring; see Week 11 for the matching client.

```python
from mcp.server import MCPServer   # mcp 1.x: from mcp.server.fastmcp import FastMCP as MCPServer

# Create a simple MCP server
mcp = MCPServer("FileReader", instructions="Reads files from the local filesystem")

# Define and implement a tool (the schema is derived from the signature)
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

### MCP scenarios
**Document analysis** — a lawyer reviewing legal documents.
1. Claude recognizes it needs access to local files.
2. The MCP host connects to a FileReader MCP server.
3. The server securely accesses the documents.
4. Claude receives the contents in a standardized format.
5. Claude analyzes the legal language and provides insights.

**Integrated development** — an AI assistant in an IDE.
1. The assistant needs to understand the codebase context.
2. MCP connects it to Git and GitHub MCP servers.
3. The Git server provides local repository information.
4. The GitHub server provides issue and pull-request context.
5. The assistant generates contextually appropriate code.

**Pre-built MCP servers:** GitHub, Slack, Google Drive, PostgreSQL, Git, Puppeteer.

## How A2A and MCP complement each other

- **MCP** connects AI models to the tools, data, and systems they need — "the protocol to connect agents with their structured tools."
- **A2A** lets end-users or other agents work with specialized agents (one that finds job candidates, one that summarizes meeting notes) and lets those agents collaborate.

### Feature comparison

| Aspect | A2A (Google) | MCP (Anthropic) |
|---|---|---|
| Primary purpose | Communication between different AI agents | Connecting AI models to data sources and tools |
| Focus | Agent-to-agent collaboration | Agent-to-data / agent-to-tool connection |
| Communication pattern | Task-based messages between agents | Request–response for tools and data |
| Key components | Agent Cards, Tasks, Messages, Parts, Artifacts | Hosts, Clients, Servers, Tools, Resources, Prompts |
| Discovery mechanism | Agent Cards and capability advertising | Host discovers available servers |
| Transport | HTTP(S) with JSON-RPC 2.0 (plus gRPC and HTTP+JSON/REST bindings); SSE for streaming | stdio; Streamable HTTP (the earlier HTTP+SSE transport is legacy) — both carry JSON-RPC 2.0 |
| Ideal use case | Multi-agent workflows requiring collaboration | A single agent that needs external systems |
| Announced | April 2025 | November 2024 |

## Use cases

| Domain | MCP | A2A | Both |
|---|---|---|---|
| Software development | Code analysis with repository context | Collaborative code review; multi-agent debugging | |
| Data analysis | Querying data from various sources | Collaborative exploration; multi-agent reporting | |
| Enterprise workflows | Access to enterprise data sources | Cross-department agent collaboration | Secure information sharing |
| Healthcare | Secure access to medical records; patient-history summarization | Specialized diagnostic collaboration | |
| Education | Access to educational resources | Collaborative tutoring agents | Personalized learning experiences |

### Real-world examples
- **Multi-agent customer support** — specialized agents handle inquiry routing, knowledge-base search, ticket creation, and follow-up coordination (A2A for agent communication, MCP for data access).
- **Supply-chain optimization** — agents coordinate inventory management, logistics planning, demand forecasting, and supplier relations (primarily A2A).
- **Research assistant** — assistants access academic papers, experimental data, and simulation results to analyze and synthesize findings (heavily MCP).

## Resources
- A2A official website — https://a2a-protocol.org/
- A2A GitHub repository — https://github.com/a2aproject/A2A
- MCP official website — https://modelcontextprotocol.io/
- MCP GitHub repository — https://github.com/modelcontextprotocol
- A2A discussions — https://github.com/a2aproject/A2A/discussions
- MCP discussions — https://github.com/modelcontextprotocol/modelcontextprotocol/discussions
- Google A2A announcement — https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/
- Anthropic MCP announcement — https://www.anthropic.com/news/model-context-protocol

## Key concepts worth simulating

- **Agent Card discovery**: a client fetches `/.well-known/agent-card.json` from several agents, parses skills/capabilities, and picks the agent whose skill matches the task.
- **Task lifecycle state machine**: submitted → working → (input-required / auth-required ↔ working) → completed | failed | canceled | rejected; animate transitions and which ones are terminal.
- **SSE streaming of status updates**: a long-running task emitting `TaskStatusUpdateEvent` chunks over a single HTTP connection versus polling.
- **Messages made of Parts**: a message carrying a TextPart + FilePart + DataPart; show how a multimodal-incapable agent negotiates a text-only format.
- **Multi-agent research handoff**: primary agent splits a task into subtasks, dispatches them to three remote agents in parallel via JSON-RPC, collects artifacts, and merges them.
- **A2A vs MCP layering**: one diagram with agents talking horizontally over A2A and each agent reaching down to its tools over MCP.

## Test code

```python
# 1. Parse an Agent Card and check whether it advertises a needed skill
import json

card = json.loads("""{
  "name": "Recruitment Assistant", "version": "1.0.0",
  "url": "https://recruitment-agent.example.com/a2a",
  "capabilities": {"streaming": true, "pushNotifications": false},
  "defaultInputModes": ["text/plain", "application/json"],
  "defaultOutputModes": ["application/json"],
  "skills": [{"id": "find-candidates", "name": "findCandidates",
              "description": "Find candidates matching job requirements",
              "tags": ["recruiting", "search"]}]
}""")

def has_skill(card, name):
    return any(s["name"] == name or s["id"] == name for s in card["skills"])

print(card["name"], "supports findCandidates:", has_skill(card, "findCandidates"))
print("streams:", card["capabilities"]["streaming"])
```

```python
# 2. Task lifecycle state machine
TRANSITIONS = {
    "submitted": {"working", "auth-required", "rejected", "canceled"},
    "working": {"input-required", "auth-required", "completed", "failed", "canceled"},
    "input-required": {"working", "canceled"},   # interrupted: waiting on the client
    "auth-required": {"working", "canceled"},    # interrupted: waiting on credentials
}
TERMINAL = {"completed", "failed", "canceled", "rejected"}

class Task:
    def __init__(self):
        self.state = "submitted"
        self.history = [self.state]
    def move(self, new):
        if self.state in TERMINAL or new not in TRANSITIONS[self.state]:
            raise ValueError(f"illegal {self.state} -> {new}")
        self.state = new
        self.history.append(new)

t = Task()
for s in ["working", "input-required", "working", "completed"]:
    t.move(s)
print(" -> ".join(t.history))
```

```python
# 3. Build a JSON-RPC 2.0 'message/send' request with a multi-part message
import json, uuid

def message_send(text, data):
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "messageId": str(uuid.uuid4()),
                "role": "user",
                "parts": [
                    {"kind": "text", "text": text},
                    {"kind": "data", "data": data},
                ],
            },
        },
    }

print(json.dumps(message_send("Find Python developers in Amman",
                            {"jobTitle": "Data Engineer", "skills": ["Python", "Spark"]}),
                 indent=2))
```

```python
# 4. Simulate SSE status updates from a long-running remote agent
import json, time

def remote_agent_stream():
    for i, state in enumerate(["working", "working", "working", "completed"]):
        time.sleep(0.2)
        event = {"kind": "status-update", "taskId": "task-1", "contextId": "ctx-1",
                 "status": {"state": state}, "final": state == "completed"}
        # each SSE chunk carries one JSON-RPC response whose result is a TaskStatusUpdateEvent
        yield "data: " + json.dumps({"jsonrpc": "2.0", "id": 1, "result": event}) + "\n\n"

for chunk in remote_agent_stream():
    print(chunk, end="")
```
