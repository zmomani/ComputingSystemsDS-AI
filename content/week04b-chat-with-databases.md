---
week: 4
title: Chat with Databases
source: https://albanna-tutorials.com/database.html
---
# Chat with Databases


## Introduction

LangChain provides powerful tools to interact with your databases using natural language. This application demonstrates how to create a chat interface that allows users to query databases in conversational language instead of writing SQL.

**Why use LangChain for database interactions?**

- Query databases using natural language
- Convert user questions to optimized SQL
- Interpret results in human-readable format
- Handle follow-up questions with context
- Protect against SQL injection with built-in safeguards

## Chat Interface

A sample conversation with the demo assistant over the Chinook sample database (a digital music store):

> **Assistant:** Hello! I'm your database assistant. I can help you query the sample Chinook database.
>
> **User:** Which country's customers spent the most?
>
> **Assistant:** Looking at the data, customers from the USA spent the most, with a total of $523.06.
>
> **User:** Who are the top 3 best selling artists?
>
> **Assistant:** The top 3 best selling artists are:
> 1. Iron Maiden - 140 units sold
> 2. U2 - 107 units sold
> 3. Metallica - 91 units sold

## Database Schema

The Chinook schema is organised around these main tables.

### Customer

| Column | Type | Constraints |
|---|---|---|
| CustomerId | INTEGER | Primary key |
| FirstName | NVARCHAR(40) | Not null |
| LastName | NVARCHAR(20) | Not null |
| Country | NVARCHAR(40) | — |
| Email | NVARCHAR(60) | Not null |

### Invoice

| Column | Type | Constraints |
|---|---|---|
| InvoiceId | INTEGER | Primary key |
| CustomerId | INTEGER | Foreign key → Customer |
| InvoiceDate | DATETIME | Not null |
| BillingCountry | NVARCHAR(40) | — |
| Total | NUMERIC(10,2) | Not null |

### Artist

| Column | Type | Constraints |
|---|---|---|
| ArtistId | INTEGER | Primary key |
| Name | NVARCHAR(120) | — |

### Album

| Column | Type | Constraints |
|---|---|---|
| AlbumId | INTEGER | Primary key |
| Title | NVARCHAR(160) | Not null |
| ArtistId | INTEGER | Foreign key → Artist |

### Key relationships

- Each Customer can have many Invoices.
- Each Artist can have many Albums.
- Each Album can have many Tracks.
- Tracks and Invoices are linked through InvoiceLine — each invoice line connects one Invoice to one Track.

## How LangChain Connects to Databases

LangChain's `SQLDatabaseToolkit` gives an AI agent a set of tools for querying and working with SQL databases:

| Tool | Purpose |
|---|---|
| `QuerySQLDatabaseTool` | Executes SQL queries |
| `InfoSQLDatabaseTool` | Gets schema information |
| `ListSQLDatabaseTool` | Lists available tables |
| `QuerySQLCheckerTool` | Validates SQL queries before execution |

### The role of the language model

The LLM acts as the bridge between natural language and SQL. It:

1. Interprets the user's question.
2. Writes a valid SQL query based on the schema.
3. Reviews the query for errors or security issues.
4. Turns raw results into a readable answer.
5. Keeps conversation context for follow-up questions.

### Architecture Overview — System Components

- **Database Connection:** SQLAlchemy connection to your database (SQLite, MySQL, PostgreSQL, etc.)
- **LLM Integration:** Connection to OpenAI, Anthropic, or other LLM APIs
- **LangChain Tools:** Specialized tools for SQL query generation and validation
- **Agent Framework:** LangChain's agent system for orchestrating complex interactions
- **Chat Interface:** Frontend for user interactions with the system

## Security Considerations

Building Q&A systems over SQL databases means executing model-generated SQL, which carries inherent risk. Recommendations:

- Always use read-only database connections when possible
- Scope database permissions as narrowly as possible
- Consider implementing additional validation layers
- Monitor query patterns and implement rate limiting

## Process Flow

Worked example: *"Which country's customers spent the most?"*

1. **User Query Input** — The user enters a natural language question about the database data.
2. **Database Schema Analysis** — The system uses `ListSQLDatabaseTool` and `InfoSQLDatabaseTool` to understand the database structure. For this question the relevant tables are Customer and Invoice.
3. **SQL Query Generation** — The LLM converts the natural language question into a SQL query based on the schema:

```sql
SELECT c.Country, SUM(i.Total) AS TotalSpent
FROM Customer c
JOIN Invoice i ON c.CustomerId = i.CustomerId
GROUP BY c.Country
ORDER BY TotalSpent DESC
LIMIT 1;
```

4. **Query Validation** — The `QuerySQLCheckerTool` validates the SQL query for errors and potential improvements.
5. **Database Execution** — The `QuerySQLDatabaseTool` executes the SQL query against the database. Result: `USA, 523.06`.
6. **Result Interpretation** — The LLM translates the raw query results into a natural language response.

## Sample Queries

| Question | Tags |
|---|---|
| Which country's customers spent the most? | Sales · Geography · Aggregation |
| Who are the top 3 best selling artists? | Artists · Sales · Ranking |
| What's the most popular music genre by sales? | Genres · Popularity · Analysis |
| Which customer has purchased the most tracks? | Customers · Purchases · Loyalty |
| What was the highest sales month in 2021? | Time · Sales · Seasonal |
| Which employee has the highest sales through their customers? | Employees · Customers · Relationships |

## Implementation Code

### Setting Up the Environment

```python
# Install required packages (run in a terminal):
#   pip install langchain langchain-community langchain-openai langgraph sqlalchemy requests

# Import necessary modules
import os
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.prompt import SQL_PREFIX
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.prebuilt import create_react_agent

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### Database Connection

```python
# Connect to an SQLite Database
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

# For an existing SQLite database
db_path = "chinook.db"  # Path to your SQLite database
engine = create_engine(f"sqlite:///{db_path}")
db = SQLDatabase(engine)

# Alternatively, for in-memory testing with example data
def get_in_memory_db():
    import sqlite3
    import requests

    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)

    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

engine = get_in_memory_db()
db = SQLDatabase(engine)
```

### Creating the LangChain Agent

```python
# Initialize Language Model
llm = ChatOpenAI(model="gpt-4o", temperature=0)  # You can also use gpt-4o-mini for lower cost

# Initialize SQL Database Toolkit
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Get the tools from the toolkit
tools = toolkit.get_tools()

# Build the system prompt from LangChain's standard SQL-agent prompt
# (the same text as the "langchain-ai/sql-agent-system-prompt" prompt on LangChain Hub)
system_message = SQL_PREFIX.format(dialect="SQLite", top_k=5)

# Create the agent
agent_executor = create_react_agent(
    llm,
    tools,
    prompt=system_message
)
```

### Django Integration (views.py)

```python
# views.py for Django integration
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def chat_with_db(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')

            # Process the user message with our agent
            response = agent_executor.invoke(
                {"messages": [("user", user_message)]}
            )

            # Extract the final AI message
            ai_message = response["messages"][-1].content

            return JsonResponse({
                'success': True,
                'message': ai_message
            })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })

    return JsonResponse({
        'success': False,
        'error': 'Only POST requests are supported'
    })
```

### Frontend JavaScript

```javascript
// JavaScript for the chat interface
document.addEventListener('DOMContentLoaded', function() {
    const messageInput = document.querySelector('.chat-input input');
    const sendButton = document.querySelector('.chat-input button');
    const messagesContainer = document.querySelector('.chat-messages');
    const sampleQueries = document.querySelectorAll('.sample-query');

    // Function to add a message to the chat
    function addMessage(message, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = isUser ? 'user-message' : 'bot-message';
        messageDiv.innerHTML = message;
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Function to send a message to the backend
    async function sendMessage(message) {
        addMessage(message, true);

        // Show loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'bot-message';
        loadingDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Thinking...';
        messagesContainer.appendChild(loadingDiv);

        try {
            const response = await fetch('/chat_with_db/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message }),
            });

            const data = await response.json();

            // Remove loading indicator
            messagesContainer.removeChild(loadingDiv);

            if (data.success) {
                addMessage(data.message);
            } else {
                addMessage('Error: ' + data.error);
            }
        } catch (error) {
            // Remove loading indicator
            messagesContainer.removeChild(loadingDiv);
            addMessage('An error occurred while connecting to the server.');
        }
    }

    // Event listener for send button
    sendButton.addEventListener('click', function() {
        const message = messageInput.value.trim();
        if (message) {
            sendMessage(message);
            messageInput.value = '';
        }
    });

    // Event listener for enter key
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            const message = messageInput.value.trim();
            if (message) {
                sendMessage(message);
                messageInput.value = '';
            }
        }
    });

    // Event listeners for sample queries
    sampleQueries.forEach(query => {
        query.addEventListener('click', function() {
            const questionText = this.querySelector('p').textContent;
            messageInput.value = questionText;
            messageInput.focus();
        });
    });
});
```

## Closing note

Always implement proper permissions and validation when connecting LLMs to databases.

*Built with LangChain, Python, and modern web technologies.*

## Key concepts worth simulating

- **The six-step process flow as an animated pipeline**: question → `ListSQLDatabaseTool` / `InfoSQLDatabaseTool` (schema) → LLM writes SQL → `QuerySQLCheckerTool` → `QuerySQLDatabaseTool` executes → LLM interprets. Show the message list growing with each tool call and tool result.
- **Schema exploration by the agent**: an ER diagram of Chinook (Customer–Invoice–InvoiceLine–Track–Album–Artist, plus Genre and Employee) where the tables the agent "touches" for a given question light up; let students pick one of the six sample queries and watch the join path highlight.
- **Natural language → SQL side by side**: type a question, see the generated SQL, then a live result table; toggle to show how `top_k`/`LIMIT` and the dialect string in the system prompt change the query.
- **The ReAct loop inside the SQL agent**: agent node ↔ tools node cycle; show the model deciding to call another tool after a checker error and retry.
- **Security sandbox**: a simulation of what happens with a read-only connection vs a read-write one when the model generates `DROP TABLE` or an `UPDATE`; demonstrate why narrow permissions and query validation matter.
- **Request/response across tiers**: browser JS `fetch('/chat_with_db/')` → Django view → `agent_executor.invoke` → JSON back → DOM update, including the "Thinking..." placeholder lifecycle.

## Test code

```python
# 1. Load Chinook into an in-memory SQLite DB and inspect it with SQLDatabase (no LLM needed)
# pip install langchain-community sqlalchemy requests
import sqlite3, requests
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langchain_community.utilities import SQLDatabase

url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
conn = sqlite3.connect(":memory:", check_same_thread=False)
conn.executescript(requests.get(url).text)
engine = create_engine("sqlite://", creator=lambda: conn, poolclass=StaticPool,
                       connect_args={"check_same_thread": False})
db = SQLDatabase(engine)

print(db.dialect)
print(db.get_usable_table_names())
print(db.get_table_info(["Customer", "Invoice"])[:800])
```

```python
# 2. Run the Process Flow SQL by hand and confirm the "USA, 523.06" result
sql = """
SELECT c.Country, SUM(i.Total) AS TotalSpent
FROM Customer c
JOIN Invoice i ON c.CustomerId = i.CustomerId
GROUP BY c.Country
ORDER BY TotalSpent DESC
LIMIT 1;
"""
print(db.run(sql))
```

```python
# 3. Look at the four tools the SQLDatabaseToolkit exposes
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
for t in toolkit.get_tools():
    print(f"{t.name}: {t.description[:90]}...")
```

```python
# 4. Ask the SQL agent a question and watch every intermediate step
from langchain_community.agent_toolkits.sql.prompt import SQL_PREFIX
from langgraph.prebuilt import create_react_agent

system_message = SQL_PREFIX.format(dialect="SQLite", top_k=5)
agent_executor = create_react_agent(llm, toolkit.get_tools(), prompt=system_message)

for step in agent_executor.stream(
    {"messages": [("user", "Who are the top 3 best selling artists?")]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```
