---
week: 9
title: AI Agents & ReAct — Building a Reasoning and Acting Agent from Scratch
source: https://albanna-tutorials.com/react-agent.html
---
# AI Agents & ReAct — Building a Reasoning and Acting Agent from Scratch

> Based on Dr. Abdulkarim Albanna's lesson "ReAct Agent Architecture — A visual guide to building a Reasoning and Acting agent from scratch" (albanna-tutorials.com). Code blocks 1–6 (environment_setup.py, agent.py, react_prompt.py, tools.py, agent_loop.py, complex_example.py) are reproduced verbatim. The final combined listing (react_agent.py) could not be fetched verbatim and is reassembled below from the verbatim parts plus the `__main__` entry point the page describes. Explanatory prose was extracted via a fetcher that paraphrases, so wording is close to but not identical to the original.

## 1. Introduction to ReAct Agents

The **ReAct (Reasoning + Acting)** pattern lets a language model solve complex tasks by alternating between *reasoning* about the problem and *acting* with tools. The model breaks a problem down, takes an action, observes the result, and uses that observation to guide its next step.

The cycle is:

```
Thought → Action → Observation → Thought → … → Answer
```

> **Key Insight:** separating reasoning from tool use lets a language model handle problems that neither capability could solve alone.

## 2. Setting Up the Environment

Import the libraries, load the API key from a `.env` file, create an OpenAI client, and send a test message to confirm the connection works.

```python
# environment_setup.py
import openai
import re
import httpx
import os
from dotenv import load_dotenv

_ = load_dotenv()
from openai import OpenAI

client = OpenAI()

# Test the OpenAI client
chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello world"}]
)

print(chat_completion.choices[0].message.content)
```

## 3. Creating the Basic Agent Class

The `Agent` class keeps the **conversation history** and an optional system message. The constructor sets up that state; `__call__` appends the user's message, queries the model and stores the reply; `execute` sends the full history to the API.

> **Design Pattern:** the agent is *stateful* — it keeps the full conversation history internally, so each new call builds on everything that came before.

```python
# agent.py
class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
                        model="gpt-4o",
                        temperature=0,
                        messages=self.messages)
        return completion.choices[0].message.content
```

## 4. Defining the ReAct Prompt

The system prompt teaches the model the **Thought, Action, PAUSE, Observation** loop, lists the available tools (a calculator and a dog-breed weight lookup), and includes a worked example session.

> **Important:** the `PAUSE` keyword tells the model to stop generating after it requests an action, so the program can run the tool and feed the result back as an `Observation`.

```python
# react_prompt.py
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()
```

## 5. Implementing the Tools

Two tools: a **calculator** that uses Python's `eval()`, and a **breed-weight lookup** with fixed sample values. A dictionary (`known_actions`) maps each action name to its function. The tools are deliberately simple; real agents would connect to APIs, databases or other external systems.

```python
# tools.py
def calculate(what):
    return eval(what)

def average_dog_weight(name):
    if name in "Scottish Terrier":
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")

known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}
```

> Teaching note: `eval()` on model output is unsafe outside a demo, and `name in "Scottish Terrier"` is a *substring* test (so `"Terrier"` matches but `"scottish terrier"` in lower case does not). Both are good discussion points for the Extensions section.

## 6. Creating the Agent Loop

The `query` function starts an agent with the ReAct prompt and sends it the question. It uses a **regular expression** to find any requested `Action:` line, runs the matching tool, and sends the result back as an `Observation:`. The loop ends when the model produces no further action.

> **Safety Feature:** the `max_turns` limit stops the loop from running indefinitely.

```python
# agent_loop.py
action_re = re.compile('^Action: (\w+): (.*)$')   # python regular expression to select action

def query(question, max_turns=5):
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [
            action_re.match(a)
            for a in result.split('\n')
            if action_re.match(a)
        ]
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return
```

Agent loop diagram:

```
User Question → Agent Reasoning → Action Extraction (regex) → Tool Execution → Return Observation
                     ↑                                                              |
                     └──────────────────────────────────────────────────────────────┘
```

## 7. Complete Example

A multi-step question about the combined weight of two dogs. The agent looks up each breed, adds the weights, and returns the total — showing how a multi-step problem is split into smaller steps.

```python
# complex_example.py
question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""

query(question)
# The agent will:
# 1. Look up the weight of a border collie
# 2. Look up the weight of a scottish terrier
# 3. Calculate the sum of the weights
# 4. Provide the final answer
```

Example session trace:

```
Question:    I have 2 dogs, a border collie and a scottish terrier. What is their combined weight?
Thought:     I need to find the weight of a border collie and a scottish terrier, then add them together.
Action:      average_dog_weight: Border Collie
PAUSE
Observation: a Border Collies average weight is 37 lbs
Thought:     Now I need to find the weight of a scottish terrier.
Action:      average_dog_weight: Scottish Terrier
PAUSE
Observation: Scottish Terriers average 20 lbs
Thought:     Now I can calculate the total weight by adding the two weights together: 37 + 20 = 57 lbs.
Action:      calculate: 37 + 20
PAUSE
Observation: 57
Answer:      The combined weight of your border collie and scottish terrier is 57 lbs.
```

## 8. Complete Code Architecture

All the pieces combined into a single script with a main entry point.

```python
# react_agent.py
import re
from dotenv import load_dotenv
from openai import OpenAI

_ = load_dotenv()
client = OpenAI()


class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
                        model="gpt-4o",
                        temperature=0,
                        messages=self.messages)
        return completion.choices[0].message.content


prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()


def calculate(what):
    return eval(what)

def average_dog_weight(name):
    if name in "Scottish Terrier":
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")

known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}

action_re = re.compile('^Action: (\w+): (.*)$')   # python regular expression to select action

def query(question, max_turns=5):
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [
            action_re.match(a)
            for a in result.split('\n')
            if action_re.match(a)
        ]
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return


if __name__ == "__main__":
    question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
    query(question)
```

## 9. Extensions

- **More tools:** web search, API calls, database queries.
- **Robust error handling** for tool failures and invalid input.
- **Memory:** persistent memory across conversations.
- **Prompt refinement** to improve reasoning and tool use.
- **Smarter tool selection** logic for choosing which tool fits a given context.
- **Multiple actions per turn** for better efficiency.

> **Advanced Implementation:** for more sophisticated agents, a framework such as **LangChain** offers many of these extensions ready-made.

## 10. Key Takeaways

1. Pairing **reasoning with tool use** gives a strong problem-solving pattern.
2. The **iterative loop** breaks complex problems into smaller steps.
3. A well-written **system prompt** strongly shapes model behaviour.
4. **External tools** greatly expand what a language model can do.
5. **Regular expressions** offer a simple way to parse actions from model output.
6. The **Agent class** preserves conversation state across the interaction.

## Key Terms

- **ReAct (Reasoning + Acting):** a framework where an LLM alternates between reasoning about a task and taking actions with tools.
- **Thought:** the agent's reasoning about the current state and next step.
- **Action:** a tool call the agent selects and emits, followed by `PAUSE`.
- **Observation:** the result of running an action, fed back to the agent.
- **PAUSE:** the keyword telling the model to stop generating so the system can run the tool.
- **Answer:** the final output, produced when no further action is needed.
- **known_actions:** dictionary mapping action names to Python functions.
- **max_turns:** iteration cap that prevents endless loops.

## Key concepts worth simulating

- **ReAct loop: Thought → Action → PAUSE → Observation cycle** — step-through animation of the dog-weight example where each click advances one phase, highlighting which party (LLM vs. Python harness) produced the text.
- **Message-history growth in the stateful Agent** — visualize `self.messages` as a stack that grows by two entries (user/observation + assistant) per turn; show token count climbing and why `max_turns` matters.
- **Regex action extraction** — live regex tester for `^Action: (\w+): (.*)$`: students type model output and see which lines match and the captured `(action, action_input)` groups; try malformed lines to see the loop terminate.
- **Dispatch table (`known_actions`)** — show the string from the regex being used as a dictionary key to select and call a Python function; show the `Unknown action` exception path.
- **Termination conditions** — a flowchart with two exits: "no Action line → return (Answer)" and "i == max_turns → stop"; simulate a model that never answers.
- **Temperature = 0 determinism** — compare repeated runs at T = 0 vs. T = 1 and show divergence in the Thought/Action text.
- **Tool-call safety** — contrast `eval("37 + 20")` with `eval("__import__('os').system('ls')")` and a safe arithmetic parser (`ast.literal_eval` / a restricted evaluator).
- **Substring-match bug in `average_dog_weight`** — show `"Terrier" in "Scottish Terrier"` is True but `"scottish terrier" in "Scottish Terrier"` is False; motivate a dictionary lookup.

## Test code

```python
# 1. Regex action extraction, no API key needed
import re
action_re = re.compile(r'^Action: (\w+): (.*)$')
sample = """Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE"""
actions = [action_re.match(a) for a in sample.split('\n') if action_re.match(a)]
print(actions[0].groups())            # ('average_dog_weight', 'Bulldog')
print(action_re.match("Answer: 57 lbs"))   # None → loop would terminate
```

```python
# 2. Tools + dispatch table, exercised directly
def calculate(what):
    return eval(what)

def average_dog_weight(name):
    if name in "Scottish Terrier":
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")

known_actions = {"calculate": calculate, "average_dog_weight": average_dog_weight}
for action, arg in [("average_dog_weight", "Border Collie"),
                    ("average_dog_weight", "Scottish Terrier"),
                    ("calculate", "37 + 20"),
                    ("average_dog_weight", "scottish terrier")]:   # note the lowercase bug
    print(f"{action}({arg!r}) -> {known_actions[action](arg)}")
```

```python
# 3. Run the full ReAct loop offline with a scripted fake LLM (same Agent/query code, no OpenAI call)
import re
scripted = iter([
    "Thought: I need both weights.\nAction: average_dog_weight: Border Collie\nPAUSE",
    "Thought: Now the terrier.\nAction: average_dog_weight: Scottish Terrier\nPAUSE",
    "Thought: Add them.\nAction: calculate: 37 + 20\nPAUSE",
    "Answer: The combined weight is 57 lbs.",
])

class Agent:
    def __init__(self, system=""):
        self.messages = [{"role": "system", "content": system}] if system else []
    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = next(scripted)                     # stand-in for client.chat.completions.create
        self.messages.append({"role": "assistant", "content": result})
        return result

known_actions = {"calculate": lambda w: eval(w),
                 "average_dog_weight": lambda n: {"Border Collie": "a Border Collies average weight is 37 lbs",
                                                  "Scottish Terrier": "Scottish Terriers average 20 lbs"}.get(n, "An average dog weights 50 lbs")}
action_re = re.compile(r'^Action: (\w+): (.*)$')

def query(question, max_turns=5):
    bot, next_prompt = Agent("ReAct prompt"), question
    for i in range(max_turns):
        result = bot(next_prompt); print(result)
        actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
        if not actions:
            return len(bot.messages)
        action, arg = actions[0].groups()
        obs = known_actions[action](arg); print(" -- running", action, arg); print("Observation:", obs)
        next_prompt = f"Observation: {obs}"

print("messages in history:", query("I have 2 dogs, a border collie and a scottish terrier. What is their combined weight"))
```

```python
# 4. Live run against OpenAI (needs OPENAI_API_KEY in .env): the lesson's own entry point
#    pip install openai python-dotenv
from dotenv import load_dotenv; _ = load_dotenv()
from openai import OpenAI
client = OpenAI()
# paste the Agent class, prompt, tools, known_actions and query() from section 8 here, then:
# query("I have 2 dogs, a border collie and a scottish terrier. What is their combined weight")
```
