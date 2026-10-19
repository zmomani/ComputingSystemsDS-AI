# Browser-runnable companion examples, one group per week.
# Format: lines "#@ week N | Title" start a new example; everything until the next marker is its code.
# Every example is self-contained and uses only the standard library, NumPy, pandas, scikit-learn or matplotlib,
# so it runs in the in-browser Python (Pyodide). build/build.py adds them to each week's "Try it yourself".

#@ week 1 | Scaled dot-product attention for a 4-token sentence (NumPy)
import numpy as np

tokens = ["The", "cat", "sat", "down"]
Q = np.array([[1, 0, 1], [0, 2, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
K = np.array([[1, 0, 0], [0, 2, 1], [0, 1, 1], [1, 0, 1]], dtype=float)
V = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)

d_k = Q.shape[1]
scores = Q @ K.T / np.sqrt(d_k)                      # (4, 4)
weights = np.exp(scores - scores.max(axis=1, keepdims=True))
weights /= weights.sum(axis=1, keepdims=True)         # softmax per row
output = weights @ V                                  # (4, 2)

np.set_printoptions(precision=2, suppress=True)
print("attention weights (rows = queries):")
for t, row in zip(tokens, weights):
    print(f"  {t:>5}", row)
print("\noutput vectors:\n", output)
print("\nEach row of weights sums to 1:", weights.sum(axis=1))

#@ week 1 | Sinusoidal positional encoding
import numpy as np

def positional_encoding(n_positions, d_model):
    pos = np.arange(n_positions)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    return np.where(i % 2 == 0, np.sin(angle), np.cos(angle))

pe = positional_encoding(6, 8)
np.set_printoptions(precision=3, suppress=True)
print(pe)
# Nearby positions get similar vectors; the dot product falls off with distance
print("\nsimilarity of position 0 with positions 0..5:", pe @ pe[0])

#@ week 1 | Matrix multiplication by hand vs NumPy
import numpy as np

A = np.array([[-1, 4, 2], [4, -2, 3]])      # 2 x 3
B = np.array([[0, -2], [-2, 4], [4, 1]])    # 3 x 2

C = np.zeros((A.shape[0], B.shape[1]), dtype=int)
for i in range(A.shape[0]):
    for j in range(B.shape[1]):
        terms = [A[i, k] * B[k, j] for k in range(A.shape[1])]
        C[i, j] = sum(terms)
        print(f"c[{i+1},{j+1}] = " + " + ".join(f"({A[i,k]})({B[k,j]})" for k in range(A.shape[1])) + f" = {C[i, j]}")

print("\nby hand:\n", C)
print("A @ B:\n", A @ B)
print("same result:", np.array_equal(C, A @ B))

#@ week 2 | Sentiment classifier: TF-IDF features + logistic regression (scikit-learn)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

train_texts = [
    "I love this course", "What a wonderful lecture", "The project was fun and useful",
    "Great explanation, very clear", "I enjoyed the lab today", "This tutorial is excellent",
    "I hate waiting for the model to train", "The lecture was boring", "This bug is terrible",
    "I am disappointed with the results", "The exam was awful", "Nothing works, very frustrating",
]
train_labels = ["pos"] * 6 + ["neg"] * 6
test_texts = ["The lab was clear and fun", "The results are terrible", "I love the project", "Boring and frustrating exam"]
test_labels = ["pos", "neg", "pos", "neg"]

vec = TfidfVectorizer()
X_train = vec.fit_transform(train_texts)
clf = LogisticRegression().fit(X_train, train_labels)

pred = clf.predict(vec.transform(test_texts))
for t, p in zip(test_texts, pred):
    print(f"{p:>3}  <- {t}")
print()
print(classification_report(test_labels, pred, zero_division=0))

#@ week 2 | Precision, recall and F1 by hand vs scikit-learn
from sklearn.metrics import precision_score, recall_score, f1_score

y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
y_pred = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]

tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print(f"TP={tp} FP={fp} FN={fn}")
print(f"by hand:      precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
print(f"scikit-learn: precision={precision_score(y_true, y_pred):.3f} "
      f"recall={recall_score(y_true, y_pred):.3f} F1={f1_score(y_true, y_pred):.3f}")

#@ week 3 | From BIO tags to entity spans, and entity-level F1
tokens = ["Dr.", "Albanna", "teaches", "at", "the", "University", "of", "Petra", "in", "Amman"]
gold   = ["B-PER", "I-PER", "O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "O", "B-LOC"]
pred   = ["B-PER", "I-PER", "O", "O", "O", "B-ORG", "I-ORG", "B-LOC", "O", "B-LOC"]

def spans(tags):
    out, cur = [], None
    for i, t in enumerate(tags):
        if t.startswith("B-"):
            if cur: out.append(cur)
            cur = [t[2:], i, i]
        elif t.startswith("I-") and cur and cur[0] == t[2:]:
            cur[2] = i
        else:
            if cur: out.append(cur)
            cur = None
    if cur: out.append(cur)
    return {(typ, s, e) for typ, s, e in out}

G, P = spans(gold), spans(pred)
for typ, s, e in sorted(G | P, key=lambda x: x[1]):
    mark = "correct" if (typ, s, e) in G and (typ, s, e) in P else ("missed" if (typ, s, e) in G else "spurious")
    print(f"{typ:4} {' '.join(tokens[s:e+1]):28} {mark}")

tp = len(G & P)
precision, recall = tp / len(P), tp / len(G)
f1 = 2 * precision * recall / (precision + recall)
token_acc = sum(g == p for g, p in zip(gold, pred)) / len(gold)
print(f"\ntoken accuracy = {token_acc:.2f}, entity P={precision:.2f} R={recall:.2f} F1={f1:.2f}")

#@ week 4 | Retrieval for RAG: embed chunks, rank by cosine similarity, build the prompt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

chunks = [
    "Transformers use self-attention to relate all tokens in a sequence.",
    "BERT is an encoder-only model pre-trained with masked language modelling.",
    "GPT models are decoder-only and generate text left to right.",
    "Amman is the capital of Jordan and its largest city.",
    "Petra is a historical city in southern Jordan carved into rock.",
    "A vector store returns the chunks whose embeddings are nearest to the query.",
]
question = "How do GPT and BERT models differ?"
k = 2

# TF-IDF stands in for an embedding model here: it matches shared words, whereas a real
# embedding model (e.g. all-MiniLM-L6-v2) also matches meaning, such as "capital" ~ "largest city".
vec = TfidfVectorizer().fit(chunks + [question])
sims = cosine_similarity(vec.transform([question]), vec.transform(chunks))[0]
top = sims.argsort()[::-1][:k]

print("ranking:")
for i in sims.argsort()[::-1]:
    print(f"  {sims[i]:.3f}  {chunks[i]}")

context = "\n".join(f"- {chunks[i]}" for i in top)
prompt = f"Answer using only the context below.\n\nContext:\n{context}\n\nQuestion: {question}"
print("\nprompt sent to the LLM:\n" + prompt)

#@ week 4 | A tiny "chain": prompt template -> model -> output parser (pure Python)
def prompt_template(topic):
    return f"List three key facts about {topic}, one per line."

def fake_llm(prompt):                 # stands in for ChatOpenAI / Ollama
    facts = {"MCP": ["Uses JSON-RPC 2.0", "Servers expose tools, resources and prompts", "Transports: stdio and Streamable HTTP"]}
    topic = prompt.split("about ")[1].split(",")[0]
    return "\n".join(facts.get(topic, ["(no facts)"]))

def output_parser(text):
    return [line.strip() for line in text.splitlines() if line.strip()]

def chain(topic):                     # LCEL equivalent: prompt | llm | parser
    return output_parser(fake_llm(prompt_template(topic)))

print(chain("MCP"))

#@ week 5 | A state graph with a conditional edge and a loop (pure Python, LangGraph-style)
from typing import TypedDict

class State(TypedDict):
    topic: str
    sources: list
    score: float
    draft: str

def research(state):
    found = ["blog A", "paper B", "docs C", "paper D"][: len(state["sources"]) + 2]
    return {"sources": found}

def grade(state):
    return {"score": round(0.25 * len(state["sources"]), 2)}

def write(state):
    return {"draft": f"Report on {state['topic']} using {len(state['sources'])} sources"}

def route(state):                     # conditional edge: returns the next node's name
    return "write" if state["score"] >= 0.75 else "research"

nodes = {"research": research, "grade": grade, "write": write}
edges = {"research": "grade", "write": "END"}

state = {"topic": "MoE routing", "sources": [], "score": 0.0, "draft": ""}
node, step = "research", 0
while node != "END":
    update = nodes[node](state)
    state = {**state, **update}        # partial update merged into the shared state
    step += 1
    print(f"step {step}: {node:8} -> {update}")
    node = route(state) if node == "grade" else edges[node]
print("\nfinal state:", state)

#@ week 6 | Temperature, top-k and top-p sampling (NumPy)
import numpy as np

words = ["sunny", "cold", "nice", "rainy", "perfect", "Tuesday", "banana"]
logits = np.array([3.2, 2.6, 2.4, 2.1, 1.5, -0.8, -2.5])
rng = np.random.default_rng(0)

def next_token_probs(logits, temperature=1.0, top_k=None, top_p=None):
    z = logits / temperature
    p = np.exp(z - z.max()); p /= p.sum()
    keep = np.ones_like(p, dtype=bool)
    order = np.argsort(p)[::-1]
    if top_k is not None:
        keep[order[top_k:]] = False
    if top_p is not None:
        cum = np.cumsum(p[order])
        cutoff = np.searchsorted(cum, top_p) + 1          # smallest set with mass >= top_p
        keep[order[cutoff:]] = False
    p = np.where(keep, p, 0); return p / p.sum()

for T in (0.2, 1.0, 2.0):
    p = next_token_probs(logits, T)
    print(f"T={T}: " + ", ".join(f"{w}={pi:.2f}" for w, pi in zip(words, p)))

p = next_token_probs(logits, 1.0, top_p=0.9)
print("\ntop-p=0.9 keeps:", [w for w, pi in zip(words, p) if pi > 0])
print("10 samples:", [str(w) for w in rng.choice(words, size=10, p=p)])

#@ week 7 | A Mixture-of-Experts layer with top-2 routing (NumPy)
import numpy as np

rng = np.random.default_rng(42)
d_model, d_ff, n_experts, top_k, n_tokens = 8, 16, 4, 2, 6

W_gate = rng.normal(0, 0.5, (d_model, n_experts))
experts = [(rng.normal(0, 0.3, (d_model, d_ff)), rng.normal(0, 0.3, (d_ff, d_model))) for _ in range(n_experts)]
x = rng.normal(size=(n_tokens, d_model))

logits = x @ W_gate
probs = np.exp(logits - logits.max(axis=1, keepdims=True)); probs /= probs.sum(axis=1, keepdims=True)
chosen = np.argsort(probs, axis=1)[:, ::-1][:, :top_k]

y = np.zeros_like(x)
load = np.zeros(n_experts, dtype=int)
for t in range(n_tokens):
    g = probs[t, chosen[t]]; g = g / g.sum()               # renormalise over the chosen experts
    for gate, e in zip(g, chosen[t]):
        W1, W2 = experts[e]
        y[t] += gate * (np.maximum(0, x[t] @ W1) @ W2)      # FFN expert with ReLU
        load[e] += 1
    print(f"token {t}: experts {chosen[t].tolist()} with weights {np.round(g, 2).tolist()}")

params_per_expert = d_model * d_ff + d_ff * d_model
print("\nload per expert:", load)
print(f"total expert parameters: {n_experts * params_per_expert}, used per token: {top_k * params_per_expert} "
      f"({top_k / n_experts:.0%})")

#@ week 9 | A ReAct loop with a scripted model and real tools
import ast, operator, re

def calculate(expr):
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops: return ops[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr, mode="eval").body)

def lookup(term):
    facts = {"amman population": "about 4.06 million (2024 estimate)"}
    return facts.get(term.lower(), "no result")

tools = {"calculate": calculate, "lookup": lookup}

script = iter([                       # what an LLM would generate, one turn at a time
    "Thought: I need the population first.\nAction: lookup: Amman population",
    "Thought: Now compute 17% of 4.06 million.\nAction: calculate: 0.17 * 4060000",
    "Thought: I have the answer.\nAnswer: about 690,200 people",
])
action_re = re.compile(r"^Action: (\w+): (.*)$")

for turn in range(1, 6):
    reply = next(script)
    print(reply)
    if reply.splitlines()[-1].startswith("Answer:"):
        break
    m = next(action_re.match(l) for l in reply.splitlines() if action_re.match(l))
    tool, arg = m.groups()
    print(f"Observation: {tools[tool](arg)}\n")

#@ week 10 | Writer-Critic loop: the evaluator-optimizer pattern (pure Python)
def writer(topic, notes=None):
    draft = f"{topic} replaces the feed-forward layer with several experts."
    if notes:
        draft += " A router picks the top-2 experts per token (Shazeer et al., 2017)."
    return draft

def critic(draft):
    score = 5 + 3 * ("(" in draft) + 2 * ("router" in draft)
    notes = [] if score >= 8 else ["add how routing works", "cite a source"]
    return score, notes

topic, notes = "Mixture of Experts", None
for round_ in range(1, 4):
    draft = writer(topic, notes)
    score, notes = critic(draft)
    print(f"round {round_}: score {score}/10  notes={notes}\n  {draft}")
    if score >= 8:
        print("accepted")
        break

#@ week 11 | What an MCP server does with JSON-RPC messages (pure Python)
import json

TOOLS = {"add": {"description": "Add two numbers",
                 "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}}}

def handle(message):
    req = json.loads(message)
    if req["method"] == "initialize":
        result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "demo", "version": "1.0"}}
    elif req["method"] == "tools/list":
        result = {"tools": [{"name": n, **spec} for n, spec in TOOLS.items()]}
    elif req["method"] == "tools/call":
        args = req["params"]["arguments"]
        result = {"content": [{"type": "text", "text": str(args["a"] + args["b"])}], "isError": False}
    else:
        return json.dumps({"jsonrpc": "2.0", "id": req["id"], "error": {"code": -32601, "message": "Method not found"}})
    return json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result})

for i, (method, params) in enumerate([("initialize", {}), ("tools/list", {}),
                                       ("tools/call", {"name": "add", "arguments": {"a": 2, "b": 40}}), ("ping/nope", {})], 1):
    msg = json.dumps({"jsonrpc": "2.0", "id": i, "method": method, "params": params})
    print("->", msg)
    print("<-", handle(msg), "\n")

#@ week 12 | A browser agent's safety gate: allowlist and forbidden actions (pure Python)
from urllib.parse import urlparse

ALLOWED_DOMAINS = {"uop.edu.jo", "en.wikipedia.org"}
FORBIDDEN_WORDS = ("pay", "checkout", "login", "password", "buy")

def gate(action):
    host = urlparse(action.get("url", "https://uop.edu.jo")).hostname or ""
    if not any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS):
        return False, f"domain {host} is not on the allowlist"
    label = action.get("label", "").lower()
    if any(w in label for w in FORBIDDEN_WORDS):
        return False, f"'{action['label']}' looks like a login or payment action"
    return True, "allowed"

plan = [
    {"type": "click", "label": "Faculty of IT", "url": "https://uop.edu.jo/fit"},
    {"type": "click", "label": "Data Science & AI", "url": "https://uop.edu.jo/fit/ds-ai"},
    {"type": "click", "label": "Pay application fee", "url": "https://pay.example-bank.com/"},
    {"type": "extract", "label": "Contact email", "url": "https://uop.edu.jo/fit/ds-ai"},
]
for step in plan:
    ok, why = gate(step)
    print(f"{'EXECUTE' if ok else 'BLOCK  '}  {step['type']:7} {step['label']:22} ({why})")

#@ week 13 | Forward diffusion: noising an image in closed form (NumPy + matplotlib)
import numpy as np
import matplotlib.pyplot as plt

T = 1000
betas = np.linspace(1e-4, 0.02, T)                 # linear schedule (Ho et al., 2020)
alpha_bar = np.cumprod(1 - betas)

yy, xx = np.mgrid[-1:1:64j, -1:1:64j]
x0 = np.where(xx**2 + yy**2 < 0.5, 1.0, -1.0)      # a white disk on black, values in [-1, 1]
rng = np.random.default_rng(0)
eps = rng.standard_normal(x0.shape)

steps = [0, 50, 150, 300, 600, 999]
fig, axes = plt.subplots(1, len(steps), figsize=(12, 2.4))
for ax, t in zip(axes, steps):
    xt = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps
    ax.imshow(xt, cmap="gray", vmin=-2.5, vmax=2.5); ax.axis("off")
    ax.set_title(f"t={t}\nsignal {np.sqrt(alpha_bar[t]):.2f}")
plt.tight_layout()
plt.show()
print("alpha_bar at t = 0, 250, 500, 999:", np.round(alpha_bar[[0, 250, 500, 999]], 4))
