#!/usr/bin/env python3
"""Build the static course site from content/*.md.

Usage:  python3 build/build.py
Output: index.html, lessons/*.html (served by GitHub Pages from the repo root).
"""
import html, json, re, sys
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / 'content'
OUT_LESSONS = ROOT / 'lessons'
OUT_LESSONS.mkdir(exist_ok=True)

SITE_TITLE = 'Computing Systems for DS & AI'
COURSE_CODE = '606363'
INSTRUCTOR = 'Dr. Zaid G. Momani'
ORIGINAL_AUTHOR = 'Dr. Abdulkarim Albanna'
ORIGINAL_SITE = 'https://albanna-tutorials.com/cs/'

# ---------------------------------------------------------------- course map
# Each week: title, ILO, topics, milestone, lesson content files (in order), simulations.
WEEKS = [
  dict(n=1, slug='week01-transformers', title='Transformer Architecture', ilo='K1',
       topics='Transformer architecture, encoder-decoder framework, Hugging Face ecosystem',
       milestone='Form teams, create the GitHub repo, set up the project and API keys, write a Transformer analysis',
       files=['week01-transformer.md', 'week01b-transformer-step-by-step.md', 'week01c-transformer-qa.md'],
       sims=['attention', 'matrix-multiply', 'vectors', 'tokenizer', 'positional']),
  dict(n=2, slug='week02-text-classification', title='Text Classification', ilo='K1, P1',
       topics='Text classification, sentiment analysis, feature extraction vs. fine-tuning, evaluation metrics',
       milestone='Classification Agent that labels sources by type and relevance; evaluate on 20+ sources',
       files=['week02-text-classification.md'], sims=['classifier', 'tokenizer', 'vectors']),
  dict(n=3, slug='week03-ner', title='Named Entity Recognition (NER)', ilo='P1, T1', soon=True,
       topics='Token classification, BIO tagging, multilingual NER, per-entity evaluation',
       milestone='NER Agent extracting people, organisations, technologies, dates and locations; test on 5+ articles',
       files=[], sims=['bio-tagging', 'classifier']),
  dict(n=4, slug='week04-langchain', title='LangChain Fundamentals', ilo='K1, P1',
       topics='LangChain chains, prompts, memory, loaders, splitters, vector stores, RAG',
       milestone='Prompt templates for each agent, document loaders, ChromaDB store, basic RAG chain',
       files=['week04-langchain.md', 'week04b-chat-with-databases.md'], sims=['rag', 'vectors']),
  dict(n=5, slug='week05-langgraph', title='LangGraph', ilo='I1, P2',
       topics='LangGraph: nodes, edges, state, routing, checkpointing, human-in-the-loop',
       milestone='StateGraph orchestrator with conditional routing, state checkpointing and 3+ connected agents',
       files=['week05-langgraph.md'], sims=['langgraph', 'multi-agent']),
  dict(n=6, slug='week06-llms', title='Large Language Models (LLMs)', ilo='K1, K2',
       topics='LLM evolution, model families, open vs. closed models, prompting, Ollama',
       milestone='Configure LLM parameters, benchmark 2+ models, track token usage and cost',
       files=['week06-llms.md'], sims=['sampling', 'tokenizer', 'attention']),
  dict(n=7, slug='week07-moe', title='Mixture of Experts (MoE)', ilo='K1, T1',
       topics='MoE architecture, routing, Mixtral, DeepSeek-MoE, dense vs. MoE trade-offs',
       milestone='MoE comparison report, optional Mixtral integration, model selection logic. Checkpoint 1: demo the NLP pipeline',
       files=['week07-moe.md'], sims=['moe', 'sampling']),
  dict(n=8, slug='week08-midterm', title='Midterm Exam', ilo='—', exam=True,
       topics='Covers Weeks 1–7: transformers, classification, NER, LangChain, LangGraph, LLMs, MoE',
       milestone='—', files=[], sims=['attention', 'classifier', 'rag', 'langgraph', 'moe']),
  dict(n=9, slug='week09-react-agents', title='AI Agents & ReAct', ilo='K2, P2',
       topics='AI agents, ReAct loop, tool use, short- and long-term memory',
       milestone='Research Agent using ReAct with search and scraping tools, relevance scoring and ChromaDB storage; 10+ sources per topic',
       files=['week09-react-agents.md'], sims=['react-loop', 'rag']),
  dict(n=10, slug='week10-multi-agent', title='Multi-Agent Systems', ilo='K2, I1, P2',
       topics='Agentic design patterns, workflow patterns, CrewAI/AutoGen, A2A protocol',
       milestone='Full pipeline: Research → Classification → NER → Analyzer → Writer → Critic, with a human checkpoint before writing',
       files=['week10-multi-agent.md', 'week10b-a2a.md'], sims=['multi-agent', 'langgraph', 'react-loop']),
  dict(n=11, slug='week11-mcp', title='Model Context Protocol (MCP)', ilo='K2, P2',
       topics='MCP architecture, resources, tools, prompts, building MCP servers',
       milestone='Expose agent tools as MCP servers (at least 2), add error handling and retries. Checkpoint 2: demo with peer feedback',
       files=['week11-mcp.md'], sims=['mcp', 'react-loop']),
  dict(n=12, slug='week12-lams', title='Large Action Models (LAMs)', ilo='K2, P2, T1', soon=True,
       topics='LAMs, function calling, GUI understanding, browser automation, safety',
       milestone='Browser Agent using Playwright or Selenium with URL allowlists, timeouts and no login or payment actions; extract data from 3+ sites',
       files=[], sims=['lam', 'react-loop']),
  dict(n=13, slug='week13-diffusion', title='Diffusion Models', ilo='K1, P1',
       topics='Diffusion models, denoising architectures, Stable Diffusion, DALL-E',
       milestone='Illustration Agent that generates figures for reports, plus a Streamlit UI with PDF download',
       files=['week13-diffusion.md'], sims=['diffusion']),
  dict(n=14, slug='week14-project', title='Project Development & Final Testing', ilo='—', meta=True,
       topics='Final development and testing',
       milestone='Test on 5 topics, benchmark speed/cost/quality, write docs, record a 5-minute demo, prepare slides',
       files=[], sims=['multi-agent']),
  dict(n=15, slug='week15-competition', title='Competition Day & Presentations', ilo='—', meta=True,
       topics='Competition day',
       milestone='Teams solve 3 unseen topics (30 min each), present for 10 minutes and peer-review two other teams',
       files=[], sims=['multi-agent']),
  dict(n=16, slug='week16-final', title='Final Exam', ilo='—', exam=True,
       topics='Covers all material, emphasising agents, multi-agent systems, MCP, LAMs and diffusion',
       milestone='—', files=[], sims=['react-loop', 'multi-agent', 'mcp', 'lam', 'diffusion']),
]
EXTRA_PAGES = [
  dict(slug='syllabus', title='Course Syllabus', files=['syllabus.md'], sims=[]),
  dict(slug='project', title='Final Project: Multi-Agent Research Assistant', files=['project.md'], sims=['multi-agent', 'langgraph', 'react-loop']),
]

# Simulation catalogue (mirrors the JS registry; used for cards and the index page)
SIMS = {
  'matrix-multiply': ('Matrix multiplication, step by step', 'Linear algebra', 'Row of A × column of B, one cell at a time, with the running sum.'),
  'vectors': ('Vectors, dot product & cosine similarity', 'Linear algebra', 'Drag two vectors; see u+v, u·v, θ and cos θ — the geometry behind embeddings.'),
  'attention': ('Self-attention: Q·Kᵀ → softmax → ·V', 'Transformers', 'Five stages of scaled dot-product attention on a 4-token sentence, with heatmaps.'),
  'tokenizer': ('Subword tokenisation', 'Transformers', 'Type text, see WordPiece-style tokens, IDs and the embedding lookup.'),
  'positional': ('Positional encoding', 'Transformers', 'Sinusoid heatmap and per-dimension waves; why attention needs position.'),
  'classifier': ('Decision boundary, threshold & F1', 'Classification', 'Train a linear classifier live over 2-D embeddings; slide the threshold.'),
  'bio-tagging': ('NER: BIO tags & entity-level F1', 'NER', 'Tag a sentence token by token; see why NER is scored per entity.'),
  'rag': ('RAG pipeline', 'LangChain', 'Load → split → embed → store → query → top-k → grounded prompt, in 2-D embedding space.'),
  'langgraph': ('LangGraph StateGraph', 'LangGraph', 'State flowing through nodes, conditional edges, a loop, an interrupt and checkpoints.'),
  'sampling': ('Temperature, top-k, top-p', 'LLMs', 'Probability bars that sharpen or flatten; truncate and sample the next token.'),
  'moe': ('Mixture of Experts router', 'MoE', 'Gate softmax picks top-k experts per token; watch load and active-parameter share.'),
  'react-loop': ('ReAct agent loop', 'Agents', 'Thought → Action → Observation on a growing scratchpad, including a tool error.'),
  'multi-agent': ('Multi-agent research pipeline', 'Multi-agent', 'The course project: six agents, a human checkpoint and a Writer–Critic loop.'),
  'mcp': ('MCP message flow', 'MCP', 'Sequence diagram of host, client, server and JSON-RPC messages for one tool call.'),
  'lam': ('Browser agent: perceive → plan → act', 'LAMs', 'A mock browsing task with allowlist and no-payment gates.'),
  'diffusion': ('Diffusion: noising & denoising', 'Diffusion', 'Scrub t under a β schedule, read ᾱ_t and SNR, then denoise step by step.'),
}
SIM_FILES = ['matrix-multiply', 'vectors', 'attention', 'tokenizer', 'positional', 'classifier', 'bio-tagging', 'rag', 'langgraph', 'sampling', 'moe', 'react-loop', 'multi-agent', 'mcp', 'lam', 'diffusion']

# ---------------------------------------------------------------- markdown
MD = markdown.Markdown(extensions=[
  'pymdownx.superfences', 'pymdownx.arithmatex', 'tables', 'toc', 'attr_list', 'md_in_html', 'sane_lists', 'pymdownx.tilde'
], extension_configs={
  'pymdownx.arithmatex': {'generic': True, 'inline_syntax': ['round'], 'block_syntax': ['dollar', 'square']},
  'toc': {'permalink': False, 'toc_depth': '2-3'},
})

def parse(path: Path):
    text = path.read_text(encoding='utf-8')
    meta = {}
    m = re.match(r'^---\n(.*?)\n---\n', text, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ':' in line:
                k, v = line.split(':', 1); meta[k.strip()] = v.strip()
        text = text[m.end():]
    # split off "Key concepts worth simulating" and "Test code"
    def cut(section):
        nonlocal text
        mm = re.search(r'\n## ' + section + r'[^\n]*\n', text)
        if not mm: return ''
        start = mm.start()
        nxt = re.search(r'\n## ', text[mm.end():])
        end = mm.end() + nxt.start() if nxt else len(text)
        chunk = text[mm.end():end]
        text = text[:start] + text[end:]
        return chunk
    concepts = cut('Key concepts worth simulating')
    testcode = cut('Test code')
    # drop the top-level H1 (we render our own header)
    text = re.sub(r'^\s*# [^\n]*\n', '', text, count=1)
    return meta, text, concepts, testcode

def snippets_from(testcode: str):
    """Return [(title, code, runnable)] from fenced python blocks; a preceding line/heading is the title."""
    out = []
    for m in re.finditer(r'(?:###?\s*([^\n]*)\n|\*\*([^*\n]+)\*\*[^\n]*\n|^([^\n`#][^\n]{0,90})\n)?\s*```(?:python|py)\n(.*?)```', testcode, re.S | re.M):
        title = (m.group(1) or m.group(2) or m.group(3) or '').strip().rstrip(':')
        code = m.group(4).rstrip('\n')
        first = code.lstrip().split('\n', 1)[0]
        if (not title or title.lower().startswith('snippet')) and first.startswith('#'):
            title = re.sub(r'^#+\s*(\d+[.)]\s*)?', '', first).strip()
        title = title or f'Snippet {len(out) + 1}'
        runnable = not re.search(r'^\s*(import|from)\s+(torch|transformers|langchain|langgraph|langchain_\w+|datasets|sentence_transformers|chromadb|openai|anthropic|mcp|requests|sklearn|diffusers|ollama|dotenv|playwright|streamlit|evaluate|seqeval|tavily|httpx|a2a)', code, re.M) and not re.search(r'os\.environ|API_KEY|input\(', code)
        out.append((title[:80], code, runnable))
    return out[:6]

# ---------------------------------------------------------------- templates
def head(title, depth=0):
    base = '../' if depth else ''
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · {SITE_TITLE}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<link rel="stylesheet" href="{base}assets/css/site.css">
</head>
<body>'''

def topbar(depth, prev=None, nxt=None):
    base = '../' if depth else ''
    nav = ''
    if prev: nav += f'<a class="btn-icon" href="{prev[0]}" title="{html.escape(prev[1])}">← Wk {prev[2]}</a>'
    if nxt: nav += f'<a class="btn-icon" href="{nxt[0]}" title="{html.escape(nxt[1])}">Wk {nxt[2]} →</a>'
    return f'''<header class="topbar">
  <button class="btn-icon" id="menu-toggle" aria-label="Menu">☰</button>
  <a class="brand" href="{base}index.html">{COURSE_CODE} · {SITE_TITLE}<small>University of Petra</small></a>
  <span class="spacer"></span>
  <nav class="week-nav">{nav}</nav>
  <a class="btn-icon" href="{base}index.html#curriculum">Curriculum</a>
  <a class="btn-icon" href="{base}index.html#simulations">Simulations</a>
  <button class="btn-icon" id="theme-toggle" aria-label="Toggle dark mode">◐</button>
</header>'''

def sidebar(active_slug, depth):
    base = '../' if depth else ''
    items = ''
    for w in WEEKS:
        cls = 'active' if w['slug'] == active_slug else ''
        if w.get('soon'): cls += ' soon'
        items += f'<li><a class="{cls.strip()}" href="{base}lessons/{w["slug"]}.html"><span class="wk">{w["n"]:02d}</span><span>{html.escape(w["title"])}</span></a></li>'
    extra = ''.join(f'<li><a class="{"active" if p["slug"] == active_slug else ""}" href="{base}lessons/{p["slug"]}.html"><span class="wk">·</span><span>{html.escape(p["title"].split(":")[0])}</span></a></li>' for p in EXTRA_PAGES)
    return f'''<aside class="sidebar">
  <h4>Course</h4><ul class="outline">{extra}</ul>
  <h4>16-week curriculum</h4><ul class="outline">{items}</ul>
  <h4>On this page</h4><ul class="toc" id="toc"></ul>
</aside>'''

def sim_card(sid):
    t, kind, desc = SIMS[sid]
    return f'''<button class="sim-card" data-sim="{sid}"><span class="sim-kind">{html.escape(kind)} · {sim_icon(sid)}</span><span class="sim-title">{html.escape(t)}</span><span class="sim-desc">{html.escape(desc)}</span></button>'''

def sim_icon(sid):
    return {'matrix-multiply': 'animation', 'vectors': 'interactive', 'attention': 'animation', 'tokenizer': 'interactive', 'positional': 'interactive', 'classifier': 'simulation', 'bio-tagging': 'animation', 'rag': 'animation', 'langgraph': 'animation', 'sampling': 'simulation', 'moe': 'simulation', 'react-loop': 'animation', 'multi-agent': 'animation', 'mcp': 'animation', 'lam': 'animation', 'diffusion': 'simulation'}[sid]

def rail(sims, snippets):
    cards = ''.join(sim_card(s) for s in sims) or '<p class="rail-intro">No simulation for this page.</p>'
    sn = ''
    for i, (title, code, runnable) in enumerate(snippets):
        run = '<button class="btn-xs run-btn">Run ▶</button>' if runnable else '<span class="btn-xs" style="cursor:default;opacity:.6" title="Needs a local Python environment with the course libraries">local</span>'
        sn += f'''<div class="snippet"><div class="snippet-head"><span>{html.escape(title)}</span><span class="spacer"></span>{run}<button class="btn-xs copy-snip">Copy</button></div><pre><code class="language-python">{html.escape(code)}</code></pre><div class="out"></div></div>'''
    if not sn: sn = '<p class="rail-intro">No test code for this page.</p>'
    return f'''<aside class="rail">
  <div class="rail-tabs"><button class="active" data-panel="panel-sims">Simulations</button><button data-panel="panel-code">Test code</button></div>
  <div class="rail-panel active" id="panel-sims"><p class="rail-intro">Interactive teaching aids for this week. Each opens full-size with step controls and teaching notes.</p>{cards}</div>
  <div class="rail-panel" id="panel-code"><p class="rail-intro"><b>Run ▶</b> executes pure-Python/NumPy snippets in your browser (Pyodide). Snippets marked <i>local</i> need the course environment (PyTorch, Transformers, LangChain…) — copy them into VS Code.</p>{sn}</div>
</aside>'''

def overlay():
    return '''<div class="sim-overlay" id="sim-overlay" role="dialog" aria-modal="true">
  <div class="sim-window"><div class="sim-head"><div><div class="sim-kind" id="sim-kind"></div><h3 id="sim-title"></h3></div><span class="spacer"></span><button class="btn-xs" id="sim-close">✕ Close (Esc)</button></div><div class="sim-body" id="sim-body"></div></div>
</div>'''

def scripts(depth):
    base = '../' if depth else ''
    sims = ''.join(f'<script src="{base}assets/js/sims/{s}.js"></script>' for s in SIM_FILES)
    return f'''{overlay()}
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
<script>document.addEventListener('DOMContentLoaded',()=>{{if(window.hljs)hljs.highlightAll();if(window.renderMathInElement)renderMathInElement(document.body,{{delimiters:[{{left:'\\\\[',right:'\\\\]',display:true}},{{left:'\\\\(',right:'\\\\)',display:false}}],throwOnError:false}});}});</script>
<script src="{base}assets/js/sims/core.js"></script>
{sims}
<script src="{base}assets/js/site.js"></script>
</body></html>'''

def footer():
    return f'''<footer style="margin-top:3rem;padding-top:1rem;border-top:1px solid var(--border);font-size:.84rem;color:var(--muted)">
Course materials by <a href="{ORIGINAL_SITE}">{ORIGINAL_AUTHOR}</a> (University of Petra), adapted for the section taught by {INSTRUCTOR}. Interactive simulations and test code added for this site. Source on <a href="https://github.com/zmomani/ComputingSystemsDS-AI">GitHub</a>.
</footer>'''

# ---------------------------------------------------------------- pages
def render_body(files):
    parts, snippets = [], []
    for f in files:
        p = CONTENT / f
        if not p.exists():
            print('missing', p, file=sys.stderr); continue
        meta, text, concepts, testcode = parse(p)
        MD.reset()
        body = MD.convert(text)
        if len(files) > 1:
            parts.append(f'<h2 id="part-{slugify(meta.get("title", f))}">{html.escape(meta.get("title", f))}</h2><p class="credit">Source page: <a href="{meta.get("source", "")}">{meta.get("source", "")}</a></p>')
        parts.append(body)
        snippets += snippets_from(testcode)
    return '\n'.join(parts), snippets

def slugify(s): return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')

def lesson_page(w, prev, nxt):
    body, snippets = render_body(w['files'])
    sims_html = '<div class="callout sim"><div class="sim-strip"><b style="margin-right:4px">Simulations for this week:</b>' + ''.join(f'<button class="chip" data-sim="{s}">▶ {html.escape(SIMS[s][0])}</button>' for s in w['sims']) + '</div></div>' if w['sims'] else ''
    if not w['files']:
        note = 'This lesson is marked <b>Coming Soon</b> on the original course site. Until it is published, use the topics, milestone and simulations below.' if w.get('soon') else ('Exam week. Use the simulations below to revise the mechanisms covered so far.' if w.get('exam') else 'Project week. The milestone and the pipeline simulation below are the reference for what is due.')
        body = f'<div class="callout warn">{note}</div><h2 id="topics">Topics</h2><p>{html.escape(w["topics"])}</p><h2 id="milestone">Project milestone</h2><p>{html.escape(w["milestone"])}</p>'
    content = f'''<main class="main">
  <div class="lesson-head"><div class="eyebrow">Week {w["n"]} · ILO {html.escape(w["ilo"])}</div><h1>{html.escape(w["title"])}</h1>
  <p class="credit"><b>Topics:</b> {html.escape(w["topics"])}<br><b>Project milestone:</b> {html.escape(w["milestone"])}</p></div>
  {sims_html}
  {body}
  <div class="pager">{f'<a href="{prev[0]}"><small>Previous</small>Week {prev[2]}: {html.escape(prev[1])}</a>' if prev else '<span></span>'}{f'<a class="next" href="{nxt[0]}"><small>Next</small>Week {nxt[2]}: {html.escape(nxt[1])}</a>' if nxt else '<span></span>'}</div>
  {footer()}
</main>'''
    return head(f'Week {w["n"]}: {w["title"]}', 1) + topbar(1, prev, nxt) + '<div class="layout">' + sidebar(w['slug'], 1) + content + rail(w['sims'], snippets) + '</div>' + scripts(1)

def extra_page(p):
    body, snippets = render_body(p['files'])
    sims_html = '<div class="callout sim"><div class="sim-strip"><b style="margin-right:4px">Related simulations:</b>' + ''.join(f'<button class="chip" data-sim="{s}">▶ {html.escape(SIMS[s][0])}</button>' for s in p['sims']) + '</div></div>' if p['sims'] else ''
    content = f'<main class="main"><div class="lesson-head"><div class="eyebrow">{COURSE_CODE}</div><h1>{html.escape(p["title"])}</h1></div>{sims_html}{body}{footer()}</main>'
    return head(p['title'], 1) + topbar(1) + '<div class="layout">' + sidebar(p['slug'], 1) + content + rail(p['sims'], snippets) + '</div>' + scripts(1)

def index_page():
    weeks = ''
    for w in WEEKS:
        pill = '<span class="pill">coming soon</span>' if w.get('soon') else ('<span class="pill">exam</span>' if w.get('exam') else ('<span class="pill">project</span>' if w.get('meta') else '<span class="pill">lesson</span>'))
        sims = ' '.join(f'<a class="pill sim" href="lessons/{w["slug"]}.html#sim={s}">{html.escape(SIMS[s][1])}</a>' for s in w['sims'][:3])
        weeks += f'<li><span class="wk">Week {w["n"]}</span><div><a class="wt" href="lessons/{w["slug"]}.html">{html.escape(w["title"])}</a><div class="wd">{html.escape(w["topics"])}</div><div class="wm"><b>Milestone:</b> {html.escape(w["milestone"])}</div><div style="margin-top:4px">{sims}</div></div>{pill}</li>'
    simcards = ''.join(f'<div class="card"><div class="sim-kind" style="font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:var(--accent-2);font-weight:700">{html.escape(SIMS[s][1])}</div><h3 style="font-size:1.05rem">{html.escape(SIMS[s][0])}</h3><p style="font-size:.9rem;color:var(--muted);margin:0 0 .6rem">{html.escape(SIMS[s][2])}</p><button class="btn secondary" data-sim="{s}">Open</button></div>' for s in SIM_FILES)
    quiz = [
      ('What mechanism lets transformers process sequences effectively?', ['Recurrence', 'Self-attention', 'Convolution', 'Pooling'], 1, 'Self-attention lets every token look at every other token in parallel — try the attention simulation.'),
      ('In ReAct, what does the cycle consist of?', ['Input, Process, Output', 'Thought, Action, Observation', 'Encode, Decode, Generate', 'Query, Key, Value'], 1, 'Reasoning traces interleaved with tool actions and their observations — see the ReAct loop simulation.'),
      ('What is the main advantage of MoE models?', ['Smaller model size', 'Better training data', 'Sparse activation reduces compute while maintaining performance', 'Simpler architecture'], 2, 'Only the top-k experts run per token — see the MoE router simulation.'),
    ]
    quiz_html = ''.join(f'<div class="q" data-answer="{a}"><p><b>{i + 1}. {html.escape(q)}</b></p>' + ''.join(f'<span class="opt" data-v="{j}">{chr(65 + j)}) {html.escape(o)}</span>' for j, o in enumerate(opts)) + f'<p class="fb" style="display:none;font-size:.9rem;color:var(--muted);margin-top:6px">{html.escape(fb)}</p></div>' for i, (q, opts, a, fb) in enumerate(quiz))
    content = f'''<main class="main" style="max-width:1000px">
  <section class="hero"><div class="eyebrow" style="font-size:.78rem;text-transform:uppercase;letter-spacing:.1em;color:var(--accent-2);font-weight:700">{COURSE_CODE} · Faculty of Information Technology · University of Petra</div>
  <h1>Computing Systems for Data Science and Artificial Intelligence</h1>
  <p class="sub">A 16-week, project-driven course from transformers to agents: LLMs, Mixture of Experts, ReAct, multi-agent systems, MCP, Large Action Models and diffusion — built around a multi-agent research assistant that competes live in Week 15.</p>
  <p class="credit">Section taught by <b>{INSTRUCTOR}</b>. Course materials by <a href="{ORIGINAL_SITE}">{ORIGINAL_AUTHOR}</a>; this site adds interactive simulations, animations and runnable test code for each week.</p>
  <p><a class="btn" href="lessons/week01-transformers.html">Start with Week 1 →</a> <a class="btn secondary" href="lessons/syllabus.html">Syllabus</a> <a class="btn secondary" href="lessons/project.html">Final project</a></p></section>

  <h2 id="overview">Course overview</h2>
  <div class="grid-2">
    <div class="card"><h3>What you'll learn</h3><p>Transformers, LLMs, Mixture of Experts, AI agents and ReAct, multi-agent systems, Model Context Protocol (MCP), Large Action Models (LAMs) and diffusion models.</p></div>
    <div class="card"><h3>What you'll build</h3><p>NER systems, RAG pipelines, LangGraph agents, multi-agent workflows, LAM applications and generative-AI projects, culminating in a multi-agent research assistant.</p></div>
    <div class="card"><h3>Technologies</h3><p>Python 3.12+, PyTorch, Hugging Face, LangChain &amp; LangGraph, OpenAI API, Ollama, Playwright, VS Code.</p></div>
    <div class="card"><h3>Who should take it</h3><p>University of Petra students in Data Science and AI degrees interested in LLMs and agents. Basic Python is recommended; see the syllabus for prerequisites.</p></div>
  </div>

  <h2 id="curriculum">16-week curriculum</h2>
  <p style="color:var(--muted)">Each week lists the topics, the ILO, the project milestone and the simulations available in that week's sidebar.</p>
  <ul class="week-list">{weeks}</ul>

  <h2 id="simulations">Simulations &amp; animations</h2>
  <p style="color:var(--muted)">Sixteen interactive teaching aids, in the style of textbook figures and Khan-Academy-type explorables: step controls, live computation, legends and teaching notes. Each is also linked from its week.</p>
  <div class="grid-2">{simcards}</div>

  <h2 id="quiz">Test your knowledge</h2>
  <div class="quiz card">{quiz_html}</div>

  <h2 id="projects">Project portfolio</h2>
  <div class="grid-2">
    <div class="card"><h3>Arabic QA System <small style="color:var(--muted);font-weight:400">· Week 6</small></h3><p>Question answering over Arabic documents with RAG and LangChain.</p></div>
    <div class="card"><h3>Research Assistant <small style="color:var(--muted);font-weight:400">· Week 10</small></h3><p>A multi-agent assistant that searches, summarises and synthesises information.</p></div>
    <div class="card"><h3>Smart Tutor <small style="color:var(--muted);font-weight:400">· Week 9</small></h3><p>A tutoring agent that adapts to students and gives personalised feedback.</p></div>
    <div class="card"><h3>Document Analyzer <small style="color:var(--muted);font-weight:400">· Week 4</small></h3><p>Document analysis combining RAG with a vector database.</p></div>
    <div class="card"><h3>Code Review Agent <small style="color:var(--muted);font-weight:400">· Week 9</small></h3><p>A ReAct-based agent that reviews code, suggests improvements and explains issues.</p></div>
    <div class="card"><h3>Browser Automation Agent <small style="color:var(--muted);font-weight:400">· Week 12</small></h3><p>A Large Action Model that automates web tasks through a browser.</p></div>
  </div>

  <h2 id="assessment">Assessment &amp; grading</h2>
  <div class="grid-2">
    <div class="card"><h3>Weights</h3><ul><li>Assignments — 10%</li><li>Project (rubric-based) — 20%</li><li>Midterm (Week 8, covers Weeks 1–7) — 30%</li><li>Final (Week 16, all material; emphasis on agents, ReAct, multi-agent, MCP, LAMs, diffusion) — 40%</li></ul></div>
    <div class="card"><h3>Weekly workload</h3><p>3 hours on campus (theory, demos, hands-on) plus 4–6 hours of self-study: 7–9 hours per week. Teams of 2–3; every week has a milestone, so falling behind early is the main failure mode.</p></div>
  </div>

  <h2 id="resources">Resources</h2>
  <div class="grid-2">
    <div class="card"><h3>Required software</h3><p>Python 3.12+, PyTorch, Hugging Face Transformers, LangChain &amp; LangGraph, Ollama, VS Code. Keep API keys in a <code>.env</code> file and never commit them.</p></div>
    <div class="card"><h3>Hardware</h3><p><b>Minimum:</b> 8 GB RAM, i5, 50 GB storage, webcam.<br><b>Recommended:</b> 16 GB RAM, i7/Ryzen 7, dedicated GPU, 100 GB SSD.</p></div>
    <div class="card" style="grid-column:1/-1"><h3>References &amp; reading</h3><ol>
      <li>Tunstall, von Werra &amp; Wolf, <i>Natural Language Processing with Transformers</i>, O'Reilly, 2022.</li>
      <li>Yao et al., <i>ReAct: Synergizing Reasoning and Acting in Language Models</i>, ICLR 2023.</li>
      <li>Chip Huyen, <i>AI Engineering</i>, O'Reilly, 2025.</li>
      <li>LangChain &amp; LangGraph official documentation.</li>
      <li>Hugging Face blog, <i>Mixture of Experts Explained</i>.</li>
      <li>Maarten Grootendorst, <i>A Visual Guide to LLM Agents</i>, 2025.</li>
      <li>Vaswani et al., <i>Attention Is All You Need</i>, NeurIPS 2017.</li>
      <li>Calvin Luo, <i>Understanding Diffusion Models: A Unified Perspective</i>, 2022.</li></ol></div>
  </div>
  {footer()}
</main>'''
    return head('Home') + topbar(0) + '<div class="layout" style="grid-template-columns:var(--sidebar-w) minmax(0,1fr)">' + sidebar('', 0) + content + '</div>' + scripts(0)

def main():
    nav = [(f'{w["slug"]}.html', w['title'], w['n']) for w in WEEKS]
    for i, w in enumerate(WEEKS):
        prev = nav[i - 1] if i > 0 else None
        nxt = nav[i + 1] if i < len(WEEKS) - 1 else None
        (OUT_LESSONS / f'{w["slug"]}.html').write_text(lesson_page(w, prev, nxt), encoding='utf-8')
    for p in EXTRA_PAGES:
        (OUT_LESSONS / f'{p["slug"]}.html').write_text(extra_page(p), encoding='utf-8')
    (ROOT / 'index.html').write_text(index_page(), encoding='utf-8')
    (ROOT / '.nojekyll').write_text('')
    print(f'built {len(WEEKS) + len(EXTRA_PAGES)} lesson pages + index')

if __name__ == '__main__':
    main()
