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


# ---------------------------------------------------------------- runnability
# Modules that run in the browser: the Python standard library plus Pyodide packages loaded on demand.
BROWSER_MODULES = set(getattr(sys, 'stdlib_module_names', ())) | {
    'numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'networkx', 'sympy', 'regex', 'PIL', 'statsmodels'}
NO_BROWSER = {'requests', 'urllib3', 'httpx', 'socket', 'subprocess', 'multiprocessing', 'asyncio_subprocess', 'webbrowser'}

def imported_modules(code):
    mods = set()
    for m in re.finditer(r'^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w., ]+))', code, re.M):
        if m.group(1): mods.add(m.group(1).split('.')[0])
        else:
            for part in m.group(2).split(','):
                part = part.strip().split(' ')[0]
                if part: mods.add(part.split('.')[0])
    return mods

LOCAL_NAMES = re.compile(r'\b(torch|nn|F|transformers|langchain\w*|langgraph|ChatOpenAI|OpenAI|AutoModel\w*|AutoTokenizer|pipeline|Trainer|tf|keras|ollama|chromadb)\.')

def runs_in_browser(code):
    if LOCAL_NAMES.search(code): return False      # uses a local-only library without importing it here
    mods = imported_modules(code)
    if any(m not in BROWSER_MODULES or m in NO_BROWSER for m in mods): return False
    if re.search(r'os\.environ|getenv\(|API_KEY|input\(|open\([\'"][^\'"]+\.(pdf|db|sqlite|png|jpg)', code): return False
    return True

def missing_modules(code):
    mods = {m for m in imported_modules(code) if m not in BROWSER_MODULES or m in NO_BROWSER}
    for m in LOCAL_NAMES.finditer(code):
        n = m.group(1)
        mods.add({'nn': 'torch', 'F': 'torch', 'tf': 'tensorflow', 'ChatOpenAI': 'langchain_openai', 'OpenAI': 'openai',
                  'pipeline': 'transformers', 'Trainer': 'transformers', 'AutoTokenizer': 'transformers'}.get(n, 'transformers' if n.startswith('AutoModel') else n))
    return sorted(mods)

# ---------------------------------------------------------------- markdown
MD = markdown.Markdown(extensions=[
  'pymdownx.arithmatex', 'tables', 'toc', 'attr_list', 'md_in_html', 'sane_lists', 'pymdownx.tilde'
], extension_configs={
  'pymdownx.arithmatex': {'generic': True, 'inline_syntax': ['round'], 'block_syntax': ['dollar', 'square']},
  'toc': {'permalink': False, 'toc_depth': '2-3'},
})

FENCE = re.compile(r'^([ \t]*)(```+|~~~+)[ \t]*([\w+#-]*)[^\n]*\n(.*?)^\1\2[ \t]*$', re.S | re.M)
LANG_ALIAS = {'py': 'python', 'sh': 'bash', 'shell': 'bash', 'console': 'bash', 'js': 'javascript', 'yml': 'yaml', 'text': 'plaintext', 'txt': 'plaintext', '': 'plaintext'}

PAGE_BLOCKS = []   # python blocks of the page being built: (code, title)

def code_block_html(code, lang, title=None):
    lang = LANG_ALIAS.get(lang.lower(), lang.lower())
    esc = html.escape(code)
    pre = f'<pre><code class="language-{lang}">{esc}</code></pre>'
    head = f'<div class="code-title">{html.escape(title)}</div>' if title else ''
    if lang != 'python' or re.search(r'^\s*[!%]', code, re.M):
        return f'<div class="codeblock">{head}<button class="copy">Copy</button>{pre}</div>'
    PAGE_BLOCKS.append((code, title))
    return f'PYBLOCK{len(PAGE_BLOCKS) - 1}XEND'

CACHE_FILE = ROOT / 'build' / 'runnable-cache.json'
try: CACHE = json.loads(CACHE_FILE.read_text())
except Exception: CACHE = {}

def classify_page():
    import hashlib, subprocess
    payload = [{'code': c, 'static_ok': runs_in_browser(c)} for c, _ in PAGE_BLOCKS]
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if key in CACHE: return CACHE[key]
    try:
        r = subprocess.run([sys.executable, '-I', str(ROOT / 'build' / 'classify.py')], input=json.dumps(payload),
                           capture_output=True, text=True, timeout=900, cwd=str(ROOT / 'build'))
        res = json.loads(r.stdout)
    except Exception as e:
        print('  (classification unavailable, using import check only:', str(e)[:80], ')', file=sys.stderr)
        return [{'run': p['static_ok'], 'setup': '', 'reason': ''} for p in payload]
    CACHE[key] = res
    return res

REPO = 'zmomani/ComputingSystemsDS-AI'
NB_DIR = ROOT / 'notebooks'
PIP_NAMES = {'dotenv': 'python-dotenv', 'sentence_transformers': 'sentence-transformers', 'a2a': 'a2a-sdk', 'tavily': 'tavily-python',
             'langchain_openai': 'langchain-openai', 'langchain_core': 'langchain-core', 'langchain_community': 'langchain-community',
             'langchain_ollama': 'langchain-ollama', 'langchain_chroma': 'langchain-chroma', 'langchain_text_splitters': 'langchain-text-splitters',
             'langchain_huggingface': 'langchain-huggingface', 'langchain_anthropic': 'langchain-anthropic', 'langchain_tavily': 'langchain-tavily',
             'langgraph': 'langgraph', 'mcp': 'mcp', 'chromadb': 'chromadb', 'datasets': 'datasets', 'evaluate': 'evaluate', 'seqeval': 'seqeval',
             'gradio': 'gradio', 'diffusers': 'diffusers', 'openai': 'openai', 'langchain': 'langchain', 'ollama': 'ollama', 'playwright': 'playwright',
             'accelerate': 'accelerate', 'bitsandbytes': 'bitsandbytes', 'umap': 'umap-learn', 'streamlit': 'streamlit', 'faiss': 'faiss-cpu'}
COLAB_PREINSTALLED = {'torch', 'torchvision', 'transformers', 'numpy', 'pandas', 'sklearn', 'matplotlib', 'scipy', 'requests', 'httpx', 'PIL', 'tqdm', 'huggingface_hub', 'seaborn', 'tensorflow', 'keras', 'IPython', 'google'}

try: NB_FRAGMENTS = json.loads((ROOT / 'build' / 'notebook-fragments.json').read_text())
except Exception: NB_FRAGMENTS = {}
SCRIPT_FILE = re.compile(r'Path\(__file__\)|\bdjango\b|INSTALLED_APPS|urlpatterns|mcp\.run\(|\bstreamlit\b|uvicorn\.run\(|app\.run\(|\bst\.(title|write|button)\(')

def colab_url(slug):
    return f'https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{slug}.ipynb'

def write_notebook(slug, title, page_html, res):
    """One notebook per page: every Python example in page order, under the heading it appears in."""
    if not PAGE_BLOCKS: return False
    md = lambda t: {'cell_type': 'markdown', 'metadata': {}, 'source': t}
    code = lambda t: {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': t}
    mods = set()
    for c, _ in PAGE_BLOCKS: mods |= imported_modules(c)
    pip = sorted({PIP_NAMES[m] for m in mods if m in PIP_NAMES and m not in COLAB_PREINSTALLED})
    cells = [md(f'# {title}\n\n606363 Computing Systems for Data Science and AI — every code example from the lesson page, in order.\n\n'
                f'Lesson page: https://zmomani.github.io/ComputingSystemsDS-AI/lessons/{slug}.html\n\n'
                '**How to use:** run the cells from the top (Runtime → Run all, or Shift+Enter cell by cell). Later examples often reuse '
                'variables from earlier ones. Examples that call OpenAI or other APIs need a key: add it with the 🔑 Secrets panel '
                'on the left, or set `os.environ["OPENAI_API_KEY"] = "..."` in a cell. Examples that use Ollama need a local Ollama server and '
                'will not work in Colab. For GPU examples choose Runtime → Change runtime type → T4 GPU.')]
    if pip: cells.append(code('# Libraries this lesson uses that are not preinstalled in Colab\n%pip install -q ' + ' '.join(pip)))
    alltext = '\n'.join(c for c, _ in PAGE_BLOCKS)
    common = []
    if re.search(r'\bmath\.', alltext): common.append('import math')
    if re.search(r'\bnp\.', alltext): common.append('import numpy as np')
    if re.search(r'\bpd\.', alltext): common.append('import pandas as pd')
    if re.search(r'\bplt\.', alltext): common.append('import matplotlib.pyplot as plt')
    if re.search(r'\b(torch|nn|F)\.', alltext): common += ['import torch', 'import torch.nn as nn', 'import torch.nn.functional as F']
    if 'asyncio.run(' in alltext or re.search(r'^await ', alltext, re.M):
        common += ['import nest_asyncio', 'nest_asyncio.apply()   # lets asyncio.run() work inside a notebook']
    if common: cells.append(code('# Imports that some of the lesson\'s code fragments assume\n' + '\n'.join(common)))
    frags = set(NB_FRAGMENTS.get(slug, []))
    import hashlib
    used_files = set()
    positions = [(m.start(), int(m.group(1))) for m in re.finditer(r'PYBLOCK(\d+)XEND', page_html)]
    last_head = None
    for pos, i in positions:
        heads = re.findall(r'<h[23][^>]*>(.*?)</h[23]>', page_html[:pos], re.S)
        head = re.sub(r'<[^>]+>', '', heads[-1]).strip() if heads else ''
        c, t = PAGE_BLOCKS[i]
        if head and head != last_head: cells.append(md(f'## {html.unescape(head)}')); last_head = head
        if t: cells.append(md(f'**{t}**'))
        if SCRIPT_FILE.search(c):
            fm = (re.search(r'[Ss]ave (?:it )?as ([\w./-]+\.py)', '\n'.join(c.split('\n')[:4]))
                  or re.search(r'([\w./-]+\.py)\b', f'{t or ""} {head} ' + c.split('\n', 1)[0]))
            fname = fm.group(1).lstrip('./') if fm else ('mcp_server.py' if 'mcp.run(' in c else f'example_{i + 1}.py')
            if fname in used_files: fname = fname.replace('.py', f'_{i + 1}.py')
            used_files.add(fname)
            note = (f'This code is a file of an application, not a notebook step. The cell saves it as `{fname}`; '
                    'run it from a terminal on your computer (or with `!python ' + fname + '` here, if it is a script).')
            cells.append(md(note))
            if '/' in fname: cells.append(code(f'!mkdir -p {fname.rsplit("/", 1)[0]}'))
            cells.append(code(f'%%writefile {fname}\n' + c))
        elif hashlib.sha256(c.encode()).hexdigest()[:16] in frags:
            cells.append(md('*Illustrative snippet — it shows the pattern explained above and is not meant to run on its own '
                            '(it uses names from a larger program).*\n\n```python\n' + c + '\n```'))
        else:
            cells.append(code(c))
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
          'language_info': {'name': 'python'}, 'colab': {'provenance': []}}, 'nbformat': 4, 'nbformat_minor': 5}
    for k, cell in enumerate(nb['cells']):
        cell['id'] = f'{slug[:12]}-{k:03d}'.replace('_', '-')
        cell['source'] = [l for l in cell['source'].splitlines(keepends=True)]
    NB_DIR.mkdir(exist_ok=True)
    (NB_DIR / f'{slug}.ipynb').write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return True

def finalize(page_html, slug=None, title=None):
    res = classify_page()
    has_nb = write_notebook(slug, title, page_html, res) if slug else False
    colab = colab_url(slug) if has_nb else ''
    page_html = page_html.replace('COLABLINKX', (f'<div class="colab-bar"><a class="btn colab" href="{colab}" target="_blank" rel="noopener">'
        '<span class="colab-logo">CO</span> Open this week\'s code in Google Colab</a><span class="muted small">Runs the PyTorch, Transformers and LangChain '
        'examples that cannot run in the browser — free, nothing to install.</span></div>') if has_nb else '')
    def block(i):
        code, title = PAGE_BLOCKS[i]; r = res[i]
        return python_block_html(code, title, r['run'], r.get('setup', ''), r.get('reason', ''))
    out = re.sub(r'<p>\s*PYBLOCK(\d+)XEND\s*</p>', lambda m: block(int(m.group(1))), page_html)
    out = re.sub(r'PYBLOCK(\d+)XEND', lambda m: block(int(m.group(1))), out)
    PAGE_BLOCKS.clear()
    return out

def python_block_html(code, title, run, setup, reason):
    esc = html.escape(code)
    pre = f'<pre><code class="language-python">{esc}</code></pre>'
    head = f'<div class="code-title">{html.escape(title)}</div>' if title else ''
    setup_ta = f'<textarea class="setup" hidden>{html.escape(setup)}</textarea>' if setup else ''
    if run:
        btn = f'<button class="try" data-run="1">▶ Try it</button>'
        return f'<div class="runnable">{head}{btn}<button class="copy">Copy</button><textarea class="src" hidden>{esc}</textarea>{setup_ta}{pre}</div>'
    miss = ', '.join(missing_modules(code)) or ('code from the earlier examples in this lesson' if 'NameError' in reason else 'an API key or local files')
    btn = (f'<span class="local-btns"><a class="try colab-try" data-colab-link="1" target="_blank" rel="noopener" title="Run it in Google Colab (opens this week\'s notebook)">▶ Run in Colab</a>'
           f'<button class="try local" data-run="0" data-needs="{html.escape(miss)}" title="Needs {html.escape(miss)} — opens in the editor">✎ Editor</button></span>')
    return f'<div class="runnable local">{head}{btn}<button class="copy">Copy</button><textarea class="src" hidden>{esc}</textarea>{pre}</div>'

def md_to_html(text):
    blocks = []
    def stash(m):
        indent, _, lang, code = m.group(1), m.group(2), m.group(3), m.group(4)
        if indent: code = '\n'.join(l[len(indent):] if l.startswith(indent) else l for l in code.split('\n'))
        blocks.append(code_block_html(code.rstrip('\n'), lang))
        return f'{indent}\n{indent}CODEBLOCK{len(blocks) - 1}XEND\n'
    text = FENCE.sub(stash, text)
    MD.reset()
    out = MD.convert(text)
    out = re.sub(r'<p>\s*CODEBLOCK(\d+)XEND\s*</p>', lambda m: blocks[int(m.group(1))], out)
    out = re.sub(r'CODEBLOCK(\d+)XEND', lambda m: blocks[int(m.group(1))], out)
    return out

def parse(path: Path):
    text = path.read_text(encoding='utf-8')
    meta = {}
    m = re.match(r'^---\n(.*?)\n---\n', text, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ':' in line:
                k, v = line.split(':', 1); meta[k.strip()] = v.strip()
        text = text[m.end():]
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
    cut('Key concepts worth simulating')
    testcode = cut('Test code')
    text = re.sub(r'^\s*# [^\n]*\n', '', text, count=1)
    first = re.search(r'\n## ', text)
    if first:
        pre, rest = text[:first.start()], text[first.start():]
        pre = '\n'.join(l for l in pre.split('\n') if not re.search(r'albanna|al-banna|ilmy|extraction note|material by|course site', l, re.I))
        pre = re.sub(r'^>\s*$', '', pre, flags=re.M)
        text = pre + rest
    return meta, text, testcode

def snippets_from(testcode: str):
    out = []
    for m in re.finditer(r'(?:###?\s*([^\n]*)\n|\*\*([^*\n]+)\*\*[^\n]*\n|^([^\n`#][^\n]{0,90})\n)?\s*```(?:python|py)\n(.*?)```', testcode, re.S | re.M):
        title = (m.group(1) or m.group(2) or m.group(3) or '').strip().rstrip(':')
        code = m.group(4).rstrip('\n')
        first = code.lstrip().split('\n', 1)[0]
        if (not title or title.lower().startswith('snippet')) and first.startswith('#'):
            title = re.sub(r'^#+\s*(\d+[.)]\s*)?', '', first).strip()
        out.append((title[:90] or f'Example {len(out) + 1}', code))
    return out[:6]

URL_MAP = {
  'transformer.html': 'week01-transformers.html', 'transformer-step-by-step.html': 'week01-transformers.html#part-transformer-complete-step-by-step-guide-with-one-example',
  'transformer-step-by-step-ar.html': 'week01-transformers.html#part-transformer-complete-step-by-step-guide-with-one-example', 'transformer-qa.html': 'week01-transformers.html#part-transformer-questions-answers',
  'textClassification.html': 'week02-text-classification.html', 'langchain.html': 'week04-langchain.html', 'database.html': 'week04-langchain.html#part-chat-with-databases',
  'langgraph.html': 'week05-langgraph.html', 'llama4.html': 'week06-llms.html', 'moe.html': 'week07-moe.html', 'react-agent.html': 'week09-react-agents.html',
  'llm_agents.html': 'week10-multi-agent.html', 'a2a.html': 'week10-multi-agent.html#part-agent-to-agent-protocols-a2a-and-mcp', 'mcp.html': 'week11-mcp.html', 'diffusion.html': 'week13-diffusion.html',
  'cs/syllabus.html': 'syllabus.html', 'cs/project_research_assistant.html': 'project.html', 'cs/': '../index.html',
}
def localise_links(body):
    return re.sub(r'https?://(?:www\.)?(?:albanna-tutorials\.com|ilmy\.me)/([A-Za-z0-9_./-]+)', lambda m: URL_MAP.get(m.group(1), m.group(0)), body)

def slugify(s): return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')

def render_body(files):
    parts, snippets = [], []
    for f in files:
        p = CONTENT / f
        if not p.exists():
            print('missing', p, file=sys.stderr); continue
        meta, text, testcode = parse(p)
        if len(files) > 1:
            parts.append(f'<h2 class="part" id="part-{slugify(meta.get("title", f))}">{html.escape(meta.get("title", f))}</h2>')
        parts.append(localise_links(md_to_html(text)))
        snippets += snippets_from(testcode)
    return '\n'.join(parts), snippets

# ---------------------------------------------------------------- page chrome
def head(title, depth=0):
    base = '../' if depth else ''
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · {SITE_TITLE}</title>
<meta name="description" content="Interactive course site for {COURSE_CODE} Computing Systems for Data Science and AI, University of Petra. Python runs in your browser; every week has simulations and animations.">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='6' fill='%231f3a5f'/><text x='16' y='22' font-size='14' font-family='monospace' fill='white' text-anchor='middle'>AI</text></svg>">
<link rel="stylesheet" href="{base}assets/vendor/codemirror/codemirror.css">
<link rel="stylesheet" href="{base}assets/vendor/codemirror/material-darker.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<link rel="stylesheet" href="{base}assets/css/site.css">
</head>
<body>'''

def lesson_select(active_slug, depth):
    base = '' if depth else 'lessons/'
    opts = '<option value="">Lessons…</option>'
    for p in EXTRA_PAGES:
        opts += f'<option value="{base}{p["slug"]}.html"{" selected" if p["slug"] == active_slug else ""}>{html.escape(p["title"].split(":")[0])}</option>'
    for w in WEEKS:
        opts += f'<option value="{base}{w["slug"]}.html"{" selected" if w["slug"] == active_slug else ""}>Week {w["n"]} · {html.escape(w["title"])}</option>'
    return f'<select id="lesson-select" aria-label="Choose lesson">{opts}</select>'

def topbar(depth, active_slug='', prev=None, nxt=None, workbench=False):
    base = '../' if depth else ''
    p = f'<a class="btn ghost" href="{prev[0]}" title="{html.escape(prev[1])}">‹ Prev</a>' if prev else '<button class="btn ghost" disabled>‹ Prev</button>'
    n = f'<a class="btn ghost" href="{nxt[0]}" title="{html.escape(nxt[1])}">Next ›</a>' if nxt else '<button class="btn ghost" disabled>Next ›</button>'
    nav = f'{p}{lesson_select(active_slug, depth)}{n}' if depth else lesson_select('', 0)
    wb = ('<button id="btn-panel" class="btn accent" title="Show or hide the editor / animation panel">⇆ Panel</button>'
          '<div class="status" id="py-status"><span class="dot"></span><span id="py-status-text">Python loads on first run</span></div>') if workbench else ''
    return f'''<header class="topbar">
  <a class="brand" href="{base}index.html">
    <span class="brand-mark">AI</span>
    <span class="brand-text"><strong>Computing Systems for DS &amp; AI</strong><span class="muted"> · {COURSE_CODE} · University of Petra</span></span>
  </a>
  <nav class="topnav">{nav}</nav>
  {wb}
</header>'''

def sim_buttons(sims, label):
    if not sims: return ''
    items = ''.join(f'<button class="sim-chip" data-display="{s}"><span class="sim-chip-kind">{html.escape(SIMS[s][1])}</span>▶ Display: {html.escape(SIMS[s][0])}</button>' for s in sims)
    return f'<div class="sim-strip"><div class="sim-strip-head">{label}</div><div class="sim-strip-items">{items}</div></div>'

def workbench(sims):
    opts = ''.join(f'<option value="{s}">{html.escape(SIMS[s][0])}</option>' for s in sims)
    others = ''.join(f'<option value="{s}">{html.escape(SIMS[s][0])}</option>' for s in SIM_FILES if s not in sims)
    return f'''<aside class="workbench" id="workbench">
  <div class="wb-tabs">
    <button class="wb-tab active" data-mode="code">⌨ Code</button>
    <button class="wb-tab" data-mode="anim">◉ Animation</button>
    <span class="spacer"></span>
    <button class="btn ghost wb-close" id="btn-close" title="Hide panel">✕</button>
  </div>
  <section class="wb-mode wb-code active" data-mode="code">
    <div class="wb-toolbar">
      <span class="wb-title" id="wb-title">Editor</span>
      <span class="spacer"></span>
      <button id="btn-run" class="btn primary" title="Run (Ctrl+Enter)">▶ Run</button>
      <button id="btn-reset" class="btn ghost" title="Restore the original code">Reset</button>
      <button id="btn-copy" class="btn ghost" title="Copy code">Copy</button>
      <button id="btn-clear" class="btn ghost" title="Clear output">Clear</button>
    </div>
    <div class="editor-wrap"><textarea id="editor"># Press "▶ Try it" on any example in the lesson,
# or write Python here and press Ctrl+Enter.
import numpy as np
print(np.round(np.linspace(0, 1, 5), 2))</textarea></div>
    <div class="output-head"><span>Output</span><span id="run-time" class="muted small"></span></div>
    <div id="output" class="output" aria-live="polite"></div>
  </section>
  <section class="wb-mode wb-anim" data-mode="anim">
    <div class="wb-toolbar light">
      <select id="anim-select" aria-label="Choose animation"><optgroup label="This week">{opts}</optgroup><optgroup label="Other weeks">{others}</optgroup></select>
      <span class="spacer"></span>
      <button id="btn-full" class="btn ghost" title="Open full screen">⤢ Full screen</button>
    </div>
    <div class="anim-host" id="anim-host"><p class="muted anim-empty">Press <b>▶ Display</b> on an animation in the lesson, or choose one above.</p></div>
  </section>
</aside>'''

def overlay():
    return '''<div class="sim-overlay" id="sim-overlay" role="dialog" aria-modal="true">
  <div class="sim-window"><div class="sim-head"><div><div class="sim-kind" id="sim-kind"></div><h3 id="sim-title"></h3></div><span class="spacer"></span><button class="btn ghost" id="sim-close">✕ Close (Esc)</button></div><div class="sim-body" id="sim-body"></div></div>
</div>'''

def scripts(depth, workbench=False):
    base = '../' if depth else ''
    sims = ''.join(f'<script src="{base}assets/js/sims/{s}.js"></script>' for s in SIM_FILES)
    cm = f'''<script src="{base}assets/vendor/codemirror/codemirror.js"></script>
<script src="{base}assets/vendor/codemirror/python.js"></script>
<script src="{base}assets/vendor/codemirror/matchbrackets.js"></script>
<script src="{base}assets/vendor/codemirror/closebrackets.js"></script>
<script>window.CS_PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";</script>''' if workbench else ''
    return f'''{overlay()}
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
{cm}
<script src="{base}assets/js/sims/core.js"></script>
{sims}
<script src="{base}assets/js/site.js"></script>
</body></html>'''

def footer():
    return f'''<footer class="site-footer">
This site is based on the course site <a href="{ORIGINAL_SITE}">albanna-tutorials.com/cs</a> by {ORIGINAL_AUTHOR}, used with the author's permission.
</footer>'''

def contents_box(body):
    heads = re.findall(r'<h2[^>]*id="([^"]+)"[^>]*>(.*?)</h2>', body)
    if len(heads) < 4: return ''
    items = ''.join(f'<li><a href="#{i}">{re.sub(r"<[^>]+>", "", t)}</a></li>' for i, t in heads)
    return f'<details class="contents"><summary>Contents ({len(heads)} sections)</summary><ol>{items}</ol></details>'

def load_browser_examples():
    src = (CONTENT / 'browser-examples.py').read_text(encoding='utf-8')
    parts = re.split(r'^#@ week (\d+) \| (.*)$', src, flags=re.M)
    ex = {}
    for i in range(1, len(parts), 3):
        ex.setdefault(int(parts[i]), []).append((parts[i + 1].strip(), parts[i + 2].strip('\n')))
    return ex
BROWSER_EXAMPLES = load_browser_examples()

def try_section(snippets, week=None):
    browser = BROWSER_EXAMPLES.get(week, [])
    if not snippets and not browser: return ''
    out = '''<h2 id="try-it-yourself">Try it yourself</h2>
<p class="lesson-intro"><b>▶ Try it</b> runs an example in the editor panel, inside your browser — change the code and press <kbd>Ctrl</kbd>+<kbd>Enter</kbd> to run it again. Examples that use PyTorch, Transformers, LangChain or an API cannot run in a browser: press <b>▶ Run in Colab</b> to run them in Google Colab (free, nothing to install), or copy them to VS Code.</p>'''
    if browser:
        out += '<h3 id="run-in-your-browser">Run in your browser</h3><p class="lesson-intro">Self-contained versions of this week\'s mechanisms, written with NumPy, scikit-learn or plain Python so they run here.</p>'
        out += ''.join(code_block_html(code, 'python', title) for title, code in browser)
    if snippets:
        out += '<h3 id="test-code">Test code for the course environment</h3>'
        out += ''.join(code_block_html(code, 'python', title) for title, code in snippets)
    return out

# ---------------------------------------------------------------- pages
def lesson_shell(title, active_slug, prev, nxt, inner, sims):
    has_nb = bool(PAGE_BLOCKS)
    inner = finalize(inner, active_slug, title)
    data = f' data-colab="{colab_url(active_slug)}"' if has_nb else ''
    return (head(title, 1) + topbar(1, active_slug, prev, nxt, workbench=True) +
            f'<main class="lesson panel-closed" id="lesson"{data}><article class="lesson-content" id="lesson-content">{inner}</article>'
            f'<div id="splitter" class="splitter" role="separator" aria-orientation="vertical" title="Drag to resize"></div>{workbench(sims)}</main>' +
            scripts(1, workbench=True))

def lesson_page(w, prev, nxt):
    body, snippets = render_body(w['files'])
    if not w['files']:
        note = ('This lesson is marked <b>Coming Soon</b> on the original course site. Until it is published, use the topics, the milestone and the animations below.' if w.get('soon')
                else 'Exam week. Use the animations below to revise the mechanisms covered so far.' if w.get('exam')
                else 'Project week. The milestone and the pipeline animation are the reference for what is due.')
        body = f'<blockquote><p>{note}</p></blockquote><h2 id="topics">Topics</h2><p>{html.escape(w["topics"])}</p><h2 id="milestone">Project milestone</h2><p>{html.escape(w["milestone"])}</p>'
    nav = f'''<div class="lesson-nav">{f'<a href="{prev[0]}">‹ Week {prev[2]}: {html.escape(prev[1])}</a>' if prev else '<a href="../index.html">‹ All lessons</a>'}{f'<a href="{nxt[0]}">Week {nxt[2]}: {html.escape(nxt[1])} ›</a>' if nxt else '<a href="../index.html">All lessons ›</a>'}</div>'''
    inner = f'''<div class="lesson-head"><div class="eyebrow">Week {w["n"]} · ILO {html.escape(w["ilo"])}</div><h1>{html.escape(w["title"])}</h1>
<p class="lesson-intro"><b>Topics:</b> {html.escape(w["topics"])}<br><b>Project milestone:</b> {html.escape(w["milestone"])}</p></div>
COLABLINKX
{sim_buttons(w["sims"], "Animations for this week — open them in the panel")}
{contents_box(body)}
{body}
{try_section(snippets, w['n'])}
{nav}
{footer()}'''
    return lesson_shell(f'Week {w["n"]}: {w["title"]}', w['slug'], prev, nxt, inner, w['sims'])

def extra_page(p):
    body, snippets = render_body(p['files'])
    inner = f'''<div class="lesson-head"><div class="eyebrow">{COURSE_CODE}</div><h1>{html.escape(p["title"])}</h1></div>
COLABLINKX
{sim_buttons(p["sims"], "Related animations")}
{contents_box(body)}{body}{try_section(snippets)}
<div class="lesson-nav"><a href="../index.html">‹ All lessons</a><span></span></div>{footer()}'''
    return lesson_shell(p['title'], p['slug'], None, None, inner, p['sims'] or ['multi-agent'])

def index_page():
    cards = ''
    for w in WEEKS:
        kind = 'coming soon' if w.get('soon') else 'exam' if w.get('exam') else 'project' if w.get('meta') else 'lesson'
        cards += f'''<li><a class="lesson-card{" dim" if w.get("soon") or w.get("exam") else ""}" href="lessons/{w["slug"]}.html">
  <span class="lesson-num">{w["n"]:02d}</span>
  <span><h3>{html.escape(w["title"])}</h3><p>{html.escape(w["topics"])}</p></span>
  <span class="lesson-meta">Week {w["n"]}<br><span class="progress-pill{" sim" if w["sims"] else ""}">{len(w["sims"])} animation{"s" if len(w["sims"]) != 1 else ""}</span><br><span class="kind">{kind}</span></span></a></li>'''
    simcards = ''.join(f'''<div class="card sim-card"><div class="sim-kind">{html.escape(SIMS[s][1])}</div><h3>{html.escape(SIMS[s][0])}</h3><p>{html.escape(SIMS[s][2])}</p><button class="btn primary" data-display="{s}">▶ Display</button></div>''' for s in SIM_FILES)
    quiz = [
      ('What mechanism lets transformers process sequences effectively?', ['Recurrence', 'Self-attention', 'Convolution', 'Pooling'], 1, 'Self-attention lets every token look at every other token in parallel — see the self-attention animation in Week 1.'),
      ('In ReAct, what does the cycle consist of?', ['Input, Process, Output', 'Thought, Action, Observation', 'Encode, Decode, Generate', 'Query, Key, Value'], 1, 'Reasoning traces interleaved with tool actions and their observations — see the ReAct loop animation in Week 9.'),
      ('What is the main advantage of MoE models?', ['Smaller model size', 'Better training data', 'Sparse activation reduces compute while maintaining performance', 'Simpler architecture'], 2, 'Only the top-k experts run for each token — see the MoE router animation in Week 7.'),
    ]
    quiz_html = ''.join(f'<div class="q" data-answer="{a}"><p><b>{i + 1}. {html.escape(q)}</b></p>' + ''.join(f'<span class="opt" data-v="{j}">{chr(65 + j)}) {html.escape(o)}</span>' for j, o in enumerate(opts)) + f'<p class="fb">{html.escape(fb)}</p></div>' for i, (q, opts, a, fb) in enumerate(quiz))
    inner = f'''<div class="home-inner">
  <div class="eyebrow">{COURSE_CODE} · Faculty of Information Technology · University of Petra</div>
  <h1>Computing Systems for Data Science and Artificial Intelligence</h1>
  <p class="lead">A 16-week, project-driven course from transformers to agents: LLMs, Mixture of Experts, ReAct, multi-agent systems, MCP, Large Action Models and diffusion — built around a multi-agent research assistant that competes live in Week 15.</p>
  <p class="lead small-lead">Each lesson has the explanation on the left and a panel on the right. <b>▶ Try it</b> runs an example in a live Python editor inside your browser — nothing to install. <b>▶ Display</b> opens an animation of the concept you are reading about. Section taught by <b>{INSTRUCTOR}</b>.</p>
  <p><a class="btn primary" href="lessons/week01-transformers.html">Start with Week 1 ›</a> <a class="btn ghost" href="lessons/syllabus.html">Syllabus</a> <a class="btn ghost" href="lessons/project.html">Final project</a></p>

  <h2 id="curriculum">16-week curriculum</h2>
  <ol class="lesson-list">{cards}</ol>

  <h2 id="simulations">Animations &amp; simulations</h2>
  <p class="muted">Step-by-step animations in the style of textbook figures and Khan-Academy explorables. Each one also appears in its week's lesson.</p>
  <div class="grid">{simcards}</div>

  <h2 id="overview">Course overview</h2>
  <div class="grid">
    <div class="card"><h3>What you'll learn</h3><p>Transformers, LLMs, Mixture of Experts, AI agents and ReAct, multi-agent systems, Model Context Protocol (MCP), Large Action Models (LAMs) and diffusion models.</p></div>
    <div class="card"><h3>What you'll build</h3><p>NER systems, RAG pipelines, LangGraph agents, multi-agent workflows, LAM applications and generative-AI projects, culminating in a multi-agent research assistant.</p></div>
    <div class="card"><h3>Technologies</h3><p>Python 3.12+, PyTorch, Hugging Face, LangChain &amp; LangGraph, OpenAI API, Ollama, Playwright, VS Code.</p></div>
    <div class="card"><h3>Who should take it</h3><p>University of Petra students in Data Science and AI degrees interested in LLMs and agents. Basic Python is recommended.</p></div>
  </div>

  <h2 id="quiz">Test your knowledge</h2>
  <div class="card quiz">{quiz_html}</div>

  <h2 id="projects">Project portfolio</h2>
  <div class="grid">
    <div class="card"><h3>Arabic QA System <span class="muted small">· Week 6</span></h3><p>Question answering over Arabic documents with RAG and LangChain.</p></div>
    <div class="card"><h3>Research Assistant <span class="muted small">· Week 10</span></h3><p>A multi-agent assistant that searches, summarises and synthesises information.</p></div>
    <div class="card"><h3>Smart Tutor <span class="muted small">· Week 9</span></h3><p>A tutoring agent that adapts to students and gives personalised feedback.</p></div>
    <div class="card"><h3>Document Analyzer <span class="muted small">· Week 4</span></h3><p>Document analysis combining RAG with a vector database.</p></div>
    <div class="card"><h3>Code Review Agent <span class="muted small">· Week 9</span></h3><p>A ReAct-based agent that reviews code, suggests improvements and explains issues.</p></div>
    <div class="card"><h3>Browser Automation Agent <span class="muted small">· Week 12</span></h3><p>A Large Action Model that automates web tasks through a browser.</p></div>
  </div>

  <h2 id="assessment">Assessment &amp; grading</h2>
  <div class="grid">
    <div class="card"><h3>Weights</h3><ul><li>Assignments — 10%</li><li>Project (rubric-based) — 20%</li><li>Midterm (Week 8, covers Weeks 1–7) — 30%</li><li>Final (Week 16, all material; emphasis on agents, ReAct, multi-agent, MCP, LAMs, diffusion) — 40%</li></ul></div>
    <div class="card"><h3>Weekly workload</h3><p>3 hours on campus (theory, demos, hands-on) plus 4–6 hours of self-study. Teams of 2–3; every week has a project milestone.</p></div>
  </div>

  <h2 id="resources">Resources</h2>
  <div class="grid">
    <div class="card"><h3>Required software</h3><p>Python 3.12+, PyTorch, Hugging Face Transformers, LangChain &amp; LangGraph, Ollama, VS Code. Keep API keys in a <code>.env</code> file and never commit them.</p></div>
    <div class="card"><h3>Hardware</h3><p><b>Minimum:</b> 8 GB RAM, i5, 50 GB storage, webcam.<br><b>Recommended:</b> 16 GB RAM, i7/Ryzen 7, dedicated GPU, 100 GB SSD.</p></div>
  </div>
  <div class="card" style="margin-top:.75rem"><h3>References &amp; reading</h3><ol>
      <li>Tunstall, von Werra &amp; Wolf, <i>Natural Language Processing with Transformers</i>, O'Reilly, 2022.</li>
      <li>Yao et al., <i>ReAct: Synergizing Reasoning and Acting in Language Models</i>, ICLR 2023.</li>
      <li>Chip Huyen, <i>AI Engineering</i>, O'Reilly, 2025.</li>
      <li>LangChain &amp; LangGraph official documentation.</li>
      <li>Hugging Face blog, <i>Mixture of Experts Explained</i>.</li>
      <li>Maarten Grootendorst, <i>A Visual Guide to LLM Agents</i>, 2025.</li>
      <li>Vaswani et al., <i>Attention Is All You Need</i>, NeurIPS 2017.</li>
      <li>Calvin Luo, <i>Understanding Diffusion Models: A Unified Perspective</i>, 2022.</li></ol></div>
  {footer()}
</div>'''
    return head('Home') + topbar(0) + f'<main class="home">{inner}</main>' + scripts(0)

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
    CACHE_FILE.write_text(json.dumps(CACHE, indent=0))
    print(f'built {len(WEEKS) + len(EXTRA_PAGES)} lesson pages + index')

if __name__ == '__main__':
    main()
