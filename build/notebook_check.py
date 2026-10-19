#!/usr/bin/env python3
"""Execute every generated notebook and record which code cells are illustrative fragments.

  python3 build/notebook_check.py         (needs nbclient, ipykernel and the course libraries)

A cell is a *fragment* when it fails for a reason that would also fail in Google Colab: it uses names
that no earlier cell defines, or it reuses objects in a way only meaningful inside a larger program.
Failures caused by this machine (no internet, no API key, no GPU, missing pip packages, interactive
input) are not fragments. Results go to build/notebook-fragments.json, which build.py reads to show
fragments as read-only code in the notebooks. Run build.py again afterwards.
"""
import ast, hashlib, json, re, sys, warnings
from pathlib import Path
import nbformat
from nbclient import NotebookClient

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'build' / 'notebook-fragments.json'

ENV_ERRORS = re.compile(r'huggingface|hf\.co|Connection|connect|resolve|getaddrinfo|Proxy|api[_ ]?key|OPENAI|Authentication|'
                        r'Ollama|CUDA|cuda|GPU|StdinNotImplemented|Timeout|timed out|No module named|not installed|'
                        r'Can\'t load|401|403|404|429|SSL|certificate|Tavily|TAVILY|MaxRetry|HTTPError|ReadError|'
                        r'Already running asyncio|cannot be called from a running event loop', re.I)

def defined_names(code):
    try: tree = ast.parse(code)
    except SyntaxError: return set()
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): names.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store): names.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names: names.add((a.asname or a.name).split('.')[0])
    return names

def key(code): return hashlib.sha256(code.encode()).hexdigest()[:16]

def main():
    result = {}
    for f in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        nb = nbformat.read(f, as_version=4)
        for c in nb.cells:
            if c.cell_type == 'code' and c.source.lstrip().startswith(('%pip', '%%writefile', '# Libraries')):
                c.metadata['skip'] = True; c.source_orig = c.source; c.source = 'pass'
        try:
            NotebookClient(nb, timeout=150, kernel_name='python3', allow_errors=True, resources={'metadata': {'path': '/tmp'}}).execute()
        except Exception as e:
            print(f'{f.name}: stopped early ({type(e).__name__})')
        failed_defs, frag = set(), []
        for c in nb.cells:
            if c.cell_type != 'code' or c.metadata.get('skip'): continue
            errs = [o for o in c.outputs if o.output_type == 'error']
            if not errs: continue
            msg = f'{errs[0].ename}: {errs[0].evalue}'
            if ENV_ERRORS.search(msg):
                failed_defs |= defined_names(c.source); continue
            m = re.search(r"name '(\w+)' is not defined", msg)
            if m and m.group(1) in failed_defs:            # cascade from an environment failure
                failed_defs |= defined_names(c.source); continue
            frag.append(key(c.source)); failed_defs |= defined_names(c.source)
            print(f'  fragment {f.name}: {msg[:100]}')
        result[f.stem] = frag
        print(f'{f.name}: {len(frag)} fragment cell(s)')
    OUT.write_text(json.dumps(result, indent=1))

if __name__ == '__main__':
    sys.exit(main())
