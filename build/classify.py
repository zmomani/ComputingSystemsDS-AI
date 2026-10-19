#!/usr/bin/env python3
"""Decide which Python examples can run in the browser, and what each needs first.

Reads JSON from stdin: a list of {"code": str, "static_ok": bool} for one lesson, in page order.
Writes JSON to stdout: a list of {"run": bool, "setup": str, "reason": str}.

A block is runnable when it executes in CPython (with numpy/pandas/sklearn/matplotlib, the same
libraries the browser loads) either on its own, or after stdlib/numpy/pandas imports it forgot,
or after the earlier runnable examples of the same lesson (a notebook-style "setup" that the page
runs silently before the example). Everything else is marked local.
"""
import io, json, sys, signal, contextlib, re

AUTO_IMPORTS = {
    'np': 'import numpy as np', 'pd': 'import pandas as pd', 'plt': 'import matplotlib.pyplot as plt',
    'math': 'import math', 're': 'import re', 'json': 'import json', 'random': 'import random', 'time': 'import time',
    'collections': 'import collections', 'itertools': 'import itertools', 'operator': 'import operator', 'ast': 'import ast',
    'Counter': 'from collections import Counter', 'defaultdict': 'from collections import defaultdict',
    'TypedDict': 'from typing import TypedDict', 'Annotated': 'from typing import Annotated', 'List': 'from typing import List',
    'Optional': 'from typing import Optional', 'Dict': 'from typing import Dict', 'Any': 'from typing import Any',
}

class Timeout(Exception): pass
def _alarm(*_): raise Timeout('took longer than 20 s')
signal.signal(signal.SIGALRM, _alarm)

def run(code, ns):
    buf = io.StringIO()
    signal.alarm(20)
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(code, '<example>', 'exec'), ns)
        return None
    except BaseException as e:
        return e
    finally:
        signal.alarm(0)
        try:
            import matplotlib.pyplot as plt; plt.close('all')
        except Exception: pass

def try_with_imports(code, prefix_ns_factory):
    """Run code; on NameError for a known name, add the import and retry. Returns (err, imports)."""
    imports = []
    for _ in range(6):
        ns = prefix_ns_factory()
        if imports: run('\n'.join(imports), ns)
        err = run(code, ns)
        if isinstance(err, NameError):
            m = re.search(r"name '(\w+)' is not defined", str(err))
            if m and m.group(1) in AUTO_IMPORTS and AUTO_IMPORTS[m.group(1)] not in imports:
                imports.append(AUTO_IMPORTS[m.group(1)]); continue
        return err, imports
    return err, imports

def main():
    import matplotlib; matplotlib.use('agg')
    blocks = json.load(sys.stdin)
    out, chain = [], []          # chain: setup+code of earlier runnable blocks, in order
    for b in blocks:
        if not b['static_ok']:
            out.append({'run': False, 'setup': '', 'reason': ''}); continue
        code = b['code']
        err, imps = try_with_imports(code, lambda: {'__name__': '__main__'})
        setup = '\n'.join(imps)
        if err is not None and chain:
            prior = '\n\n'.join(chain)
            def with_chain():
                ns = {'__name__': '__main__'}; run(prior, ns); return ns
            err2, imps2 = try_with_imports(code, with_chain)
            if err2 is None:
                err, imps = None, imps2
                setup = prior + ('\n' + '\n'.join(imps2) if imps2 else '')
        own = '\n'.join(imps)
        if err is not None and chain:
            pass
        if err is None:
            out.append({'run': True, 'setup': setup, 'reason': ''})
            chain.append((own + '\n' + code) if own else code)
        else:
            reason = f'{type(err).__name__}: {str(err)[:120]}'
            out.append({'run': False, 'setup': '', 'reason': reason})
    json.dump(out, sys.stdout)

if __name__ == '__main__':
    main()
