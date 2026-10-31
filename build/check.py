#!/usr/bin/env python3
"""Smoke-test the built site in headless Chromium.

  python3 build/check.py [screenshot_dir] [local_pyodide_dir]

Loads every page (no JS errors), opens every animation in the panel, presses "Try it" on a
pure-Python example and checks the editor output. If local_pyodide_dir is given (the extracted
npm `pyodide` package), requests to the Pyodide CDN are served from it, so the test works offline.
"""
import sys, threading, http.server, functools, mimetypes
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'build' / 'shots'
PYO = Path(sys.argv[2]) if len(sys.argv) > 2 else None
SHOTS.mkdir(parents=True, exist_ok=True)

class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
srv = http.server.ThreadingHTTPServer(('127.0.0.1', 8765), functools.partial(Quiet, directory=str(ROOT)))
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = 'http://127.0.0.1:8765/'
errors, fails = [], []

def serve_pyodide(route):
    name = route.request.url.split('/full/')[-1].split('?')[0]
    f = PYO / name
    if f.exists():
        route.fulfill(status=200, body=f.read_bytes(), headers={'Content-Type': mimetypes.guess_type(name)[0] or 'application/octet-stream', 'Access-Control-Allow-Origin': '*'})
    else:
        route.fulfill(status=404, body='not available offline')

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    if PYO: pg.route('https://cdn.jsdelivr.net/pyodide/**', serve_pyodide)
    pg.on('pageerror', lambda e: errors.append((pg.url, str(e))))
    pg.on('console', lambda m: errors.append((pg.url, m.text)) if m.type == 'error' and 'Failed to load resource' not in m.text and 'net::' not in m.text else None)

    pages = ['index.html'] + sorted(f'lessons/{f.name}' for f in (ROOT / 'lessons').glob('*.html'))
    for path in pages:
        pg.goto(base + path, wait_until='load'); pg.wait_for_timeout(250)
        info = pg.evaluate('''() => ({try: document.querySelectorAll('.try[data-run="1"]').length,
            local: document.querySelectorAll('.try[data-run="0"]').length, display: document.querySelectorAll('[data-display]').length})''')
        print(f'ok {path:45s} try={info["try"]:<3} local={info["local"]:<3} display={info["display"]}')
    pg.goto(base + 'index.html'); pg.screenshot(path=str(SHOTS / 'home.png'))

    # every animation inside the panel
    pg.goto(base + 'lessons/week01-transformers.html', wait_until='load')
    pg.screenshot(path=str(SHOTS / 'lesson_closed.png'))
    ids = pg.evaluate('Sims.list().map(s => s.id)')
    for sid in ids:
        pg.evaluate(f'document.querySelector("#anim-select").value = "{sid}"; document.querySelector("#anim-select").dispatchEvent(new Event("change"))')
        if sid == ids[0]: pg.click('[data-display]')   # also exercises the Display button
        pg.wait_for_timeout(150)
        for _ in range(3):
            nxt = pg.locator('#anim-host button', has_text='Next')
            if nxt.count() and nxt.first.is_enabled(): nxt.first.click(); pg.wait_for_timeout(100)
        if not pg.locator('#anim-host canvas, #anim-host svg').count(): fails.append(f'animation {sid} did not render')
        pg.locator('#workbench').screenshot(path=str(SHOTS / f'panel_{sid}.png'))
    print('animations rendered in panel:', len(ids) - sum('animation' in f for f in fails), '/', len(ids))
    pg.screenshot(path=str(SHOTS / 'lesson_anim.png'))

    # Try it on a pure-Python example (week 9 regex parser), then a local-only one
    pg.goto(base + 'lessons/week09-react-agents.html', wait_until='load')
    pure = pg.evaluate('''() => { const t = [...document.querySelectorAll('.try[data-run="1"]')];
        const i = t.findIndex(b => !/import (numpy|pandas|sklearn|matplotlib)/.test(b.closest('.runnable').querySelector('.src').value)); return i; }''')
    if PYO and pure >= 0:
        pg.locator('.try[data-run="1"]').nth(pure).scroll_into_view_if_needed()
        pg.locator('.try[data-run="1"]').nth(pure).click()
        try:
            pg.wait_for_function('document.querySelector("#output").textContent.trim().length > 0 && !document.querySelector("#btn-run").disabled', timeout=120000)
            txt = pg.inner_text('#output')
            print('Try it output:', txt[:200].replace('\n', ' | '))
            if 'Error' in txt and 'Traceback' in txt: fails.append('Try it raised: ' + txt[:200])
        except Exception as e:
            fails.append('Try it did not finish: ' + str(e)[:120])
        pg.screenshot(path=str(SHOTS / 'lesson_code.png'))
        # edit + Ctrl+Enter
        pg.evaluate('document.querySelector(".CodeMirror").CodeMirror.setValue("x = sum(i*i for i in range(10))\\nprint(\\"sum of squares:\\", x)")')
        pg.click('.CodeMirror'); pg.keyboard.press('Control+Enter')
        pg.wait_for_function('document.querySelector("#output").textContent.includes("sum of squares")', timeout=30000)
        print('edited code output:', pg.inner_text('#output').strip())
    local = pg.locator('.try[data-run="0"]')
    if local.count():
        local.first.scroll_into_view_if_needed(); local.first.click(); pg.wait_for_timeout(200)
        print('local example note:', pg.inner_text('#output').strip()[:120].replace('\n', ' | '))
    b.close()
srv.shutdown()
print('\nJS errors:', len(errors)); [print('  ', e) for e in errors[:20]]
print('Failures:', len(fails)); [print('  ', f) for f in fails]
sys.exit(1 if errors or fails else 0)
