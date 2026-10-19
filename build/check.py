#!/usr/bin/env python3
"""Smoke-test the built site in headless Chromium: console errors, every simulation mounts, stepper works, screenshots."""
import sys, time, threading, http.server, functools
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'build' / 'shots'
SHOTS.mkdir(parents=True, exist_ok=True)
PORT = 8765
Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
srv = http.server.ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f'http://127.0.0.1:{PORT}/'
errors = []
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1400, 'height': 900})
    pg.on('pageerror', lambda e: errors.append(('pageerror', pg.url, str(e))))
    pg.on('console', lambda m: errors.append(('console', pg.url, m.text)) if m.type == 'error' and 'net::ERR' not in m.text and 'Failed to load resource' not in m.text else None)
    pages = ['index.html'] + sorted(f'lessons/{f.name}' for f in (ROOT / 'lessons').glob('*.html'))
    for path in pages:
        pg.goto(base + path, wait_until='load'); pg.wait_for_timeout(400)
        pg.screenshot(path=str(SHOTS / (path.replace('/', '_') + '.png')))
        print('ok', path, 'h2:', pg.locator('.main h2').count(), 'snippets:', pg.locator('.snippet').count(), 'sims:', pg.locator('.sim-card').count())
    # every simulation, from the index
    pg.goto(base + 'index.html', wait_until='load')
    sims = pg.evaluate('Sims.list().map(s=>s.id)')
    print('registered sims:', len(sims), sims)
    for sid in sims:
        pg.evaluate(f'openSim("{sid}")'); pg.wait_for_timeout(300)
        # press Next a few times / nudge sliders if present
        for _ in range(4):
            nxt = pg.locator('#sim-body button', has_text='Next')
            if nxt.count() and nxt.first.is_enabled(): nxt.first.click(); pg.wait_for_timeout(120)
        rng = pg.locator('#sim-body input[type=range]')
        if rng.count(): rng.first.evaluate('e=>{e.value=e.max*0.6;e.dispatchEvent(new Event("input"))}'); pg.wait_for_timeout(120)
        pg.locator('.sim-window').screenshot(path=str(SHOTS / f'sim_{sid}.png'))
        pg.evaluate('closeSim()')
    # test the Pyodide runner on one runnable snippet (network permitting)
    pg.goto(base + 'lessons/week01-transformers.html', wait_until='load')
    pg.click('.rail-tabs button[data-panel=panel-code]')
    runs = pg.locator('.run-btn')
    print('runnable snippets on week 1:', runs.count())
    if runs.count():
        runs.first.click()
        try:
            pg.wait_for_function('document.querySelector(".snippet .out").textContent.length>0 && !document.querySelector(".run-btn").disabled', timeout=90000)
            print('pyodide output:', pg.locator('.snippet .out').first.inner_text()[:300].replace('\n', ' | '))
        except Exception as e: print('pyodide run did not finish (network?):', str(e)[:100])
    b.close()
srv.shutdown()
print('\nJS errors:', len(errors))
for e in errors[:30]: print(' ', e)
