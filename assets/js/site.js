/* Site behaviour: theme, sidebar, rail tabs, TOC highlight, copy buttons, Python runner, simulation launcher */
(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

  /* ---- theme ---- */
  const root = document.documentElement;
  try { const t = localStorage.getItem('cs-theme'); if (t) root.setAttribute('data-theme', t); } catch (e) {}
  const themeBtn = $('#theme-toggle');
  if (themeBtn) themeBtn.addEventListener('click', () => {
    const dark = root.getAttribute('data-theme') === 'dark' || (!root.getAttribute('data-theme') && matchMedia('(prefers-color-scheme: dark)').matches);
    const next = dark ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('cs-theme', next); } catch (e) {}
    document.dispatchEvent(new CustomEvent('cs-theme'));
  });

  /* ---- mobile sidebar ---- */
  const menuBtn = $('#menu-toggle'), sidebar = $('.sidebar');
  if (menuBtn && sidebar) menuBtn.addEventListener('click', () => sidebar.classList.toggle('open'));

  /* ---- rail tabs ---- */
  $$('.rail-tabs button').forEach(btn => btn.addEventListener('click', () => {
    $$('.rail-tabs button').forEach(b => b.classList.toggle('active', b === btn));
    $$('.rail-panel').forEach(p => p.classList.toggle('active', p.id === btn.dataset.panel));
  }));

  /* ---- TOC: build from headings + highlight on scroll ---- */
  const toc = $('#toc');
  if (toc) {
    const heads = $$('.main h2, .main h3').filter(h => h.id);
    heads.forEach(h => {
      const li = document.createElement('li'); li.className = h.tagName.toLowerCase();
      const a = document.createElement('a'); a.href = '#' + h.id; a.textContent = h.textContent.replace(/¶$/, '');
      li.appendChild(a); toc.appendChild(li);
    });
    const links = $$('a', toc);
    const obs = new IntersectionObserver(entries => {
      entries.forEach(en => { if (en.isIntersecting) {
        links.forEach(l => l.classList.toggle('active', l.getAttribute('href') === '#' + en.target.id));
      } });
    }, { rootMargin: '-60px 0px -70% 0px', threshold: 0 });
    heads.forEach(h => obs.observe(h));
  }

  /* ---- copy buttons on every <pre> ---- */
  $$('pre').forEach(pre => {
    if (pre.closest('.snippet')) return;
    const b = document.createElement('button'); b.className = 'btn-xs copy-btn'; b.textContent = 'Copy';
    b.addEventListener('click', () => copyText(pre.innerText, b));
    pre.appendChild(b);
  });
  function copyText(text, btn) {
    navigator.clipboard.writeText(text).then(() => { const o = btn.textContent; btn.textContent = 'Copied'; setTimeout(() => btn.textContent = o, 1200); });
  }
  window.csCopy = copyText;

  /* ---- Python runner (Pyodide, loaded lazily; only for snippets marked runnable) ---- */
  let pyodideReady = null;
  function loadPyodide() {
    if (pyodideReady) return pyodideReady;
    pyodideReady = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = 'https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js';
      s.onload = async () => { try { const py = await window.loadPyodide(); await py.loadPackage(['numpy']); resolve(py); } catch (e) { reject(e); } };
      s.onerror = () => reject(new Error('Could not load Pyodide'));
      document.head.appendChild(s);
    });
    return pyodideReady;
  }
  $$('.snippet').forEach(sn => {
    const code = $('code', sn), out = $('.out', sn), run = $('.run-btn', sn), copy = $('.copy-snip', sn);
    if (copy) copy.addEventListener('click', () => copyText(code.innerText, copy));
    if (run) run.addEventListener('click', async () => {
      run.disabled = true; run.textContent = 'Loading Python…'; out.style.display = 'block'; out.classList.remove('err'); out.textContent = '';
      try {
        const py = await loadPyodide();
        run.textContent = 'Running…';
        py.setStdout({ batched: s => { out.textContent += s + '\n'; } });
        py.setStderr({ batched: s => { out.textContent += s + '\n'; } });
        await py.runPythonAsync(code.innerText);
        if (!out.textContent) out.textContent = '(no output)';
      } catch (e) { out.classList.add('err'); out.textContent += String(e); }
      run.disabled = false; run.textContent = 'Run ▶';
    });
  });

  /* ---- Simulation overlay ---- */
  const overlay = $('#sim-overlay');
  let current = null;
  function openSim(id) {
    const sim = window.Sims && window.Sims.get(id);
    if (!sim || !overlay) return;
    closeSim();
    $('#sim-title').textContent = sim.title;
    $('#sim-kind').textContent = sim.kind || 'Interactive simulation';
    const body = $('#sim-body'); body.innerHTML = '';
    current = window.Sims.mount(id, body);
    overlay.classList.add('open'); document.body.style.overflow = 'hidden';
    try { history.replaceState(null, '', '#sim=' + id); } catch (e) {}
  }
  function closeSim() {
    if (current && current.destroy) current.destroy();
    current = null;
    if (overlay) { overlay.classList.remove('open'); $('#sim-body').innerHTML = ''; }
    document.body.style.overflow = '';
    if (location.hash.startsWith('#sim=')) try { history.replaceState(null, '', location.pathname); } catch (e) {}
  }
  window.openSim = openSim; window.closeSim = closeSim;
  $$('[data-sim]').forEach(el => el.addEventListener('click', () => openSim(el.dataset.sim)));
  if (overlay) {
    $('#sim-close').addEventListener('click', closeSim);
    overlay.addEventListener('click', e => { if (e.target === overlay) closeSim(); });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeSim(); });
    if (location.hash.startsWith('#sim=')) setTimeout(() => openSim(location.hash.slice(5)), 50);
  }

  /* ---- Home quiz ---- */
  $$('.quiz .q').forEach(q => {
    const ans = q.dataset.answer;
    $$('.opt', q).forEach(o => o.addEventListener('click', () => {
      $$('.opt', q).forEach(x => x.classList.remove('right', 'wrong'));
      o.classList.add(o.dataset.v === ans ? 'right' : 'wrong');
      $$('.opt', q).find(x => x.dataset.v === ans).classList.add('right');
      const fb = $('.fb', q); if (fb) fb.style.display = 'block';
    }));
  });
})();
