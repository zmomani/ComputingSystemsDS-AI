/* Simulation framework: registry, canvas helpers, standard layout, step player. */
window.Sims = (function () {
  const registry = new Map();

  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  const theme = () => ({
    bg: css('--surface'), bg2: css('--surface-2'), border: css('--border'), text: css('--text'), muted: css('--muted'),
    accent: css('--sim-accent') || css('--accent'), accent2: css('--accent-2'), green: css('--green'), purple: css('--purple'),
    yellow: css('--yellow'), red: css('--red'), mono: css('--font-mono') || 'monospace', body: css('--font-body') || 'sans-serif'
  });

  /* Hi-DPI canvas */
  function canvas(w, h) {
    const c = document.createElement('canvas');
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = w * dpr; c.height = h * dpr; c.style.width = w + 'px'; c.style.height = h + 'px';
    const ctx = c.getContext('2d'); ctx.scale(dpr, dpr);
    c.w = w; c.h = h; c.ctx = ctx;
    return c;
  }
  function clear(c) { c.ctx.clearRect(0, 0, c.w, c.h); }
  function text(ctx, s, x, y, o = {}) {
    ctx.save(); ctx.font = `${o.weight || 400} ${o.size || 13}px ${o.mono ? theme().mono : theme().body}`;
    ctx.fillStyle = o.color || theme().text; ctx.textAlign = o.align || 'left'; ctx.textBaseline = o.base || 'alphabetic';
    ctx.fillText(s, x, y); ctx.restore();
  }
  function rect(ctx, x, y, w, h, o = {}) {
    ctx.save(); ctx.beginPath();
    const r = o.r || 0;
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
    if (o.fill) { ctx.fillStyle = o.fill; ctx.fill(); }
    if (o.stroke) { ctx.strokeStyle = o.stroke; ctx.lineWidth = o.lw || 1; ctx.stroke(); }
    ctx.restore();
  }
  function arrow(ctx, x1, y1, x2, y2, o = {}) {
    ctx.save(); ctx.strokeStyle = o.color || theme().text; ctx.fillStyle = o.color || theme().text; ctx.lineWidth = o.lw || 1.6;
    if (o.dash) ctx.setLineDash(o.dash);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.setLineDash([]);
    const a = Math.atan2(y2 - y1, x2 - x1), s = o.head || 9;
    ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x2 - s * Math.cos(a - .45), y2 - s * Math.sin(a - .45)); ctx.lineTo(x2 - s * Math.cos(a + .45), y2 - s * Math.sin(a + .45)); ctx.closePath(); ctx.fill();
    ctx.restore();
  }
  function hexA(hex, a) { // hex color + alpha
    hex = hex.replace('#', ''); if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
    const n = parseInt(hex, 16); return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }
  // Sequential colormap (viridis-like, 5 anchors) for heatmaps
  function heat(t) {
    const stops = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
    t = Math.max(0, Math.min(1, t)); const i = Math.min(3, Math.floor(t * 4)), f = t * 4 - i;
    const a = stops[i], b = stops[i + 1];
    return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},${Math.round(a[1] + (b[1] - a[1]) * f)},${Math.round(a[2] + (b[2] - a[2]) * f)})`;
  }
  const fmt = (v, d = 2) => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(d);
  function softmax(arr, T = 1) { const m = Math.max(...arr); const e = arr.map(v => Math.exp((v - m) / T)); const s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); }
  function seeded(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }

  /* Standard layout: stage on the left, controls / explanation / notes on the right */
  function layout(root, opts = {}) {
    root.innerHTML = `<div class="sim-layout">
      <div><div class="sim-stage"></div>${opts.below ? '<div class="sim-below" style="margin-top:12px"></div>' : ''}</div>
      <div class="sim-side">
        <div class="sim-controls"></div>
        <div class="sim-explain"><span class="step-label">What is happening</span><div class="explain-text"></div></div>
        ${opts.legend ? '<div class="sim-legend"></div>' : ''}
        <details class="sim-notes"><summary>Teaching notes</summary><div class="notes-text" style="margin-top:6px"></div></details>
      </div></div>`;
    const L = {
      stage: root.querySelector('.sim-stage'), controls: root.querySelector('.sim-controls'),
      explainEl: root.querySelector('.explain-text'), notes: root.querySelector('.notes-text'), legend: root.querySelector('.sim-legend'), below: root.querySelector('.sim-below'),
      explain(html, label) { this.explainEl.innerHTML = html; if (label) root.querySelector('.step-label').textContent = label; },
      setNotes(html) { this.notes.innerHTML = html; },
      setLegend(items) { if (!this.legend) return; this.legend.innerHTML = items.map(([c, l]) => `<span style="--c:${c}">${l}</span>`).join(''); },
      button(label, fn, cls = 'btn') { const b = document.createElement('button'); b.className = cls; b.textContent = label; b.addEventListener('click', fn); this.controls.appendChild(b); return b; },
      range(label, min, max, step, val, fn) {
        const w = document.createElement('label'); w.innerHTML = `${label} <input type="range" min="${min}" max="${max}" step="${step}" value="${val}"> <output>${val}</output>`;
        const i = w.querySelector('input'), o = w.querySelector('output');
        i.addEventListener('input', () => { o.textContent = i.value; fn(parseFloat(i.value)); }); this.controls.appendChild(w); return i;
      },
      select(label, options, val, fn) {
        const w = document.createElement('label'); w.innerHTML = `${label} <select>${options.map(o => `<option value="${o[0]}"${o[0] == val ? ' selected' : ''}>${o[1]}</option>`).join('')}</select>`;
        const s = w.querySelector('select'); s.addEventListener('change', () => fn(s.value)); this.controls.appendChild(w); return s;
      },
      number(label, min, max, val, fn) {
        const w = document.createElement('label'); w.innerHTML = `${label} <input type="number" min="${min}" max="${max}" value="${val}">`;
        const i = w.querySelector('input'); i.addEventListener('change', () => { let v = Math.max(min, Math.min(max, parseInt(i.value) || min)); i.value = v; fn(v); }); this.controls.appendChild(w); return i;
      },
      textInput(label, val, fn, placeholder = '') {
        const w = document.createElement('label'); w.style.flex = '1'; w.innerHTML = `${label} <input type="text" value="${val.replace(/"/g, '&quot;')}" placeholder="${placeholder}" style="flex:1;min-width:140px">`;
        const i = w.querySelector('input'); i.addEventListener('input', () => fn(i.value)); this.controls.appendChild(w); return i;
      }
    };
    return L;
  }

  /* Step player: Prev / Next / Play controls over a list of steps */
  function stepper(L, getSteps, render, opts = {}) {
    let i = 0, timer = null;
    const prev = L.button('◀ Prev', () => go(i - 1), 'btn secondary');
    const next = L.button('Next ▶', () => go(i + 1));
    const play = L.button('▶ Play', () => { if (timer) stop(); else { play.textContent = '❚❚ Pause'; timer = setInterval(() => { if (i >= getSteps().length - 1) stop(); else go(i + 1); }, opts.interval || 1100); } }, 'btn secondary');
    const reset = L.button('↺ Reset', () => { stop(); go(0); }, 'btn secondary');
    function stop() { clearInterval(timer); timer = null; play.textContent = '▶ Play'; }
    function go(n) { const steps = getSteps(); i = Math.max(0, Math.min(steps.length - 1, n)); prev.disabled = i === 0; next.disabled = i === steps.length - 1; render(steps[i], i, steps.length); }
    go(0);
    return { go, get index() { return i; }, stop, destroy: stop, refresh: () => go(i) };
  }

  function register(id, def) { registry.set(id, Object.assign({ id }, def)); }
  function get(id) { return registry.get(id); }
  function mount(id, el) { const d = registry.get(id); if (!d) return null; const inst = d.mount(el, { theme, canvas, clear, text, rect, arrow, hexA, heat, fmt, softmax, seeded, layout, stepper }); return inst || {}; }
  function list() { return Array.from(registry.values()); }

  return { register, get, mount, list, theme, canvas, clear, text, rect, arrow, hexA, heat, fmt, softmax, seeded, layout, stepper };
})();
