/* Sinusoidal positional encoding on a short sentence: position, angle, sine/cosine pair, filling the vector, adding it to the embedding. */
Sims.register('posenc', {
  title: 'Positional encoding: one word, step by step',
  kind: 'Transformers · Week 1',
  short: 'Follow a word in "I love cats very much": its position, the angle for each frequency, the sine/cosine pair on a circle, the full vector, and the sum with the embedding.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    const toks = ['I', 'love', 'cats', 'very', 'much'], n = toks.length;
    const steps = [
      { label: 'Step 1 · Position' },
      { label: 'Step 2 · Frequency and angle' },
      { label: 'Step 3 · Sine and cosine' },
      { label: 'Step 4 · Fill the vector' },
      { label: 'Step 5 · Add to the embedding' }
    ];
    let p = 1, D = 8, pair = 0, layout = 'int', stage = 0, probe = 30, showNums = true, tips = [], St;
    const PURPLE = '#6a3fa0', GREEN = '#1e7f4f', NAVY = '#1f3a5f', ORANGE = '#c8862b';

    const stops = [[59, 76, 192], [141, 176, 254], [221, 221, 221], [244, 154, 123], [180, 4, 38]];
    const rgb = t => { t = Math.max(0, Math.min(1, t)); const i = Math.min(3, Math.floor(t * 4)), f = t * 4 - i, a = stops[i], b = stops[i + 1]; return [0, 1, 2].map(k => Math.round(a[k] + (b[k] - a[k]) * f)); };
    const heat = t => 'rgb(' + rgb(t).join(',') + ')';
    const isDark = t => { const c = rgb(t); return (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) < 150; };
    const norm = v => (Math.max(-1, Math.min(1, v)) + 1) / 2;

    const omega = i => 1 / Math.pow(10000, 2 * i / D);
    const fw = w => w >= 0.01 ? w.toFixed(4).replace(/0+$/, '').replace(/\.$/, '') : w.toExponential(2).replace('e-', ' × 10⁻').replace('e+', ' × 10');
    const fv = v => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(3);
    const short = v => (v < 0 ? '−' : '') + Math.abs(v).toFixed(2).replace(/^0/, '');
    // slot -> (pair index, isCosine) for the chosen layout; and back
    const slotInfo = d => layout === 'int' ? { i: Math.floor(d / 2), cos: d % 2 === 1 } : { i: d % (D / 2), cos: d >= D / 2 };
    const slotOf = (i, cos) => layout === 'int' ? 2 * i + (cos ? 1 : 0) : i + (cos ? D / 2 : 0);
    const pe = (pos, d) => { const s = slotInfo(d), a = pos * omega(s.i); return s.cos ? Math.cos(a) : Math.sin(a); };
    // illustrative embedding values (fixed per word and slot, not from a trained model)
    const emb = (k, d) => { const x = Math.sin((k + 1) * 12.9898 + (d + 1) * 78.233 + D * 3.1) * 43758.5453; return Math.round(((x - Math.floor(x)) - 0.5) * 120) / 100; };

    L.stage.innerHTML = `<div class="pr-wrap hd-wrap pe-wrap">
      <div class="hd-pipe"></div>
      <div class="hd-two">
        <div class="hd-left">
          <div class="pr-hint">Click a word to choose its position. Hover any mark for the exact number.</div>
          <svg class="pr-svg pe-svg" viewBox="0 0 560 400" role="img" aria-label="Position, angle, and the sine and cosine of the angle for one word"></svg>
        </div>
        <div class="hd-right">
          <div class="pr-mtitle"></div>
          <div class="pr-hint">Each row is one word's positional vector. Click a column to choose the pair <i>i</i> (its two slots are outlined).</div>
          <div class="pr-cols"><span class="pr-gap" style="flex-basis:76px"></span><div class="pr-colnums"></div></div>
          <div class="pr-mxbody"><div class="pr-rl" style="flex-basis:76px"></div><div class="pr-brk l"></div><div class="pr-cells pe-cells"></div><div class="pr-brk r"></div></div>
          <label class="pr-num"><input type="checkbox"> Show the numbers in the cells</label>
          <div class="pe-add"></div>
        </div>
      </div>
      <div class="pr-tip"></div>
    </div>`;
    const wrap = L.stage.querySelector('.pe-wrap'), tip = wrap.querySelector('.pr-tip'), svg = wrap.querySelector('.pe-svg');
    const pipe = wrap.querySelector('.hd-pipe'), right = wrap.querySelector('.hd-right'), addBox = wrap.querySelector('.pe-add');
    const rl = wrap.querySelector('.pr-rl'), cells = wrap.querySelector('.pe-cells'), colnums = wrap.querySelector('.pr-colnums'), mtitle = wrap.querySelector('.pr-mtitle');
    const numBox = wrap.querySelector('.pr-num input'); numBox.checked = true;

    const T = s => { tips.push(s); return tips.length - 1; };
    function show(e, i) {
      tip.innerHTML = tips[i]; tip.style.display = 'block';
      const r = wrap.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, th = tip.offsetHeight, tw = tip.offsetWidth;
      tip.style.left = Math.max(4, Math.min(r.width - tw - 4, x - tw / 2)) + 'px';
      tip.style.top = (y - th - 14 < 0 ? y + 18 : y - th - 14) + 'px';
    }
    function bind(el) {
      const i = +el.getAttribute('data-t');
      el.addEventListener('pointerenter', e => show(e, i));
      el.addEventListener('pointermove', e => { if (e.pointerType === 'mouse') show(e, i); });
      el.addEventListener('pointerleave', e => { if (e.pointerType === 'mouse') tip.style.display = 'none'; });
      el.addEventListener('click', e => show(e, i));
    }

    function paintPipe() {
      pipe.innerHTML = '';
      steps.forEach((s, i) => {
        const b = document.createElement('button');
        b.className = 'hd-pill' + (i === stage ? ' on' : '') + (i < stage ? ' done' : '');
        b.textContent = (i + 1) + ' ' + s.label.split('· ')[1];
        b.addEventListener('click', () => { St.stop(); St.go(i); });
        pipe.appendChild(b);
      });
    }

    function paintSvg() {
      const w = omega(pair), th = p * w, sv = Math.sin(th), cv = Math.cos(th);
      const f1 = stage < 1 ? 0.35 : 1, f2 = stage < 2 ? 0.35 : 1;
      let s = '';
      // tokens
      const xs = toks.map((_, k) => 58 + k * 111);
      toks.forEach((t, k) => {
        const on = k === p;
        s += `<rect x="${xs[k] - 42}" y="10" width="84" height="30" rx="7" fill="${on ? NAVY : '#fff'}" stroke="${on ? NAVY : '#c9d0da'}" stroke-width="1.5" data-sel="${k}" style="cursor:pointer"/>`;
        s += `<text x="${xs[k]}" y="30" text-anchor="middle" font-size="14" font-weight="700" fill="${on ? '#fff' : '#1b1f24'}" pointer-events="none">${t}</text>`;
        s += `<text x="${xs[k]}" y="56" text-anchor="middle" font-size="11" fill="#7a8291">pos = ${k}</text>`;
      });
      // formulas
      s += `<g opacity="${f1}">`;
      s += `<text x="20" y="84" font-size="12.5" font-family="monospace" fill="#1b1f24">ω = 1 / 10000^(2·${pair}/${D}) = <tspan font-weight="700" fill="${NAVY}">${fw(w)}</tspan>   <tspan fill="#7a8291">(pair i = ${pair})</tspan></text>`;
      s += `<text x="20" y="104" font-size="12.5" font-family="monospace" fill="#1b1f24">θ = pos × ω = ${p} × ${fw(w)} = <tspan font-weight="700" fill="${NAVY}">${th.toFixed(3)}</tspan> rad</text>`;
      s += `</g>`;
      // wave chart
      const bx = 24, by = 124, bw = 300, bh = 148, mid = by + bh / 2, amp = bh / 2 - 10, maxPos = 24;
      const X = pos => bx + pos / maxPos * bw, Y = v => mid - v * amp;
      s += `<g opacity="${f2}">`;
      s += `<rect x="${bx}" y="${by}" width="${bw}" height="${bh}" fill="#fff" stroke="#e3e6ea"/>`;
      s += `<line x1="${bx}" y1="${mid}" x2="${bx + bw}" y2="${mid}" stroke="#c9d0da" stroke-dasharray="3 3"/>`;
      let ds = '', dc = '';
      for (let q = 0; q <= maxPos * 5; q++) { const pos = q / 5; ds += (q ? 'L' : 'M') + X(pos).toFixed(1) + ' ' + Y(Math.sin(pos * w)).toFixed(1); dc += (q ? 'L' : 'M') + X(pos).toFixed(1) + ' ' + Y(Math.cos(pos * w)).toFixed(1); }
      s += `<path d="${ds}" fill="none" stroke="${PURPLE}" stroke-width="1.6" opacity=".55"/><path d="${dc}" fill="none" stroke="${GREEN}" stroke-width="1.6" opacity=".55"/>`;
      [0, 5, 10, 15, 20].forEach(q => { s += `<text x="${X(q)}" y="${by + bh + 14}" text-anchor="middle" font-size="10" fill="#7a8291">${q}</text>`; });
      s += `<text x="${bx + bw / 2}" y="${by + bh + 28}" text-anchor="middle" font-size="10.5" fill="#7a8291">position</text>`;
      s += `<text x="${bx}" y="${by - 6}" font-size="10.5" fill="${PURPLE}" font-weight="700">● sin(pos·ω)</text><text x="${bx + 90}" y="${by - 6}" font-size="10.5" fill="${GREEN}" font-weight="700">● cos(pos·ω)</text>`;
      toks.forEach((t, k) => {
        const a = k * w, on = k === p;
        const ts = T(`<b>“${t}” (pos ${k}), pair i = ${pair}</b><br>θ = ${k} × ${fw(w)} = ${a.toFixed(3)} rad<br>sin θ = ${fv(Math.sin(a))}<br>cos θ = ${fv(Math.cos(a))}`);
        s += `<circle cx="${X(k)}" cy="${Y(Math.sin(a))}" r="${on ? 6 : 3.6}" fill="${PURPLE}" stroke="${on ? '#fff' : 'none'}" stroke-width="1.5" data-t="${ts}"/>`;
        s += `<circle cx="${X(k)}" cy="${Y(Math.cos(a))}" r="${on ? 6 : 3.6}" fill="${GREEN}" stroke="${on ? '#fff' : 'none'}" stroke-width="1.5" data-t="${ts}"/>`;
        if (on) s += `<line x1="${X(k)}" y1="${by}" x2="${X(k)}" y2="${by + bh}" stroke="${NAVY}" stroke-width="1" stroke-dasharray="2 3" opacity=".6"/>`;
      });
      // circle
      const cx = 446, cy = mid, r = 62;
      s += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#c9d0da" stroke-width="2"/>`;
      s += `<line x1="${cx - r - 10}" y1="${cy}" x2="${cx + r + 10}" y2="${cy}" stroke="#d9dde3"/><line x1="${cx}" y1="${cy - r - 10}" x2="${cx}" y2="${cy + r + 10}" stroke="#d9dde3"/>`;
      s += `<text x="${cx + r + 12}" y="${cy + 4}" font-size="10" fill="${GREEN}">cos</text><text x="${cx + 4}" y="${cy - r - 13}" font-size="10" fill="${PURPLE}">sin</text>`;
      toks.forEach((t, k) => {
        const a = k * w, px = cx + r * Math.cos(a), py = cy - r * Math.sin(a), on = k === p;
        const ts = T(`<b>“${t}” on the circle</b><br>the point (cos θ, sin θ) for θ = ${a.toFixed(3)} rad`);
        if (on) {
          s += `<line x1="${cx}" y1="${cy}" x2="${px}" y2="${py}" stroke="${NAVY}" stroke-width="2"/>`;
          s += `<line x1="${px}" y1="${py}" x2="${px}" y2="${cy}" stroke="${GREEN}" stroke-width="1.5" stroke-dasharray="3 3"/><line x1="${px}" y1="${py}" x2="${cx}" y2="${py}" stroke="${PURPLE}" stroke-width="1.5" stroke-dasharray="3 3"/>`;
        }
        s += `<circle cx="${px}" cy="${py}" r="${on ? 6 : 3.6}" fill="${on ? NAVY : '#7a8291'}" data-t="${ts}"/>`;
        s += `<text x="${px + (Math.cos(a) >= 0 ? 8 : -8)}" y="${py - 7}" text-anchor="${Math.cos(a) >= 0 ? 'start' : 'end'}" font-size="10" fill="${on ? NAVY : '#7a8291'}" font-weight="${on ? 700 : 400}">${k}</text>`;
      });
      // probe: two angles that share a sine
      if (stage === 2) {
        const al = probe * Math.PI / 180, al2 = Math.PI - al;
        const p1x = cx + r * Math.cos(al), p1y = cy - r * Math.sin(al), p2x = cx + r * Math.cos(al2), p2y = cy - r * Math.sin(al2);
        const tp = T(`<b>Probe angle ${probe}°</b><br>sin = ${fv(Math.sin(al))}, cos = ${fv(Math.cos(al))}`);
        const tq = T(`<b>The other angle with the same sine: ${(180 - probe)}°</b><br>sin = ${fv(Math.sin(al2))} (same), cos = ${fv(Math.cos(al2))} (opposite sign)`);
        s += `<line x1="${p2x}" y1="${p1y}" x2="${p1x}" y2="${p1y}" stroke="${ORANGE}" stroke-width="1.2" stroke-dasharray="4 3"/>`;
        s += `<circle cx="${p1x}" cy="${p1y}" r="5.5" fill="${ORANGE}" data-t="${tp}"/><circle cx="${p2x}" cy="${p2y}" r="5.5" fill="#fff" stroke="${ORANGE}" stroke-width="2" data-t="${tq}"/>`;
        s += `<text x="${cx}" y="${cy + r + 22}" text-anchor="middle" font-size="10.5" fill="${ORANGE}">${probe}° and ${180 - probe}° share sin = ${Math.sin(al).toFixed(2)}</text><text x="${cx}" y="${cy + r + 35}" text-anchor="middle" font-size="10.5" fill="${ORANGE}">cos tells them apart</text>`;
      }
      s += `</g>`;
      // the two slots this pair fills
      const sS = slotOf(pair, false), sC = slotOf(pair, true);
      s += `<g opacity="${f2}">`;
      s += `<rect x="20" y="318" width="250" height="46" rx="8" fill="#fff" stroke="${PURPLE}" stroke-width="1.6"/><text x="30" y="337" font-size="11.5" fill="#444">slot ${sS} (even) = sin θ</text><text x="30" y="355" font-size="14" font-family="monospace" font-weight="700" fill="${PURPLE}">${fv(sv)}</text>`;
      s += `<rect x="290" y="318" width="250" height="46" rx="8" fill="#fff" stroke="${GREEN}" stroke-width="1.6"/><text x="300" y="337" font-size="11.5" fill="#444">slot ${sC} (odd) = cos θ</text><text x="300" y="355" font-size="14" font-family="monospace" font-weight="700" fill="${GREEN}">${fv(cv)}</text>`;
      s += `</g>`;
      s = s.replace(/slot (\d+) \(even\)/, layout === 'int' ? 'slot $1 (even)' : 'slot $1 (first half)').replace(/slot (\d+) \(odd\)/, layout === 'int' ? 'slot $1 (odd)' : 'slot $1 (second half)');
      s += `<text x="20" y="386" font-size="10.5" fill="#7a8291">Pair i = ${pair} uses frequency ω = ${fw(w)}; every word at every position is placed on the same circle at angle pos × ω.</text>`;
      svg.innerHTML = s;
      svg.querySelectorAll('[data-t]').forEach(bind);
      svg.querySelectorAll('[data-sel]').forEach(el => el.addEventListener('click', () => { p = +el.getAttribute('data-sel'); paint(); }));
    }

    function cellGrid(host, rowsVals, opts) {
      // rowsVals: array of {label, vals[], cls}; builds cells inside host (a grid with D columns)
      host.innerHTML = '';
      rowsVals.forEach(r => r.vals.forEach((v, d) => {
        const el = document.createElement('div'), a = norm(v / (opts.scale || 1));
        el.style.background = heat(a); el.style.color = isDark(a) ? '#fff' : '#222';
        if (opts.faded) el.style.opacity = 0.55;
        el.textContent = showNums && !numBox.disabled ? short(v) : '';
        if (r.cls) el.className = r.cls;
        if (opts.pairCols) { const s = slotInfo(d); if (s.i === pair) el.classList.add('pe-pair'); el.setAttribute('data-col', d); }
        if (r.tip) el.setAttribute('data-t', T(r.tip(d, v)));
        host.appendChild(el);
      }));
    }

    function paintMatrix() {
      const f3 = stage < 3 ? 0.55 : 1;
      mtitle.innerHTML = 'Positional vectors · d<sub>model</sub> = ' + D + ' <span style="font-weight:400;color:#7a8291">(the paper uses 512)</span>';
      colnums.style.gridTemplateColumns = cells.style.gridTemplateColumns = 'repeat(' + D + ', var(--c))';
      colnums.innerHTML = Array.from({ length: D }, (_, d) => { const s = slotInfo(d); return `<div data-col="${d}" style="cursor:pointer${s.i === pair ? ';color:#1f3a5f;font-weight:700' : ''}">${d}</div>`; }).join('');
      rl.innerHTML = '';
      toks.forEach((t, k) => { const d = document.createElement('div'); d.textContent = t; if (k === p) d.className = 'pr-rsel'; d.addEventListener('click', () => { p = k; paint(); }); rl.appendChild(d); });
      const rows = toks.map((t, k) => ({
        vals: Array.from({ length: D }, (_, d) => pe(k, d)),
        cls: k === p ? 'pr-qrow' : '',
        tip: (d, v) => { const s = slotInfo(d); return `<b>PE(pos ${k}, slot ${d})</b><br>${s.cos ? 'cos' : 'sin'}(${k} × ${fw(omega(s.i))}) = ${fv(v)}<br>pair i = ${s.i}`; }
      }));
      cells.style.opacity = f3;
      cellGrid(cells, rows, { pairCols: true });
      cells.querySelectorAll('[data-col]').forEach(el => el.addEventListener('click', () => { pair = slotInfo(+el.getAttribute('data-col')).i; paint(); }));
      colnums.querySelectorAll('[data-col]').forEach(el => el.addEventListener('click', () => { pair = slotInfo(+el.getAttribute('data-col')).i; paint(); }));
      cells.querySelectorAll('[data-t]').forEach(bind);
    }

    function paintAdd() {
      const f4 = stage < 4 ? 0.35 : 1;
      const E = Array.from({ length: D }, (_, d) => emb(p, d)), P = Array.from({ length: D }, (_, d) => pe(p, d)), S = E.map((v, d) => v + P[d]);
      addBox.style.opacity = f4;
      addBox.innerHTML = `<div class="pr-mtitle" style="margin-top:14px">Input for “${toks[p]}” = embedding + positional vector</div>
        <div class="pr-hint">The embedding numbers are illustrative (a trained model has learned values). Added entry by entry, so the size stays ${D}.</div>
        <div class="pe-addrow"><span>embedding</span><div class="pr-cells pe-g1"></div></div>
        <div class="pe-addrow"><span>+ position</span><div class="pr-cells pe-g2"></div></div>
        <div class="pe-addrow pe-sum"><span>= input</span><div class="pr-cells pe-g3"></div></div>`;
      const g = [addBox.querySelector('.pe-g1'), addBox.querySelector('.pe-g2'), addBox.querySelector('.pe-g3')];
      g.forEach(x => x.style.gridTemplateColumns = 'repeat(' + D + ', var(--c))');
      const mk = (host, vals, name) => cellGrid(host, [{ vals, tip: (d, v) => `<b>${name}, slot ${d}</b><br>${fv(v)}` }], { scale: name === 'input' ? 1.5 : 1 });
      mk(g[0], E, 'embedding'); mk(g[1], P, 'position'); mk(g[2], S, 'input');
      addBox.querySelectorAll('[data-t]').forEach(bind);
    }

    function explain() {
      const w = omega(pair), th = p * w, sv = Math.sin(th), cv = Math.cos(th), wd = toks[p];
      const txt = [
        `The sentence enters the model all at once, so the model must be told where each word is. Each word gets its position number <b>pos</b> = 0, 1, 2, … “<b>${wd}</b>” is at pos = ${p}. Click another word to change it.`,
        `Each pair of slots <b>i</b> has its own frequency ω<sub>i</sub> = 1 / 10000<sup>2i/d<sub>model</sub></sup> (a fixed constant, not learned). The angle is <b>θ = pos × ω<sub>i</sub></b> = ${p} × ${fw(w)} = ${th.toFixed(3)} rad. Click a column in the table to try another pair: small i is fast, large i is slow.`,
        `The pair fills two slots with <b>sin θ = ${fv(sv)}</b> and <b>cos θ = ${fv(cv)}</b>, the coordinates of a point on a circle. Values are positive or negative because they are positions on a wave. Drag the probe angle: two different angles can have the same sine, but their cosines differ in sign, so the pair identifies the angle.`,
        `Repeat for every pair i and every word: each row is one word's positional vector (${D} numbers, ${D / 2} pairs). The first pairs change quickly from word to word, the last pairs barely change, so no two rows are the same. Change the layout to see the same numbers in another order.`,
        `The positional vector is <b>added</b>, entry by entry, to the word's embedding. Both have ${D} numbers, so the sum also has ${D} and nothing is doubled. This sum is what enters the first attention layer.`
      ][stage];
      L.explain(txt, steps[stage].label);
    }

    function paint() { tips = []; tip.style.display = 'none'; paintPipe(); paintSvg(); paintMatrix(); paintAdd(); explain(); }

    L.select('Layout', [['int', 'Interleaved (sin, cos, sin, cos…)'], ['half', 'Halves (all sin, then all cos)']], layout, v => { layout = v; paint(); });
    L.select('d_model', [[4, '4'], [8, '8'], [16, '16']], D, v => { D = +v; pair = Math.min(pair, D / 2 - 1); fit(); paint(); });
    L.range('Probe angle (°)', 0, 180, 5, probe, v => { probe = v; paint(); });
    numBox.addEventListener('change', e => { showNums = e.target.checked; paint(); });
    St = H.stepper(L, () => steps, (st, i) => { stage = i; paint(); }, { interval: 2200 });
    L.setLegend([[PURPLE, 'sine slot'], [GREEN, 'cosine slot'], [NAVY, 'selected word'], [ORANGE, 'probe angle']]);
    L.setNotes(`<p><b>What is real and what is illustrative.</b> The positional values come straight from the formula of Vaswani et al. (2017), §3.5, with a small d<sub>model</sub> so every number is visible (the paper uses 512, with 256 pairs). The embedding numbers are made up for the demonstration; in a trained model they are learned.</p><p><b>ω is a frequency, not a weight.</b> It is fixed by the formula and never trained. The learned weight matrices (W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>) later learn to use the patterns this vector adds.</p><p><b>Why a pair?</b> One sine value alone repeats within a cycle, and a sine/cosine pair at the same frequency also makes a shift of k positions a rotation of the pair, which the authors hoped would let attention learn relative offsets. The layout (interleaved or halves) is a convention; both appear in code.</p>`);

    let autoOff = false, lastC = 0;
    function fit() {
      const W = right.clientWidth; if (!W) return false;
      const c = Math.max(14, Math.min(30, Math.floor((W - 108) / D - 2)));
      wrap.style.setProperty('--c', c + 'px');
      const tight = c < 24, before = showNums;
      if (tight && !numBox.disabled) { numBox.checked = false; showNums = false; autoOff = true; }
      else if (!tight && numBox.disabled && autoOff) { numBox.checked = true; showNums = true; autoOff = false; }
      numBox.disabled = tight; numBox.parentElement.style.opacity = tight ? 0.5 : 1;
      numBox.parentElement.title = tight ? 'Not enough room for numbers at this width; hover a cell instead' : '';
      const changed = c !== lastC || before !== showNums; lastC = c; return changed;
    }
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(() => { if (fit()) paint(); }) : null;
    if (ro) ro.observe(right);
    fit(); paint();
    return { destroy() { if (ro) ro.disconnect(); St.destroy(); } };
  }
});
