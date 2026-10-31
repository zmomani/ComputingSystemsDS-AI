/* What does "it" refer to? Two sentences that differ by one word, with attention arcs and a heat-map matrix. */
Sims.register('pronoun', {
  title: 'Attention: what does “it” refer to?',
  kind: 'Transformers · Week 1',
  short: 'Change one word (tired / soft) and watch the attention of “it” move between “cat” and “mat”. Hover for exact numbers.',
  mount(root, H) {
    const L = H.layout(root);
    const base = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was'];
    const sents = { tired: { last: 'tired', ref: 1 }, soft: { last: 'soft', ref: 5 } };
    let cur = 'tired', q = 7, showNums = false, tips = [];
    const stops = [[59, 76, 192], [141, 176, 254], [221, 221, 221], [244, 154, 123], [180, 4, 38]];
    const rgb = t => { t = Math.max(0, Math.min(1, t)); const i = Math.min(3, Math.floor(t * 4)), f = t * 4 - i, a = stops[i], b = stops[i + 1]; return [0, 1, 2].map(k => Math.round(a[k] + (b[k] - a[k]) * f)); };
    const heat = t => 'rgb(' + rgb(t).join(',') + ')';
    const isDark = t => { const c = rgb(t); return (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) < 150; };
    const words = k => base.concat([sents[k].last]);

    function matrix(k) {
      const n = 10, ref = sents[k].ref, M = [];
      for (let i = 0; i < n; i++) {
        const r = new Array(n).fill(0.2);
        r[i] += 3;
        if (i > 0) r[i - 1] += 1.5;
        if (i < n - 1) r[i + 1] += 1.5;
        if (i > 1) r[i - 2] += 0.6;
        if (i < n - 2) r[i + 2] += 0.6;
        const add = (j, v) => { r[j] += v; };
        if (i === 7) { add(ref, 9); add(9, 4); }
        if (i === 9) { add(7, 4); add(ref, 6); add(8, 2); }
        if (i === 1) add(2, 3);
        if (i === 2) { add(1, 4); add(3, 2); }
        if (i === 5) { add(3, 3); add(4, 3); }
        if (i === 8) { add(7, 4); add(9, 2); }
        if (i === 6) { add(2, 2); add(7, 1.5); }
        const t = r.reduce((a, b) => a + b, 0);
        M.push(r.map(v => v / t));
      }
      return M;
    }

    L.stage.innerHTML = `<div class="pr-wrap">
      <div class="pr-two">
        <div class="pr-left">
          <div class="pr-hint">Click a word to see where it looks. Hover an arc or cell for the exact number.</div>
          <svg class="pr-svg" viewBox="0 0 380 178" role="img" aria-label="Attention arcs from the selected word"></svg>
          <div class="pr-legend"><span>low</span><span class="pr-scale"></span><span>high attention</span><span class="pr-star-key">★ vital: hover for why</span></div>
          <div class="pr-cards">
            <div class="pr-card"><div class="pr-card-l">Attention on “cat”</div><div class="pr-card-v" data-k="cat"></div></div>
            <div class="pr-card"><div class="pr-card-l">Attention on “mat”</div><div class="pr-card-v" data-k="mat"></div></div>
          </div>
        </div>
        <div class="pr-right">
          <div class="pr-mtitle">Attention matrix</div>
          <div class="pr-hint">Row = the word looking (its query). Column = the word looked at (its key). Each row sums to 1.</div>
          <div class="pr-cols"><span class="pr-gap"></span><div class="pr-colnums"></div></div>
          <div class="pr-mxbody"><div class="pr-rl"></div><div class="pr-brk l"></div><div class="pr-cells"></div><div class="pr-brk r"></div></div>
          <label class="pr-num"><input type="checkbox"> Show the numbers in the cells</label>
        </div>
      </div>
      <div class="pr-tip"></div>
    </div>`;
    const wrap = L.stage.querySelector('.pr-wrap'), tip = wrap.querySelector('.pr-tip'), svg = wrap.querySelector('.pr-svg');
    const rl = wrap.querySelector('.pr-rl'), cells = wrap.querySelector('.pr-cells'), colnums = wrap.querySelector('.pr-colnums');
    wrap.querySelector('.pr-scale').innerHTML = [0, 0.25, 0.5, 0.75, 1].map(t => `<i style="background:${heat(t)}"></i>`).join('');

    const T = h => { tips.push(h); return tips.length - 1; };
    function show(e, i) {
      tip.innerHTML = tips[i]; tip.style.display = 'block';
      const r = wrap.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, h = tip.offsetHeight;
      tip.style.left = Math.max(4, Math.min(r.width - 228, x - 110)) + 'px';
      tip.style.top = (y - h - 14 < 0 ? y + 18 : y - h - 14) + 'px';
    }
    function bind(el) {
      const i = +el.getAttribute('data-t');
      el.addEventListener('pointerenter', e => show(e, i));
      el.addEventListener('pointermove', e => { if (e.pointerType === 'mouse') show(e, i); });
      el.addEventListener('pointerleave', e => { if (e.pointerType === 'mouse') tip.style.display = 'none'; });
      el.addEventListener('click', e => show(e, i));
    }
    const arcPath = (x1, x2, y) => { const dx = Math.abs(x2 - x1), h = Math.min(86, 16 + dx * 0.4); return 'M' + x1 + ' ' + y + ' Q' + ((x1 + x2) / 2) + ' ' + (y - h * 2) + ' ' + x2 + ' ' + y; };

    function paint() {
      tips = []; tip.style.display = 'none';
      const oth = cur === 'tired' ? 'soft' : 'tired';
      const w = words(cur), M = matrix(cur), Mo = matrix(oth), row = M[q], ref = sents[cur].ref;
      const wd = w.map((t, i) => t.length * 6.4 + (i === 9 ? 10 : 0)), gap = 12, total = wd.reduce((a, b) => a + b, 0) + gap * 9;
      let x0 = (380 - total) / 2; const xs = wd.map(v => { const c = x0 + v / 2; x0 += v + gap; return c; });
      const y0 = 122, mxo = Math.max(...row.filter((v, i) => i !== q));
      let s = '';
      w.forEach((t, j) => {
        if (j === q) return;
        const wt = row[j], nm = wt / mxo, id = T('<b>“' + w[q] + '” → “' + t + '”</b><br>attention ' + wt.toFixed(2) + ' (' + Math.round(wt * 100) + '% of its total)');
        const d = arcPath(xs[q], xs[j], y0);
        s += `<path d="${d}" fill="none" stroke="${heat(nm)}" stroke-width="${(1 + wt * 16).toFixed(1)}" stroke-linecap="round" opacity="${(0.3 + 0.7 * nm).toFixed(2)}"/>`;
        s += `<path d="${d}" fill="none" stroke="transparent" stroke-width="14" data-t="${id}" style="cursor:pointer"/>`;
      });
      const sid = T('<b>“' + w[q] + '” → itself</b><br>attention ' + row[q].toFixed(2) + '. Words usually keep some attention for themselves; the short tick shows it.');
      s += `<line x1="${xs[q]}" y1="${y0 - 8}" x2="${xs[q]}" y2="${y0 - 8 - row[q] * 60}" stroke="${heat(row[q] / Math.max(...row))}" stroke-width="4" stroke-linecap="round" data-t="${sid}"/>`;
      w.forEach((t, i) => {
        const isq = i === q;
        s += `<circle cx="${xs[i]}" cy="${y0}" r="${isq ? 7 : 4}" fill="${isq ? '#1f3a5f' : '#7a8291'}"/>`;
        s += `<text x="${xs[i]}" y="${y0 + 24}" text-anchor="middle" font-size="12" fill="#1b1f24" font-weight="${isq ? 700 : 400}">${t}</text>`;
        s += `<text x="${xs[i]}" y="${y0 + 40}" text-anchor="middle" font-size="11" fill="#7a8291">${i + 1}</text>`;
        s += `<rect x="${xs[i] - wd[i] / 2 - 5}" y="${y0 + 8}" width="${wd[i] + 10}" height="38" fill="transparent" data-sel="${i}" style="cursor:pointer"/>`;
      });
      const sid2 = T('<b>★ The only word that differs</b><br>Switch between “tired” and “soft” and watch where “it” looks. This single word decides whether “it” means the cat or the mat.');
      s += `<text x="${xs[9] + wd[9] / 2 + 1}" y="${y0 + 20}" font-size="12" data-t="${sid2}" fill="#BA7517" style="cursor:pointer">★</text>`;
      svg.innerHTML = s;
      wrap.querySelector('[data-k=cat]').textContent = row[1].toFixed(2);
      wrap.querySelector('[data-k=mat]').textContent = row[5].toFixed(2);

      const top = row.map((v, i) => [v, i]).filter(a => a[1] !== q).sort((a, b) => b[0] - a[0])[0][1];
      let msg, label;
      if (q === 7) {
        const why = cur === 'tired' ? 'Being <b>tired</b> is something a living thing can be. A mat cannot get tired, so “it” is linked to <b>cat</b>.' : 'Being <b>soft</b> describes a material or surface. A cat is not the natural thing to call soft here, so “it” is linked to <b>mat</b>.';
        label = 'Reading the “it” row';
        msg = `<b>“it” puts most of its attention on “${w[ref]}” (${row[ref].toFixed(2)}).</b> ${why} Switch to “…because it was ${oth}” and the attention moves to the other word.`;
      } else {
        label = 'Reading the “' + w[q] + '” row';
        msg = `<b>“${w[q]}” puts most of its attention on “${w[top]}”.</b> Most words attend to themselves and their neighbours. Click “it” (word 8) to see the interesting case: the word it refers to is several words away.`;
      }
      L.explain(msg, label);

      colnums.innerHTML = w.map((x, i) => `<div>${i + 1}</div>`).join('');
      rl.innerHTML = ''; cells.innerHTML = '';
      w.forEach((x, i) => {
        const d = document.createElement('div');
        d.setAttribute('data-sel', i);
        d.setAttribute('data-t', T('<b>Row ' + (i + 1) + ': “' + x + '” looking</b><br>Its attention over all 10 words sums to 1.00.'));
        if (i === q) d.className = 'pr-rsel';
        d.textContent = x; rl.appendChild(d);
      });
      const mxAll = Math.max(...M.flat());
      M.forEach((r, i) => r.forEach((v, j) => {
        const a = Math.min(1, v / mxAll);
        let vital = false, note = '<b>“' + w[i] + '” → “' + w[j] + '”</b><br>attention ' + v.toFixed(2) + ' (' + Math.round(v * 100) + '%)';
        if (i === 7 && j === ref) { vital = true; note = '<b>★ Key attention: “it” → “' + w[j] + '” ' + v.toFixed(2) + '</b><br>This is the model working out what “it” refers to. In the “' + oth + '” sentence this same cell is only ' + Mo[7][ref].toFixed(2) + '.'; }
        if (i === 9 && j === ref) { vital = true; note = '<b>★ “' + w[9] + '” → “' + w[j] + '” ' + v.toFixed(2) + '</b><br>The describing word looks back at the thing it describes. That gives “it” extra evidence to link to the same word.'; }
        const el = document.createElement('div');
        el.setAttribute('data-t', T(note));
        el.style.background = heat(a); el.style.color = isDark(a) ? '#fff' : '#222';
        if (i === q) el.className = 'pr-qrow';
        el.textContent = (showNums ? v.toFixed(2).slice(1) : '') + (vital ? '★' : '');
        if (vital) el.classList.add('pr-vital');
        cells.appendChild(el);
      }));
      wrap.querySelectorAll('[data-sel]').forEach(el => el.addEventListener('click', () => { q = +el.getAttribute('data-sel'); paint(); }));
      wrap.querySelectorAll('[data-t]').forEach(bind);
    }

    L.select('Sentence', [['tired', '…because it was tired'], ['soft', '…because it was soft']], cur, v => { cur = v; paint(); });
    wrap.querySelector('.pr-num input').addEventListener('change', e => { showNums = e.target.checked; wrap.classList.toggle('pr-nums', showNums); paint(); });
    L.setNotes(`<p><b>Why two sentences.</b> “The cat sat on the mat because it was comfortable.” is ambiguous: “comfortable” fits both the cat and the mat. Changing one word (tired / soft) gives a pair where world knowledge decides what “it” refers to (a Winograd schema).</p><p><b>Two meanings of “weight”.</b> W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> are learned weight matrices (parameters), fixed after training. The numbers in this matrix are <i>attention</i> (attention weights), computed for each sentence as softmax(QKᵀ/√d<sub>k</sub>).</p><p><b>What the matrix uses.</b> It comes from Q and K only. V is used in the next step, when each word’s attention values mix the V vectors into its output.</p><p><b>Illustrative values.</b> The attention here is hand-set to show the idea. A trained model gives messier numbers, and high attention on a word such as “the” does not by itself explain what the model understood.</p>`);

    const mcol = wrap.querySelector('.pr-right');
    function fit() {
      const W = mcol.clientWidth; if (!W) return;
      const c = Math.max(17, Math.min(34, Math.floor((W - 96) / 10 - 2)));
      wrap.style.setProperty('--c', c + 'px');
    }
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(fit) : null;
    if (ro) ro.observe(mcol);
    fit(); paint();
    return { destroy() { if (ro) ro.disconnect(); } };
  }
});
