/* Multi-head attention on one sentence: input, project, attend, mix, concatenate, with arcs and a heat-map matrix per head. */
Sims.register('heads', {
  title: 'Multi-head attention: one sentence, three heads',
  kind: 'Transformers · Week 1',
  short: 'Sentence from the original Transformer paper. Follow “its” through three heads: coreference, determiner and possessive relations.',
  mount(root, H) {
    const L = H.layout(root);
    const toks = ['The', 'Law', 'will', 'never', 'be', 'perfect', ',', 'but', 'its', 'application', 'should', 'be', 'just'];
    const n = toks.length;
    const heads = [
      { n: 'Head 1', rel: 'Coreference', col: '#6a3fa0', pair: [8, 1] },
      { n: 'Head 2', rel: 'Determiner → noun', col: '#1e7f4f', pair: [0, 1] },
      { n: 'Head 3', rel: 'Possessive → noun', col: '#c8862b', pair: [8, 9] }
    ];
    const sub = ['₁', '₂', '₃'];
    const vitalNote = [
      v => `<b>★ Coreference: “its” → “Law” ${v}</b><br>The pronoun looks back at the noun it refers to. Vaswani et al. (2017, Fig. 4) show heads in layer 5 that attend sharply from “its” to “Law”. Clark et al. (2019) report a BERT head (5-4) that finds antecedents with about 65% accuracy.`,
      v => `<b>★ Determiner → noun: “The” → “Law” ${v}</b><br>Clark et al. (2019): BERT head 8-11 attends from determiners to the noun they belong to, with 94.3% accuracy on a dependency test.`,
      v => `<b>★ Possessive → noun: “its” → “application” ${v}</b><br>Clark et al. (2019): BERT head 7-6 attends from possessive pronouns to the noun they modify, with 80.5% accuracy.`
    ];
    const steps = [
      { label: 'Step 1 · Input' },
      { label: 'Step 2 · Project' },
      { label: 'Step 3 · Attend' },
      { label: 'Step 4 · Mix' },
      { label: 'Step 5 · Concatenate' }
    ];
    let h = 0, q = 8, stage = 0, showNums = false, tips = [], St;

    const stops = [[59, 76, 192], [141, 176, 254], [221, 221, 221], [244, 154, 123], [180, 4, 38]];
    const rgb = t => { t = Math.max(0, Math.min(1, t)); const i = Math.min(3, Math.floor(t * 4)), f = t * 4 - i, a = stops[i], b = stops[i + 1]; return [0, 1, 2].map(k => Math.round(a[k] + (b[k] - a[k]) * f)); };
    const heat = t => 'rgb(' + rgb(t).join(',') + ')';
    const isDark = t => { const c = rgb(t); return (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) < 150; };
    function matrix(k) {
      const [qi, ki] = heads[k].pair, M = [];
      for (let i = 0; i < n; i++) {
        const r = new Array(n).fill(0.2);
        r[i] += 3;
        if (i > 0) r[i - 1] += 1.5;
        if (i < n - 1) r[i + 1] += 1.5;
        if (i > 1) r[i - 2] += 0.6;
        if (i < n - 2) r[i + 2] += 0.6;
        if (i === qi) r[ki] += 9;
        const t = r.reduce((a, b) => a + b, 0);
        M.push(r.map(v => v / t));
      }
      return M;
    }
    const Ms = heads.map((_, k) => matrix(k));

    L.stage.innerHTML = `<div class="pr-wrap hd-wrap">
      <div class="hd-pipe"></div>
      <div class="hd-two">
        <div class="hd-left">
          <div class="pr-hint">Click a word to choose the query word. Hover an arc or cell for the exact number.</div>
          <svg class="pr-svg" viewBox="0 0 560 200" role="img" aria-label="Attention arcs from the selected word in the selected head"></svg>
          <div class="pr-legend"><span>low</span><span class="pr-scale"></span><span>high attention</span><span class="pr-star-key">★ vital: hover for why</span></div>
        </div>
        <div class="hd-right">
          <div class="pr-mtitle"></div>
          <div class="pr-hint">Row = the word looking (its query). Column = the word looked at (its key). Each row sums to 1.</div>
          <div class="pr-cols"><span class="pr-gap" style="flex-basis:76px"></span><div class="pr-colnums"></div></div>
          <div class="pr-mxbody"><div class="pr-rl" style="flex-basis:76px"></div><div class="pr-brk l"></div><div class="pr-cells"></div><div class="pr-brk r"></div></div>
          <label class="pr-num"><input type="checkbox"> Show the numbers in the cells</label>
        </div>
      </div>
      <div class="hd-out"></div>
      <div class="pr-tip"></div>
    </div>`;
    const wrap = L.stage.querySelector('.hd-wrap'), tip = wrap.querySelector('.pr-tip'), svg = wrap.querySelector('.pr-svg');
    const pipe = wrap.querySelector('.hd-pipe'), out = wrap.querySelector('.hd-out'), right = wrap.querySelector('.hd-right');
    const rl = wrap.querySelector('.pr-rl'), cells = wrap.querySelector('.pr-cells'), colnums = wrap.querySelector('.pr-colnums'), mtitle = wrap.querySelector('.pr-mtitle');
    const numBox = wrap.querySelector('.pr-num input');
    colnums.style.gridTemplateColumns = cells.style.gridTemplateColumns = 'repeat(' + n + ', var(--c))';
    wrap.querySelector('.pr-scale').innerHTML = [0, 0.25, 0.5, 0.75, 1].map(t => `<i style="background:${heat(t)}"></i>`).join('');

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
    const arcPath = (x1, x2, y) => { const dx = Math.abs(x2 - x1), hh = Math.min(86, 16 + dx * 0.3); return 'M' + x1 + ' ' + y + ' Q' + ((x1 + x2) / 2) + ' ' + (y - hh * 2) + ' ' + x2 + ' ' + y; };
    const top3 = row => row.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
    const topOther = (row, i0) => row.map((v, i) => [v, i]).filter(a => a[1] !== i0).sort((a, b) => b[0] - a[0])[0];

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

    function paintOut() {
      const w = toks[q];
      let rows = '';
      heads.forEach((hd, k) => {
        const row = Ms[k][q], to = topOther(row, q);
        let body = '';
        if (stage === 1) body = `own W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> → q${sub[k]}(“${w}”), and k${sub[k]}, v${sub[k]} for every word`;
        if (stage === 2) body = `“${w}” looks most at “${toks[to[1]]}” (${to[0].toFixed(2)})`;
        if (stage === 3) body = `z${sub[k]} = ` + top3(row).map(a => `${a[0].toFixed(2)}·v(${toks[a[1]]})`).join(' + ') + ' + …';
        rows += `<div class="hd-row${k === h ? ' sel' : ''}" data-head="${k}" style="--hc:${hd.col}"><b>${hd.n}</b> <span class="hd-rel">${hd.rel}</span><div>${body}</div></div>`;
      });
      if (stage === 4) {
        rows = `<div class="hd-concat">` + heads.map((hd, k) => `<span class="hd-z" style="--hc:${hd.col}">z${sub[k]}</span>`).join('') + `<span class="hd-arrow">→ concatenate → × W<sub>O</sub> →</span><span class="hd-new">new vector for “${w}”</span></div>`;
      }
      out.innerHTML = stage === 0 ? '' : rows;
      out.style.display = stage === 0 ? 'none' : 'block';
      out.querySelectorAll('[data-head]').forEach(el => el.addEventListener('click', () => { h = +el.getAttribute('data-head'); L_head.value = h; paint(); }));
    }

    function paint() {
      tips = []; tip.style.display = 'none';
      const M = Ms[h], row = M[q], hd = heads[h];
      const wd = toks.map(t => t.length * 6.8 + (t === 'its' ? 10 : 0)), gap = 10, total = wd.reduce((a, b) => a + b, 0) + gap * (n - 1);
      let x0 = (560 - total) / 2; const xs = wd.map(v => { const c = x0 + v / 2; x0 += v + gap; return c; });
      const y0 = 130, mxo = Math.max(...row.filter((v, i) => i !== q));
      let s = '';
      if (stage >= 2) {
        toks.forEach((t, j) => {
          if (j === q) return;
          const wt = row[j], nm = wt / mxo, id = T('<b>' + hd.n + ': “' + toks[q] + '” → “' + t + '”</b><br>attention ' + wt.toFixed(2) + ' (' + Math.round(wt * 100) + '% of its total)');
          const d = arcPath(xs[q], xs[j], y0);
          s += `<path d="${d}" fill="none" stroke="${heat(nm)}" stroke-width="${(1 + wt * 16).toFixed(1)}" stroke-linecap="round" opacity="${(0.3 + 0.7 * nm).toFixed(2)}"/>`;
          s += `<path d="${d}" fill="none" stroke="transparent" stroke-width="14" data-t="${id}" style="cursor:pointer"/>`;
        });
        const sid = T('<b>' + hd.n + ': “' + toks[q] + '” → itself</b><br>attention ' + row[q].toFixed(2) + '. Words usually keep some attention for themselves; the short tick shows it.');
        s += `<line x1="${xs[q]}" y1="${y0 - 8}" x2="${xs[q]}" y2="${y0 - 8 - row[q] * 60}" stroke="${heat(row[q] / Math.max(...row))}" stroke-width="4" stroke-linecap="round" data-t="${sid}"/>`;
      }
      toks.forEach((t, i) => {
        const isq = i === q;
        s += `<circle cx="${xs[i]}" cy="${y0}" r="${isq ? 7 : 4}" fill="${isq ? '#1f3a5f' : '#7a8291'}"/>`;
        s += `<text x="${xs[i]}" y="${y0 + 24}" text-anchor="middle" font-size="13" fill="#1b1f24" font-weight="${isq ? 700 : 400}">${t}</text>`;
        s += `<text x="${xs[i]}" y="${y0 + 40}" text-anchor="middle" font-size="11" fill="#7a8291">${i + 1}</text>`;
        s += `<rect x="${xs[i] - wd[i] / 2 - 5}" y="${y0 + 8}" width="${wd[i] + 10}" height="38" fill="transparent" data-sel="${i}" style="cursor:pointer"/>`;
      });
      const sid2 = T('<b>★ One word, two heads</b><br>“its” is handled by Head 1 (it points back to “Law”, the noun it refers to) and by Head 3 (it points forward to “application”, the noun it belongs to). Switch heads and keep “its” selected.');
      s += `<text x="${xs[8]}" y="${y0 + 57}" text-anchor="middle" font-size="14" data-t="${sid2}" fill="#BA7517" style="cursor:pointer">★</text>`;
      svg.innerHTML = s;

      mtitle.textContent = 'Attention matrix · ' + hd.n + ' (' + hd.rel + ')';
      colnums.innerHTML = toks.map((x, i) => `<div>${i + 1}</div>`).join('');
      rl.innerHTML = ''; cells.innerHTML = '';
      toks.forEach((x, i) => {
        const d = document.createElement('div');
        d.setAttribute('data-sel', i);
        d.setAttribute('data-t', T('<b>Row ' + (i + 1) + ': “' + x + '” looking</b><br>Its attention over all ' + n + ' words sums to 1.00.'));
        if (i === q) d.className = 'pr-rsel';
        d.textContent = x; rl.appendChild(d);
      });
      const mxAll = Math.max(...M.flat());
      M.forEach((r, i) => r.forEach((v, j) => {
        const a = Math.min(1, v / mxAll), el = document.createElement('div');
        const vital = i === hd.pair[0] && j === hd.pair[1];
        let note = '<b>' + hd.n + ': “' + toks[i] + '” → “' + toks[j] + '”</b><br>attention ' + v.toFixed(2) + ' (' + Math.round(v * 100) + '%)';
        if (vital) note = vitalNote[h](v.toFixed(2));
        el.setAttribute('data-t', T(note));
        if (stage >= 2) { el.style.background = heat(a); el.style.color = isDark(a) ? '#fff' : '#222'; }
        else { el.style.background = '#eef0f3'; }
        if (i === q) el.className = 'pr-qrow';
        el.textContent = stage >= 2 ? ((showNums ? v.toFixed(2).slice(1) : '') + (vital ? '★' : '')) : '';
        if (vital) el.classList.add('pr-vital');
        cells.appendChild(el);
      }));
      wrap.querySelectorAll('[data-sel]').forEach(el => el.addEventListener('click', () => { q = +el.getAttribute('data-sel'); paint(); }));
      wrap.querySelectorAll('[data-t]').forEach(bind);
      paintOut();
      paintPipe();
      explain();
    }

    function explain() {
      const w = toks[q], hd = heads[h], row = Ms[h][q], to = topOther(row, q);
      const txt = [
        `The sentence enters all at once (the first clause of the sentence in Figure 4 of the original Transformer paper). Choose the <b>query word</b>; “its” is the interesting one. Press Next.`,
        `Every head has its <b>own</b> W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>. The same word “<b>${w}</b>” becomes a different query in each head, and every word gets a different key and value per head.`,
        `Each head scores the words differently, then softmax. <b>${hd.n} (${hd.rel})</b>: “${w}” puts most of its attention on “${toks[to[1]]}” (${to[0].toFixed(2)})` + (to[0] > 0.3 ? '. Switch to another head and watch where “its” looks.' : '. This head has no special relation for this word, so it stays near itself and its neighbours. Try “its”.'),
        `Each head returns a weighted sum of <i>its own</i> value vectors: <b>z<sub>h</sub> = Σ attention · V<sub>h</sub></b>. That gives one vector per head for “<b>${w}</b>”.`,
        `The head outputs are joined, <b>[z₁ z₂ z₃]</b>, and multiplied by <b>W<sub>O</sub></b>. The result is one new vector for “<b>${w}</b>”, the same size as its input. In the original paper: 8 heads of 64 dimensions, 8 × 64 = 512.`
      ][stage];
      L.explain(txt, steps[stage].label);
    }

    const L_head = L.select('Head', heads.map((x, k) => [k, x.n + ' · ' + x.rel]), h, v => { h = +v; paint(); });
    numBox.addEventListener('change', e => { showNums = e.target.checked; paint(); });
    St = H.stepper(L, () => steps, (st, i) => { stage = i; paint(); }, { interval: 2000 });
    L.setLegend([['#6a3fa0', 'Head 1 · coreference'], ['#1e7f4f', 'Head 2 · determiner'], ['#c8862b', 'Head 3 · possessive']]);
    L.setNotes(`<p><b>Where the sentence comes from.</b> “The Law will never be perfect, but its application should be just” is the start of the sentence used in Figures 4 and 5 of Vaswani et al. (2017), <i>Attention Is All You Need</i>, which show heads in layer 5 that attend sharply from “its” to “Law”.</p><p><b>Where the head types come from.</b> Clark et al. (2019), <i>What Does BERT Look At?</i>, found BERT heads that attend from determiners to nouns (head 8-11, 94.3%), from possessive pronouns to the noun they modify (7-6, 80.5%), and that resolve coreference (5-4, about 65%).</p><p><b>What is simplified.</b> Those findings come from different models and layers. Here they are shown together as three heads of one layer, with hand-set attention values, to teach the idea. Real heads are rarely this clean, and many are redundant or hard to interpret.</p><p><b>Two meanings of “weight”.</b> W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> and W<sub>O</sub> are learned weight matrices. The numbers in the matrix are <i>attention</i>, computed per sentence from Q and K.</p>`);

    function fit() {
      const W = right.clientWidth; if (!W) return;
      const c = Math.max(14, Math.min(30, Math.floor((W - 108) / n - 2)));
      wrap.style.setProperty('--c', c + 'px');
      numBox.disabled = c < 22; numBox.parentElement.style.opacity = c < 22 ? 0.5 : 1;
      numBox.parentElement.title = c < 22 ? 'Not enough room for numbers at this width; hover a cell instead' : '';
    }
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(fit) : null;
    if (ro) ro.observe(right);
    fit();
    return { destroy() { if (ro) ro.disconnect(); St.destroy(); } };
  }
});
