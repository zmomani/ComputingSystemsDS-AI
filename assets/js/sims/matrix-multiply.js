/* Matrix multiplication, stepped cell by cell — row of A × column of B, as drawn in linear-algebra textbooks. */
Sims.register('matrix-multiply', {
  title: 'Matrix multiplication, step by step',
  kind: 'Linear algebra · Week 1',
  short: 'Watch each cell of C = A·B form as a row of A meets a column of B. Change sizes and values.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    let m = 2, n = 3, p = 2; // A is m×n, B is n×p
    const rnd = H.seeded(7);
    let A, B, C;
    function newMatrices() {
      A = Array.from({ length: m }, () => Array.from({ length: n }, () => Math.floor(rnd() * 7) - 2));
      B = Array.from({ length: n }, () => Array.from({ length: p }, () => Math.floor(rnd() * 7) - 2));
      C = Array.from({ length: m }, () => Array(p).fill(null));
    }
    newMatrices();
    const cv = H.canvas(680, 360); L.stage.appendChild(cv); const ctx = cv.ctx;
    const cell = 46;

    function steps() { // one step per (i,j,k) term, then a "cell done" step
      const s = [{ type: 'intro' }];
      for (let i = 0; i < m; i++) for (let j = 0; j < p; j++) {
        for (let k = 0; k < n; k++) s.push({ type: 'term', i, j, k });
        s.push({ type: 'done', i, j });
      }
      s.push({ type: 'end' }); return s;
    }
    function drawMatrix(M, x0, y0, label, hl, color, rows, cols, dimLabel) {
      const T = H.theme();
      H.text(ctx, label, x0 + cols * cell / 2, y0 - 26, { align: 'center', weight: 700, size: 15 });
      H.text(ctx, dimLabel, x0 + cols * cell / 2, y0 - 10, { align: 'center', size: 11, color: T.muted });
      // brackets
      ctx.save(); ctx.strokeStyle = T.text; ctx.lineWidth = 1.6;
      const bh = rows * cell, bw = cols * cell;
      ctx.beginPath(); ctx.moveTo(x0 - 6, y0); ctx.lineTo(x0 - 12, y0); ctx.lineTo(x0 - 12, y0 + bh); ctx.lineTo(x0 - 6, y0 + bh); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x0 + bw + 6, y0); ctx.lineTo(x0 + bw + 12, y0); ctx.lineTo(x0 + bw + 12, y0 + bh); ctx.lineTo(x0 + bw + 6, y0 + bh); ctx.stroke(); ctx.restore();
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        const x = x0 + c * cell, y = y0 + r * cell; const h = hl && hl(r, c);
        if (h) H.rect(ctx, x + 2, y + 2, cell - 4, cell - 4, { fill: H.hexA(h === 2 ? T.accent2 : color, h === 2 ? .35 : .22), stroke: h === 2 ? T.accent2 : color, lw: 2, r: 6 });
        const v = M[r][c];
        H.text(ctx, v === null ? '·' : String(v), x + cell / 2, y + cell / 2, { align: 'center', base: 'middle', size: 16, mono: true, color: v === null ? T.muted : T.text, weight: h ? 700 : 400 });
      }
    }
    function render(st) {
      const T = H.theme(); H.clear(cv);
      const gap = 56, totalW = n * cell + p * cell + p * cell + gap * 2 + 60;
      if (st.type === 'done') C[st.i][st.j] = A[st.i].reduce((acc, a, k) => acc + a * B[k][st.j], 0);
      let x = (cv.w - totalW) / 2 + 14, y = 110;
      const ax = x, bx = ax + n * cell + gap, cx = bx + p * cell + gap;
      const hlA = st.type === 'term' || st.type === 'done' ? (r, c) => r === st.i ? (st.type === 'term' && c === st.k ? 2 : 1) : 0 : null;
      const hlB = st.type === 'term' || st.type === 'done' ? (r, c) => c === st.j ? (st.type === 'term' && r === st.k ? 2 : 1) : 0 : null;
      const hlC = st.type === 'term' || st.type === 'done' ? (r, c) => (r === st.i && c === st.j ? 2 : 0) : null;
      drawMatrix(A, ax, y, 'A', hlA, T.accent, m, n, `${m} × ${n}`);
      H.text(ctx, '×', bx - gap / 2, y + m * cell / 2, { align: 'center', base: 'middle', size: 22 });
      drawMatrix(B, bx, y, 'B', hlB, T.green, n, p, `${n} × ${p}`);
      H.text(ctx, '=', cx - gap / 2, y + m * cell / 2, { align: 'center', base: 'middle', size: 22 });
      drawMatrix(C, cx, y, 'C = A·B', hlC, T.accent2, m, p, `${m} × ${p}`);
      // formula strip
      const fy = y + Math.max(m, n) * cell + 46;
      if (st.type === 'term' || st.type === 'done') {
        const terms = A[st.i].map((a, k) => ({ a, b: B[k][st.j], k }));
        const upto = st.type === 'done' ? n : st.k + 1;
        let s = `c[${st.i + 1},${st.j + 1}] = ` + terms.slice(0, upto).map(t => `(${t.a})(${t.b})`).join(' + ');
        const partial = terms.slice(0, upto).reduce((acc, t) => acc + t.a * t.b, 0);
        if (st.type === 'done') s += ` = ${partial}`; else s += `  → running sum ${partial}`;
        H.text(ctx, s, cv.w / 2, fy, { align: 'center', mono: true, size: 14, color: T.text });
      } else if (st.type === 'intro') {
        H.text(ctx, `Inner dimensions must match: (${m}×${n})·(${n}×${p}) → result is ${m}×${p}`, cv.w / 2, fy, { align: 'center', size: 14, color: T.muted });
      } else {
        H.text(ctx, `Done: ${m * p} cells, each a dot product of length ${n} → ${m * n * p} multiplications total`, cv.w / 2, fy, { align: 'center', size: 14, color: T.muted });
      }
    }
    function explain(st) {
      if (st.type === 'intro') L.explain(`To multiply <b>A</b> (${m}×${n}) by <b>B</b> (${n}×${p}), the number of <i>columns</i> of A must equal the number of <i>rows</i> of B. Each entry of the result is the <b>dot product</b> of one row of A with one column of B. Press <b>Next</b> to compute the first term.`, 'Setup');
      else if (st.type === 'term') L.explain(`Row <b>${st.i + 1}</b> of A meets column <b>${st.j + 1}</b> of B. Term ${st.k + 1} of ${n}: multiply the ${st.k + 1}-th element of the row by the ${st.k + 1}-th element of the column, <code>${A[st.i][st.k]} × ${B[st.k][st.j]} = ${A[st.i][st.k] * B[st.k][st.j]}</code>, and add it to the running sum.`, `Cell (${st.i + 1},${st.j + 1}) · term ${st.k + 1}/${n}`);
      else if (st.type === 'done') L.explain(`All ${n} products are summed: <b>c<sub>${st.i + 1}${st.j + 1}</sub> = ${C[st.i][st.j]}</b>. In a transformer this exact operation computes attention scores (Q·K<sup>T</sup>) and every linear layer (X·W).`, `Cell (${st.i + 1},${st.j + 1}) complete`);
      else L.explain(`C is complete. Cost grows as m·n·p — this is why GPUs, which do thousands of multiply-adds in parallel, matter so much for deep learning.`, 'Finished');
    }
    let S;
    function rebuild() { newMatrices(); S.go(0); }
    L.number('rows of A (m)', 1, 4, m, v => { m = v; rebuild(); });
    L.number('cols of A = rows of B (n)', 1, 4, n, v => { n = v; rebuild(); });
    L.number('cols of B (p)', 1, 4, p, v => { p = v; rebuild(); });
    L.button('🎲 New values', () => rebuild(), 'btn secondary');
    S = H.stepper(L, steps, st => { if (st.type === 'intro') C = Array.from({ length: m }, () => Array(p).fill(null)); render(st); explain(st); }, { interval: 900 });
    L.setLegend([[H.theme().accent, 'row of A in use'], [H.theme().green, 'column of B in use'], [H.theme().accent2, 'current term / target cell']]);
    L.setNotes(`<p>Ask students to predict the result's shape <i>before</i> pressing Next. Then ask what happens if n of A ≠ rows of B (undefined). Link to the lesson: Q·K<sup>T</sup> is (tokens × d<sub>k</sub>)·(d<sub>k</sub> × tokens) → a tokens × tokens score matrix.</p><p>Convention follows standard textbooks (Strang; Khan Academy): highlight the row and column, show the running sum, fill the cell.</p>`);
    document.addEventListener('cs-theme', S.refresh);
    return { destroy() { S.destroy(); document.removeEventListener('cs-theme', S.refresh); } };
  }
});
