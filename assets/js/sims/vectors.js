/* Vectors in the plane: drag two vectors, see addition, dot product, angle and cosine similarity. */
Sims.register('vectors', {
  title: 'Vectors in the plane: dot product & cosine similarity',
  kind: 'Linear algebra · Weeks 1, 2, 4',
  short: 'Drag the arrow tips. See u+v, u·v, the angle between them and cos θ — the measure behind embeddings and vector search.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    const W = 560, Hh = 420, ox = W / 2, oy = Hh / 2, unit = 36; // pixels per unit
    let u = { x: 3, y: 1 }, v = { x: 1, y: 3 };
    let showSum = true, showProj = true, snap = true;
    const cv = H.canvas(W, Hh); L.stage.appendChild(cv); const ctx = cv.ctx;
    const toPx = p => ({ x: ox + p.x * unit, y: oy - p.y * unit });
    const toU = px => ({ x: (px.x - ox) / unit, y: (oy - px.y) / unit });

    function draw() {
      const T = H.theme(); H.clear(cv);
      // grid
      ctx.save(); ctx.strokeStyle = H.hexA(T.text.startsWith('#') ? T.text : '#888', .08); ctx.lineWidth = 1;
      for (let x = ox % unit; x < W; x += unit) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, Hh); ctx.stroke(); }
      for (let y = oy % unit; y < Hh; y += unit) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
      ctx.restore();
      H.arrow(ctx, 10, oy, W - 10, oy, { color: T.muted, lw: 1.2 }); H.arrow(ctx, ox, Hh - 10, ox, 10, { color: T.muted, lw: 1.2 });
      H.text(ctx, 'x', W - 16, oy + 16, { color: T.muted, size: 12 }); H.text(ctx, 'y', ox + 8, 18, { color: T.muted, size: 12 });
      for (let i = -7; i <= 7; i++) if (i) { H.text(ctx, i, ox + i * unit, oy + 14, { color: T.muted, size: 10, align: 'center' }); H.text(ctx, i, ox - 6, oy - i * unit + 3, { color: T.muted, size: 10, align: 'right' }); }
      const pu = toPx(u), pv = toPx(v), o = { x: ox, y: oy };
      if (showSum) { // parallelogram + sum vector
        const s = { x: u.x + v.x, y: u.y + v.y }, ps = toPx(s);
        ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = T.muted; ctx.beginPath(); ctx.moveTo(pu.x, pu.y); ctx.lineTo(ps.x, ps.y); ctx.lineTo(pv.x, pv.y); ctx.stroke(); ctx.restore();
        H.arrow(ctx, o.x, o.y, ps.x, ps.y, { color: T.purple, lw: 2.2 });
        H.text(ctx, `u+v = (${H.fmt(s.x, 1)}, ${H.fmt(s.y, 1)})`, ps.x + 8, ps.y - 6, { color: T.purple, size: 12, weight: 700 });
      }
      const dot = u.x * v.x + u.y * v.y, nu = Math.hypot(u.x, u.y), nv = Math.hypot(v.x, v.y);
      const cos = nu && nv ? dot / (nu * nv) : 0, ang = Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI;
      if (showProj && nv > 0) { // projection of u onto v
        const k = dot / (nv * nv), pr = { x: v.x * k, y: v.y * k }, pp = toPx(pr);
        ctx.save(); ctx.setLineDash([3, 3]); ctx.strokeStyle = T.accent2; ctx.beginPath(); ctx.moveTo(pu.x, pu.y); ctx.lineTo(pp.x, pp.y); ctx.stroke(); ctx.restore();
        ctx.save(); ctx.strokeStyle = T.accent2; ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(pp.x, pp.y); ctx.stroke(); ctx.restore();
        H.text(ctx, `proj = ${H.fmt(k * nv)}`, pp.x + 6, pp.y + 16, { color: T.accent2, size: 11 });
      }
      // angle arc
      const a1 = Math.atan2(-u.y, u.x), a2 = Math.atan2(-v.y, v.x);
      ctx.save(); ctx.strokeStyle = T.yellow; ctx.lineWidth = 2; ctx.beginPath();
      let s0 = a1, e0 = a2; if (((e0 - s0) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) > Math.PI) [s0, e0] = [e0, s0];
      ctx.arc(ox, oy, 30, s0, e0); ctx.stroke(); ctx.restore();
      H.arrow(ctx, o.x, o.y, pu.x, pu.y, { color: T.accent, lw: 3 }); H.arrow(ctx, o.x, o.y, pv.x, pv.y, { color: T.green, lw: 3 });
      H.text(ctx, `u = (${H.fmt(u.x, 1)}, ${H.fmt(u.y, 1)})`, pu.x + 8, pu.y + 4, { color: T.accent, size: 13, weight: 700 });
      H.text(ctx, `v = (${H.fmt(v.x, 1)}, ${H.fmt(v.y, 1)})`, pv.x + 8, pv.y + 4, { color: T.green, size: 13, weight: 700 });
      // handles
      [pu, pv].forEach((p, i) => { ctx.save(); ctx.fillStyle = i ? T.green : T.accent; ctx.beginPath(); ctx.arc(p.x, p.y, 7, 0, 7); ctx.fill(); ctx.strokeStyle = T.bg; ctx.lineWidth = 2; ctx.stroke(); ctx.restore(); });

      const sim = cos > .7 ? 'similar (point the same way)' : cos < -.7 ? 'opposite' : Math.abs(cos) < .15 ? 'nearly orthogonal (unrelated)' : 'somewhat related';
      L.explain(`<b>u · v</b> = (${H.fmt(u.x, 1)})(${H.fmt(v.x, 1)}) + (${H.fmt(u.y, 1)})(${H.fmt(v.y, 1)}) = <b>${H.fmt(dot)}</b><br>
        |u| = ${H.fmt(nu)}, |v| = ${H.fmt(nv)}, θ = <b>${H.fmt(ang, 1)}°</b><br>
        <b>cos θ = u·v / (|u||v|) = ${H.fmt(cos, 3)}</b> → the vectors are <i>${sim}</i>.<br>
        <span style="color:var(--muted);font-size:.85em">In vector search, documents whose embeddings have cos θ close to 1 are returned for a query; in attention, q·k scores play the same role.</span>`, 'Live computation');
    }
    // dragging
    let drag = null;
    const pos = e => { const r = cv.getBoundingClientRect(); const t = e.touches ? e.touches[0] : e; return { x: (t.clientX - r.left) * (W / r.width), y: (t.clientY - r.top) * (Hh / r.height) }; };
    const down = e => { const p = pos(e); const du = Math.hypot(p.x - toPx(u).x, p.y - toPx(u).y), dv = Math.hypot(p.x - toPx(v).x, p.y - toPx(v).y); if (Math.min(du, dv) < 18) { drag = du < dv ? 'u' : 'v'; e.preventDefault(); } };
    const move = e => { if (!drag) return; let q = toU(pos(e)); if (snap) q = { x: Math.round(q.x * 2) / 2, y: Math.round(q.y * 2) / 2 }; q.x = Math.max(-7, Math.min(7, q.x)); q.y = Math.max(-5.5, Math.min(5.5, q.y)); if (drag === 'u') u = q; else v = q; draw(); };
    const up = () => drag = null;
    cv.addEventListener('mousedown', down); cv.addEventListener('mousemove', move); window.addEventListener('mouseup', up);
    cv.addEventListener('touchstart', down, { passive: false }); cv.addEventListener('touchmove', move, { passive: true }); window.addEventListener('touchend', up);
    cv.style.cursor = 'grab';

    const presets = [['same', 'Same direction (cos ≈ 1)', [3, 1], [4.5, 1.5]], ['orth', 'Orthogonal (cos = 0)', [3, 1], [-1, 3]], ['opp', 'Opposite (cos ≈ −1)', [3, 1], [-3, -1]], ['default', 'Default', [3, 1], [1, 3]]];
    L.select('Preset', presets.map(p => [p[0], p[1]]), 'default', k => { const p = presets.find(x => x[0] === k); u = { x: p[2][0], y: p[2][1] }; v = { x: p[3][0], y: p[3][1] }; draw(); });
    const cb = (label, val, fn) => { const w = document.createElement('label'); w.innerHTML = `<input type="checkbox" ${val ? 'checked' : ''}> ${label}`; w.querySelector('input').addEventListener('change', e => { fn(e.target.checked); draw(); }); L.controls.appendChild(w); };
    cb('show u+v', showSum, x => showSum = x); cb('show projection of u on v', showProj, x => showProj = x); cb('snap to grid', snap, x => snap = x);
    L.setLegend([[H.theme().accent, 'u'], [H.theme().green, 'v'], [H.theme().purple, 'u + v'], [H.theme().accent2, 'projection of u onto v'], [H.theme().yellow, 'angle θ']]);
    L.setNotes(`<p>Drag the tips. Start with the <i>Orthogonal</i> preset and ask: why is the dot product zero? Then show that scaling a vector changes u·v but not cos θ — this is why cosine similarity, not the raw dot product, is used to compare embeddings of different lengths.</p><p>Embeddings live in hundreds of dimensions, but the geometry is identical: the formula cos θ = u·v/(|u||v|) does not care how many components there are.</p>`);
    draw();
    const onTheme = () => draw(); document.addEventListener('cs-theme', onTheme);
    return { destroy() { window.removeEventListener('mouseup', up); window.removeEventListener('touchend', up); document.removeEventListener('cs-theme', onTheme); } };
  }
});
