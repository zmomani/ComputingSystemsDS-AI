/* Retrieval-Augmented Generation: chunk → embed → store → embed query → nearest-k → prompt. 2-D embedding space. */
Sims.register('rag', {
  title: 'RAG pipeline: chunks, embeddings, vector search, grounded prompt',
  kind: 'LangChain · Week 4',
  short: 'Documents become chunks, chunks become points in embedding space; a query lands among them and the k nearest are stuffed into the prompt.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    // Toy corpus: chunks with hand-placed 2-D "embeddings" clustered by topic
    const chunks = [
      { t: 'Transformers use self-attention to relate all tokens in a sequence.', topic: 'nlp', x: 1.8, y: 2.2 },
      { t: 'BERT is an encoder-only model pre-trained with masked language modelling.', topic: 'nlp', x: 2.4, y: 1.6 },
      { t: 'GPT models are decoder-only and generate text left to right.', topic: 'nlp', x: 2.9, y: 2.3 },
      { t: 'Petra is a historical city in southern Jordan carved into rose-coloured rock.', topic: 'jordan', x: -2.3, y: 1.9 },
      { t: 'Amman is the capital of Jordan and its largest city.', topic: 'jordan', x: -2.8, y: 1.2 },
      { t: 'The Dead Sea is the lowest point on Earth, on the Jordan–Israel border.', topic: 'jordan', x: -1.9, y: 0.9 },
      { t: 'ChromaDB stores embeddings and returns the nearest vectors for a query.', topic: 'rag', x: 0.6, y: -1.8 },
      { t: 'A text splitter cuts documents into overlapping chunks before embedding.', topic: 'rag', x: 1.3, y: -2.4 },
      { t: 'Cosine similarity compares the direction of two embedding vectors.', topic: 'rag', x: 0.2, y: -2.6 },
      { t: 'LangChain chains compose a prompt template, an LLM and an output parser.', topic: 'lc', x: -0.9, y: -1.1 },
      { t: 'A retriever wraps a vector store and exposes get_relevant_documents().', topic: 'lc', x: 0.0, y: -0.6 },
      { t: 'Ollama runs open models like Llama locally through a simple HTTP API.', topic: 'lc', x: -1.6, y: -1.9 }
    ];
    const queries = [
      { q: 'How does BERT differ from GPT?', x: 2.6, y: 2.0 },
      { q: 'What is the capital of Jordan?', x: -2.6, y: 1.5 },
      { q: 'How does a vector store find relevant chunks?', x: 0.5, y: -2.1 },
      { q: 'What does a LangChain chain contain?', x: -0.7, y: -1.0 }
    ];
    let qi = 0, k = 3;
    const cv = H.canvas(680, 400); L.stage.appendChild(cv); const ctx = cv.ctx;
    const steps = ['load', 'split', 'embed', 'query', 'search', 'prompt'];
    const topicColor = () => ({ nlp: H.theme().accent, jordan: H.theme().accent2, rag: H.theme().green, lc: H.theme().purple });
    const cos = (a, b) => (a.x * b.x + a.y * b.y) / (Math.hypot(a.x, a.y) * Math.hypot(b.x, b.y));
    function nearest() { const q = queries[qi]; return chunks.map((c, i) => ({ i, s: cos(c, q) })).sort((a, b) => b.s - a.s).slice(0, k); }
    function render(st) {
      const T = H.theme(); H.clear(cv); const tc = topicColor();
      // pipeline strip
      const names = ['Load docs', 'Split into chunks', 'Embed + store', 'Embed query', 'Top-k search', 'Prompt → LLM'];
      names.forEach((nm, i) => { const x = 14 + i * 110, on = steps.indexOf(st) >= i, cur = steps[i] === st; H.rect(ctx, x, 10, 102, 28, { fill: cur ? T.accent2 : on ? H.hexA(T.accent, .25) : T.bg, stroke: cur ? T.accent2 : T.border, r: 6 }); H.text(ctx, nm, x + 51, 28, { align: 'center', size: 11, weight: cur ? 700 : 500, color: cur ? '#fff' : T.text }); if (i < 5) H.arrow(ctx, x + 103, 24, x + 109, 24, { color: T.muted, head: 5 }); });
      const ox = 240, oy = 230, u = 44; // embedding plane
      const px = p => ({ x: ox + p.x * u, y: oy - p.y * u });
      if (st === 'load' || st === 'split') {
        // document icons
        [['transformers.md', 'nlp'], ['jordan_guide.pdf', 'jordan'], ['langchain_docs.html', 'rag'], ['tooling.txt', 'lc']].forEach((d, i) => {
          const x = 40 + i * 160, y = 70; H.rect(ctx, x, y, 120, 150, { fill: T.bg, stroke: T.border, r: 6 }); H.text(ctx, d[0], x + 60, y + 18, { align: 'center', size: 11, weight: 700 });
          const mine = chunks.filter(c => c.topic === d[1]);
          mine.forEach((c, j) => { const cy = y + 32 + j * 38; if (st === 'split') H.rect(ctx, x + 8, cy - 4, 104, 32, { fill: H.hexA(tc[c.topic], .18), stroke: tc[c.topic], r: 4 }); for (let l = 0; l < 3; l++) H.rect(ctx, x + 14, cy + l * 8, 92 - (l === 2 ? 30 : 0), 3, { fill: H.hexA(T.muted, .5) }); });
        });
        H.text(ctx, st === 'load' ? '4 documents loaded (PDF, markdown, HTML, text) via LangChain document loaders' : `RecursiveCharacterTextSplitter → ${chunks.length} chunks with overlap, metadata keeps the source file`, 340, 255, { align: 'center', size: 12, color: T.muted });
      } else {
        // axes
        H.arrow(ctx, 40, oy, 440, oy, { color: T.border, lw: 1 }); H.arrow(ctx, ox, 390, ox, 60, { color: T.border, lw: 1 });
        H.text(ctx, 'embedding space (2 of 384 dims)', 440, oy + 14, { align: 'right', size: 10, color: T.muted });
        const near = (st === 'search' || st === 'prompt') ? nearest() : [];
        const nearSet = new Set(near.map(n => n.i));
        chunks.forEach((c, i) => { const p = px(c); ctx.save(); ctx.fillStyle = tc[c.topic]; ctx.globalAlpha = near.length && !nearSet.has(i) ? .35 : 1; ctx.beginPath(); ctx.arc(p.x, p.y, 7, 0, 7); ctx.fill(); ctx.restore(); H.text(ctx, String(i + 1), p.x, p.y + 4, { align: 'center', size: 9, color: '#fff', weight: 700 }); });
        if (st !== 'embed') { const q = px(queries[qi]); ctx.save(); ctx.fillStyle = T.yellow; ctx.strokeStyle = T.text; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(q.x, q.y - 11); ctx.lineTo(q.x + 10, q.y + 7); ctx.lineTo(q.x - 10, q.y + 7); ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.restore(); H.text(ctx, 'query', q.x + 13, q.y + 4, { size: 11, weight: 700 }); }
        near.forEach(n => { const a = px(queries[qi]), b = px(chunks[n.i]); ctx.save(); ctx.strokeStyle = T.yellow; ctx.setLineDash([4, 3]); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); ctx.restore(); });
        // right panel: retrieved chunks / prompt
        const rx = 460; H.rect(ctx, rx, 60, 210, 330, { fill: T.bg, stroke: T.border, r: 8 });
        if (st === 'embed') { H.text(ctx, 'Vector store (ChromaDB)', rx + 10, 80, { size: 12, weight: 700 }); chunks.forEach((c, i) => H.text(ctx, `${String(i + 1).padStart(2)}  [${H.fmt(c.x, 1)}, ${H.fmt(c.y, 1)}, …]`, rx + 10, 102 + i * 20, { size: 10, mono: true, color: tc[c.topic] })); }
        else if (st === 'query') { H.text(ctx, 'Query', rx + 10, 80, { size: 12, weight: 700 }); wrap(queries[qi].q, rx + 10, 100, 190, 12); H.text(ctx, `→ embed → [${H.fmt(queries[qi].x, 1)}, ${H.fmt(queries[qi].y, 1)}, …]`, rx + 10, 160, { size: 10, mono: true, color: T.muted }); }
        else { H.text(ctx, st === 'search' ? `Top-${k} by cosine similarity` : 'Prompt sent to the LLM', rx + 10, 80, { size: 12, weight: 700 }); let y = 100;
          if (st === 'prompt') { y = wrap('Answer using only the context below.\n\nContext:', rx + 10, y, 190, 10, T.muted) + 4; }
          near.forEach(n => { H.text(ctx, `#${n.i + 1}  cos = ${n.s.toFixed(3)}`, rx + 10, y, { size: 10, mono: true, color: tc[chunks[n.i].topic], weight: 700 }); y = wrap(chunks[n.i].t, rx + 10, y + 14, 190, 10) + 8; });
          if (st === 'prompt') { y = wrap('Question: ' + queries[qi].q, rx + 10, y + 2, 190, 10, T.muted); }
        }
      }
      function wrap(txt, x, y, w, size, color) { ctx.font = `${size}px ${T.body}`; const words = txt.split(/(\s+)/); let line = ''; txt.split('\n').forEach(par => { par.split(' ').forEach(wd => { const test = line ? line + ' ' + wd : wd; if (ctx.measureText(test).width > w) { H.text(ctx, line, x, y, { size, color }); y += size + 4; line = wd; } else line = test; }); H.text(ctx, line, x, y, { size, color }); y += size + 4; line = ''; }); return y; }
      const ex = {
        load: ['1 · Load', 'Document loaders turn files of any format into <code>Document</code> objects (page_content + metadata). Nothing is lost yet, but whole documents are too long to embed or to fit in a prompt.'],
        split: ['2 · Split', 'A text splitter cuts each document into chunks of ~500–1000 characters with some overlap so a sentence cut in half still appears whole in one chunk. Chunk size is the first RAG hyper-parameter students tune.'],
        embed: ['3 · Embed & store', 'An embedding model (e.g. <code>all-MiniLM-L6-v2</code>, 384 dims) maps each chunk to a vector; semantically similar chunks land near each other. Vectors + text + metadata are written to ChromaDB. Notice the four topic clusters.'],
        query: ['4 · Embed the query', 'The <i>same</i> embedding model maps the question into the same space. Because the model places questions near the passages that answer them, geometry does the retrieval.'],
        search: ['5 · Nearest neighbours', `Cosine similarity (see the vectors simulation) ranks every chunk; the top-k = ${k} are returned with their scores and sources. Chunks from other clusters are ignored even if they share words.`],
        prompt: ['6 · Grounded generation', 'The retrieved chunks are pasted into a prompt template with the question. The LLM answers from the provided context, which cuts hallucination and lets it use private or recent documents it was never trained on.']
      }[st];
      L.explain(ex[1], ex[0]);
    }
    let S;
    L.select('Query', queries.map((q, i) => [i, q.q]), qi, v => { qi = +v; S.refresh(); });
    L.range('k (chunks retrieved)', 1, 6, 1, k, v => { k = v; S.refresh(); });
    S = H.stepper(L, () => steps, render, { interval: 1800 });
    L.setLegend([[H.theme().accent, 'transformers chunks'], [H.theme().accent2, 'Jordan chunks'], [H.theme().green, 'RAG / vector-store chunks'], [H.theme().purple, 'LangChain / tooling chunks'], [H.theme().yellow, 'query & retrieved links']]);
    L.setNotes(`<p>Set k = 1 with the Jordan question and ask whether the single chunk is enough. Then raise k to 6 and point out that off-topic chunks now enter the prompt — more context is not always better (noise, cost, "lost in the middle"). Mention re-ranking and metadata filtering as the next tools.</p><p>Maps to the Week 4 milestone: loaders → splitter → ChromaDB → <code>RetrievalQA</code>/LCEL chain.</p>`);
    document.addEventListener('cs-theme', S.refresh);
    return { destroy() { S.destroy(); document.removeEventListener('cs-theme', S.refresh); } };
  }
});
