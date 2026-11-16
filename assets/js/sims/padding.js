/* Padding and attention masks: three texts of different lengths become one rectangular input batch.
   Follows Figure 2-3 of Tunstall, von Werra & Wolf, Natural Language Processing with Transformers (Chapter 2). */
Sims.register('padding', {
  title: 'Padding and attention masks: building the input batch',
  kind: 'Tokenization · Week 2',
  short: 'Three texts of different lengths get [CLS] and [SEP], are padded with zeros to the longest one, receive an attention mask, and are stacked into the input batch.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    // The token IDs of the textbook figure. 101 = [CLS], 102 = [SEP], 0 = [PAD]; the other numbers are placeholders.
    const texts = [[23, 74, 2, 67], [14, 66, 53, 7, 87, 14, 37, 31, 17, 9, 21], [91, 20, 15, 98, 36, 81, 85, 23]];
    const W = Math.max(...texts.map(t => t.length)) + 2;            // 13: the longest text with its two special tokens
    const TIP = {
      cls: '[CLS], ID 101. The classification token. The tokenizer puts it at the start of every text. Later in this lesson, the vector the model produces at this position stands for the whole text.',
      sep: '[SEP], ID 102. The separator token. It marks the end of the text (and the border between two texts when a model receives a pair).',
      pad: '[PAD], ID 0. The padding token. It carries no meaning: it only fills the row up to the length of the longest text.',
      id: n => `Token ID ${n}: the number of one word or word-piece in the vocabulary. The IDs in this picture are small placeholders; real DistilBERT IDs run from 0 to 30,521.`,
      gap: 'An empty position. This text is shorter than the longest one, so the row cannot yet be part of a rectangular matrix.',
      m1: 'Mask value 1: a real token (a word-piece, [CLS] or [SEP]). The model pays attention to this position.',
      m0: 'Mask value 0: a padding position. The model ignores it, so the zeros added for padding cannot change the result.',
      ids: 'input_ids: the text as a sequence of integers, one per token. This is the first thing the tokenizer returns and the first thing the model reads.',
      mask: 'attention_mask: one number per position of input_ids. 1 means "real token, use it"; 0 means "padding, ignore it". The tokenizer returns it together with input_ids.',
      text: n => `Text ${n}: one text of the batch (in this lesson, one tweet) after the tokenizer has turned it into token IDs.`,
      batch: `Input batch: the rows stacked into one matrix with ${texts.length} rows (texts) and ${W} columns (positions). The model processes all rows at once, which is why they must have the same length.`,
      len: 'The number of positions in this row. A matrix needs the same number in every row.'
    };
    const SPECIAL = [['[PAD]', 0, 'Padding token. Fills short texts up to the common length. Used in this picture.'],
                     ['[UNK]', 100, 'Unknown token. Stands for a piece of text that cannot be built from the vocabulary at all. Rare with subword tokenization, and not used in this picture.'],
                     ['[CLS]', 101, 'Classification token. Added at the start of every text. Used in this picture.'],
                     ['[SEP]', 102, 'Separator token. Added at the end of every text. Used in this picture.'],
                     ['[MASK]', 103, 'Mask token. Hides a word during pre-training so that the model learns to predict it. Not used when you classify text, and not in this picture.']];
    const steps = [
      { k: 'raw', label: 'Step 1 · Token IDs', html: 'The tokenizer has turned three texts into token IDs. They have <b>4</b>, <b>11</b> and <b>8</b> tokens: texts are not the same length. Move the pointer over any number or label to see what it is.' },
      { k: 'special', label: 'Step 2 · Special tokens', html: 'The tokenizer adds <b>101</b> (<code>[CLS]</code>) at the start and <b>102</b> (<code>[SEP]</code>) at the end of each text. The lengths are now 6, 13 and 10.' },
      { k: 'longest', label: 'Step 3 · The problem', html: 'A model receives a batch as one <b>matrix</b>, and a matrix must be rectangular. Text 2 is the longest, with <b>13</b> positions. Text 1 is 7 positions short and Text 3 is 3 short (dashed boxes).' },
      { k: 'pad1', label: 'Step 4 · Pad Text 1', html: '<code>padding=True</code> fills Text 1 with seven <b>0</b>s. Zero is the ID of <code>[PAD]</code>. The zeros come <i>after</i> <code>[SEP]</code>: they are not part of the text.' },
      { k: 'pad3', label: 'Step 5 · Pad Text 3', html: 'Text 3 gets three <b>0</b>s. All three rows now have 13 positions. Text 2 needs no padding: it set the length.' },
      { k: 'mask1', label: 'Step 6 · Attention mask, Text 1', html: 'The model must not treat the zeros as words. So the tokenizer writes a second row, the <b>attention mask</b>: <b>1</b> under every real token, <b>0</b> under every padding position. Text 1: six 1s, seven 0s.' },
      { k: 'mask2', label: 'Step 7 · Attention mask, Text 2', html: 'Text 2 has no padding, so its mask is thirteen <b>1</b>s.' },
      { k: 'mask3', label: 'Step 8 · Attention mask, Text 3', html: 'Text 3: ten <b>1</b>s and three <b>0</b>s. Note that <code>[CLS]</code> and <code>[SEP]</code> count as real tokens: their mask value is 1.' },
      { k: 'stack', label: 'Step 9 · Stack into matrices', html: 'The rows are stacked. The result is two matrices of the same shape, <b>3 × 13</b>: <code>input_ids</code> and <code>attention_mask</code>. This pair is the <b>input batch</b>, and it is exactly what <code>tokenize(batch)</code> returns in the lesson.' },
      { k: 'model', label: 'Step 10 · What the model does', html: 'Inside the model, every position with mask 0 is left out of the attention calculation (faded here). The prediction for Text 1 is therefore the same as if it had been sent alone, without padding.' }
    ];
    const order = steps.map(s => s.k), at = k => order.indexOf(k);

    L.stage.style.display = 'block';
    L.stage.innerHTML = `<div class="pad-sim">
      <table class="pad-special"><tr><th>Special token</th>${SPECIAL.map(s => `<td data-tip="${s[2]}" data-tip-pos="over" data-sp="${s[1]}"><code>${s[0]}</code></td>`).join('')}</tr>
        <tr><th>ID</th>${SPECIAL.map(s => `<td data-tip="${s[2]}" data-tip-pos="over" data-sp="${s[1]}">${s[1]}</td>`).join('')}</tr></table>
      <div class="pad-blocks"><div class="pad-block" data-b="ids"></div><div class="pad-block" data-b="mask"></div></div></div>`;
    const bIds = L.stage.querySelector('[data-b="ids"]'), bMask = L.stage.querySelector('[data-b="mask"]');
    const cell = (cls, txt, tip) => `<span class="pad-cell ${cls}" data-tip="${tip}" data-tip-pos="over">${txt}</span>`;

    function render(st) {
      const n = at(st.k), stacked = n >= at('stack'), fade = st.k === 'model';
      const padded = i => i === 1 || (i === 0 && n >= at('pad1')) || (i === 2 && n >= at('pad3'));
      const masked = i => n >= at('mask' + (i + 1));
      const hot = { pad1: 0, pad3: 2, mask1: 0, mask2: 1, mask3: 2 }[st.k];
      let ids = '', mask = '';
      texts.forEach((t, i) => {
        const special = n >= at('special'), len = t.length + (special ? 2 : 0);
        let r = '';
        if (special) r += cell('cls' + (st.k === 'special' ? ' new' : ''), 101, TIP.cls);
        r += t.map(v => cell('tok', v, TIP.id(v))).join('');
        if (special) r += cell('sep' + (st.k === 'special' ? ' new' : ''), 102, TIP.sep);
        for (let j = len; j < W; j++) {
          if (padded(i)) r += cell('pad' + (hot === i && st.k.startsWith('pad') ? ' new' : '') + (fade ? ' faded' : ''), 0, TIP.pad);
          else if (n >= at('longest')) r += cell('gap', '', TIP.gap);
        }
        const count = padded(i) ? W : len;
        ids += `<div class="pad-row${st.k === 'longest' && i === 1 ? ' longest' : ''}"><span class="pad-name" data-tip="${TIP.text(i + 1)}" data-tip-pos="over">${stacked ? '' : 'Text ' + (i + 1)}</span>${r}` +
               `<span class="pad-len" data-tip="${TIP.len}" data-tip-pos="over">${stacked ? '' : count}</span></div>`;
        let m = '';
        if (masked(i)) for (let j = 0; j < W; j++) { const real = j < t.length + 2; m += cell((real ? 'm1' : 'm0' + (fade ? ' faded' : '')) + (hot === i && st.k.startsWith('mask') ? ' new' : ''), real ? 1 : 0, real ? TIP.m1 : TIP.m0); }
        mask += `<div class="pad-row"><span class="pad-name" data-tip="${TIP.text(i + 1)}" data-tip-pos="over">${stacked ? '' : 'Text ' + (i + 1)}</span>${m}</div>`;
      });
      const head = (name, tip, shape) => `<div class="pad-head"><span data-tip="${tip}" data-tip-pos="over">${name}</span>${stacked ? ` <span class="pad-shape" data-tip="${TIP.batch}" data-tip-pos="over">input batch · ${shape}</span>` : ''}</div>`;
      bIds.className = 'pad-block' + (stacked ? ' stacked' : '');
      bIds.innerHTML = head('Input IDs', TIP.ids, `${texts.length} × ${W}`) + `<div class="pad-rows">${ids}</div>`;
      bMask.className = 'pad-block' + (stacked ? ' stacked' : '');
      bMask.style.visibility = n >= at('mask1') ? 'visible' : 'hidden';
      bMask.innerHTML = head('Attention masks', TIP.mask, `${texts.length} × ${W}`) + `<div class="pad-rows">${mask}</div>`;
      const used = { 0: n >= at('pad1'), 101: n >= at('special'), 102: n >= at('special') };
      L.stage.querySelectorAll('[data-sp]').forEach(td => td.classList.toggle('on', !!used[td.dataset.sp]));
      L.explain(st.html, st.label);
    }
    const S = H.stepper(L, () => steps, render, { interval: 2600 });
    L.setLegend([['#f7941d', 'token ID (101 = [CLS], 102 = [SEP])'], ['#c9ccd1', '0 = [PAD] (padding)'], ['#5aa9de', 'mask 1: real token'], ['#c9302c', 'mask 0: ignore']]);
    L.setNotes(`<p>The picture reproduces Figure 2-3 of the textbook (Chapter 2, "Tokenizing the Whole Dataset"), with its numbers.</p>
      <p>Points worth saying aloud: (1) padding goes to the longest text <i>of the batch</i>, not of the dataset, which is why the lesson uses <code>batch_size=None</code> to pad the whole split to one length; (2) <code>truncation=True</code> is the opposite operation, cutting texts longer than the model's limit of 512 tokens; (3) the mask is needed because 0 is an ordinary row of the embedding table: without the mask the model would read seven "words" at the end of Text 1.</p>
      <p>Question for the class: what would the masks look like if all three texts had the same length?</p>`);
    return { destroy() { S.destroy(); } };
  }
});
