/* The course notebook: lesson and Jupyter notebook side by side.
 *
 * On GitHub Pages this file does nothing. When the site is served by the Jupyter server of the course codespace
 * (scripts/start-lesson-lab.sh), every lesson gets a "Notebook" tab in its side panel that holds the lesson's real
 * notebook, running on the codespace with PyTorch, Transformers and the other course libraries.
 * Every ▶ Try it and 📓 Open in notebook button on a Python example then sends that example to a cell of the
 * notebook: it selects the cell that already holds the example, or fills a new cell with it.
 */
(function () {
  const where = location.pathname.match(/^(.*)\/static\/(?:index\.html|lessons\/([\w-]+)\.html)$/);
  if (!where) return;                              // the public site: nothing changes
  const base = where[1], slug = where[2];
  const lesson = document.getElementById("lesson"), wb = document.getElementById("workbench");

  if (!slug || !lesson || !wb) {                   // home page inside the codespace: the notebook is already running
    document.querySelectorAll("a.btn.codespaces").forEach((a) => {
      const note = document.createElement("p");
      note.className = "muted small"; note.textContent = "You are in the course notebook. Open any lesson to see its notebook beside it.";
      a.replaceWith(note);
    });
    return;
  }
  const nbPath = `notebooks/${slug}.ipynb`;

  const style = document.createElement("style");
  style.textContent = `
    .wb-nb { grid-template-rows: auto 1fr; min-height: 0; background: #fff; }
    .wb-nb .nb-bar { display: flex; flex-wrap: wrap; align-items: center; gap: .4rem; padding: .4rem .6rem; background: #171d25; border-bottom: 1px solid #263040; }
    .wb-nb .nb-status { flex: 1 1 220px; min-width: 0; font-size: .82rem; color: #c8d3e0; }
    .wb-nb .nb-status b { color: #fff; }
    .wb-nb iframe { width: 100%; height: 100%; border: 0; background: #fff; }
    .runnable.nb-sent { outline: 2px solid var(--accent); outline-offset: 2px; }`;
  document.head.appendChild(style);

  // ---- the Notebook tab and its panel
  const tab = document.createElement("button");
  tab.className = "wb-tab"; tab.dataset.mode = "nb"; tab.textContent = "📓 Notebook";
  wb.querySelector(".wb-tabs").prepend(tab);
  const pane = document.createElement("section");
  pane.className = "wb-mode wb-nb"; pane.dataset.mode = "nb";
  pane.innerHTML = `<div class="nb-bar">
      <span class="nb-status" id="nb-status">This lesson's notebook, running in your codespace.</span>
      <button class="btn primary" id="nb-run" title="Run the selected cell (Shift+Enter inside the notebook)">▶ Run cell</button>
      <button class="btn ghost" id="nb-above" title="Run every cell above the selected one, then the selected one">⏫ Run all above, then this</button>
      <a class="btn ghost" id="nb-tab" target="_blank" rel="noopener" title="Open the notebook in its own browser tab, with the file browser and a terminal">↗ Full tab</a>
    </div><div class="nb-frame"></div>`;
  wb.appendChild(pane);
  const status = pane.querySelector("#nb-status");
  pane.querySelector("#nb-tab").href = `${base}/lab/tree/${nbPath}`;

  function showNotebookTab() {
    lesson.classList.remove("panel-closed");
    wb.querySelectorAll(".wb-tab").forEach((t) => t.classList.toggle("active", t === tab));
    wb.querySelectorAll(".wb-mode").forEach((m) => m.classList.toggle("active", m === pane));
    frame();
  }
  tab.addEventListener("click", showNotebookTab);

  // ---- the notebook itself: JupyterLab's single-document view of this lesson's notebook, same origin as this page
  let iframe = null;
  function frame() {
    if (!iframe) {
      iframe = document.createElement("iframe");
      iframe.title = "Notebook"; iframe.src = `${base}/doc/tree/${nbPath}`;
      pane.querySelector(".nb-frame").replaceWith(iframe);
    }
    return iframe;
  }
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  async function notebook() {                    // waits until JupyterLab has opened the notebook
    frame();
    for (let i = 0; i < 240; i++) {
      try {
        const app = iframe.contentWindow.jupyterapp, panel = app && app.shell.currentWidget;
        const nb = panel && panel.content;
        if (nb && nb.model && nb.model.sharedModel.cells.length && nb.widgets.length) { await panel.context.ready; return { app, nb }; }
      } catch (_) { /* still loading */ }
      await sleep(250);
    }
    throw new Error("The notebook did not open. In the codespace terminal run: bash scripts/start-lesson-lab.sh");
  }

  const norm = (s) => s.replace(/\r/g, "").trim();
  async function sendToCell(code) {
    showNotebookTab();
    status.textContent = "Opening the notebook…";
    const { nb } = await notebook();
    const cells = nb.model.sharedModel.cells, want = norm(code);
    let index = cells.findIndex((c) => {
      if (c.cell_type !== "code") return false;
      const src = norm(c.getSource());
      return src === want || norm(src.replace(/^%%writefile[^\n]*\n/, "")) === want;
    });
    let filled = false;
    if (index < 0) {                             // not in the notebook (edited, removed, or text-only): fill a new cell
      index = Math.min(nb.activeCellIndex + 1, cells.length);
      nb.model.sharedModel.insertCell(index, { cell_type: "code", source: code });
      filled = true;
      await sleep(50);
    }
    nb.deselectAll(); nb.activeCellIndex = index; nb.mode = "command";
    try { await nb.scrollToItem(index, "start"); } catch (_) { /* older JupyterLab: selection alone is enough */ }
    status.innerHTML = filled
      ? `<b>Filled a new cell (number ${index + 1})</b> with this example. Press ▶ Run cell.`
      : `<b>Cell ${index + 1}</b> holds this example. Press ▶ Run cell, or ⏫ if it needs the earlier ones.`;
  }

  async function command(id) {
    try { const { app } = await notebook(); iframe.contentWindow.focus(); await app.commands.execute(id); }
    catch (e) { status.textContent = String(e.message || e); }
  }
  pane.querySelector("#nb-run").addEventListener("click", () => command("notebook:run-cell"));
  pane.querySelector("#nb-above").addEventListener("click", async () => { await command("notebook:run-all-above"); await command("notebook:run-cell"); });

  // ---- every button on a Python example now sends the example to the notebook
  document.addEventListener("click", (e) => {
    const top = e.target.closest && e.target.closest("a.btn.codespaces");
    if (top) { e.preventDefault(); e.stopImmediatePropagation(); showNotebookTab(); return; }   // "Open this week's notebook"
    const btn = e.target.closest && e.target.closest(".runnable .try");
    if (!btn) return;
    const box = btn.closest(".runnable"), src = box.querySelector("textarea.src");
    if (!src || box.dataset.lang === "javascript") return;      // JavaScript examples keep running in the browser editor
    if (btn.matches("[data-colab-link]")) return;               // GPU training still opens in Colab
    e.preventDefault(); e.stopImmediatePropagation();
    document.querySelectorAll(".runnable.nb-sent").forEach((b) => b.classList.remove("nb-sent"));
    box.classList.add("nb-sent");
    sendToCell(src.value).catch((err) => { status.textContent = String(err.message || err); });
  }, true);
  document.querySelectorAll(".runnable:not([data-lang='javascript']) .try:not([data-colab-link])").forEach((b) => {
    b.title = "Takes you to this example's cell in the notebook beside the lesson";
    b.textContent = "📓 To notebook";
  });
  // an example that had two buttons (open + editor) needs only one here
  document.querySelectorAll(".runnable:not([data-lang='javascript']) .local-btns").forEach((g) => {
    if (g.querySelector("a.try:not([data-colab-link])")) g.querySelectorAll("button.try.local").forEach((b) => { b.hidden = true; });
  });
})();
