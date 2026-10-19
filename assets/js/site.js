/* 606363 course site — lesson on the left, a panel on the right with a live Python editor
 * (Pyodide, as in the Programming for Data Science tutorial) or an animation.
 *  ▶ Try it   → opens the panel, loads the example into the editor and runs it
 *  ✎ Open in editor → loads code that needs local libraries (explains why it can't run here)
 *  ▶ Display  → opens the panel on an animation
 */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

  /* ------------------------------------------------ common: highlight, maths, nav, quiz */
  document.addEventListener("DOMContentLoaded", () => {
    if (window.hljs) $$("pre code[class*=language-]").forEach((c) => { if (!c.classList.contains("language-plaintext")) hljs.highlightElement(c); });
    if (window.renderMathInElement) renderMathInElement(document.body, {
      delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }], throwOnError: false });
  });
  const sel = $("#lesson-select");
  if (sel) sel.addEventListener("change", () => { if (sel.value) location.href = sel.value; });

  $$(".quiz .q").forEach((q) => {
    const ans = q.dataset.answer;
    $$(".opt", q).forEach((o) => o.addEventListener("click", () => {
      $$(".opt", q).forEach((x) => x.classList.remove("right", "wrong"));
      o.classList.add(o.dataset.v === ans ? "right" : "wrong");
      $$(".opt", q).find((x) => x.dataset.v === ans).classList.add("right");
      const fb = $(".fb", q); if (fb) fb.style.display = "block";
    }));
  });

  function copyText(text, btn) {
    navigator.clipboard.writeText(text).then(() => { const o = btn.textContent; btn.textContent = "Copied"; setTimeout(() => (btn.textContent = o), 1200); }).catch(() => {});
  }
  $$(".runnable .copy, .codeblock .copy").forEach((b) => b.addEventListener("click", () => {
    const src = b.parentElement.querySelector(".src");
    copyText(src ? src.value : b.parentElement.querySelector("pre").innerText, b);
  }));

  /* ------------------------------------------------ full-screen animation overlay */
  const overlay = $("#sim-overlay");
  let overlayInst = null;
  function openOverlay(id) {
    const sim = window.Sims && Sims.get(id); if (!sim || !overlay) return;
    closeOverlay();
    $("#sim-title").textContent = sim.title; $("#sim-kind").textContent = sim.kind || "Animation";
    const body = $("#sim-body"); body.innerHTML = "";
    overlayInst = Sims.mount(id, body);
    overlay.classList.add("open"); document.body.style.overflow = "hidden";
  }
  function closeOverlay() {
    if (overlayInst && overlayInst.destroy) overlayInst.destroy();
    overlayInst = null;
    if (overlay) { overlay.classList.remove("open"); $("#sim-body").innerHTML = ""; }
    document.body.style.overflow = "";
  }
  if (overlay) {
    $("#sim-close").addEventListener("click", closeOverlay);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) closeOverlay(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeOverlay(); });
  }

  const lesson = $("#lesson");
  if (!lesson) {                        // home page: Display opens the full-screen view
    $$("[data-display]").forEach((b) => b.addEventListener("click", () => openOverlay(b.dataset.display)));
    return;
  }

  /* ------------------------------------------------ panel open / close / modes */
  const wb = $("#workbench");
  function openPanel(mode) {
    lesson.classList.remove("panel-closed");
    if (mode) setMode(mode);
    setTimeout(() => editor && editor.refresh(), 0);
  }
  function closePanel() { lesson.classList.add("panel-closed"); }
  $("#btn-panel").addEventListener("click", () => lesson.classList.contains("panel-closed") ? openPanel() : closePanel());
  $("#btn-close").addEventListener("click", closePanel);
  function setMode(mode) {
    $$(".wb-tab", wb).forEach((t) => t.classList.toggle("active", t.dataset.mode === mode));
    $$(".wb-mode", wb).forEach((m) => m.classList.toggle("active", m.dataset.mode === mode));
    if (mode === "code") setTimeout(() => editor.refresh(), 0);
    if (mode === "anim" && !animInst) showAnim($("#anim-select").value);
  }
  $$(".wb-tab", wb).forEach((t) => t.addEventListener("click", () => setMode(t.dataset.mode)));

  // draggable splitter (as in the template)
  (() => {
    const sp = $("#splitter"); let dragging = false;
    sp.addEventListener("mousedown", (e) => { dragging = true; e.preventDefault(); document.body.style.cursor = "col-resize"; });
    window.addEventListener("mouseup", () => { if (dragging) { dragging = false; document.body.style.cursor = ""; editor.refresh(); if (animInst && animInst.refresh) animInst.refresh(); } });
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      const pct = Math.min(75, Math.max(25, (e.clientX / window.innerWidth) * 100));
      lesson.style.setProperty("--left", pct + "%");
    });
  })();

  /* ------------------------------------------------ animations in the panel */
  const host = $("#anim-host"), animSel = $("#anim-select");
  let animInst = null, animId = null;
  function showAnim(id) {
    const sim = window.Sims && Sims.get(id); if (!sim) return;
    if (animInst && animInst.destroy) animInst.destroy();
    host.innerHTML = `<div class="anim-kind">${sim.kind || "Animation"}</div><h3 class="anim-title">${sim.title}</h3><p class="muted small" style="margin:.1rem 0 .8rem">${sim.short || ""}</p><div class="anim-mount"></div>`;
    animInst = Sims.mount(id, host.querySelector(".anim-mount")) || {};
    animId = id; animSel.value = id;
    $$(".sim-chip").forEach((c) => c.classList.toggle("active", c.dataset.display === id));
  }
  animSel.addEventListener("change", () => showAnim(animSel.value));
  $("#btn-full").addEventListener("click", () => openOverlay(animId || animSel.value));
  $$("[data-display]").forEach((b) => b.addEventListener("click", () => { openPanel("anim"); showAnim(b.dataset.display); }));

  /* ------------------------------------------------ Google Colab links */
  const COLAB = lesson.dataset.colab || "";
  $$("[data-colab-link]").forEach((a) => { if (COLAB) a.href = COLAB; else a.remove(); });
  function vscodeNote(what) {
    const d = document.createElement("div"); d.className = "colab-note";
    d.innerHTML = `<p>${what}</p><p>It starts programs on your own computer (a local model, a server, a web app or a real browser), which neither this page nor Google Colab can do.</p>` +
      `<p><button class="btn ghost" data-copy-editor>Copy the code</button> then paste it into VS Code and run it with the course environment.</p>`;
    el.out.appendChild(d);
    d.querySelector("[data-copy-editor]").addEventListener("click", (e) => copyText(editor.getValue(), e.currentTarget));
  }
  function colabNote(what) {
    const browser = document.getElementById("run-in-your-browser");
    const d = document.createElement("div"); d.className = "colab-note";
    d.innerHTML = `<p>${what}</p><p>` +
      (COLAB ? `<a class="btn colab" href="${COLAB}" target="_blank" rel="noopener"><span class="colab-logo">CO</span> Run it in Google Colab</a> ` : "") +
      (browser ? `<button class="btn ghost" data-goto-browser>Run the in-browser version</button>` : "") +
      `</p><p class="small">In Colab the examples are in the same order as on this page — run the cells from the top so earlier code is defined. Or copy the code and run it in VS Code.</p>`;
    el.out.appendChild(d);
    const g = d.querySelector("[data-goto-browser]");
    if (g) g.addEventListener("click", () => browser.scrollIntoView({ behavior: "smooth", block: "start" }));
  }

  /* ------------------------------------------------ editor + Python (Pyodide) */
  const el = {
    run: $("#btn-run"), reset: $("#btn-reset"), clear: $("#btn-clear"), copy: $("#btn-copy"),
    out: $("#output"), runTime: $("#run-time"), title: $("#wb-title"),
    status: $("#py-status"), statusText: $("#py-status-text"),
  };
  const editor = CodeMirror.fromTextArea($("#editor"), {
    mode: "python", theme: "material-darker", lineNumbers: true, indentUnit: 4,
    matchBrackets: true, autoCloseBrackets: true,
    extraKeys: {
      "Ctrl-Enter": () => runEditor(), "Cmd-Enter": () => runEditor(),
      Tab: (cm) => (cm.somethingSelected() ? cm.indentSelection("add") : cm.replaceSelection("    ", "end")),
    },
  });
  let current = { code: editor.getValue(), local: false, needs: "", setup: "", lang: "python" };

  const PRELUDE = `
import sys, io, base64, builtins
def _cs_repr(obj):
    try:
        import pandas as _pd
        if isinstance(obj, _pd.DataFrame):
            return {"kind": "html", "value": obj.to_html(max_rows=60, max_cols=25, border=0)}
        if isinstance(obj, _pd.Series):
            return {"kind": "html", "value": obj.to_frame().to_html(max_rows=60, border=0)}
    except Exception:
        pass
    return {"kind": "text", "value": repr(obj)}
def _cs_figs():
    if "matplotlib.pyplot" not in sys.modules:
        return []
    import matplotlib.pyplot as plt
    out = []
    for n in plt.get_fignums():
        buf = io.BytesIO(); plt.figure(n).savefig(buf, format="png", dpi=110, bbox_inches="tight")
        out.append(base64.b64encode(buf.getvalue()).decode())
    plt.close("all")
    return out
def input(prompt=""):
    raise RuntimeError("input() is not available in the browser. Assign the value to a variable instead.")
builtins.input = input
`;
  const BROWSER_PKGS = { numpy: "numpy", pandas: "pandas", scipy: "scipy", sklearn: "scikit-learn", matplotlib: "matplotlib", networkx: "networkx", sympy: "sympy", regex: "regex", PIL: "pillow", statsmodels: "statsmodels" };
  const S = { py: null, ready: null, loaded: new Set(), running: false };

  function setStatus(text, cls) { el.statusText.textContent = text; el.status.className = "status" + (cls ? " " + cls : ""); }
  function loadScript(src) { return new Promise((res, rej) => { const s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = () => rej(new Error("Could not load " + src)); document.head.appendChild(s); }); }
  function ensurePython() {
    if (S.ready) return S.ready;
    S.ready = (async () => {
      setStatus("Loading Python (≈10 MB, once)…", "loading");
      if (!window.loadPyodide) await loadScript(window.CS_PYODIDE_URL + "pyodide.js");
      const py = await loadPyodide({ indexURL: window.CS_PYODIDE_URL });
      await py.runPythonAsync(PRELUDE);
      S.py = py; setStatus("Python ready", "ready");
      return py;
    })().catch((e) => { S.ready = null; setStatus("Python failed to load — check your connection", "error"); throw e; });
    return S.ready;
  }
  async function loadPackagesFor(py, code) {
    const mods = py.pyodide_py.code.find_imports(code).toJs();
    const pkgs = [...new Set(mods.map((m) => BROWSER_PKGS[m]).filter(Boolean))].filter((p) => !S.loaded.has(p));
    if (pkgs.length) {
      setStatus("Loading " + pkgs.join(", ") + "…", "loading");
      await py.loadPackage(pkgs);
      pkgs.forEach((p) => S.loaded.add(p));
      if (pkgs.includes("matplotlib")) await py.runPythonAsync('import matplotlib; matplotlib.use("agg")');
      setStatus("Python ready", "ready");
    }
    return mods;
  }

  function out(kind, text) { const s = document.createElement("span"); s.className = kind; s.textContent = text; el.out.appendChild(s); el.out.scrollTop = el.out.scrollHeight; }
  function outHTML(h) { const d = document.createElement("div"); d.innerHTML = h; el.out.appendChild(d); el.out.scrollTop = el.out.scrollHeight; }
  function clearOut() { el.out.innerHTML = ""; el.runTime.textContent = ""; }
  function cleanTraceback(msg) {
    const keep = []; let skip = false;
    for (const ln of String(msg).split("\n")) {
      if (/^\s+File "/.test(ln)) skip = /File "\/lib\/python|pyodide/.test(ln);
      if (skip) continue;
      keep.push(ln.replace('File "<exec>"', 'File "<your code>"'));
    }
    return keep.join("\n").trim();
  }

  function runJavaScript(code) {
    return new Promise((resolve) => {
      const src = `const fmt = (a) => a.map((x) => typeof x === "string" ? x : (() => { try { return JSON.stringify(x); } catch (_) { return String(x); } })()).join(" ");
        const send = (kind) => (...a) => postMessage({ kind, text: fmt(a) });
        console.log = send("stdout"); console.info = send("stdout"); console.warn = send("stderr"); console.error = send("stderr");
        onmessage = (e) => { try { (0, eval)(e.data); } catch (err) { postMessage({ kind: "stderr", text: (err && err.name ? err.name + ": " : "") + (err && err.message || err) }); } postMessage({ kind: "done" }); };`;
      const url = URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
      const w = new Worker(url);
      const timer = setTimeout(() => { w.terminate(); out("stderr", "Stopped: the code ran for more than 10 seconds.\n"); resolve(); }, 10000);
      w.onmessage = (e) => { if (e.data.kind === "done") { clearTimeout(timer); w.terminate(); URL.revokeObjectURL(url); resolve(); } else out(e.data.kind, e.data.text + "\n"); };
      w.onerror = (e) => { clearTimeout(timer); out("stderr", e.message + "\n"); w.terminate(); resolve(); };
      w.postMessage(code);
    });
  }

  async function runEditor() {
    if (S.running) return;
    if (current.lang === "javascript") {
      S.running = true; el.run.disabled = true; clearOut();
      const t0 = performance.now();
      try { await runJavaScript(editor.getValue()); } finally { S.running = false; el.run.disabled = false; }
      if (!el.out.querySelector(".stdout, .stderr")) out("result", "(no output — add console.log(...) to print values)\n");
      el.runTime.textContent = ((performance.now() - t0) / 1000).toFixed(2) + " s · JavaScript";
      return;
    }
    S.running = true; el.run.disabled = true;
    const keepNote = el.out.querySelector(".note") && current.setup ? el.out.querySelector(".note").textContent : ""; clearOut(); if (keepNote) out("note", keepNote);
    const code = editor.getValue(), t0 = performance.now();
    try {
      const py = await ensurePython();
      const setup = current.setup || "";
      const mods = await loadPackagesFor(py, setup + "\n" + code);
      const unavailable = mods.filter((m) => !BROWSER_PKGS[m] && !py.runPython(`__import__("importlib.util").util.find_spec(${JSON.stringify(m)}) is not None`));
      if (unavailable.length) {
        if (current.target === "vscode") vscodeNote(`This example needs <b>${unavailable.join(", ")}</b> and runs in <b>VS Code</b>.`);
        else colabNote(`This example needs <b>${unavailable.join(", ")}</b>, which cannot run inside a web browser.`);
        return;
      }
      const ns = py.globals.get("dict")();
      if (setup) {   // earlier examples this one builds on — run silently first
        py.setStdout({ batched: () => {} }); py.setStderr({ batched: () => {} });
        try { await py.runPythonAsync(setup, { globals: ns }); } catch (_) { /* reported below if it matters */ }
        try { py.runPython("_cs_figs()"); } catch (_) {}
      }
      py.setStdout({ batched: (s) => out("stdout", s + "\n") });
      py.setStderr({ batched: (s) => out("stderr", s + "\n") });
      try {
        const result = await py.runPythonAsync(code, { globals: ns });
        if (result !== undefined && result !== null) {
          const r = py.globals.get("_cs_repr")(result); const info = r.toJs({ dict_converter: Object.fromEntries }); r.destroy();
          if (info.kind === "html") outHTML(info.value); else out("result", info.value + "\n");
          if (result && typeof result.destroy === "function") result.destroy();
        }
      } catch (e) {
        out("stderr", cleanTraceback(e.message) + "\n");
      } finally {
        try { for (const b64 of py.runPython("_cs_figs()").toJs()) outHTML(`<img alt="figure" src="data:image/png;base64,${b64}">`); } catch (_) { /* no figures */ }
        ns.destroy();
      }
      if (!el.out.querySelector(".stdout, .stderr, .result, img, table")) out("result", "(no output — this example defines functions, classes or data that later examples use; add a print() to inspect them)\n");
      el.runTime.textContent = ((performance.now() - t0) / 1000).toFixed(2) + " s";
    } catch (e) {
      out("stderr", "Runtime error: " + e.message + "\n");
    } finally { S.running = false; el.run.disabled = false; }
  }

  function loadIntoEditor(code, { title = "Example", local = false, needs = "", setup = "", lang = "python", target = "colab" } = {}) {
    current = { code, local, needs, setup, lang, target };
    editor.setOption("mode", lang === "javascript" ? "javascript" : "python");
    editor.setOption("indentUnit", lang === "javascript" ? 2 : 4);
    editor.setValue(code); clearOut();
    el.title.textContent = title;
    if (setup) out("note", "This example continues an earlier one in the lesson; that code runs first, silently.\n");
    if (local && target === "vscode") vscodeNote(`This example needs <b>${needs}</b> and runs in <b>VS Code</b>.`);
    else if (local) colabNote(`This example needs <b>${needs}</b>, so it cannot run in the browser.`);
    setTimeout(() => { editor.refresh(); editor.focus(); }, 0);
  }

  $$(".runnable .try").forEach((b) => b.addEventListener("click", () => {
    const box = b.closest(".runnable"), code = box.querySelector(".src").value;
    const titleEl = box.querySelector(".code-title");
    let title = titleEl ? titleEl.textContent : "";
    if (!title) { const h = [...$$(".lesson-content h2, .lesson-content h3")].filter((x) => x.compareDocumentPosition(box) & Node.DOCUMENT_POSITION_FOLLOWING).pop(); title = h ? h.textContent : "Example"; }
    const local = b.dataset.run === "0";
    openPanel("code");
    const setupEl = box.querySelector(".setup");
    loadIntoEditor(code, { title, local, needs: b.dataset.needs || "", setup: setupEl ? setupEl.value : "", lang: b.dataset.lang || "python", target: b.dataset.target || "colab" });
    if (!local) runEditor();
  }));

  el.run.addEventListener("click", runEditor);
  el.clear.addEventListener("click", clearOut);
  el.reset.addEventListener("click", () => { editor.setValue(current.code); clearOut(); });
  el.copy.addEventListener("click", () => copyText(editor.getValue(), el.copy));

  // first animation of the week is ready in the selector; open the panel on wide screens only when asked
  if (location.hash.startsWith("#display=")) { openPanel("anim"); showAnim(location.hash.slice(9)); }
})();
