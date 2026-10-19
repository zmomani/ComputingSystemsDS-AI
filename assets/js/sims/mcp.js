/* Model Context Protocol: sequence diagram of Host ↔ Client ↔ Server with JSON-RPC messages (initialize, tools/list, tools/call). */
Sims.register('mcp', {
  title: 'MCP message flow: host, client, server and JSON-RPC',
  kind: 'Model Context Protocol · Week 11',
  short: 'A sequence diagram that builds one message at a time: initialize handshake, tool discovery, a tool call and the result reaching the LLM.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    const lanes = [['user', 'User'], ['host', 'Host app\n(LLM + MCP client)'], ['server', 'MCP server\n(your tools)'], ['ext', 'External system\n(DB / API / files)']];
    const msgs = [
      { f: 'host', t: 'server', txt: 'initialize {protocolVersion, capabilities, clientInfo}', kind: 'req', ex: ['Handshake', 'The client (inside the host) opens a transport — <b>stdio</b> for a local process or <b>Streamable HTTP</b> for remote — and sends <code>initialize</code>. Both sides declare what they support.'] },
      { f: 'server', t: 'host', txt: 'result {capabilities: {tools, resources, prompts}, serverInfo}', kind: 'res', ex: ['Handshake', 'The server answers with its capabilities. MCP has three primitives a server may expose: <b>tools</b> (model-controlled actions), <b>resources</b> (application-controlled data, like GET endpoints) and <b>prompts</b> (user-controlled templates).'] },
      { f: 'host', t: 'server', txt: 'notifications/initialized', kind: 'note', ex: ['Handshake', 'A notification (no id, no reply) confirms the session is ready. All messages are <b>JSON-RPC 2.0</b>: requests carry an <code>id</code>, responses echo it, notifications have none.'] },
      { f: 'host', t: 'server', txt: 'tools/list', kind: 'req', ex: ['Discovery', 'The client asks what tools exist. This happens at run time, so the host does not need to be rebuilt when the server adds a tool.'] },
      { f: 'server', t: 'host', txt: 'result {tools: [{name: "query_db", description, inputSchema}]}', kind: 'res', ex: ['Discovery', 'Each tool comes with a <b>JSON Schema</b> for its arguments. The host converts these into the function-calling format of whatever LLM it uses — that translation is the whole point: one server works with Claude, GPT, Llama, LangChain…'] },
      { f: 'user', t: 'host', txt: '"How many students enrolled in 606363?"', kind: 'user', ex: ['User turn', 'The user asks a question. The host puts the tool list (from the previous step) into the LLM request alongside the message.'] },
      { f: 'host', t: 'host', txt: 'LLM decides: call query_db(sql="SELECT COUNT(*) …")', kind: 'self', ex: ['Model decision', 'The <i>model</i> chooses the tool and fills its arguments from the schema. The host validates the arguments against the schema before doing anything.'] },
      { f: 'host', t: 'server', txt: 'tools/call {name: "query_db", arguments: {sql: "…"}}', kind: 'req', ex: ['Tool call', 'The client forwards the call. Note the LLM never talks to the server directly; the host mediates, which is where permissions and logging live.'] },
      { f: 'server', t: 'ext', txt: 'SQL query', kind: 'ext', ex: ['Execution', 'Server-side Python runs the real operation — a database query here. This code is ordinary; MCP only standardises how it is described and invoked.'] },
      { f: 'ext', t: 'server', txt: '42', kind: 'ext', ex: ['Execution', 'The result returns to the server, which wraps it as MCP <code>content</code> (text, image, or embedded resource).'] },
      { f: 'server', t: 'host', txt: 'result {content: [{type: "text", text: "42"}], isError: false}', kind: 'res', ex: ['Tool result', 'The response matches the request <code>id</code>. Errors are reported in-band with <code>isError: true</code> so the model can read them and recover (compare the ReAct simulation).'] },
      { f: 'host', t: 'host', txt: 'LLM reads result → composes answer', kind: 'self', ex: ['Second model call', 'The tool result is appended to the conversation and the model is called again to produce the final text.'] },
      { f: 'host', t: 'user', txt: '"42 students are enrolled in 606363."', kind: 'user', ex: ['Answer', 'Done. Two model calls, one tool call, four JSON-RPC exchanges. Multiple servers can be connected at once; the host merges their tool lists.'] }
    ];
    const cv = H.canvas(760, 440); L.stage.appendChild(cv); const ctx = cv.ctx;
    const lx = { user: 70, host: 270, server: 500, ext: 690 };
    function render(st, idx) {
      const T = H.theme(); H.clear(cv);
      lanes.forEach(([id, label]) => { const x = lx[id]; H.rect(ctx, x - 70, 10, 140, 44, { fill: id === 'host' ? H.hexA(T.accent, .2) : id === 'server' ? H.hexA(T.green, .2) : T.bg, stroke: id === 'host' ? T.accent : id === 'server' ? T.green : T.border, r: 8 }); label.split('\n').forEach((l, i) => H.text(ctx, l, x, 28 + i * 14 - (label.includes('\n') ? 4 : -4), { align: 'center', size: i ? 10 : 12, weight: i ? 400 : 700, color: i ? T.muted : T.text })); ctx.save(); ctx.strokeStyle = T.border; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(x, 54); ctx.lineTo(x, 430); ctx.stroke(); ctx.restore(); });
      const colorOf = k => ({ req: T.accent, res: T.green, note: T.muted, user: T.purple, self: T.accent2, ext: T.yellow })[k];
      for (let i = 0; i <= idx; i++) {
        const m = msgs[i], y = 76 + i * 27, cur = i === idx, c = colorOf(m.kind);
        if (m.f === m.t) { ctx.save(); ctx.strokeStyle = c; ctx.lineWidth = cur ? 2.5 : 1.3; ctx.beginPath(); ctx.moveTo(lx[m.f], y - 8); ctx.lineTo(lx[m.f] + 30, y - 8); ctx.lineTo(lx[m.f] + 30, y + 6); ctx.stroke(); ctx.restore(); H.arrow(ctx, lx[m.f] + 30, y + 6, lx[m.f] + 2, y + 6, { color: c, lw: cur ? 2.5 : 1.3 }); H.text(ctx, m.txt, lx[m.f] + 38, y + 2, { size: 10, mono: true, color: cur ? c : T.text, weight: cur ? 700 : 400 }); }
        else { H.arrow(ctx, lx[m.f], y, lx[m.t], y, { color: c, lw: cur ? 2.5 : 1.3, dash: m.kind === 'res' || m.kind === 'note' ? [5, 3] : null }); H.text(ctx, m.txt, (lx[m.f] + lx[m.t]) / 2, y - 5, { align: 'center', size: 10, mono: true, color: cur ? c : T.text, weight: cur ? 700 : 400 }); }
      }
      L.explain(st.ex[1], `${st.ex[0]} · message ${idx + 1}/${msgs.length}`);
    }
    const S = H.stepper(L, () => msgs, render, { interval: 1700 });
    L.setLegend([[H.theme().accent, 'JSON-RPC request (client → server)'], [H.theme().green, 'response (dashed)'], [H.theme().accent2, 'LLM reasoning inside the host'], [H.theme().yellow, 'server-side execution'], [H.theme().purple, 'user messages']]);
    L.setNotes(`<p>Analogy that works: MCP is a USB-C port for AI tools — one connector, any device. Emphasise the separation of roles: the <b>server author</b> writes Python functions with type hints (<code>@mcp.tool()</code> in FastMCP), the <b>host</b> handles the model, and the protocol sits between. Week 11 milestone: expose two of the project's agent tools as MCP servers, add error handling and retries — show where retries belong (host side, on <code>isError</code> or transport failures).</p>`);
    document.addEventListener('cs-theme', S.refresh);
    return { destroy() { S.destroy(); document.removeEventListener('cs-theme', S.refresh); } };
  }
});
