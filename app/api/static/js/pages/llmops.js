function mockNotice(prov){
  if(prov !== "mock") return "";
  return `<div class="note" style="margin-bottom:14px"><b>Provider: offline mock.</b>
    The mock provider is deterministic and runs locally — it is <b>not a language model</b>.
    Evaluation scores below measure the harness, not model quality, and cost is $0 because no
    external API is called.</div>`;
}

PAGES["llm-overview"] = {
  title: "LLMOps — Overview",
  intro: "Provider, traffic, token consumption and spend for the foundation-model side of the platform.",
  async render(){
    const r = await loadAll({ dash:"/api/v1/dashboard", prov:"/api/v1/llm/providers",
      tok:"/api/v1/llm/tokens", cost:"/api/v1/llm/cost" }, 8000);
    const llm = (r.dash.ok ? r.dash.data.llm : {}) || {};
    const t = (r.tok.ok ? r.tok.data.totals : {}) || {};

    const head = `<div class="grid g6" style="margin-bottom:14px">
      ${kpi("Provider", badge(llm.provider||"—", llm.is_mock?"warn":"info"),
            llm.model?`<span class="mono">${esc(llm.model)}</span>`:"")}
      ${kpi("Calls", int(llm.calls))}
      ${kpi("Total tokens", int(llm.total_tokens))}
      ${kpi("Avg latency", ms(llm.avg_latency_ms))}
      ${kpi("Cost today", usd(llm.today_cost_usd), `month ${has(llm.month_cost_usd)?"$"+llm.month_cost_usd.toFixed(4):"—"}`)}
      ${kpi("Error rate", pct(llm.error_rate,2))}
    </div>`;

    const budget = card("Budget", `<div class="grid g2">
      <div><div class="dim" style="font-size:11px;margin-bottom:6px">Daily
        (${has(llm.daily_budget_usd)?"$"+llm.daily_budget_usd:"—"})</div>
        <div class="bar"><i class="${(llm.daily_budget_used_pct||0)>80?"bad":"ok"}"
          style="width:${Math.min(100,llm.daily_budget_used_pct||0)}%"></i></div>
        <div class="dim" style="margin-top:5px">${(llm.daily_budget_used_pct||0).toFixed(2)}% used</div></div>
      <div><div class="dim" style="font-size:11px;margin-bottom:6px">Monthly</div>
        <div class="bar"><i class="${(llm.monthly_budget_used_pct||0)>80?"bad":"ok"}"
          style="width:${Math.min(100,llm.monthly_budget_used_pct||0)}%"></i></div>
        <div class="dim" style="margin-top:5px">${(llm.monthly_budget_used_pct||0).toFixed(2)}% used</div></div>
    </div>`);

    const provs = sect(r.prov, d => card("Provider adapters", table([
      { label:"Provider", render:([k]) => `<span class="mono">${esc(k)}</span>` +
          (k === d.configured ? " " + badge("configured","info") : "") },
      { label:"Available", render:([,v]) => boolBadge(v && v.available, "available","not configured") },
      { label:"Detail", render:([,v]) => `<span class="dim">${esc((v && (v.detail || v.reason)) || "")}</span>` },
    ], Object.entries(d.providers || {}), { empty:"No adapters reported." }),
    { flush:true, sub:`configured: ${esc(d.configured||"—")}` }), "providers");

    return mockNotice(llm.provider) + head + `<div class="grid g2">${budget}${
      card("Token split", `<div class="grid g2">
        ${kpi("Input", int(t.input_tokens ?? llm.input_tokens))}
        ${kpi("Output", int(t.output_tokens ?? llm.output_tokens))}</div>
      <div class="dim" style="margin-top:10px">Avg ${has(llm.avg_tokens_per_call)?
        llm.avg_tokens_per_call.toFixed(1):"—"} tokens/call</div>`)}</div>` + provs;
  }
};

PAGES["llm-prompts"] = {
  title: "LLMOps — Prompts",
  intro: "Versioned prompt registry. Each version is content-hashed, so a prompt change is a tracked, comparable event.",
  async render(){
    const r = await loadAll({ p:"/api/v1/llm/prompts", dash:"/api/v1/dashboard" }, 15000);
    return sect(r.p, d => {
      const entries = Object.entries(d.prompts || {});
      if(!entries.length) return emptyState("No prompts registered.");
      return entries.map(([name, info]) => {
        const versions = info.versions || [];
        return card(name, table([
          { label:"Version", render:v => `<b class="mono">${esc(v.version||v)}</b>` +
              ((info.latest === (v.version||v)) ? " " + badge("latest","ok") : "") },
          { label:"Hash", render:v => v.content_hash ?
              `<span class="mono dim">${esc(String(v.content_hash).slice(0,16))}</span>` : NA },
          { label:"Variables", render:v => (v.variables||[]).length ?
              (v.variables||[]).map(x => `<span class="mono">${esc(x)}</span>`).join(", ") : NA },
          { label:"Description", render:v => `<span class="dim">${esc(v.description||"")}</span>` },
        ], versions.map(v => typeof v === "string" ? { version:v } : v),
          { empty:"No versions." }), { flush:true, sub:`latest ${esc(info.latest||"—")}` });
      }).join("") + `<div class="note">Prompt bodies are served by
        <span class="mono">GET /api/v1/llm/prompts/{name}/{version}</span>; a diff between two
        versions is available at <span class="mono">/diff/{a}/{b}</span>.</div>`;
    }, "prompts");
  }
};

PAGES["llm-evals"] = {
  title: "LLMOps — Evaluations",
  intro: "Prompt and model evaluation runs, scored by the configured scorers.",
  async render(){
    const r = await loadAll({ e:"/api/v1/llm/evaluations", dash:"/api/v1/dashboard" }, 8000);
    const llm = (r.dash.ok ? r.dash.data.llm : {}) || {};
    return mockNotice(llm.provider) + sect(r.e, d => {
      const evals = d.evaluations || [], sets = d.datasets || [];
      return card("Evaluation runs", table([
        { label:"ID", render:x => `<span class="mono">${esc(String(x.id||"").slice(0,12))}</span>` },
        { label:"Suite", render:x => esc(x.suite||"—") },
        { label:"Dataset", render:x => `<span class="mono">${esc(x.dataset||"—")}</span>` },
        { label:"Prompt", render:x => `<span class="mono">${esc(x.prompt_version||x.prompt||"—")}</span>` },
        { label:"Score", num:true, render:x => num(x.overall_score ?? x.overall) },
        { label:"Cases", num:true, render:x => int(x.cases) },
        { label:"Failed", num:true, render:x => int(x.failed) },
        { label:"Tokens", num:true, render:x => int(x.total_tokens ?? x.tokens) },
        { label:"When", render:x => when(x.created_at) },
      ], evals, { empty:"No evaluation runs recorded. Trigger one with POST /api/v1/llm/evaluations/run." }),
      { flush:true, sub:`${sets.length} dataset(s) available` });
    }, "evaluations");
  }
};

PAGES["llm-cost"] = {
  title: "LLMOps — Tokens & Cost",
  intro: "Token consumption and estimated spend. Costs are derived from token counts and a configured price table, not from a billing API.",
  async render(){
    const r = await loadAll({ tok:"/api/v1/llm/tokens", cost:"/api/v1/llm/cost", dash:"/api/v1/dashboard" }, 8000);
    const llm = (r.dash.ok ? r.dash.data.llm : {}) || {};
    const t = (r.tok.ok ? r.tok.data.totals : {}) || {};
    const c = r.cost.ok ? r.cost.data : {};

    const head = `<div class="grid g4" style="margin-bottom:14px">
      ${kpi("Total tokens", int(t.total_tokens))}
      ${kpi("Input", int(t.input_tokens))}
      ${kpi("Output", int(t.output_tokens))}
      ${kpi("Cost today", usd(c.today_cost_usd), `month ${has(c.month_cost_usd)?"$"+c.month_cost_usd.toFixed(6):"—"}`)}
    </div>`;

    const daily = (c.daily || []);
    const costChart = card("Estimated cost, daily", daily.length > 1
      ? lineChart(daily.map(x => ({ x: x.day || x.date, y: x.cost_usd ?? x.cost })), { dp:4 })
      : emptyState("Not enough daily cost history to plot."), { sub:`${daily.length} day(s)` });

    const byModel = card("By model", barList(Object.entries(c.by_model || {}).map(([k,v]) => ({
      label:k, value: typeof v === "number" ? v : (v.cost_usd || 0),
      tag: (typeof v === "object" && v.priced === false) ? badge("not priced","warn") : ""
    })), { empty:"No per-model spend recorded." }));

    const byPrompt = sect(r.tok, d => card("By prompt version", table([
      { label:"Prompt", render:x => `<span class="mono">${esc(x.prompt_version||x.prompt||"—")}</span>` },
      { label:"Calls", num:true, render:x => int(x.calls) },
      { label:"Tokens", num:true, render:x => int(x.total_tokens ?? x.tokens) },
      { label:"Cost", num:true, render:x => usd(x.cost_usd) },
    ], d.by_prompt_version || [], { empty:"No prompt-level usage recorded yet." }), { flush:true }), "tokens");

    return mockNotice(llm.provider) + head + costChart +
      `<div class="grid g2">${byModel}${byPrompt}</div>` +
      `<div class="note"><b>Cost figures are estimates.</b> They are computed from token counts and a
        configured price table — this is not a billing system. A model with no price entry is flagged
        rather than silently counted as $0.</div>`;
  }
};

PAGES["llm-safety"] = {
  title: "LLMOps — Safety",
  intro: "The safety screen applied to model input and output, and an explicit statement of what it does and does not detect.",
  async render(){
    const d = await api.get("/api/v1/llm/safety/checks", 20000);
    const checks = d.checks || [];
    return card("Screening configuration", `<dl class="kv">
        <dt>Enabled</dt><dd>${boolBadge(d.enabled,"enabled","disabled")}</dd>
        <dt>Block on violation</dt><dd>${boolBadge(d.block_on_violation,"blocking","flag only")}</dd>
        <dt>Active checks</dt><dd class="mono">${checks.length}</dd>
      </dl>`)
      + card("Checks", table([
        { label:"Check", render:c => `<b class="mono">${esc(c.name || c.id || "—")}</b>` },
        { label:"Severity", render:c => c.severity ? badge(c.severity,
            c.severity==="critical"?"bad":c.severity==="high"?"warn":"info") : NA },
        { label:"Blocking", render:c => has(c.blocking) ? boolBadge(c.blocking,"blocks","flags") : NA },
        { label:"Description", render:c => `<span class="dim">${esc(c.description||"")}</span>` },
      ], checks, { empty:"No checks configured." }), { flush:true })
      + `<div class="note"><b>This is heuristic pattern matching, not a content-safety classifier.</b>
        ${esc(d.limitations || "")} It cannot detect hallucination — only stylistic correlates of
        unsupported claims — and its false-negative rate is unmeasured. A production system needs a
        real moderation model in addition to this.</div>`;
  }
};
