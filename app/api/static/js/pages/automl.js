const AUTOML = { version: null, profile: null, target: null, selection: [], tune: false,
                 stage: "Staging", metric: "roc_auc" };

PAGES.automl = {
  title: "AutoML",
  intro: "Profile a dataset, confirm the target, then search candidate models. Recommendations come from the data, not from a model — every one shows the evidence behind it, and you can override any of them.",
  async render(){
    const r = await loadAll({ ds:"/api/v1/datasets", runs:"/api/v1/automl/runs?limit=20" }, 4000);
    const versions = r.ds.ok ? (r.ds.data.versions || []) : [];

    if(!versions.length){
      return card("AutoML", `<div class="state">
        <div class="big">No datasets yet</div>
        Upload a CSV and AutoML will profile it, suggest a target and recommend models.
        <div style="margin-top:12px"><a class="btn pri" href="#/datasets">Upload a dataset</a></div>
      </div>`);
    }

    const picker = card("1. Choose a dataset", `
      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:flex-end">
        <div style="flex:1;min-width:240px">
          <label class="lbl" for="amds">Dataset version</label>
          <select id="amds">${versions.map(v =>
            `<option value="${esc(v.version)}" ${AUTOML.version===v.version?"selected":""}>${esc(v.version)} — ${v.n_rows} rows × ${v.n_columns} cols</option>`).join("")}</select>
        </div>
        <button class="btn pri" id="amprofile">Profile dataset</button>
      </div>`, { sub: "step 1 of 4" });

    const runs = sect(r.runs, d => card("AutoML runs", table([
      { label:"Run", render:x => `<span class="mono">${esc(String(x.run_id).replace("automl-","").slice(0,12))}</span>` },
      { label:"Status", render:x => runStatusBadge(x.status) },
      { label:"Dataset", render:x => `<span class="mono">${esc(x.dataset_version)}</span>` },
      { label:"Target", render:x => `<span class="mono">${esc(x.target_column)}</span>` },
      { label:"Candidates", num:true, render:x => int((x.algorithms||[]).length) },
      { label:"Best", render:x => x.best_algorithm
          ? `<b class="mono">${esc(x.best_algorithm)}</b>` + (has(x.best_model_version)?` <span class="dim">v${esc(x.best_model_version)}</span>`:"") : NA },
      { label:"Duration", num:true, render:x => has(x.duration_seconds) ?
          `<span class="mono">${Number(x.duration_seconds).toFixed(1)}s</span>` : NA },
      { label:"Started", render:x => when(x.created_at) },
      { label:"", render:x => `<button class="btn" data-act="am-detail" data-id="${esc(x.run_id)}">Detail</button>` },
    ], d.runs || [], { empty:"No AutoML runs yet. Profile a dataset above to start one." }),
    { flush:true, sub:`${d.count||0} run(s)`,
      right:`<button class="btn" data-retry="${esc(location.hash)}">Refresh</button>` }), "automl runs");

    return picker + `<div id="amwizard"></div>` + runs + `<div id="amdetail"></div>`;
  }
};

/* Run status now comes from core.js runStatusBadge: one state->colour map
   for every page, so a rejected run is the same colour wherever it appears. */

function renderWizard(d){
  const prof = d.profile, target = prof.suggested_target;
  AUTOML.profile = d;
  AUTOML.metric = d.primary_metric;
  if(!AUTOML.target) AUTOML.target = target ? target.column : null;
  if(!AUTOML.selection.length) AUTOML.selection = (d.default_selection || []).slice();

  const stats = `<div class="grid g4" style="margin-bottom:14px">
    ${kpi("Rows", int(prof.n_rows))}
    ${kpi("Columns", int(prof.n_columns), `${prof.n_numeric} numeric · ${prof.n_categorical} categorical`)}
    ${kpi("Missing cells", int(prof.missing_cells))}
    ${kpi("Duplicate rows", int(prof.duplicate_rows))}
  </div>`;

  const warn = (prof.warnings || []).length ? card("Data quality", `<div style="display:grid;gap:8px">${
    prof.warnings.map(w => `<div style="display:flex;gap:10px;align-items:flex-start">
      ${badge(w.type.replace(/_/g," "), w.severity === "error" ? "bad" : "warn")}
      <span style="flex:1">${w.column?`<span class="mono">${esc(w.column)}</span> — `:""}${esc(w.detail)}</span>
    </div>`).join("")}</div>`) : "";

  const columns = card("Column profile", table([
    { label:"Column", render:c => `<span class="mono">${esc(c.name)}</span>` },
    { label:"Type", render:c => badge(c.kind, c.kind==="numeric"?"info":"mute") },
    { label:"Unique", num:true, render:c => int(c.n_unique) },
    { label:"Missing", num:true, render:c => `${c.missing_pct}%` },
    { label:"Example", render:c => c.example ? `<span class="mono dim">${esc(c.example)}</span>` : NA },
    { label:"Role", render:c => c.role === "target" ? badge("target","ok")
        : c.role === "excluded" ? `${badge("excluded","mute")} <span class="dim" style="font-size:11px">${esc(c.exclusion_reason||"")}</span>`
        : badge("feature","info") },
  ], prof.columns || [], { empty:"No columns." }), { flush:true });

  // Step 2 -- target
  const others = prof.alternative_targets || [];
  const targetCard = card("2. Confirm the target", target ? `
    <div style="display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap">
      <div style="flex:1;min-width:260px">
        <div style="display:flex;gap:9px;align-items:center;margin-bottom:8px">
          <b class="mono" style="font-size:14px">${esc(target.column)}</b>
          ${badge(target.confidence + " confidence", target.confidence==="high"?"ok":target.confidence==="medium"?"warn":"bad")}
        </div>
        <ul style="margin:0 0 10px;padding-left:18px;color:var(--ink-2)">
          ${(target.reasons||[]).slice(0,4).map(x=>`<li>${esc(x)}</li>`).join("")}
        </ul>
        ${Object.keys(target.class_balance||{}).length ? `<div class="dim" style="font-size:11.5px">
          Class balance: ${Object.entries(target.class_balance).map(([k,v])=>
            `<span class="mono">${esc(k)}</span> ${(v*100).toFixed(1)}%`).join(" · ")}</div>`:""}
      </div>
      <div style="min-width:220px">
        <label class="lbl" for="amtarget">Use a different column</label>
        <select id="amtarget">${(prof.columns||[]).map(c =>
          `<option value="${esc(c.name)}" ${c.name===AUTOML.target?"selected":""}>${esc(c.name)}</option>`).join("")}</select>
        ${others.length?`<div class="dim" style="font-size:11.5px;margin-top:7px">Also plausible:
          ${others.map(o=>`<span class="mono">${esc(o.column)}</span>`).join(", ")}</div>`:""}
      </div>
    </div>` : `<div class="state"><div class="big">No target could be recommended</div>
      Nothing in this dataset looks like a label. Choose the column to predict:
      <div style="max-width:260px;margin:12px auto 0">
        <select id="amtarget">${(prof.columns||[]).map(c=>`<option value="${esc(c.name)}">${esc(c.name)}</option>`).join("")}</select>
      </div></div>`, { sub:"step 2 of 4" });

  // Step 3 -- problem type
  const supported = d.problem_supported;
  const problemCard = card("3. Problem type", `
    <div style="display:flex;gap:10px;align-items:center;margin-bottom:10px">
      <b>${esc((d.problem_type||"unknown").replace(/_/g," "))}</b>
      ${supported ? badge("supported","ok",true) : badge("not supported by this platform","bad",true)}
    </div>
    ${supported ? `<p class="dim" style="margin:0">${esc(d.metric_note||"")}</p>
      <div class="grid g2" style="margin-top:12px">
        <div><label class="lbl" for="ammetric">Primary metric</label>
          <select id="ammetric">${[d.primary_metric, ...(d.secondary_metrics||[])].map(m =>
            `<option value="${esc(m)}" ${m===AUTOML.metric?"selected":""}>${esc(m)}</option>`).join("")}</select></div>
        <div><label class="lbl" for="amstage">If the gate passes</label>
          <select id="amstage">
            <option value="Staging">Promote to Staging</option>
            <option value="Production">Promote to Production</option>
          </select></div>
      </div>`
    : `<div class="note" style="border-color:var(--bad-line);background:var(--bad-bg)">
        <b>This platform trains binary classification only.</b> The training stack casts the label to
        an integer and evaluates with ROC-AUC over two classes, so a
        ${esc((d.problem_type||"").replace(/_/g," "))} target cannot be fitted here. Pick a different
        target column, or use a dataset whose label has two classes.</div>`}`,
    { sub:"step 3 of 4" });

  // Step 4 -- candidates
  const cands = d.candidates || [];
  const usable = cands.filter(c => c.available);
  const candCard = card("4. Candidate models", supported ? `
    <div class="grid g3">${usable.map(c => `
      <label class="cand ${AUTOML.selection.includes(c.algorithm)?"on":""}">
        <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px">
          <input type="checkbox" class="amcand" value="${esc(c.algorithm)}"
                 ${AUTOML.selection.includes(c.algorithm)?"checked":""}>
          <b>${esc(c.label)}</b>
          ${badge(c.tier, c.tier==="recommended"?"ok":c.tier==="baseline"?"info":"mute")}
        </div>
        <ul style="margin:0;padding-left:17px;color:var(--ink-3);font-size:11.5px">
          ${(c.reasons||[]).slice(0,3).map(x=>`<li>${esc(x)}</li>`).join("")}
        </ul>
      </label>`).join("")}</div>
    ${cands.filter(c=>!c.available).length ? `<p class="dim" style="margin:12px 0 0;font-size:11.5px">
      Unavailable here: ${cands.filter(c=>!c.available).map(c=>
        `<span class="mono">${esc(c.label)}</span>`).join(", ")}
      — ${esc(cands.filter(c=>!c.available)[0].unavailable_reason||"")}</p>`:""}
    <div style="display:flex;gap:14px;align-items:center;margin-top:14px;flex-wrap:wrap">
      <label style="display:flex;gap:7px;align-items:center;cursor:pointer">
        <input type="checkbox" id="amtune"> Run hyperparameter search per candidate</label>
      <span class="dim" id="amcount"></span>
      <span class="spacer"></span>
      <button class="btn pri" id="amstart">Start AutoML</button>
    </div>
    <div class="note" style="margin-top:12px">The winner is registered and then judged by the
      <b>same approval gate</b> as any other model. AutoML picks a candidate; it does not decide
      what reaches production.</div>
    <div id="amresult" style="margin-top:12px"></div>`
    : `<div class="state">Choose a binary target above to see candidate models.</div>`,
    { sub:"step 4 of 4" });

  return stats + warn + targetCard + problemCard + candCard + columns;
}

function renderAutoMLRun(run){
  const lb = run.leaderboard || [];
  const cands = run.candidates || [];
  const failed = cands.filter(c => c.status === "failed");
  const prom = run.promotion || {};
  const best = lb[0];

  const head = `<div class="grid g4" style="margin-bottom:14px">
    ${kpi("Status", runStatusBadge(run.status),
          has(run.duration_seconds)?`${Number(run.duration_seconds).toFixed(1)}s`:"")}
    ${kpi("Candidates", int(cands.length), `${lb.length} succeeded · ${failed.length} failed`)}
    ${kpi("Primary metric", `<span class="mono">${esc(run.primary_metric)}</span>`)}
    ${kpi("Best candidate", best?`<span class="mono">${esc(best.algorithm)}</span>`:NA,
          best&&has(best.model_version)?`registered v${esc(best.model_version)}`:"")}
  </div>`;

  const bestCard = best ? card("Best candidate for this run", `
    <div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:12px">
      <b class="mono" style="font-size:16px">${esc(best.algorithm)}</b>
      ${badge("rank 1","ok")}
      ${has(best.model_version)?badge("registered v"+best.model_version,"info"):""}
    </div>
    <div class="grid g4">
      ${kpi(run.primary_metric, num((best.metrics||{})[run.primary_metric]))}
      ${kpi("F1", num((best.metrics||{}).f1))}
      ${kpi("Precision", num((best.metrics||{}).precision))}
      ${kpi("Recall", num((best.metrics||{}).recall))}
    </div>
    <p class="dim" style="margin:12px 0 0">Selected because it achieved the highest
      <span class="mono">${esc(run.primary_metric)}</span> among the candidates that trained
      successfully in this run. ${esc(run.ranking_rule||"")}</p>`) : "";

  const board = card("Leaderboard", table([
    { label:"Rank", render:c => `<b>#${c.rank}</b>` },
    { label:"Model", render:c => `<span class="mono">${esc(c.algorithm)}</span>` },
    { label:run.primary_metric, num:true, render:c => num((c.metrics||{})[run.primary_metric]) },
    { label:"F1", num:true, render:c => num((c.metrics||{}).f1) },
    { label:"Precision", num:true, render:c => num((c.metrics||{}).precision) },
    { label:"Recall", num:true, render:c => num((c.metrics||{}).recall) },
    { label:"Accuracy", num:true, render:c => num((c.metrics||{}).accuracy) },
    { label:"Time", num:true, render:c => has(c.duration_seconds)?`<span class="mono">${c.duration_seconds}s</span>`:NA },
    { label:"Version", render:c => has(c.model_version)?`<span class="mono">v${esc(c.model_version)}</span>`:NA },
  ], lb, { empty:"No candidate completed successfully." }), { flush:true,
     sub: lb.length ? `ranked by ${run.primary_metric}` : "" });

  const failures = failed.length ? card("Failed candidates", table([
    { label:"Model", render:c => `<span class="mono">${esc(c.algorithm)}</span>` },
    { label:"Error", render:c => `<span class="dim">${esc(String(c.error||"").slice(0,180))}</span>` },
  ], failed, { empty:"" }), { flush:true,
    sub:"a failed candidate never enters the ranking" }) : "";

  const gate = Object.keys(prom).length ? card("Approval gate", `
    <div style="display:flex;gap:10px;align-items:center;margin-bottom:10px">
      ${badge(prom.decision, prom.decision==="approved"?"ok":"bad", true)}
      ${prom.promoted ? badge("promoted to "+esc(prom.final_stage||""),"ok")
                      : badge("not promoted","mute")}
    </div>
    <p style="margin:0;color:var(--ink-2)">${esc(prom.reason||"")}</p>
    ${(prom.failed_checks||[]).length?`<p style="margin:10px 0 0">Failed checks:
      ${prom.failed_checks.map(c=>badge(c,"bad")).join(" ")}</p>`:""}
    ${!prom.promoted?`<div class="note" style="margin-top:12px">The candidate is registered but
      <b>not eligible for production</b>. The production model is unchanged. This is the gate
      working: a candidate that does not clear the bar does not ship.</div>`:""}`)
    : "";

  const repro = card("Reproducibility", `<dl class="kv">
    <dt>Run</dt><dd class="mono">${esc(run.run_id)}</dd>
    <dt>Dataset version</dt><dd class="mono">${esc(run.dataset_version)}</dd>
    <dt>Target</dt><dd class="mono">${esc(run.target_column)}</dd>
    <dt>Problem type</dt><dd>${esc((run.problem_type||"").replace(/_/g," "))}</dd>
    <dt>Candidates</dt><dd class="mono">${(run.algorithms||[]).join(", ")}</dd>
    <dt>Primary metric</dt><dd class="mono">${esc(run.primary_metric)}</dd>
    <dt>Hyperparameter search</dt><dd>${run.tune?badge("enabled","info"):`<span class="dim">disabled</span>`}</dd>
    <dt>Target stage</dt><dd class="mono">${esc(run.target_stage||"")}</dd>
    <dt>Started</dt><dd>${when(run.started_at||run.created_at)}</dd>
    <dt>Completed</dt><dd>${when(run.completed_at)}</dd>
  </dl>`);

  const err = run.error ? `<div class="note" style="border-color:var(--bad-line);
    background:var(--bad-bg);margin-bottom:14px"><b>Run failed.</b><br>${esc(run.error)}</div>` : "";

  return err + head + bestCard + board + failures + gate + repro;
}

function wireWizard(){
  const targetSel = $("#amtarget");
  if(targetSel) targetSel.onchange = async () => {
    AUTOML.target = targetSel.value;
    AUTOML.selection = [];
    $("#amwizard").innerHTML = card("Re-profiling for the new target", skeleton(4), { flush:true });
    try {
      const d = await api.get(
        `/api/v1/automl/profile/${encodeURIComponent(AUTOML.version)}?target=${encodeURIComponent(AUTOML.target)}`, 0);
      $("#amwizard").innerHTML = renderWizard(d);
      wireWizard();
    } catch(e){
      $("#amwizard").innerHTML = card("Profiling failed", errorState(e.message, location.hash));
    }
  };

  const metric = $("#ammetric"); if(metric) metric.onchange = () => { AUTOML.metric = metric.value; };
  const stage = $("#amstage");   if(stage)  stage.onchange  = () => { AUTOML.stage  = stage.value; };
  const tune = $("#amtune");     if(tune)   tune.onchange   = () => { AUTOML.tune   = tune.checked; };

  const boxes = [...document.querySelectorAll(".amcand")];
  const count = $("#amcount");
  const sync = () => {
    AUTOML.selection = boxes.filter(b => b.checked).map(b => b.value);
    boxes.forEach(b => b.closest(".cand").classList.toggle("on", b.checked));
    if(count) count.textContent = AUTOML.selection.length
      ? `AutoML will train ${AUTOML.selection.length} candidate model(s).`
      : "Select at least one model.";
    const start = $("#amstart");
    if(start) start.disabled = !AUTOML.selection.length;
  };
  boxes.forEach(b => b.onchange = sync);
  sync();

  const start = $("#amstart");
  if(start) start.onclick = async () => {
    const payload = {
      dataset_version: AUTOML.version,
      target_column: AUTOML.target,
      algorithms: AUTOML.selection,
      primary_metric: AUTOML.metric,
      tune: AUTOML.tune,
      target_stage: AUTOML.stage,
      max_models: Math.max(1, AUTOML.selection.length),
    };
    const needsKey = await authRequired();
    const proceed = await confirmAction({
      title: "Start AutoML",
      body: `Train ${AUTOML.selection.length} candidate model(s) on ${AUTOML.version}, target `
          + `${AUTOML.target}. The best is registered and then judged by the approval gate — `
          + `promotion is not automatic.`,
      confirm: "Start AutoML", needsKey,
    });
    if(!proceed) return;
    if(needsKey && !proceed.key){ toast("An API key is required.", "bad"); return; }

    start.disabled = true; start.textContent = "Starting...";
    try {
      const res = await api.post("/api/v1/automl/runs", payload, proceed.key);
      api.bust();
      toast("AutoML started.", "ok");
      pollAutoML(res.run_id);
    } catch(e){
      const msg = (e.status === 401 || e.status === 403)
        ? "Rejected: the API key was missing or not accepted." : e.message;
      $("#amresult").innerHTML = `<div class="note" style="border-color:var(--bad-line);
        background:var(--bad-bg)"><b>Could not start.</b><br>${esc(msg)}</div>`;
      toast(msg, "bad");
    } finally {
      start.disabled = false; start.textContent = "Start AutoML";
    }
  };
}

/* Progress from the run's own state: which stage it reached and what each
   candidate is doing. Stops the moment the run is terminal. */
async function pollAutoML(runId, attempt){
  attempt = attempt || 0;
  if(attempt > 300) return;
  let run;
  try { run = await api.get(`/api/v1/automl/runs/${encodeURIComponent(runId)}`, 0); }
  catch(e){ return; }

  const stages = ["profiling","training","ranking","completed"];
  const reached = stages.indexOf(run.status === "completed_with_warnings" ? "completed" : run.status);
  const done = ["completed","completed_with_warnings","failed"].includes(run.status);
  const cands = run.candidates || [];
  const finished = cands.filter(c => c.status !== "queued" && c.status !== "training").length;
  const pct = done ? 100 : (cands.length ? Math.round((finished / cands.length) * 100) : 5);

  const box = $("#amresult");
  if(box) box.innerHTML = `
    <div style="display:flex;gap:10px;align-items:center;margin-bottom:8px">
      <b class="mono">${esc(runId.replace("automl-",""))}</b> ${runStatusBadge(run.status)}
      <span class="spacer"></span><span class="mono dim">${pct}%</span>
    </div>
    <div class="bar" style="margin-bottom:10px"><i style="width:${pct}%"></i></div>
    <div class="flow" style="margin-bottom:10px">${stages.map((s,i) =>
      `<span class="step ${done||i<reached?"done":i===reached?"on":""}">${esc(s)}</span>` +
      (i<stages.length-1?'<span class="arw">&rarr;</span>':"")).join("")}</div>
    <div style="display:grid;gap:6px">${cands.map(c =>
      `<div style="display:flex;justify-content:space-between;align-items:center;gap:10px">
        <span class="mono">${esc(c.algorithm)}</span>
        <span style="display:flex;gap:8px;align-items:center">
          ${c.metrics && c.metrics.roc_auc != null
            ? `<span class="mono dim">roc_auc ${Number(c.metrics.roc_auc).toFixed(4)}</span>` : ""}
          ${badge(c.status, c.status==="completed"?"ok":c.status==="failed"?"bad":
                            c.status==="training"?"info":"mute", c.status==="training")}
        </span></div>`).join("")}</div>`;

  if(done){
    api.bust();
    toast(`AutoML ${run.status.replace(/_/g," ")}.`,
          run.status === "failed" ? "bad" : "ok");
    const detail = $("#amdetail");
    if(detail) detail.innerHTML = renderAutoMLRun(run);
    return;
  }
  setTimeout(() => pollAutoML(runId, attempt + 1), 3000);
}
