/* Guided build workflow -- the ten step renderers.
 *
 * Each returns HTML for one step and reads only APIs that already exist.
 * State, gating and the stepper live in project.js; event handling lives in
 * project-wire.js.
 */


/* ===================================================================== */
/* 01 Dataset                                                            */
/* ===================================================================== */
async function stepDataset(){
  let limits = null;
  try { limits = await api.get("/api/v1/datasets/limits", 300000); } catch(e){ limits = null; }
  const maxMb = limits && limits.max_upload_bytes
    ? Math.floor(limits.max_upload_bytes / (1024*1024)) : null;
  const formats = limits && limits.accepted_formats ? limits.accepted_formats.join(", ") : "CSV";

  const drop = `
    <label class="drop" id="prjdrop" for="prjfile">
      <div class="icn" aria-hidden="true">&uarr;</div>
      <div class="big">Drag &amp; drop a CSV here</div>
      <div class="hint">or click to browse</div>
      <input type="file" id="prjfile" accept=".csv,text/csv">
    </label>
    <div class="grid g2" style="margin-top:14px">
      <div><span class="dim" style="font-size:12px">Maximum file size</span><br>
        ${maxMb ? `<b>${maxMb} MB</b>` : `<span class="dim">Not reported by the API</span>`}</div>
      <div><span class="dim" style="font-size:12px">Supported format</span><br>
        <b>${esc(formats)}</b></div>
    </div>
    <div style="margin-top:14px">
      <input type="text" id="prjdesc" placeholder="Description (optional)"
        style="width:100%;max-width:420px">
    </div>
    <div id="prjupstatus" style="margin-top:14px"></div>`;

  let done = "";
  if(PRJ.version){
    const v = PRJ.validation;
    done = `<div style="margin-top:20px">${card("Uploaded", `
      <div class="verdict pass" style="margin-bottom:14px">
        <span class="mark" aria-hidden="true">&check;</span>
        <span class="txt"><b>UPLOAD SUCCESSFUL</b>
          <span>Registered as an immutable, content-addressed version.</span></span></div>
      <div class="grid g4">
        ${kpi("Filename", esc(PRJ.datasetName || "-"))}
        ${kpi("Version", `<span class="mono">${esc(PRJ.version)}</span>`)}
        ${kpi("Rows", int(PRJ.rows))}
        ${kpi("Columns", int(PRJ.columns))}
      </div>
      ${v ? `<div style="margin-top:14px">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
          <b style="font-size:13px">Validation</b>
          ${v.passed ? badge("PASSED","ok",true) : badge("FAILED","bad",true)}
          <span class="dim" style="font-size:12px">${int(v.succeeded)} of
            ${int(v.expectations)} expectations passed</span>
        </div>
        ${!v.passed ? `<p class="dim" style="margin:0;font-size:12.5px">
          Validation failures do not block this walkthrough, but the training pipeline applies
          the same expectations and may reject the run.
          <a href="#/datasets">See the full report on the Datasets page.</a></p>` : ""}
      </div>` : `<p class="dim" style="margin:14px 0 0;font-size:12.5px">
        Validation was not run for this upload.</p>`}`, { flush:true })}
      <div class="wiznav">
        <button class="btn pri" data-goto="2">Continue to profile</button>
        <button class="btn" id="prjreset">Start over with a different dataset</button>
      </div></div>`;
  }

  return wizFrame(1, "Upload your data",
    `Upload the CSV you want FMOps to train on. It is parsed with pandas, registered as a
     content-addressed version, and checked against the platform validation suite &mdash; the
     same expectations the training pipeline gates on. Uploading identical bytes twice returns
     the existing version rather than creating a duplicate.`,
    drop + done);
}


/* ===================================================================== */
/* 02 Profile                                                            */
/* ===================================================================== */
async function stepProfile(){
  if(!PRJ.version) return wizFrame(2, "Profile", "No dataset has been uploaded yet.",
    unavailable("Upload a dataset first.") + `<div class="wiznav">
      <button class="btn pri" data-goto="1">Back to upload</button></div>`);

  let d;
  try { d = await api.get(`/api/v1/automl/profile/${encodeURIComponent(PRJ.version)}`, 60000); }
  catch(e){
    return wizFrame(2, "Profile", "The profiler could not read this dataset.",
      errorState(e.message, "newproject") + `<div class="wiznav">
        <button class="btn" data-goto="1">Back to upload</button></div>`);
  }

  PRJ.profile = d;
  PRJ.primaryMetric = d.primary_metric;
  if(!PRJ.target){ PRJ.problemType = d.problem_type; PRJ.problemSupported = d.problem_supported; }
  prjSave();

  const p = d.profile || {};
  const cols = p.columns || [];
  const dateLike = cols.filter(c => c.kind === "datetime").length;
  // pct() scales a fraction, so keep this a fraction.
  const missingFrac = p.n_rows && p.n_columns
    ? p.missing_cells / (p.n_rows * p.n_columns) : null;

  const overview = `<div class="grid g6">
    ${kpi("Rows", int(p.n_rows))}
    ${kpi("Columns", int(p.n_columns))}
    ${kpi("Numeric", int(p.n_numeric))}
    ${kpi("Categorical", int(p.n_categorical))}
    ${kpi("Date-like", int(dateLike))}
    ${kpi("Missing values", missingFrac == null ? NA : pct(missingFrac, 2),
          `${int(p.missing_cells)} cells`)}
  </div>`;

  const sug = p.suggested_target;
  const targetPanel = sug ? `
    <div class="rectarget">
      <div class="dim mono" style="font-size:11px;letter-spacing:.08em;margin-bottom:6px">
        POTENTIAL TARGET</div>
      <div class="nm">${esc(sug.column)}</div>
      ${confidenceBadge(sug.confidence)}
      <ul class="reasons">${(sug.reasons||[]).map(r => `<li>${esc(r)}</li>`).join("")}</ul>
      <p class="dim" style="margin:0;font-size:12.5px">You confirm the target in the next step.</p>
    </div>`
    : `<div class="note" style="border-color:var(--warn-line);background:var(--warn-bg)">
        <b>No target could be identified.</b><br>
        <span style="font-size:12.5px">Nothing in this dataset looks like an outcome column.
        You will have to choose one yourself in the next step.</span></div>`;

  const warns = p.warnings || [];
  const warnPanel = warns.length ? card("Data quality warnings", `
    <p class="dim" style="margin:0 0 12px;font-size:12.5px">Reported, never acted on
      automatically. Nothing below is resampled, dropped or corrected without you asking.</p>
    ${table([
      { label:"Warning", render:w => badge(String(w.type||"warning").replace(/_/g," "),
          w.severity === "error" ? "bad" : w.severity === "warning" ? "warn" : "mute") },
      { label:"Column", render:w => w.column ? `<span class="mono">${esc(w.column)}</span>` : NA },
      { label:"Detail", render:w => esc(w.detail || "") },
    ], warns, { empty:"" })}`, { flush:true, sub:`${warns.length} warning(s)` }) : "";

  const colTable = card("Columns", table([
    { label:"Column", render:c => `<span class="mono">${esc(c.name)}</span>` },
    { label:"Type", render:c => esc(c.kind) },
    { label:"Role", render:c => c.role === "excluded"
        ? badge("excluded","mute") + (c.exclusion_reason
            ? ` <span class="dim" style="font-size:11.5px">${esc(c.exclusion_reason)}</span>` : "")
        : badge(c.role, c.role === "target" ? "info" : "ok") },
    { label:"Unique", num:true, render:c => int(c.n_unique) },
    { label:"Missing", num:true, render:c => c.missing ? pct(c.missing_pct / 100, 2) : NA },
    { label:"Example", render:c => c.example != null
        ? `<span class="mono dim">${esc(String(c.example).slice(0,32))}</span>` : NA },
  ], cols, { empty:"No columns." }), { flush:true, sub:`${cols.length} column(s)` });

  return wizFrame(2, "Dataset profile",
    `Every column classified, every exclusion given a reason. This is a deterministic pass over
     observable properties &mdash; no model is involved in any recommendation below.`,
    prjContext() + overview + `<div style="margin:18px 0">${targetPanel}</div>` +
    warnPanel + colTable + `<div class="wiznav">
      <button class="btn pri" data-goto="3">Continue to target selection</button>
      <button class="btn" data-goto="1">Back</button></div>`);
}

async function stepTarget(){
  if(!PRJ.profile) return wizFrame(3, "Target", "Profile the dataset first.",
    unavailable("No profile available.") + `<div class="wiznav">
      <button class="btn pri" data-goto="2">Back to profile</button></div>`);

  const d = PRJ.target
    ? await profileFor(PRJ.target).catch(() => PRJ.profile)
    : PRJ.profile;
  const p = d.profile || {};
  const sug = p.suggested_target;
  const chosen = PRJ.target;

  const candidates = (p.columns || [])
    .filter(c => c.role !== "excluded" || c.role === "target")
    .map(c => c.name);
  const allCols = (p.columns || []).map(c => c.name);

  const rec = (!chosen && sug) ? `
    <div class="rectarget" style="max-width:460px">
      <div class="dim mono" style="font-size:11px;letter-spacing:.08em;margin-bottom:6px">
        RECOMMENDED TARGET</div>
      <div class="nm">${esc(sug.column)}</div>
      ${confidenceBadge(sug.confidence)}
      <ul class="reasons">${(sug.reasons||[]).map(r => `<li>${esc(r)}</li>`).join("")}</ul>
      <button class="btn pri" data-usetarget="${esc(sug.column)}">Use ${esc(sug.column)}</button>
    </div>` : "";

  const lowConfidence = sug && String(sug.confidence).toLowerCase() === "low";
  const warnNoTarget = (!chosen && !sug) ? `
    <div class="note" style="border-color:var(--warn-line);background:var(--warn-bg);max-width:640px">
      <b>We could not confidently identify the target column.</b><br>
      <span style="font-size:12.5px">Nothing in this dataset scored as a plausible outcome
      column, so there is no recommendation to accept. Choose the column you want to predict
      below. Training never starts on a guess.</span></div>` : "";
  const warnLow = (!chosen && lowConfidence) ? `
    <div class="note" style="border-color:var(--warn-line);background:var(--warn-bg);max-width:640px;margin-top:14px">
      <b>Low confidence.</b> <span style="font-size:12.5px">Another column scored almost as
      well, which is ambiguity rather than a recommendation. Check this is the column you
      actually want to predict.</span></div>` : "";

  const picker = `
    <div style="margin-top:${rec ? "20px" : "0"};max-width:460px">
      <label for="prjtarget" style="display:block;font-size:12.5px;font-weight:600;margin-bottom:6px">
        ${rec ? "Or choose a different column" : "Choose the column you want to predict"}</label>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <select id="prjtarget" style="flex:1 1 240px">
          <option value="">Select a column&hellip;</option>
          ${allCols.map(c => `<option value="${esc(c)}" ${c === chosen ? "selected" : ""}>
            ${esc(c)}${candidates.includes(c) ? "" : " (excluded from features)"}</option>`).join("")}
        </select>
        <button class="btn" id="prjsettarget">Use this column</button>
      </div>
    </div>`;

  /* Once a target is chosen the inferred problem type decides whether this
     stack can fit it at all -- and an unsupported one is refused here rather
     than attempted and quietly mangled. */
  let verdict = "";
  if(chosen){
    const supported = d.problem_supported;
    PRJ.problemType = d.problem_type;
    PRJ.problemSupported = supported;
    PRJ.primaryMetric = d.primary_metric;
    prjSave();
    const pretty = String(d.problem_type || "unknown").replace(/_/g, " ");
    verdict = supported ? `
      <div style="margin-top:20px">${card("Target confirmed", `
        <div class="grid g3">
          ${kpi("Target", `<span class="mono">${esc(chosen)}</span>`)}
          ${kpi("Problem type", esc(pretty))}
          ${kpi("Supported", badge("YES","ok",true))}
        </div>
        ${d.metric_note ? `<p class="dim" style="margin:14px 0 0;font-size:12.5px">
          ${esc(d.metric_note)}</p>` : ""}`, { flush:true })}
        <div class="wiznav">
          <button class="btn pri" data-goto="4">Continue to training</button>
          <button class="btn" data-goto="2">Back</button></div></div>`
      : `<div style="margin-top:20px">${card("Unsupported problem type", `
        <div class="verdict fail">
          <span class="mark" aria-hidden="true">&times;</span>
          <span class="txt"><b>FMOPS CANNOT TRAIN THIS TARGET</b>
            <span>Detected: ${esc(pretty)}</span></span></div>
        <p style="margin:0 0 10px;font-size:13px;line-height:1.6">
          This platform currently fits <b>binary classification</b> only. That is a property of
          the training stack, not of the profiler: the model factory builds classifiers, the
          label is cast with <span class="mono">astype(int)</span>, and evaluation reads
          <span class="mono">predict_proba[:, 1]</span>. Training a
          ${esc(pretty)} target here would produce a meaningless score, so the API refuses it.</p>
        <p class="dim" style="margin:0;font-size:12.5px">Supported:
          ${(d.supported_problem_types||[]).map(t =>
            `<span class="mono">${esc(String(t).replace(/_/g," "))}</span>`).join(", ") || "&mdash;"}</p>`,
        { flush:true })}
        <div class="wiznav">
          <button class="btn pri" id="prjcleartarget">Back to target selection</button>
        </div></div>`;
  }

  return wizFrame(3, "What do you want to predict?",
    `FMOps recommends a target from observable evidence, but you confirm it. The recommendation
     is a deterministic score over column name, cardinality, missingness and position &mdash;
     never a guess the platform acts on by itself.`,
    prjContext() + warnNoTarget + rec + warnLow + picker + verdict);
}


/* ===================================================================== */
/* 04 Training strategy                                                  */
/* ===================================================================== */
async function stepTraining(){
  if(!PRJ.target || !PRJ.problemSupported)
    return wizFrame(4, "Training", "A supported target must be confirmed first.",
      unavailable("No usable target selected.") + `<div class="wiznav">
        <button class="btn pri" data-goto="3">Back to target</button></div>`);

  const d = PRJ.profile || {};
  const cands = (d.candidates || []).filter(c => c.available);
  const preselect = d.default_selection || [];
  const maxModels = d.max_models_limit || 3;

  let algos = {};
  try { algos = (await api.get("/api/v1/models/algorithms", 300000)).algorithms || {}; }
  catch(e){ algos = {}; }
  const installed = Object.keys(algos).filter(k => algos[k]);

  const automl = `
    <div class="choice rec">
      <span class="tag">RECOMMENDED</span>
      <h4>AutoML</h4>
      <p>Let FMOps evaluate several supported algorithms on this dataset and rank them by the
        primary metric.</p>
      <ul>
        <li>Dataset analysis</li><li>Candidate selection</li><li>Multiple model training</li>
        <li>Evaluation</li><li>Deterministic ranking</li><li>Best candidate registered</li>
      </ul>
      <div class="foot">
        <div style="margin-bottom:10px">
          <label style="display:block;font-size:12px;font-weight:600;margin-bottom:6px">
            Candidates (up to ${maxModels})</label>
          <div style="display:grid;gap:6px">
            ${cands.map(c => `<label class="cand" style="display:flex;gap:8px;align-items:flex-start">
              <input type="checkbox" class="prjcand" value="${esc(c.algorithm)}"
                ${preselect.includes(c.algorithm) ? "checked" : ""}>
              <span><b style="font-size:12.5px">${esc(c.label)}</b>
                ${badge(c.tier, c.tier === "recommended" ? "ok" : c.tier === "baseline" ? "info" : "mute")}
                <br><span class="dim" style="font-size:11.5px">${esc((c.reasons||[])[0] || "")}</span>
              </span></label>`).join("") || `<span class="dim">No candidates available.</span>`}
          </div>
        </div>
        <p id="prjcandcount" class="dim" style="font-size:12px;margin:0 0 10px"></p>
        <button class="btn pri" id="prjstartautoml">Start AutoML</button>
      </div>
    </div>`;

  const manual = `
    <div class="choice">
      <h4>Manual training</h4>
      <p>Choose the algorithm yourself. Same pipeline, same gate &mdash; you pick the estimator
        instead of comparing several.</p>
      <div class="foot">
        <div style="margin-bottom:10px">
          <label for="prjalgo" style="display:block;font-size:12px;font-weight:600;margin-bottom:6px">
            Algorithm</label>
          <select id="prjalgo" style="width:100%">
            ${installed.map(a => `<option value="${esc(a)}">${esc(a.replace(/_/g," "))}</option>`).join("")}
          </select>
          <p class="dim" style="font-size:11.5px;margin:6px 0 0">Only estimators this environment
            can actually build are listed${Object.keys(algos).length > installed.length
              ? `; ${Object.keys(algos).length - installed.length} optional backend(s) are not installed`
              : ""}.</p>
        </div>
        <label style="display:flex;gap:8px;align-items:center;margin-bottom:10px;font-size:12.5px">
          <input type="checkbox" id="prjtune"> Run hyperparameter search first (slower)</label>
        <button class="btn" id="prjstartmanual">Start training</button>
      </div>
    </div>`;

  const shared = card("What happens to the winner", `
    <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end">
      <div><label for="prjstage" style="display:block;font-size:12px;font-weight:600;margin-bottom:6px">
        Promote into</label>
        <select id="prjstage">
          <option value="Staging" selected>Staging</option>
          <option value="Production">Production</option>
        </select></div>
      <p class="dim" style="flex:1 1 320px;margin:0;font-size:12.5px;line-height:1.6">
        Both paths register the resulting model and put it through the approval gate
        automatically &mdash; that happens inside the run, which is why there is no separate
        register button later. Asking for a stage only asks for the attempt: promotion still
        requires clearing the absolute thresholds and beating the incumbent.</p>
    </div>`, { flush:true });

  const note = `<div class="note" style="margin-top:16px">
    <b>Where this runs.</b> <span style="font-size:12.5px">Runs execute as background tasks
    inside the API process, so training competes with request handling for the same CPU, and a
    run in flight is lost if the process restarts. Each candidate is a full training run, so
    <i>n</i> candidates cost roughly <i>n</i> times one run.</span></div>`;

  return wizFrame(4, "How should FMOps train?",
    `Both routes use the same training pipeline, the same evaluation and the same approval gate.
     AutoML compares several estimators and ranks them; manual training fits the one you pick.`,
    prjContext() + shared + `<div class="choices" style="margin-top:16px">${automl}${manual}</div>`
    + note + `<div id="prjstartresult" style="margin-top:16px"></div>
    <div class="wiznav"><button class="btn" data-goto="3">Back</button></div>`);
}


async function stepEvaluation(){
  if(!PRJ.runId) return wizFrame(5, "Evaluation", "No run has been started.",
    unavailable("Start a training run first.") + `<div class="wiznav">
      <button class="btn pri" data-goto="4">Back to training</button></div>`);

  let run;
  try { run = await api.get(runPath(), 0); }
  catch(e){
    return wizFrame(5, "Evaluation", "The run could not be read.",
      errorState(e.message, "newproject"));
  }
  absorbRun(run);

  const live = !runDone(run.status);
  const body = `<div data-progress>`
    + (PRJ.runKind === "automl" ? automlBody(run, live) : trainingBody(run, live))
    + `</div>`;
  const nav = live ? "" : `<div class="wiznav">
    ${run.status === "failed"
      ? `<button class="btn pri" data-goto="4">Try a different configuration</button>`
      : `<button class="btn pri" data-goto="6">Continue to registry</button>`}
    <button class="btn" data-goto="4">Back</button></div>`;

  return wizFrame(5, live ? "Training candidate models" : "Evaluation",
    live
      ? `The run is executing now. This page follows it until it reaches a terminal state and
         then stops polling &mdash; there is nothing further to learn from a finished run.`
      : `Results as the platform recorded them. Nothing here is recomputed in the browser.`,
    prjContext() + body + `<div id="prjrunbox"></div>` + nav)
    + (live ? `<span id="prjpoll" data-run="${esc(PRJ.runId)}" hidden></span>` : "");
}


function automlBody(run, live){
  const cands = run.candidates || [];
  const finished = cands.filter(c => c.status !== "queued" && c.status !== "training").length;
  const pct2 = runDone(run.status) ? 100 : (cands.length ? Math.round(finished/cands.length*100) : 5);
  const stages = ["profiling", "training", "ranking", "completed"];
  const reachedName = run.status === "completed_with_warnings" ? "completed" : run.status;
  const reached = stages.indexOf(reachedName);

  const progress = `${card("Progress", `
    <div style="display:flex;gap:10px;align-items:center;margin-bottom:10px">
      ${runStatusBadge(run.status)}
      <span class="spacer"></span>
      <span class="mono dim">${pct2}%</span>
    </div>
    <div class="bar" style="margin-bottom:12px"><i style="width:${pct2}%"></i></div>
    <div class="flow" style="margin-bottom:14px">${stages.map((s,i) =>
      `<span class="step ${runDone(run.status)||i<reached ? "done" : i===reached ? "on" : ""}">
        ${esc(s)}</span>` + (i<stages.length-1 ? '<span class="arw">&rarr;</span>' : "")).join("")}</div>
    <div style="display:grid;gap:7px">${cands.map(c => `
      <div style="display:flex;justify-content:space-between;align-items:center;gap:10px">
        <span class="mono">${esc(c.algorithm)}</span>
        <span style="display:flex;gap:8px;align-items:center">
          ${c.duration_seconds != null
            ? `<span class="mono dim">${Number(c.duration_seconds).toFixed(1)}s</span>` : ""}
          ${runStatusBadge(c.status)}</span></div>`).join("")
      || `<span class="dim">No candidates recorded yet.</span>`}</div>
    ${run.duration_seconds != null
      ? `<p class="dim" style="margin:12px 0 0;font-size:12px">Elapsed
         ${Number(run.duration_seconds).toFixed(1)}s</p>` : ""}`, { flush:true })}`;

  if(live) return progress;

  if(run.status === "failed") return progress + card("Run failed", `
    <div class="verdict fail"><span class="mark" aria-hidden="true">&times;</span>
      <span class="txt"><b>NO MODEL WAS PRODUCED</b>
        <span>The error below is what the backend recorded.</span></span></div>
    <pre class="scroll" style="margin:0">${esc(run.error || "no error recorded")}</pre>`,
    { flush:true });

  const board = run.leaderboard || [];
  const metric = run.primary_metric || "roc_auc";
  const lb = card("Model leaderboard", table([
    { label:"#", render:c => c.rank === 1
        ? `<span class="ranked">1</span>` : `<span class="ranked other">${int(c.rank)}</span>` },
    { label:"Algorithm", render:c => `<b class="mono">${esc(c.algorithm)}</b>` },
    { label:metric.toUpperCase().replace(/_/g,"-"), num:true,
      render:c => num((c.metrics||{})[metric], 4) },
    { label:"F1", num:true, render:c => num((c.metrics||{}).f1, 4) },
    { label:"Precision", num:true, render:c => num((c.metrics||{}).precision, 4) },
    { label:"Recall", num:true, render:c => num((c.metrics||{}).recall, 4) },
    { label:"Time", num:true, render:c => c.duration_seconds != null
        ? `${Number(c.duration_seconds).toFixed(1)}s` : NA },
    { label:"", render:c => `<button class="btn" data-cand="${esc(c.algorithm)}">Details</button>` },
  ], board, { empty:"No ranked candidates.", rowClass:c => c.rank === 1 ? "win" : "" }),
    { flush:true, sub:`best: ${esc(run.best_algorithm || "-")}` });

  const failedCands = (run.candidates||[]).filter(c => c.status === "failed");
  const partial = failedCands.length ? `<div class="note"
    style="border-color:var(--warn-line);background:var(--warn-bg);margin-top:14px">
    <b>${failedCands.length} candidate(s) failed and were not ranked.</b>
    <div style="font-size:12.5px;margin-top:6px">${failedCands.map(c =>
      `<div><span class="mono">${esc(c.algorithm)}</span> &mdash; ${esc(c.error||"no error recorded")}</div>`
      ).join("")}</div></div>` : "";

  const rule = run.ranking_rule ? card("How ranking works", `
    <p style="margin:0 0 8px;font-size:13px">Candidates are ordered by:</p>
    <p class="mono" style="margin:0 0 10px;font-size:12.5px">${esc(run.ranking_rule)}</p>
    <p class="dim" style="margin:0;font-size:12.5px">Quoted from the run record, so what is
      shown is the rule that actually ordered these candidates.</p>`, { flush:true }) : "";

  return progress + lb + partial + rule;
}


function trainingBody(run, live){
  const m = run.metrics || {};
  const head = card("Run", `
    <div class="grid g4">
      ${kpi("Status", runStatusBadge(run.status))}
      ${kpi("Algorithm", `<span class="mono">${esc(run.algorithm||"-")}</span>`)}
      ${kpi("Tuning", run.tune ? "enabled" : "off")}
      ${kpi("Duration", run.duration_seconds != null
        ? `${Number(run.duration_seconds).toFixed(1)}s` : NA)}
    </div>`, { flush:true });

  if(live) return head + `<div class="note" style="margin-top:14px">
    <span class="spin" aria-hidden="true"></span> Training is running in the background.
    This page follows it until it finishes.</div>`;

  if(run.status === "failed" || run.status === "rejected") return head + card(
    run.status === "rejected" ? "Rejected by the pipeline" : "Run failed", `
    <div class="verdict fail"><span class="mark" aria-hidden="true">&times;</span>
      <span class="txt"><b>${run.status === "rejected"
        ? "THE PIPELINE REJECTED THIS RUN" : "NO MODEL WAS PRODUCED"}</b>
        <span>The message below is what the backend recorded.</span></span></div>
    <pre class="scroll" style="margin:0">${esc(run.error || "no error recorded")}</pre>`,
    { flush:true });

  const keys = Object.keys(m);
  return head + card("Metrics", keys.length ? `<div class="grid g4">
    ${keys.slice(0,8).map(k => kpi(k.replace(/_/g," "), metricValue(m[k]))).join("")}</div>`
    : unavailable("The run recorded no metrics."), { flush:true });
}


/* ===================================================================== */
/* 06 Registry                                                           */
/* ===================================================================== */
async function stepRegistry(){
  if(!PRJ.runId) return wizFrame(6, "Registry", "No run has been started.",
    unavailable("Nothing has been trained yet."));

  let run = null;
  try { run = await api.get(runPath(), 0); absorbRun(run); } catch(e){ /* fall through */ }

  if(!PRJ.modelVersion){
    const why = PRJ.runKind === "training"
      ? `This run was started without asking for registration, so the model was trained and
         evaluated but never entered the registry. There is no endpoint that registers a
         finished run after the fact &mdash; registration happens inside the run.`
      : `The run did not record a registered model version.`;
    return wizFrame(6, "Model registry", "Nothing was registered by this run.",
      prjContext() + `<div class="note" style="border-color:var(--warn-line);background:var(--warn-bg)">
        <b>No model version was registered.</b>
        <div style="font-size:12.5px;margin-top:6px">${why}</div></div>
      <div class="wiznav"><button class="btn pri" data-goto="4">Run again with registration</button>
        <button class="btn" data-goto="5">Back to results</button></div>`);
  }

  /* The registered name comes from the registry, never from this page: the
     platform registers every model under one configured name, so inventing a
     per-dataset name here would misrepresent what is actually stored. */
  let versions = [], modelName = PRJ.modelName;
  try {
    const models = await api.get("/api/v1/models", 15000);
    const names = (models.models || []).map(m => typeof m === "string" ? m : m.name);
    modelName = modelName || names[0] || null;
    if(modelName){
      versions = await api.get(`/api/v1/models/${encodeURIComponent(modelName)}/versions`, 5000);
    }
  } catch(e){ /* reported below */ }
  PRJ.modelName = modelName; prjSave();

  const mine = versions.find(v => Number(v.version) === Number(PRJ.modelVersion)) || null;
  if(mine) { PRJ.registeredStage = mine.stage; prjSave(); }

  const detail = mine ? card("Registered", `
    <div class="verdict pass" style="margin-bottom:14px">
      <span class="mark" aria-hidden="true">&check;</span>
      <span class="txt"><b>MODEL REGISTERED</b>
        <span>Recorded by the registry during the run.</span></span></div>
    <div class="grid g4">
      ${kpi("Model", `<span class="mono">${esc(modelName)}</span>`)}
      ${kpi("Version", `<span class="mono">v${int(mine.version)}</span>`)}
      ${kpi("Stage", badge(mine.stage, mine.stage === "Production" ? "ok"
        : mine.stage === "Staging" ? "info" : "mute"))}
      ${kpi("Algorithm", `<span class="mono">${esc(mine.algorithm
        || (PRJ.runKind === "automl" && run && run.best_algorithm) || "-")}</span>`)}
    </div>
    ${mine.metrics ? `<div class="grid g4" style="margin-top:14px">
      ${Object.keys(mine.metrics).slice(0,4).map(k =>
        kpi(k.replace(/_/g," "), num(mine.metrics[k], 4))).join("")}</div>` : ""}
    <p class="dim" style="margin:14px 0 0;font-size:12.5px">
      Every model on this platform is registered under one configured name
      (<span class="mono">${esc(modelName)}</span>) and distinguished by version. There is no
      per-dataset model namespace, so this version sits in the same lineage as every other
      model trained here.</p>`, { flush:true })
    : `<div class="note" style="border-color:var(--warn-line);background:var(--warn-bg)">
        <b>Version v${int(PRJ.modelVersion)} was recorded by the run but could not be read back
        from the registry.</b></div>`;

  return wizFrame(6, "Model registry",
    `Registration happened inside the run: both AutoML and a training run with promotion
     requested register the winner and put it through the gate before the run reports back.
     This step shows what the registry actually holds.`,
    prjContext() + detail + `<div class="wiznav">
      <button class="btn pri" data-goto="7">Continue to approval</button>
      <button class="btn" data-goto="5">Back to results</button></div>`);
}

/* The platform registers every model under one configured name, so the name is
   read from the registry rather than guessed from the dataset. */


/* ===================================================================== */
/* 07 Approval gate                                                      */
/* ===================================================================== */
/* Two sources, deliberately kept apart:
 *
 *   - what the gate decided during the run (recorded, needs no auth), and
 *   - what the gate would decide against the thresholds in force right now.
 *
 * They can differ if configuration changed since the run, and collapsing them
 * into one number would hide exactly the discrepancy an operator cares about.
 */
async function stepApproval(){
  if(!PRJ.modelVersion) return wizFrame(7, "Approval gate", "No registered model version.",
    unavailable("Nothing has been registered yet.") + `<div class="wiznav">
      <button class="btn pri" data-goto="6">Back to registry</button></div>`);

  await ensureModelName();
  if(!PRJ.modelName) return wizFrame(7, "Approval gate", "The registered model could not be named.",
    unavailable("The registry did not report any model to evaluate against."));

  let recorded = null;
  if(PRJ.runKind === "automl"){
    try {
      const run = await api.get(runPath(), 0);
      recorded = run.promotion || null;
      if(recorded && recorded.decision){ PRJ.gateDecision = recorded.decision; prjSave(); }
    } catch(e){ /* recorded outcome simply unavailable */ }
  }

  /* Ask whether a key is needed before sending a request that would need one.
     The gate endpoint is a POST, so on a key-protected deployment an
     unauthenticated probe is a guaranteed 401 -- handled, but it still logs a
     console error on every visit to this step. Skip it and show the key panel
     directly; the recorded outcome below does not need the call. */
  let gate = null, gateError = null;
  let needsKey = await authRequired();
  if(!needsKey){
    try {
      gate = await api.post(`/api/v1/models/${encodeURIComponent(PRJ.modelName)}`
        + `/versions/${PRJ.modelVersion}/evaluate-gate`);
    } catch(e){
      if(e.status === 401 || e.status === 403) needsKey = true;
      else gateError = e.message;
    }
  }
  if(gate && gate.approval && gate.approval.decision){
    PRJ.gateDecision = gate.approval.decision; prjSave();
  }

  const decision = (gate && gate.approval && gate.approval.decision)
    || (recorded && recorded.decision) || null;
  const approved = decision === "approved";

  /* Three outcomes, not two. An environment with require_manual_approval on
     returns pending_manual for a candidate that cleared every automated check,
     and calling that "not eligible" would be a straight misreading: nothing
     failed, a human simply has not signed it off yet. */
  const pending = decision === "pending_manual";
  const verdictClass = approved ? "pass" : pending ? "warn" : "fail";
  const verdictMark = approved ? "&check;" : pending ? "&#9203;" : "&times;";
  const verdictTitle = approved ? "APPROVED FOR PRODUCTION"
    : pending ? "AWAITING MANUAL APPROVAL" : "NOT ELIGIBLE FOR PRODUCTION";
  const verdictBody = approved
    ? "Every blocking check passed."
    : pending
      ? "Every automated check passed. This environment requires a human to promote the "
        + "version before it reaches Production."
      : "At least one blocking check failed. Promotion to Production is refused.";

  const verdict = decision ? `<div class="verdict ${verdictClass}">
      <span class="mark" aria-hidden="true">${verdictMark}</span>
      <span class="txt"><b>${verdictTitle}</b><span>${verdictBody}</span></span></div>`
    : `<div class="note" style="border-color:var(--warn-line);background:var(--warn-bg)">
        <b>The gate outcome could not be read.</b>
        <div style="font-size:12.5px;margin-top:6px">${needsKey
          ? "Re-running the gate is a write-method call and this API requires a key for it."
          : esc(gateError || "The endpoint returned nothing usable.")}</div></div>`;

  const keyPanel = needsKey ? card("Re-run the gate", `
    <p class="dim" style="margin:0 0 10px;font-size:12.5px">This API requires a key for
      write-method calls, and the gate endpoint is a POST. Supplying a key here re-runs the gate
      against the thresholds currently in force. The key is used for this request and is not
      stored.</p>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <input type="password" id="prjgatekey" placeholder="X-API-Key" style="flex:1 1 240px"
        autocomplete="off">
      <button class="btn" id="prjrungate">Run the gate</button></div>`, { flush:true }) : "";

  const checks = (gate && gate.approval && gate.approval.checks) || [];
  const checkTable = checks.length ? card("Production gate", `
    ${table([
      { label:"Check", render:c => `<span class="mono">${esc(c.name)}</span>` },
      { label:"Observed", num:true, render:c => typeof c.observed === "number"
          ? num(c.observed, 4) : (c.observed != null ? esc(String(c.observed)) : NA) },
      { label:"Required", num:true, render:c => typeof c.threshold === "number"
          ? num(c.threshold, 4) : (c.threshold != null ? esc(String(c.threshold)) : NA) },
      { label:"Blocking", render:c => c.blocking ? badge("blocking","warn") : badge("advisory","mute") },
      { label:"Result", render:c => c.passed ? badge("PASSED","ok") : badge("FAILED","bad") },
    ], checks, { empty:"" })}
    <p class="dim" style="margin:12px 0 0;font-size:12.5px">Thresholds come from the platform
      configuration, not from this page. They are not relaxed to make a candidate pass.</p>`,
    { flush:true, sub:`${checks.filter(c => !c.passed).length} failed` }) : "";

  const cmp = (gate && gate.comparison) || (recorded && recorded.comparison) || null;
  const champion = cmp ? card("Champion vs candidate", `
    ${table([
      { label:"", render:r => `<b>${esc(r.k)}</b>` },
      { label:"Champion" + (cmp.baseline_version != null ? ` (v${cmp.baseline_version})` : ""),
        num:true, render:r => r.a },
      { label:"Candidate" + (cmp.candidate_version != null ? ` (v${cmp.candidate_version})` : ""),
        num:true, render:r => r.b },
    ], [
      { k:String(cmp.metric || "metric").replace(/_/g," "),
        a:num(cmp.baseline_score, 4), b:num(cmp.candidate_score, 4) },
      { k:"Improvement", a:NA,
        b:`${cmp.improvement >= 0 ? "+" : ""}${num(cmp.improvement, 4)}` },
      { k:"Minimum required", a:NA, b:num(cmp.min_improvement, 4) },
    ], { empty:"" })}
    <div style="margin-top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
      ${cmp.candidate_is_better ? badge("candidate is better","ok") : badge("not an improvement","warn")}
      ${cmp.decision ? badge(String(cmp.decision).replace(/_/g," "),
        cmp.decision === "promote" ? "ok" : "mute") : ""}
    </div>
    ${cmp.reason ? `<p class="dim" style="margin:10px 0 0;font-size:12.5px">${esc(cmp.reason)}</p>` : ""}
    <p class="dim" style="margin:10px 0 0;font-size:12.5px">Beating the incumbent is necessary
      but not sufficient: a candidate must also clear the absolute thresholds above.</p>`,
    { flush:true }) : (PRJ.modelVersion ? card("Champion vs candidate",
      unavailable("No incumbent to compare against, or the comparison was not recorded."),
      { flush:true }) : "");

  const failedNames = (recorded && recorded.failed_checks) || [];
  const recordedPanel = recorded ? card("Recorded at run time", `
    <div class="grid g3">
      ${kpi("Decision", badge(String(recorded.decision||"-").toUpperCase(),
        recorded.decision === "approved" ? "ok" : "bad"))}
      ${kpi("Promoted", recorded.promoted ? badge("YES","ok") : badge("NO","mute"))}
      ${kpi("Final stage", esc(recorded.final_stage || "-"))}
    </div>
    ${recorded.reason ? `<p class="dim" style="margin:12px 0 0;font-size:12.5px">
      ${esc(recorded.reason)}</p>` : ""}
    ${failedNames.length ? `<p style="margin:10px 0 0;font-size:12.5px">Failed checks:
      ${failedNames.map(n => `<span class="mono">${esc(n)}</span>`).join(", ")}</p>` : ""}
    <p class="dim" style="margin:10px 0 0;font-size:12px">This is what the gate decided while the
      run executed. The panels above re-run it against the thresholds in force now.</p>`,
    { flush:true }) : "";

  /* Whether the next step can do anything is the deployment API's rule, not
     this page's: it accepts a version whose stage is deployable and refuses
     one that is not. A rejected candidate stays in Development and is refused
     there; a pending_manual candidate registered into Staging is genuinely
     deployable, and pretending otherwise would be inventing a restriction the
     platform does not have. */
  const nav = `<div class="wiznav">
    ${decision === "rejected"
      ? `<button class="btn pri" data-goto="4">Train another candidate</button>
         <button class="btn" data-goto="8">See deployment status</button>`
      : `<button class="btn pri" data-goto="8">Continue to deployment</button>`}
    <button class="btn" data-goto="6">Back</button>
    ${decision === "rejected" ? `<span class="dim" style="font-size:12.5px">A rejected
      candidate stays in Development, which is not a deployable stage.</span>` : ""}
    ${pending ? `<span class="dim" style="font-size:12.5px">Promotion to Production needs a
      human; deploying the Staging version does not.</span>` : ""}</div>`;

  return wizFrame(7, "Model readiness",
    `The backend owns this decision. This page reads the gate; it cannot override it, and there
     is no path through this workflow that deploys a version the gate rejected.`,
    prjContext() + verdict + keyPanel + checkTable + champion + recordedPanel + nav);
}


/* ===================================================================== */
/* 08 Deploy                                                             */
/* ===================================================================== */
async function stepDeploy(){
  if(!PRJ.modelVersion) return wizFrame(8, "Deploy", "No registered model version.",
    unavailable("Nothing has been registered yet."));

  await ensureModelName();

  /* Deployability is a property of the version's stage, which is what the
     deployment API actually checks -- read it back rather than inferring it
     from the gate decision this browser happens to remember. */
  const DEPLOYABLE = ["Staging", "Production", "Validation"];
  let stage = null, stageError = null;
  try {
    const v = await api.get(`/api/v1/models/${encodeURIComponent(PRJ.modelName)}`
      + `/versions/${PRJ.modelVersion}`, 0);
    stage = v.stage;
  } catch(e){ stageError = e.message; }
  const approved = DEPLOYABLE.includes(String(stage));

  let current = null;
  try { current = await api.get("/api/v1/deployments/current", 5000); } catch(e){ /* below */ }

  const live = current && current.deployment ? current.deployment : null;
  const state = card("Endpoint now", live ? `
    <div class="grid g4">
      ${kpi("Endpoint", `<span class="mono">${esc(live.endpoint_name)}</span>`)}
      ${kpi("Serving", `<span class="mono">v${int(live.current_version)}</span>`)}
      ${kpi("State", badge(live.state, live.state === "live" ? "ok" : "info"))}
      ${kpi("Previous", live.previous_version != null
        ? `<span class="mono">v${int(live.previous_version)}</span>` : NA)}
    </div>` : unavailable(current
      ? "No deployment has been created for this endpoint yet."
      : "The deployments API could not be reached."), { flush:true });

  const form = approved ? `
    <div style="display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end">
      <div><label for="prjstrategy" style="display:block;font-size:12px;font-weight:600;margin-bottom:6px">
        Strategy</label>
        <select id="prjstrategy">
          <option value="blue_green" selected>Blue / green</option>
          <option value="direct">Direct</option>
          <option value="canary">Canary</option>
        </select></div>
      <div style="flex:1 1 260px"><p class="dim" style="margin:0;font-size:12.5px;line-height:1.6">
        Deploys <span class="mono">${esc(PRJ.modelName||"")} v${int(PRJ.modelVersion)}</span>.
        Shadow deployments are not offered here because they mirror traffic from a live version
        rather than replacing it &mdash; start one from the
        <a href="#/deployments">Deployments page</a>.</p></div>
    </div>
    <div style="margin-top:14px"><button class="btn pri" id="prjdeploy">Deploy this version</button></div>
    <div id="prjdeployresult" style="margin-top:14px"></div>`
    : `<div class="verdict fail">
        <span class="mark" aria-hidden="true">&#128274;</span>
        <span class="txt"><b>DEPLOYMENT LOCKED</b>
          <span>${stageError
            ? "The version's stage could not be read."
            : `Version ${esc(String(PRJ.modelVersion))} is in
               <b>${esc(stage || "an unknown stage")}</b>, which is not deployable.`}</span>
        </span></div>
      <p style="margin:0;font-size:13px;line-height:1.6">The deployment API accepts only
        ${DEPLOYABLE.map(s => `<span class="mono">${esc(s)}</span>`).join(", ")}. A candidate the
        approval gate rejected stays in <span class="mono">Development</span> and is refused
        there. The API does take an override that skips this check; <b>this workflow never
        sends it</b>. Train a candidate that satisfies the policy, or promote an eligible
        version from the <a href="#/models">Model Registry</a>.</p>`;

  return wizFrame(8, "Deploy",
    `Rolling a version onto the serving endpoint. The strategy decides how traffic moves; the
     version's stage decides whether it moves at all, and the approval gate decides the stage.`,
    prjContext() + state + card("Roll out", form, { flush:true }) + `<div class="wiznav">
      ${PRJ.deployedVersion ? `<button class="btn pri" data-goto="9">Continue to predictions</button>` : ""}
      <button class="btn" data-goto="7">Back</button></div>`);
}

/* ===================================================================== */
/* 09 Predict                                                            */
/* ===================================================================== */
/* Online scoring accepts exactly one request shape. Rather than hardcode a
   copy of it here -- which would drift the first time the backend changed --
   the field list is read from the served OpenAPI document, and the step
   refuses to render a form it knows would be rejected. */


async function stepPredict(){
  if(!PRJ.deployedVersion) return wizFrame(9, "Predict", "Nothing is deployed from this workflow.",
    unavailable("Deploy a model first.") + `<div class="wiznav">
      <button class="btn pri" data-goto="8">Back to deployment</button></div>`);

  let schema = null, schemaError = null;
  try { schema = await predictSchema(); }
  catch(e){ schemaError = e.message; }

  if(!schema) return wizFrame(9, "Test predictions",
    "The request schema for online scoring could not be read.",
    prjContext() + unavailable(schemaError || "The API description did not describe /api/v1/predict."));

  const need = schema.required;
  const have = new Set((PRJ.columnNames || []).map(c => String(c)));
  const missing = need.filter(f => !have.has(f));

  /* A model trained on differently-shaped data cannot be exercised through
     this endpoint. Saying so beats rendering a form that would 422. */
  if(missing.length) return wizFrame(9, "Test predictions",
    `Online scoring on this platform accepts one fixed request shape.`,
    prjContext() + card("Not available for this dataset", `
      <div class="verdict fail"><span class="mark" aria-hidden="true">&times;</span>
        <span class="txt"><b>THIS DATASET DOES NOT MATCH THE SCORING SCHEMA</b>
          <span>The prediction endpoint would reject every request built from it.</span></span></div>
      <p style="margin:0 0 10px;font-size:13px;line-height:1.6">
        <span class="mono">POST /api/v1/predict</span> validates against a single fixed schema
        (<span class="mono">${esc(schema.name)}</span>), which belongs to this platform's
        reference model. Your dataset does not carry
        ${missing.length} of its ${need.length} required fields, so there is no honest way to
        build a request from it. The model you trained is registered and deployable; it is the
        <i>serving contract</i> that is fixed, not the training path.</p>
      <p class="dim" style="margin:0;font-size:12.5px">Missing:
        ${missing.map(m => `<span class="mono">${esc(m)}</span>`).join(", ")}</p>`, { flush:true })
    + `<div class="wiznav"><button class="btn pri" data-goto="10">Continue to monitoring</button>
       <button class="btn" data-goto="8">Back</button></div>`);

  /* Columns line up, so prefill from a real row of the uploaded data rather
     than inventing plausible-looking values. */
  let sample = null;
  try {
    const pv = await api.get(`/api/v1/datasets/${encodeURIComponent(PRJ.version)}/preview?rows=1`, 60000);
    sample = (pv.sample || [])[0] || null;
  } catch(e){ sample = null; }

  const fields = need.map(f => {
    const p = schema.properties[f] || {};
    const t = p.type || (p.anyOf ? "string" : "string");
    const numeric = t === "number" || t === "integer";
    const val = sample && sample[f] != null ? String(sample[f]) : "";
    return `<label style="display:block">
      <span style="display:block;font-size:11.5px;font-weight:600;margin-bottom:4px">
        ${esc(f)}</span>
      <input class="prjfeat" data-f="${esc(f)}" data-num="${numeric ? "1" : "0"}"
        type="${numeric ? "number" : "text"}" step="any" value="${esc(val)}"
        style="width:100%" ${p.minimum != null ? `min="${esc(p.minimum)}"` : ""}
        ${p.maximum != null ? `max="${esc(p.maximum)}"` : ""}></label>`;
  }).join("");

  return wizFrame(9, "Test predictions",
    `A real request against the live endpoint. The response reports which model version and
     which variant actually served it, so a canary or shadow rollout is visible per request.`,
    prjContext() + card("Score one record", `
      ${sample ? `<p class="dim" style="margin:0 0 12px;font-size:12.5px">Prefilled from the first
        row of <span class="mono">${esc(PRJ.version)}</span>. Edit any value before scoring.</p>`
        : `<p class="dim" style="margin:0 0 12px;font-size:12.5px">Fill in the fields below.</p>`}
      <div class="grid g3" style="gap:12px">${fields}</div>
      <div style="margin-top:14px"><button class="btn pri" id="prjpredict">Score</button></div>
      <div id="prjpredresult" style="margin-top:14px"></div>`, { flush:true })
    + `<div class="wiznav"><button class="btn pri" data-goto="10">Continue to monitoring</button>
       <button class="btn" data-goto="8">Back</button></div>`);
}


/* ===================================================================== */
/* 10 Monitor                                                            */
/* ===================================================================== */
async function stepMonitor(){
  const r = await loadAll({
    summary: "/api/v1/monitoring/summary",
    current: "/api/v1/deployments/current",
    drift:   "/api/v1/drift/latest",
  }, 5000);

  const svc = sect(r.summary, d => {
    const s = d.service || {}, lat = s.latency || {}, lp = d.live_performance || {};
    return card("Serving", `
      <div class="grid g4">
        ${kpi("Model", `<span class="mono">${esc(d.model_name||"-")}${d.model_version != null
          ? ` v${d.model_version}` : ""}</span>`, esc(d.model_stage||""))}
        ${kpi("Requests", int(s.request_count), `last ${int(s.window_minutes)} min`)}
        ${kpi("p95 latency", ms(lat.p95_ms),
          s.latency_slo_met === false ? "SLO breached" : "within SLO")}
        ${kpi("Open alerts", int(d.open_alerts))}
      </div>
      <div class="grid g4" style="margin-top:14px">
        ${kpi("Error rate", pct(s.error_rate, 2))}
        ${kpi("Positive rate", pct(d.prediction_positive_rate, 2))}
        ${kpi("Labelled samples", int(lp.labelled_samples))}
        ${kpi("Live F1", lp.available ? num(lp.f1, 4) : NA,
          lp.available ? "from returned labels" : "needs labels")}
      </div>
      <p class="dim" style="margin:14px 0 0;font-size:12.5px">${lp.available
        ? "Live performance is computed only over predictions whose true outcome came back."
        : esc(lp.detail || "Live accuracy cannot be computed without ground-truth labels.")}</p>`,
      { flush:true });
  }, "monitoring summary");

  /* /drift/latest answers {found, report}: "no scan yet" is a legitimate
     state, and reporting it as stable would be an invented reassurance. */
  const drift = sect(r.drift, d => {
    const rep = d && d.report;
    if(!d || !d.found || !rep) return card("Drift", unavailable(
      "No drift report has been produced yet. Run a scan from the Drift Detection page."),
      { flush:true });
    const drifted = (rep.drifted_features || []).length;
    const examined = (rep.feature_drift || []).length;
    const status = String(rep.concept_drift_status || "unknown");
    const measured = status === "measured";
    return card("Latest drift scan", `
      <div class="grid g4">
        ${kpi("Status", rep.drift_detected ? badge("DRIFT","warn",true) : badge("STABLE","ok",true))}
        ${kpi("Features drifted", int(drifted), `of ${examined} examined`)}
        ${kpi("Share drifted", pct(rep.dataset_drift_share, 1))}
        ${kpi("Scanned", when(rep.created_at))}
      </div>
      <div style="margin-top:14px;padding-top:12px;border-top:1px solid var(--line-2)">
        <div style="display:flex;gap:10px;align-items:center;margin-bottom:6px">
          <b style="font-size:13px">Concept drift</b>
          ${badge(status.replace(/_/g," "), measured ? "info" : "mute")}
        </div>
        <p class="dim" style="margin:0;font-size:12.5px">${esc(rep.concept_drift_detail
          || "No detail recorded.")}</p>
      </div>
      <p class="dim" style="margin:12px 0 0;font-size:12.5px">The counts above are <b>data and
        prediction drift</b> &mdash; changes in the inputs and outputs. It is not concept drift,
        which is a change in the relationship between features and outcome and is only
        measurable once true labels come back${measured
          ? "; the status above says labels were available for this scan."
          : "."}</p>`, { flush:true });
  }, "drift");

  const dep = sect(r.current, d => d.deployment ? card("Deployment", `
    <div class="grid g4">
      ${kpi("Endpoint", `<span class="mono">${esc(d.deployment.endpoint_name)}</span>`)}
      ${kpi("Serving", `<span class="mono">v${int(d.deployment.current_version)}</span>`)}
      ${kpi("State", badge(d.deployment.state, d.deployment.state === "live" ? "ok" : "info"))}
      ${kpi("Strategy", esc(d.deployment.strategy || "-"))}
    </div>`, { flush:true })
    : card("Deployment", unavailable("No deployment on this endpoint."), { flush:true }), "deployment");

  return wizFrame(10, "Monitoring",
    `The workflow ends where operations begin. Everything below is live platform state, shared
     with the Command Center &mdash; this step is a starting point, not a separate copy.`,
    prjContext() + svc + dep + drift + card("Where to go next", `
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <a class="btn" href="#/monitoring">Monitoring</a>
        <a class="btn" href="#/drift">Drift Detection</a>
        <a class="btn" href="#/retraining">Retraining</a>
        <a class="btn" href="#/gates">Quality Gates</a>
        <a class="btn" href="#/audit">Audit Log</a>
        <a class="btn" href="#/overview">Command Center</a>
      </div>
      <p class="dim" style="margin:12px 0 0;font-size:12.5px">Retraining is driven by the
        platform's own triggers and by drift, not by this workflow.</p>`, { flush:true })
    + `<div class="wiznav">
        <button class="btn" data-goto="9">Back</button>
        <button class="btn" id="prjreset">Start another project</button></div>`);
}


function candidateDetail(c, run){
  const m = c.metrics || {};
  const keys = Object.keys(m);
  return `<div class="grid g2" style="gap:10px">
      ${kpi("Algorithm", `<span class="mono">${esc(c.algorithm)}</span>`)}
      ${kpi("Status", runStatusBadge(c.status))}
      ${kpi("Rank", c.rank != null ? `#${int(c.rank)}` : NA)}
      ${kpi("Duration", c.duration_seconds != null
        ? `${Number(c.duration_seconds).toFixed(1)}s` : NA)}
    </div>
    <div style="margin-top:16px"><b style="font-size:13px">Metrics</b>
      ${keys.length ? `<div class="grid g2" style="gap:10px;margin-top:8px">
        ${keys.map(k => kpi(k.replace(/_/g," "), metricValue(m[k]))).join("")}</div>`
        : `<p class="dim" style="margin:8px 0 0;font-size:12.5px">
           No metrics were recorded for this candidate.</p>`}</div>
    <div style="margin-top:16px"><b style="font-size:13px">Provenance</b>
      <div class="kv" style="margin-top:8px">
        <div><span class="dim">Dataset</span> <span class="mono">${esc(run.dataset_version||"-")}</span></div>
        <div><span class="dim">Target</span> <span class="mono">${esc(run.target_column||"-")}</span></div>
        <div><span class="dim">Problem</span> ${esc(String(run.problem_type||"-").replace(/_/g," "))}</div>
        <div><span class="dim">Primary metric</span> <span class="mono">${esc(run.primary_metric||"-")}</span></div>
        <div><span class="dim">Tuning</span> ${run.tune ? "enabled" : "off"}</div>
        <div><span class="dim">Registered version</span> ${c.model_version != null
          ? `<span class="mono">v${int(c.model_version)}</span>` : "not registered"}</div>
      </div></div>
    ${c.error ? `<div style="margin-top:16px"><b style="font-size:13px">Error</b>
      <pre class="scroll" style="margin:8px 0 0">${esc(c.error)}</pre></div>` : ""}
    <p class="dim" style="margin:16px 0 0;font-size:12px">Hyperparameters are recorded with the
      MLflow run rather than on the candidate record; see the
      <a href="#/experiments">Experiments</a> page.</p>`;
}
