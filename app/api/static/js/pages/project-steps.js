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
        Validation was not run for this upload.</p>`}`)}
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
    const tp = p.suggested_target && p.suggested_target.column === chosen ? p.suggested_target : null;
    const classes = Object.keys((tp && tp.class_balance) || {});
    verdict = supported ? `
      <div style="margin-top:20px">${card("Target confirmed", `
        <div class="grid g3">
          ${kpi("Target", `<span class="mono">${esc(chosen)}</span>`)}
          ${kpi("Problem type", esc(pretty))}
          ${kpi("Supported", badge("YES","ok",true))}
        </div>
        ${d.metric_note ? `<p class="dim" style="margin:14px 0 0;font-size:12.5px">
          ${esc(d.metric_note)}</p>` : ""}
        <div class="grid g2" style="margin-top:16px">
          <div class="field"><label for="prjname">Model name</label>
            <input id="prjname" maxlength="64" autocomplete="off" value="${esc(PRJ.nameAsked || "")}"
              placeholder="${esc(suggestedName(chosen))}">
            <span class="hint">The trained model is registered as a version of this name. Leave it
              empty for <span class="mono">${esc(suggestedName(chosen))}</span>. A name keeps one
              target for life, so a different target needs a different name.</span></div>
          <div class="field"><label for="prjpos">Positive class</label>
            <select id="prjpos"><option value="">automatic — yes/true/1 if present, else the rarer class</option>
              ${classes.map(k => `<option value="${esc(k)}" ${k === PRJ.positive ? "selected" : ""}>${
                esc(k)} (${pct(tp.class_balance[k], 1)})</option>`).join("")}</select>
            <span class="hint">The class the model's probability is the probability of. The rule
              used is recorded with the model.</span></div>
        </div>`)}
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
          the training stack, not of the profiler: the target is encoded as two classes,
          evaluation reads the probability of the positive one, and the gate, drift on predictions
          and live quality are all defined over two classes. Training a ${esc(pretty)} target here
          would produce a meaningless score, so the API refuses it.</p>
        <p class="dim" style="margin:0;font-size:12.5px">Supported:
          ${(d.supported_problem_types||[]).map(t =>
            `<span class="mono">${esc(String(t).replace(/_/g," "))}</span>`).join(", ") || "&mdash;"}</p>`,
        {})}
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
          <option value="Production" selected>Production — may take live traffic</option>
          <option value="Staging">Staging — shadow evaluation only</option>
        </select></div>
      <p class="dim" style="flex:1 1 320px;margin:0;font-size:12.5px;line-height:1.6">
        Both paths register the resulting model and put it through the approval gate
        automatically &mdash; that happens inside the run, which is why there is no separate
        register button later. Asking for a stage only asks for the attempt. Staging needs the
        absolute thresholds; Production also needs to beat the live version on held-out rows
        neither trained on, and only a Production version takes live traffic. Where the
        environment requires it, a person approves either.</p>
    </div>`);

  const note = `<div class="note" style="margin-top:16px">
    <b>Where this runs.</b> <span style="font-size:12.5px">A run is queued as a job in the
    platform database and executed by the API's job worker, which streams its log to the
    <a href="#/jobs">Jobs</a> page. The worker shares the API container's CPU, so training
    competes with request handling. If the process restarts mid-run the job is marked failed
    on recovery and can be retried; it is never silently resumed. Each candidate is a full
    training run, so <i>n</i> candidates cost roughly <i>n</i> times one run.</span></div>`;

  const naming = `<p class="dim" style="margin:0 0 14px;font-size:12.5px">Registering as
    <b class="mono">${esc(PRJ.nameAsked || suggestedName(PRJ.target))}</b>, predicting
    <span class="mono">${esc(PRJ.target)}</span>${PRJ.positive
      ? `, positive class <b>${esc(PRJ.positive)}</b>` : ""}.
    <button class="linkbtn" data-goto="3">Change</button></p>`;

  return wizFrame(4, "How should FMOps train?",
    `Both routes use the same training pipeline, the same evaluation and the same approval gate.
     AutoML compares several estimators and ranks them; manual training fits the one you pick.`,
    prjContext() + naming + shared + `<div class="choices" style="margin-top:16px">${automl}${manual}</div>`
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
    ${PRJ.jobId ? `<a class="btn" href="#/jobs/${encodeURIComponent(PRJ.jobId)}">Open the job log</a>` : ""}
    <button class="btn" data-goto="4">Back</button></div>`;

  return wizFrame(5, live ? "Training candidate models" : "Evaluation",
    live
      ? `The run is executing now. This page follows it until it reaches a terminal state and
         then stops polling &mdash; there is nothing further to learn from a finished run.`
      : `Results as the platform recorded them. Nothing here is recomputed in the browser.`,
    prjContext() + body
    + (live && PRJ.jobId ? jobPanel({ id: PRJ.jobId, kind: PRJ.runKind, status: run.status }) : "")
    + `<div id="prjrunbox"></div>` + nav)
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
         ${Number(run.duration_seconds).toFixed(1)}s</p>` : ""}`)}`;

  if(live) return progress;

  if(run.status === "failed") return progress + card("Run failed", `
    <div class="verdict fail"><span class="mark" aria-hidden="true">&times;</span>
      <span class="txt"><b>NO MODEL WAS PRODUCED</b>
        <span>The error below is what the backend recorded.</span></span></div>
    <pre class="scroll" style="margin:0">${esc(run.error || "no error recorded")}</pre>`,
    {});

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
      shown is the rule that actually ordered these candidates.</p>`) : "";

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
    </div>`);

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
    {});

  const keys = Object.keys(m);
  return head + card("Metrics", keys.length ? `<div class="grid g4">
    ${keys.slice(0,8).map(k => kpi(k.replace(/_/g," "), metricValue(m[k]))).join("")}</div>`
    : unavailable("The run recorded no metrics."));
}


/* ===================================================================== */
/* 06 Registry                                                           */
/* ===================================================================== */
async function stepRegistry(){
  if(!PRJ.runId) return wizFrame(6, "Registry", "No run has been started.",
    unavailable("Nothing has been trained yet."));

  let run = null;
  try { run = await api.get(runPath(), 0); absorbRun(run); } catch(e){ /* fall through */ }

  if(!PRJ.modelVersion || !PRJ.modelName){
    return wizFrame(6, "Model registry", "Nothing was registered by this run.",
      prjContext() + `<div class="note warnnote"><b>No model version was registered.</b>
        <div style="font-size:12.5px;margin-top:6px">${esc((run && run.error) || "The run did not "
          + "record a registered version — every candidate may have failed. See the run's job log.")}</div></div>
      <div class="wiznav"><button class="btn pri" data-goto="4">Train again</button>
        <button class="btn" data-goto="5">Back to results</button></div>`);
  }

  const enc = encodeURIComponent(PRJ.modelName);
  let mine = null, sig = null;
  try { mine = await api.get(`/api/v1/models/${enc}/versions/${PRJ.modelVersion}`, 0); } catch(e){ mine = null; }
  try { sig = (await api.get(`/api/v1/models/${enc}/signature?version=${PRJ.modelVersion}`, 0)).signature; } catch(e){ sig = null; }
  if(mine){ PRJ.registeredStage = mine.stage; prjSave(); }

  const detail = mine ? card("Registered", `
    <div class="verdict pass" style="margin-bottom:14px">
      <span class="mark" aria-hidden="true">&check;</span>
      <span class="txt"><b>MODEL REGISTERED</b>
        <span>${esc(PRJ.modelName)} v${int(mine.version)} — recorded by the registry during the run.</span></span></div>
    <div class="grid g4">
      ${kpi("Model", `<a href="#/models/${enc}"><b>${esc(PRJ.modelName)}</b></a>`)}
      ${kpi("Version", `<a class="mono" href="#/models/${enc}/${mine.version}">v${int(mine.version)}</a>`)}
      ${kpi("Stage", stageBadge(mine))}
      ${kpi("Algorithm", `<span class="mono">${esc(mine.algorithm || "-")}</span>`)}
    </div>
    ${mine.metrics ? `<div class="grid g4" style="margin-top:14px">
      ${["roc_auc","f1","precision","recall"].map(k => kpi(k.replace(/_/g," "), num(mine.metrics[k], 4))).join("")}</div>` : ""}
    ${sig ? `<div class="eyebrow" style="margin-top:16px">Input contract recorded with this version</div>
      <p class="dim" style="margin:4px 0 0;font-size:12.5px">${int(sig.features.length)} features ·
        predicts <span class="mono">${esc(sig.target)}</span> · positive class
        <b>${esc((sig.display_labels || sig.class_labels)[1])}</b> (${esc(sig.positive_label_rule || "")}).
        Every prediction request is checked against it.</p>` : ""}`)
    : `<div class="note warnnote"><b>${esc(PRJ.modelName)} v${int(PRJ.modelVersion)} was recorded by
        the run but could not be read back from the registry.</b></div>`;

  return wizFrame(6, "Model registry",
    `Registration happens inside the run: the winner becomes a version of
     <b>${esc(PRJ.modelName)}</b> and goes through the gate before the run reports back.
     ${PRJ.runKind === "automl" ? `Every other candidate that trained is kept as a Development
     version too, so each leaderboard row stays traceable; only the winner is gated.` : ""}
     This step shows what the registry actually holds.`,
    prjContext() + detail + `<div class="wiznav">
      <button class="btn pri" data-goto="7">Continue to approval</button>
      <button class="btn" data-goto="5">Back to results</button></div>`);
}


/* ===================================================================== */
/* 07 Approval                                                           */
/* ===================================================================== */
/* The gate's recorded decision, and -- where the environment requires it --
   the human one. Nothing here re-decides: the buttons call the same /approve
   and /reject endpoints the model page uses, which re-run the gate. */
async function stepApproval(){
  if(!PRJ.modelVersion || !PRJ.modelName) return wizFrame(7, "Approval gate",
    "There is no registered version to gate yet.", unavailable("Train and register a model first."));
  const enc = encodeURIComponent(PRJ.modelName);
  let v = null, decisions = [];
  try { v = await api.get(`/api/v1/models/${enc}/versions/${PRJ.modelVersion}`, 0); } catch(e){ v = null; }
  try { decisions = (await api.get(`/api/v1/models/${enc}/decisions?version=${PRJ.modelVersion}`, 0)).decisions || []; }
  catch(e){ decisions = []; }
  if(!v) return wizFrame(7, "Approval gate", "The version could not be read.", errorState("registry unavailable", "newproject"));

  const latest = decisions[0] || null;
  const automated = decisions.find(d => d.source === "pipeline") || null;
  PRJ.gateDecision = latest ? latest.decision : null;
  PRJ.registeredStage = v.stage; prjSave();
  const deployable = ["Staging","Production"].includes(v.stage);
  const staged = v.stage === "Staging";
  const waiting = String(v.status) === "pending";

  const body = (latest ? verdictFor(latest, v) : "")
    + (automated && (automated.checks || []).length ? evidenceTable(automated.checks) : "")
    + (automated && automated.comparison ? championCard(automated.comparison) : "")
    + (waiting ? `<div class="note warnnote" style="margin-top:14px"><b>Every automated check passed.</b>
        This environment requires a person to approve promotion (<span class="mono">pending_manual</span>).
        Approval re-runs the gate first — it cannot push a failing version through.</div>
      <div class="actions" style="margin-top:12px">
        <button class="btn pri" data-approve="${esc(String(v.version))}" data-name="${esc(PRJ.modelName)}"
          data-target="Production">Approve…</button>
        <button class="btn danger" data-reject="${esc(String(v.version))}" data-name="${esc(PRJ.modelName)}">Reject…</button></div>` : "")
    + (staged ? `<div class="note warnnote" style="margin-top:14px"><b>Approved into Staging.</b>
        Staging clears the thresholds but is never compared with the live version, so it can only
        be shadowed. Promote it to Production to let it answer callers — that re-runs the gate and
        the comparison.
        <div class="actions" style="margin-top:10px"><button class="btn pri" data-approve="${esc(String(v.version))}"
          data-name="${esc(PRJ.modelName)}" data-target="Production">Promote to Production…</button></div></div>` : "")
    + (!deployable && !waiting ? `<div class="note bad" style="margin-top:14px"><b>Not eligible for deployment.</b>
        The version is in ${esc(v.stage)}${String(v.status) === "rejected" ? " and was rejected" : ""}. Only a
        version the gate moved into Production can take live traffic. Train again with more or
        better data, or a different algorithm.</div>` : "");

  return wizFrame(7, "Approval gate",
    `The gate compares the version against absolute thresholds and against the current production
     version on held-out rows neither was trained on. Its decision is recorded, with every check.`,
    prjContext() + card("Gate decision", body, { flush:false, sub: latest ? `${latest.source} · ${esc(ago(latest.created_at))}` : "no record" })
    + `<div class="wiznav">
      ${v.stage === "Production" ? `<button class="btn pri" data-goto="8">Continue to deployment</button>` : ""}
      <button class="btn" data-goto="6">Back</button>
      <a class="btn" href="#/models/${enc}/${esc(String(v.version))}?tab=evaluation">Open the full evaluation</a></div>`);
}


/* ===================================================================== */
/* 08 Deploy                                                             */
/* ===================================================================== */
async function stepDeploy(){
  if(!PRJ.modelVersion || !PRJ.modelName) return wizFrame(8, "Deploy", "Nothing to deploy yet.",
    unavailable("Register and approve a version first."));
  const enc = encodeURIComponent(PRJ.modelName);
  let v = null, current = null;
  try { v = await api.get(`/api/v1/models/${enc}/versions/${PRJ.modelVersion}`, 0); } catch(e){ v = null; }
  try { current = await api.get(`/api/v1/deployments/current?model=${enc}`, 0); } catch(e){ current = null; }
  const dep = current && current.deployment;
  const live = dep && String(dep.current_version) === String(PRJ.modelVersion) && dep.state === "live";
  if(v) PRJ.registeredStage = v.stage;
  if(live) PRJ.deployedVersion = PRJ.modelVersion;
  prjSave();
  const deployable = v && v.stage === "Production";

  const status = card("Endpoint", `<div class="grid g4">
      ${kpi("Endpoint", `<span class="mono">${esc((current && current.endpoint) || "-")}</span>`, "this model's own")}
      ${kpi("Serving", dep && dep.current_version != null ? `<span class="mono">v${int(dep.current_version)}</span>` : NA,
            dep ? esc(dep.state) : "nothing deployed")}
      ${kpi("This version", v ? stageBadge(v) : NA)}
      ${kpi("Strategy", dep ? badge(dep.strategy, "mute") : NA)}
    </div>`, { sub:"GET /api/v1/deployments/current" });

  const action = live ? `<div class="verdict pass"><span class="mark" aria-hidden="true">&check;</span>
      <span class="txt"><b>LIVE</b><span>${esc(PRJ.modelName)} v${int(PRJ.modelVersion)} is serving on its endpoint.</span></span></div>`
    : !deployable ? `<div class="note bad"><b>This version cannot take live traffic.</b> It is in
        ${esc(v ? v.stage : "an unknown stage")}; only a version approved into Production can —
        Production approval compares it with the live version on held-out rows first.
        <div style="margin-top:8px"><button class="btn" data-goto="7">Back to approval</button></div></div>`
    : `<div class="grid g2" style="align-items:end">
        <div><label for="prjstrategy">Strategy</label>
          <select id="prjstrategy">
            <option value="blue_green">blue/green — health-check, then switch all traffic</option>
            <option value="direct">direct — switch immediately</option>
            <option value="canary" ${dep ? "" : "disabled"}>canary — shift traffic in steps${dep ? "" : " (needs a live version)"}</option>
            <option value="shadow" ${dep ? "" : "disabled"}>shadow — mirror traffic${dep ? "" : " (needs a live version)"}</option>
          </select></div>
        <div><button class="btn pri" id="prjdeploy">Deploy v${esc(String(PRJ.modelVersion))}</button></div>
      </div>
      <p class="dim" style="margin:12px 0 0;font-size:12.5px">The rollout runs as a background job. A canary
        waits between steps while it watches errors and latency; cancelling it puts all traffic back on
        the live version.</p>
      <div id="prjdeployresult" style="margin-top:14px"></div>`;

  return wizFrame(8, "Deploy",
    `Deployment puts an approved version onto <b>${esc(PRJ.modelName)}</b>'s own endpoint. Other models'
     endpoints are untouched.`,
    prjContext() + status + card("Roll out", action)
    + `<div class="wiznav">${live ? `<button class="btn pri" data-goto="9">Continue to prediction</button>` : ""}
      <button class="btn" data-goto="7">Back</button></div>`);
}


/* ===================================================================== */
/* 09 Predict                                                            */
/* ===================================================================== */
/* The form is built from the serving version's recorded signature, so it
   takes exactly the features this model was trained on -- whatever dataset
   it came from. */
async function stepPredict(){
  if(!PRJ.modelName) return wizFrame(9, "Predict", "Nothing is deployed yet.", unavailable("Deploy a version first."));
  const enc = encodeURIComponent(PRJ.modelName);
  let s = null;
  try { s = await api.get(`/api/v1/models/${enc}/signature`, 0); }
  catch(e){ return wizFrame(9, "Predict", "The model's input contract could not be read.", errorState(e.message, "newproject")); }
  const sig = s.signature;
  if(!sig) return wizFrame(9, "Predict", "This version has no recorded input contract.",
    unavailable(s.detail || "Retrain to record one."));
  window.__predctx = { name: PRJ.modelName, labels: sig.display_labels || sig.class_labels,
                       features: sig.features, example: (s.example || {}).features || {} };

  return wizFrame(9, "Score a record",
    `The form below is built from the input contract recorded when v${int(s.version)} was trained:
     ${int(sig.features.length)} features, predicting <span class="mono">${esc(sig.target)}</span>.
     The request is checked against the version that actually serves it.`,
    prjContext() + card(`Predict with ${PRJ.modelName}`, `
      <form id="predform" class="formgrid" novalidate>
        ${sig.features.map(f => featureInput(f, window.__predctx.example[f.name])).join("")}
      </form>
      <div class="actions" style="margin-top:14px">
        <button class="btn pri" id="predgo">Predict</button>
        <button class="btn" id="predfill">Fill with training medians</button>
      </div>`)
    + `<div id="predresult"></div>
      <div class="wiznav"><button class="btn pri" data-goto="10">Continue to monitoring</button>
        <button class="btn" data-goto="8">Back</button>
        <a class="btn" href="#/predict?model=${enc}">Open the Predict page</a></div>`);
}


/* ===================================================================== */
/* 10 Monitor                                                            */
/* ===================================================================== */
async function stepMonitor(){
  if(!PRJ.modelName) return wizFrame(10, "Monitor", "Nothing is deployed yet.", unavailable("Deploy a version first."));
  const enc = encodeURIComponent(PRJ.modelName);
  const r = await loadAll({ sum:`/api/v1/monitoring/summary?model=${enc}`,
    drift:`/api/v1/drift/latest?model=${enc}`, trig:`/api/v1/retraining/trigger/evaluate?model=${enc}` }, 0);
  const d = r.sum.ok ? r.sum.data : {}, svc = d.service || {}, lp = d.live_performance || {};
  const dr = r.drift.ok && r.drift.data.found ? r.drift.data.report : null;
  const t = r.trig.ok ? r.trig.data : null;

  return wizFrame(10, "Watch it in production",
    `Everything below is measured for <b>${esc(PRJ.modelName)}</b> only: its endpoint's traffic, its
     drift against its own training data, and whether its retraining trigger would fire.`,
    prjContext() + `<div class="grid g2">
      ${card("Traffic", `<div class="grid g3">
        ${kpi("Requests", int(svc.request_count), "last hour")}
        ${kpi("p95 latency", ms((svc.latency || {}).p95_ms))}
        ${kpi("Error rate", pct(svc.error_rate, 2))}</div>
        <p class="dim" style="margin:12px 0 0;font-size:12.5px">${lp.available
          ? `Live F1 ${num(lp.f1, 4)} over ${int(lp.labelled_samples)} labelled predictions.`
          : esc(lp.detail || "Live quality needs ground-truth labels: record outcomes after predicting.")}</p>`)}
      ${card("Drift", dr ? `<div class="grid g3">
          ${kpi("Status", dr.drift_detected ? badge("drift","warn") : badge("stable","ok"))}
          ${kpi("Score", num(dr.dataset_drift_score, 4))}
          ${kpi("Drifted", `${(dr.drifted_features || []).length} feature(s)`)}</div>
          <p class="dim" style="margin:12px 0 0;font-size:12.5px">This measures how inputs moved. It is not
            concept drift, which needs labels: ${esc(dr.concept_drift_status)}.</p>`
        : `<div class="state"><div class="big">Not scanned yet</div>A scan needs enough logged predictions;
            the scheduler runs one automatically once they arrive.</div>`)}
    </div>`
    + card("Retraining", t ? `<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
        ${t.should_retrain ? badge("would retrain","warn",true) : badge("not needed","ok",true)}
        <span class="dim" style="font-size:12.5px">${esc(t.reason)}</span></div>` : unavailable())
    + `<div class="wiznav">
        <a class="btn pri" href="#/models/${enc}">Open ${esc(PRJ.modelName)}</a>
        <a class="btn" href="#/monitoring?model=${enc}">Observability</a>
        <a class="btn" href="#/drift?model=${enc}">Drift</a>
        <button class="btn" data-goto="9">Back</button>
        <button class="btn" id="prjreset">Start a new project</button></div>`);
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
