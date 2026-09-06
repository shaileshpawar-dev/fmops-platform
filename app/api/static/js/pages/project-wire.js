/* Guided build workflow -- event handling.
 *
 * Everything that responds to a click, a drop, a keystroke or a poll tick.
 * Renderers live in project-steps.js; state lives in project.js.
 */


/* ===================================================================== */
/* Wiring                                                                */
/* ===================================================================== */
let PRJ_POLL = null;


function wireProject(){
  if(PRJ_POLL){ clearTimeout(PRJ_POLL); PRJ_POLL = null; }

  document.querySelectorAll("[data-goto]").forEach(b => b.onclick = () => {
    const n = parseInt(b.dataset.goto, 10);
    if(stepState(n) === "locked"){ toast("That step is not available yet.", "bad"); return; }
    goStep(n);
  });

  document.querySelectorAll("#prjreset").forEach(b => b.onclick = async () => {
    const ok = await confirmAction({
      title: "Start over",
      body: "Clears this walkthrough's progress in your browser. Datasets, runs, model "
          + "versions and deployments already created stay exactly where they are.",
      confirm: "Start over",
    });
    if(!ok) return;
    prjReset(); api.bust(); goStep(1); render();
  });

  wireUpload();
  wireTarget();
  wireStart();
  wireDrawer();
  wireGate();
  wireDeploy();
  wirePredict();

  const poll = $("#prjpoll");
  if(poll) pollRunStep(poll.dataset.run, 0);
}


/* --- 01 upload ---------------------------------------------------------- */
function wireUpload(){
  const zone = $("#prjdrop"), input = $("#prjfile");
  if(!zone || !input) return;

  ["dragenter","dragover"].forEach(ev => zone.addEventListener(ev, e => {
    e.preventDefault(); zone.classList.add("over"); }));
  ["dragleave","drop"].forEach(ev => zone.addEventListener(ev, e => {
    e.preventDefault(); zone.classList.remove("over"); }));
  zone.addEventListener("drop", e => {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if(f) doUpload(f);
  });
  input.onchange = () => { const f = input.files && input.files[0]; if(f) doUpload(f); };
}


async function doUpload(file){
  const box = $("#prjupstatus");
  const fail = msg => { box.innerHTML =
    `<div class="note" style="border-color:var(--bad-line);background:var(--bad-bg)">
      <b>Upload failed.</b><br><span style="font-size:12.5px">${esc(msg)}</span></div>`;
    toast(msg, "bad"); };

  if(!/\.csv$/i.test(file.name)) return fail(
    `${file.name} is not a .csv file. This endpoint accepts CSV only.`);
  if(file.size === 0) return fail("That file is empty.");

  let limit = null;
  try { limit = (await api.get("/api/v1/datasets/limits", 300000)).max_upload_bytes; } catch(e){}
  if(limit && file.size > limit) return fail(
    `${file.name} is ${(file.size/1048576).toFixed(1)} MB, over the `
    + `${Math.floor(limit/1048576)} MB limit the API accepts.`);

  const needsKey = await authRequired();
  let key = null;
  if(needsKey){
    const proceed = await confirmAction({
      title: "Upload dataset",
      body: `Register ${file.name} (${(file.size/1024).toFixed(0)} KB) as a new dataset version `
          + `and validate it.`,
      confirm: "Upload", needsKey: true,
    });
    if(!proceed) return;
    if(!proceed.key){ toast("An API key is required to upload.", "bad"); return; }
    key = proceed.key;
  }

  const desc = ($("#prjdesc") && $("#prjdesc").value) || "";
  box.innerHTML = `<div class="loadrow"><span class="spin" aria-hidden="true"></span>
    <span>Uploading ${esc(file.name)}&hellip;</span></div>
    <div class="bar" style="margin-top:8px"><i id="prjbar" style="width:0%"></i></div>`;

  /* XHR rather than fetch: only XHR reports upload progress, and a multi-MB
     CSV on a slow link is exactly when a progress bar earns its place. */
  const url = "/api/v1/datasets/upload?filename=" + encodeURIComponent(file.name)
    + "&description=" + encodeURIComponent(desc) + "&validate=true";
  const body = await new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.setRequestHeader("Content-Type", "text/csv");
    xhr.setRequestHeader("Accept", "application/json");
    if(key) xhr.setRequestHeader("X-API-Key", key);
    xhr.upload.onprogress = e => {
      if(!e.lengthComputable) return;
      const bar = $("#prjbar");
      if(bar) bar.style.width = Math.round(e.loaded / e.total * 100) + "%";
    };
    xhr.onerror = () => reject(new Error(
      "The network dropped the request. Nothing was registered."));
    xhr.ontimeout = () => reject(new Error("The upload timed out."));
    xhr.onload = () => {
      let parsed = null;
      try { parsed = JSON.parse(xhr.responseText || "null"); } catch(e){ parsed = null; }
      if(xhr.status >= 200 && xhr.status < 300) return resolve(parsed);
      const msg = (parsed && parsed.error && parsed.error.message)
        || (parsed && parsed.detail) || `HTTP ${xhr.status}`;
      reject(new Error(msg));
    };
    xhr.send(file);
  }).catch(e => { fail(e.message); return null; });
  if(!body) return;

  PRJ = prjBlank();
  PRJ.version = body.version;
  PRJ.datasetName = file.name;
  PRJ.rows = body.rows;
  PRJ.columns = body.columns;
  PRJ.columnNames = body.column_names || [];
  PRJ.validation = body.validation || null;
  prjSave();
  api.bust();
  toast(`Registered as ${body.version}.`, "ok");
  render();
}


/* --- 03 target ---------------------------------------------------------- */
function wireTarget(){
  document.querySelectorAll("[data-usetarget]").forEach(b => b.onclick = () => {
    PRJ.target = b.dataset.usetarget; prjSave(); render();
  });
  const set = $("#prjsettarget"), sel = $("#prjtarget");
  if(set && sel) set.onclick = () => {
    if(!sel.value){ toast("Choose a column first.", "bad"); return; }
    PRJ.target = sel.value; prjSave(); render();
  };
  const clear = $("#prjcleartarget");
  if(clear) clear.onclick = () => {
    PRJ.target = null; PRJ.problemSupported = null; prjSave(); render();
  };
}


/* --- 04 start a run ----------------------------------------------------- */
function wireStart(){
  const boxes = [...document.querySelectorAll(".prjcand")];
  const count = $("#prjcandcount");
  const sync = () => {
    const n = boxes.filter(b => b.checked).length;
    if(count) count.textContent = n
      ? `AutoML will train ${n} candidate model(s), each a full training run.`
      : "Select at least one model.";
    const start = $("#prjstartautoml");
    if(start) start.disabled = !n;
  };
  boxes.forEach(b => b.onchange = sync);
  if(boxes.length) sync();

  const auto = $("#prjstartautoml");
  if(auto) auto.onclick = () => startRun("automl", {
    dataset_version: PRJ.version,
    target_column: PRJ.target,
    algorithms: boxes.filter(b => b.checked).map(b => b.value),
    primary_metric: PRJ.primaryMetric || "roc_auc",
    tune: false,
    target_stage: ($("#prjstage") && $("#prjstage").value) || "Staging",
    max_models: Math.max(1, boxes.filter(b => b.checked).length),
  });

  const man = $("#prjstartmanual");
  if(man) man.onclick = () => startRun("training", {
    dataset_version: PRJ.version,
    algorithm: ($("#prjalgo") && $("#prjalgo").value) || null,
    tune: !!($("#prjtune") && $("#prjtune").checked),
    promote: true,
    target_stage: ($("#prjstage") && $("#prjstage").value) || "Staging",
  });
}


async function startRun(kind, payload){
  const stage = payload.target_stage;
  const btn = kind === "automl" ? $("#prjstartautoml") : $("#prjstartmanual");
  const box = $("#prjstartresult");

  if(kind === "automl" && !payload.algorithms.length){
    toast("Select at least one candidate.", "bad"); return;
  }

  const needsKey = await authRequired();
  const proceed = await confirmAction({
    title: kind === "automl" ? "Start AutoML" : "Start training",
    body: kind === "automl"
      ? `Train ${payload.algorithms.length} candidate model(s) on ${PRJ.version}, target `
        + `${PRJ.target}. The best is registered and then judged by the approval gate — `
        + `promotion into ${stage} is not automatic.`
      : `Train ${payload.algorithm || "the configured default"} on ${PRJ.version}. The model is `
        + `registered and judged by the approval gate — promotion into ${stage} is not automatic.`,
    confirm: kind === "automl" ? "Start AutoML" : "Start training", needsKey,
  });
  if(!proceed) return;
  if(needsKey && !proceed.key){ toast("An API key is required.", "bad"); return; }

  const label = btn ? btn.textContent : "";
  if(btn){ btn.disabled = true; btn.textContent = "Starting…"; }
  try {
    const path = kind === "automl" ? "/api/v1/automl/runs" : "/api/v1/training/runs";
    const res = await api.post(path, payload, proceed.key);
    PRJ.mode = kind === "automl" ? "automl" : "manual";
    PRJ.runKind = kind;
    PRJ.runId = res.run_id;
    PRJ.runStatus = res.status || "queued";
    PRJ.modelVersion = null; PRJ.gateDecision = null; PRJ.deployedVersion = null;
    prjSave(); api.bust();
    toast("Run started.", "ok");
    goStep(5);
    render();
  } catch(e){
    const msg = (e.status === 401 || e.status === 403)
      ? "Rejected: the API key was missing or not accepted." : e.message;
    if(box) box.innerHTML =
      `<div class="note" style="border-color:var(--bad-line);background:var(--bad-bg)">
        <b>Could not start the run.</b><br><span style="font-size:12.5px">${esc(msg)}</span></div>`;
    toast(msg, "bad");
  } finally {
    if(btn){ btn.disabled = false; btn.textContent = label; }
  }
}

/* --- 05 polling --------------------------------------------------------- */
/* Every three seconds while the run is live, and not one request after it
   reaches a terminal state. Also stops if the user navigates away. */

async function pollRunStep(runId, attempt){
  if(!runId || runId !== PRJ.runId) return;
  if(attempt > 400) return;
  if(route() !== "newproject") return;

  let run;
  try { run = await api.get(runPath(), 0); }
  catch(e){ PRJ_POLL = setTimeout(() => pollRunStep(runId, attempt + 1), 6000); return; }

  absorbRun(run);
  if(runDone(run.status)){
    api.bust();
    toast(`Run ${String(run.status).replace(/_/g," ")}.`,
      run.status === "failed" || run.status === "rejected" ? "bad" : "ok");
    render();
    return;
  }
  // Refresh the progress panel in place: re-rendering the page would discard
  // whatever the user has in flight elsewhere on it.
  const holder = document.querySelector("[data-progress]");
  if(holder && document.visibilityState === "visible"){
    holder.innerHTML = PRJ.runKind === "automl" ? automlBody(run, true) : trainingBody(run, true);
  }
  PRJ_POLL = setTimeout(() => pollRunStep(runId, attempt + 1), 3000);
}


/* --- 05 candidate drawer ------------------------------------------------ */
function wireDrawer(){
  document.querySelectorAll("[data-cand]").forEach(b => b.onclick = async () => {
    const algo = b.dataset.cand;
    let run;
    try { run = await api.get(runPath(), 0); }
    catch(e){ toast(e.message, "bad"); return; }
    const c = (run.candidates || []).find(x => x.algorithm === algo);
    if(!c){ toast("That candidate is no longer in the run record.", "bad"); return; }
    openDrawer(algo, candidateDetail(c, run));
  });
}

/* Metrics arrive in one bag, mixing scores with counts. Rendering everything
   to four decimals turns "1200 samples" into "1200.0000", so whole numbers are
   shown as whole numbers. */


function openDrawer(title, html){
  closeDrawer();
  const scrim = document.createElement("div");
  scrim.className = "scrim"; scrim.id = "prjscrim";
  const d = document.createElement("aside");
  d.className = "drawer"; d.id = "prjdrawer";
  d.setAttribute("role", "dialog"); d.setAttribute("aria-modal", "true");
  d.setAttribute("aria-label", title);
  d.innerHTML = `<div class="dhead"><b>${esc(title)}</b><span class="spacer"></span>
      <button class="btn icon" id="prjdclose" aria-label="Close">&times;</button></div>
    <div class="dbody">${html}</div>`;
  document.body.appendChild(scrim); document.body.appendChild(d);
  /* Force a reflow so the browser records the closed position before the open
     class lands, which is what makes it animate rather than jump. A
     requestAnimationFrame here would do the same when the tab is painting, but
     it does not fire in a backgrounded or non-rendering tab -- and a drawer
     that never opens is a worse failure than one that does not slide. */
  void d.offsetWidth;
  scrim.classList.add("on"); d.classList.add("open");
  $("#prjdclose").onclick = closeDrawer;
  scrim.onclick = closeDrawer;
  /* Tab used to walk out of the open drawer into the page behind it. The
     shared trap keeps the keyboard inside and returns focus to whatever
     opened it. */
  DRAWER_RELEASE = trapFocus(d, closeDrawer);
}

let DRAWER_RELEASE = null;

function closeDrawer(){
  if(DRAWER_RELEASE){ DRAWER_RELEASE(); DRAWER_RELEASE = null; }
  ["#prjdrawer", "#prjscrim"].forEach(s => { const el = $(s); if(el) el.remove(); });
}


/* --- 07 gate ------------------------------------------------------------ */
function wireGate(){
  const btn = $("#prjrungate");
  if(!btn) return;
  btn.onclick = async () => {
    const key = ($("#prjgatekey") && $("#prjgatekey").value) || "";
    if(!key){ toast("Enter an API key.", "bad"); return; }
    btn.disabled = true;
    try {
      await api.post(`/api/v1/models/${encodeURIComponent(PRJ.modelName)}`
        + `/versions/${PRJ.modelVersion}/evaluate-gate`, null, key);
      api.bust(); render();
    } catch(e){
      toast(e.status === 401 || e.status === 403
        ? "The API key was not accepted." : e.message, "bad");
      btn.disabled = false;
    }
  };
}


/* --- 08 deploy ---------------------------------------------------------- */
function wireDeploy(){
  const btn = $("#prjdeploy");
  if(!btn) return;
  btn.onclick = async () => {
    const strategy = ($("#prjstrategy") && $("#prjstrategy").value) || "blue_green";
    const needsKey = await authRequired();
    const proceed = await confirmAction({
      title: "Deploy model version",
      body: `Roll ${PRJ.modelName} v${PRJ.modelVersion} onto the serving endpoint using the `
          + `${strategy.replace(/_/g," ")} strategy. This changes what live traffic is scored by.`,
      confirm: "Deploy", needsKey,
    });
    if(!proceed) return;
    if(needsKey && !proceed.key){ toast("An API key is required to deploy.", "bad"); return; }

    btn.disabled = true; btn.textContent = "Deploying…";
    try {
      const res = await api.post("/api/v1/deployments", {
        model_name: PRJ.modelName,
        model_version: PRJ.modelVersion,
        strategy,
        reason: "deployed from the guided build workflow",
      }, proceed.key);
      PRJ.deployedVersion = PRJ.modelVersion; prjSave(); api.bust();
      toast("Deployment created.", "ok");
      $("#prjdeployresult").innerHTML =
        `<div class="note" style="border-color:var(--ok-line);background:var(--ok-bg)">
          <b>Deployed.</b><br><span style="font-size:12.5px">
          ${esc(res.detail || res.message || "The deployment API accepted the rollout.")}</span></div>`;
      render();
    } catch(e){
      const msg = (e.status === 401 || e.status === 403)
        ? "Rejected: the API key was missing or not accepted." : e.message;
      $("#prjdeployresult").innerHTML =
        `<div class="note" style="border-color:var(--bad-line);background:var(--bad-bg)">
          <b>Deployment refused.</b><br><span style="font-size:12.5px">${esc(msg)}</span></div>`;
      toast(msg, "bad");
      btn.disabled = false; btn.textContent = "Deploy this version";
    }
  };
}


/* --- 09 predict --------------------------------------------------------- */
function wirePredict(){
  const btn = $("#prjpredict");
  if(!btn) return;
  btn.onclick = async () => {
    const features = {};
    let bad = null;
    document.querySelectorAll(".prjfeat").forEach(i => {
      const raw = i.value.trim();
      if(raw === ""){ bad = bad || `${i.dataset.f} is empty.`; return; }
      if(i.dataset.num === "1"){
        const n = Number(raw);
        if(!isFinite(n)){ bad = bad || `${i.dataset.f} is not a number.`; return; }
        features[i.dataset.f] = n;
      } else features[i.dataset.f] = raw;
    });
    if(bad){ toast(bad, "bad"); return; }

    const needsKey = await authRequired();
    let key = null;
    if(needsKey){
      const proceed = await confirmAction({
        title: "Score a record", body: "Sends one request to the live prediction endpoint.",
        confirm: "Score", needsKey: true,
      });
      if(!proceed) return;
      if(!proceed.key){ toast("An API key is required.", "bad"); return; }
      key = proceed.key;
    }

    btn.disabled = true; btn.textContent = "Scoring…";
    try {
      const r = await api.post("/api/v1/predict", { features }, key);
      $("#prjpredresult").innerHTML = card("Response", `
        <div class="grid g4">
          ${kpi("Prediction", `<b>${esc(r.prediction_label)}</b>`, `class ${int(r.prediction)}`)}
          ${kpi("Probability", num(r.probability, 4), `threshold ${num(r.threshold, 2)}`)}
          ${kpi("Served by", `<span class="mono">v${int(r.model_version)}</span>`,
            `${esc(r.model_stage||"")} · ${esc(r.variant||"primary")}`)}
          ${kpi("Latency", ms(r.inference_latency_ms))}
        </div>
        <p class="dim" style="margin:12px 0 0;font-size:12.5px">The version and variant above are
          what actually served this request, not what the registry says is current.</p>`,
        { flush:true });
    } catch(e){
      const msg = (e.status === 401 || e.status === 403)
        ? "Rejected: the API key was missing or not accepted." : e.message;
      $("#prjpredresult").innerHTML =
        `<div class="note" style="border-color:var(--bad-line);background:var(--bad-bg)">
          <b>The endpoint rejected this request.</b><br>
          <span style="font-size:12.5px">${esc(msg)}</span></div>`;
    } finally {
      btn.disabled = false; btn.textContent = "Score";
    }
  };
}
