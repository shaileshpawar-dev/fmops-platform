/* Jobs -- every long-running operation, as it runs.
 *
 * Training, AutoML, retraining, deployments and drift scans all execute as
 * jobs: queued, claimed by a worker, heartbeating, and finished as succeeded,
 * failed or cancelled. A job is the *execution*; what it produced (a run, an
 * event, a deployment) keeps its own record, linked from here.
 *
 *   #/jobs            every job, filterable, plus the automation schedule
 *   #/jobs/job-123    one job: its log as it is written, cancel, retry
 */

const JOB_KINDS = ["training","automl","retraining","deployment","drift_scan"];

function jobResourceLink(job){
  const id = job.resource_id;
  if(!id) return NA;
  const href = job.kind === "training" ? `#/training`
    : job.kind === "automl" ? `#/automl`
    : job.kind === "retraining" ? `#/retraining`
    : job.kind === "deployment" && job.model_name ? `#/deployments?model=${encodeURIComponent(job.model_name)}`
    : job.kind === "drift_scan" && job.model_name ? `#/drift?model=${encodeURIComponent(job.model_name)}`
    : null;
  return href ? `<a class="mono" href="${href}">${esc(String(id).slice(0, 22))}</a>`
    : `<span class="mono">${esc(String(id).slice(0, 22))}</span>`;
}

function jobDuration(job){
  const a = job.started_at && Date.parse(job.started_at);
  const b = job.finished_at ? Date.parse(job.finished_at) : (job.status === "running" ? Date.now() : null);
  if(!a || !b) return NA;
  const s = Math.max(0, (b - a) / 1000);
  return `<span class="mono">${s < 90 ? s.toFixed(1) + "s" : (s / 60).toFixed(1) + "m"}</span>`;
}

async function jobsIndex(){
  const q = hashQuery();
  const status = q.get("status") || "", kind = q.get("kind") || "";
  const qs = new URLSearchParams({ limit: "100" });
  if(status) qs.set("status", status);
  if(kind) qs.set("kind", kind);
  const r = await loadAll({ jobs:`/api/v1/jobs?${qs}`, auto:"/api/v1/automation" }, 3000);
  if(!r.jobs.ok) return errorState(r.jobs.error, "jobs");
  const d = r.jobs.data, c = d.counts || {};
  const auto = r.auto.ok ? r.auto.data : null;

  const strip = `<div class="grid g5 statstrip">
    ${["queued","running","succeeded","failed","cancelled"].map(s =>
      `<a class="stat ${status === s ? "on" : ""}" href="#/jobs?status=${status === s ? "" : s}${kind ? "&kind=" + kind : ""}">
        <span class="n">${int(c[s] || 0)}</span><span class="l">${runStatusBadge(s)}</span></a>`).join("")}
  </div>`;

  const filters = `<div class="pagebar"><label class="mpick" for="jobkind"><span>Kind</span>
    <select id="jobkind"><option value="">all kinds</option>${JOB_KINDS.map(k =>
      `<option value="${k}" ${k === kind ? "selected" : ""}>${k.replace("_"," ")}</option>`).join("")}</select></label></div>`;

  const list = card("Jobs", table([
    { label:"Job", sort:j => j.created_at, render:j => `<a class="mono" href="#/jobs/${encodeURIComponent(j.id)}">${esc(j.id)}</a>` },
    { label:"Kind", sort:j => j.kind, render:j => badge(String(j.kind).replace("_"," "), "mute") },
    { label:"Status", sort:j => j.status, render:j => runStatusBadge(j.status)
        + (j.cancel_requested && j.status === "running" ? ` <span class="dim">stopping…</span>` : "") },
    { label:"Model", sort:j => j.model_name, render:j => j.model_name
        ? `<a href="#/models/${encodeURIComponent(j.model_name)}">${esc(j.model_name)}</a>` : NA },
    { label:"Produced", render:j => jobResourceLink(j) },
    { label:"By", render:j => `<span class="dim">${esc(j.requested_by || "system")}</span>` },
    { label:"Duration", num:true, render:j => jobDuration(j) },
    { label:"Queued", sort:j => j.created_at, render:j => when(j.created_at) },
    { label:"Error", render:j => j.error ? `<span class="dim" title="${esc(j.error)}">${esc(String(j.error).slice(0, 70))}</span>` : "" },
  ], d.jobs, { id:"jobs", sortKey:"Queued", sortDir:"desc", filter:"Filter jobs",
    empty:"No jobs yet. Training, AutoML, retraining, deployments and drift scans appear here." }),
    { flush:true, sub:`${d.count} shown` });

  const autoCard = card("Scheduled automation", auto ? `
    <div class="grid g3">
      ${kpi("Schedule", auto.enabled ? `every ${int(auto.interval_minutes)} min` : badge("disabled","mute"))}
      ${kpi("Last pass", auto.last_tick_at ? esc(ago(auto.last_tick_at)) : `<span class="dim">never</span>`)}
      ${kpi("Runs as", "ordinary jobs", "visible and cancellable here")}
    </div>
    <ul class="plain" style="margin:13px 0 0">${(auto.does || []).map(x => `<li>${esc(x)}</li>`).join("")}</ul>
    <div style="margin-top:13px"><button class="btn" id="autorun">${icon("refresh",14)} Run the checks now</button>
      <span id="autoout" class="dim" style="margin-left:10px;font-size:12.5px"></span></div>`
    : unavailable("The automation status could not be read."), { sub:"drift · retraining · SLO watchdog" });

  return strip + filters + list + autoCard;
}

async function jobDetail(id){
  let job;
  try { job = await api.get(`/api/v1/jobs/${encodeURIComponent(id)}`, 0); }
  catch(e){ return errorState(e.message, "jobs"); }
  const live = !JOB_TERMINAL.has(job.status);
  const head = `<div class="mhead"><div class="top">
      <div style="min-width:0">
        <h1 class="mono" style="font-size:22px">${esc(job.id)}</h1>
        <div style="display:flex;gap:9px;align-items:center;margin-top:8px;flex-wrap:wrap">
          ${runStatusBadge(job.status)} ${badge(String(job.kind).replace("_"," "),"mute")}
          ${job.cancel_requested && live ? badge("cancellation requested","warn") : ""}
        </div>
        <div class="idl">
          <span>model <b>${job.model_name ? `<a href="#/models/${encodeURIComponent(job.model_name)}">${esc(job.model_name)}</a>` : "—"}</b></span>
          <span>produced <b>${jobResourceLink(job)}</b></span>
          <span>requested by <b>${esc(job.requested_by || "system")}</b></span>
          <span>attempts <b>${int(job.attempts)}</b></span>
          <span>worker <b class="mono">${esc(job.worker || "—")}</b></span>
          ${job.retry_of ? `<span>retry of <b><a class="mono" href="#/jobs/${encodeURIComponent(job.retry_of)}">${esc(job.retry_of)}</a></b></span>` : ""}
        </div>
      </div>
      <span class="spacer"></span>
      ${live ? `<button class="btn danger" id="jobcancel" data-id="${esc(job.id)}">${icon("stop",14)} Cancel</button>` : ""}
      ${["failed","cancelled"].includes(job.status) ? `<button class="btn pri" id="jobretry" data-id="${esc(job.id)}">${icon("refresh",14)} Retry</button>` : ""}
      <a class="btn" href="#/jobs">All jobs</a>
    </div></div>`;

  const timing = card("Timing", `<div class="grid g4">
      ${kpi("Queued", when(job.created_at))}
      ${kpi("Started", job.started_at ? when(job.started_at) : NA)}
      ${kpi("Finished", job.finished_at ? when(job.finished_at) : NA)}
      ${kpi("Duration", jobDuration(job))}
    </div>${job.error ? `<div class="note bad" style="margin-top:14px"><b>${esc(job.status === "cancelled" ? "Cancelled" : "Failed")}.</b><br>${esc(job.error)}</div>` : ""}`);

  const log = card("Log", `<pre class="joblog tall" aria-live="polite" id="joblogfull">Loading…</pre>`,
    { flush:true, sub: live ? "streaming while the job runs" : "captured while the job ran" });

  const result = job.result ? card("Result", `<pre class="jsonview">${esc(JSON.stringify(job.result, null, 2))}</pre>`,
    { flush:true, sub:"what the job returned" }) : "";
  const payload = card("Request", `<pre class="jsonview">${esc(JSON.stringify(job.payload, null, 2))}</pre>`,
    { flush:true, sub:"what the job was asked to do" });

  return head + timing + log + `<div class="grid g2">${payload}${result}</div>`;
}

async function streamJobLog(id){
  let after = 0;
  for(;;){
    const pre = document.getElementById("joblogfull");
    if(!pre) return;
    let chunk;
    try { chunk = await api.get(`/api/v1/jobs/${encodeURIComponent(id)}/logs?after=${after}&limit=1000`, 0); }
    catch(e){ pre.textContent = `Could not read the log: ${e.message}`; return; }
    if(after === 0) pre.textContent = chunk.lines.length ? "" : "No log lines captured.";
    if(chunk.lines.length){
      pre.textContent += chunk.lines.map(l => `${String(l.created_at || "").slice(11, 23)}  ${
        l.level.padEnd(7)} ${l.message}${l.context && Object.keys(l.context).length
          ? "  " + JSON.stringify(l.context).slice(0, 300) : ""}`).join("\n") + "\n";
      pre.scrollTop = pre.scrollHeight;
    }
    after = chunk.next_after;
    if(chunk.finished) return chunk.status;
    await new Promise(res => setTimeout(res, 1200));
  }
}

PAGES.jobs = {
  title: "Jobs",
  intro: "Every long-running operation — training, AutoML, retraining, deployments, drift scans — "
       + "as it runs: queued, running, succeeded, failed or cancelled, with its log.",
  async render(){
    const id = routeParam();
    return id ? await jobDetail(id) : await jobsIndex();
  },
  wire(){
    const id = routeParam();
    if(id){
      streamJobLog(id).then(status => {
        const was = document.querySelector(".mhead .badge");
        if(status && was && !was.textContent.toLowerCase().includes(status)) render();
      });
    }
    const kind = $("#jobkind");
    if(kind) kind.onchange = () => {
      const q = hashQuery();
      if(kind.value) q.set("kind", kind.value); else q.delete("kind");
      location.hash = `#/jobs?${q}`;
    };
    const cancel = $("#jobcancel");
    if(cancel) cancel.onclick = () => runAction({
      title: "Cancel this job",
      body: "A queued job is cancelled at once. A running job stops at its next checkpoint — between "
          + "AutoML candidates, between canary steps, around the training fit. A canary that is "
          + "interrupted puts all traffic back on the live version.",
      confirm: "Cancel job", danger: true, needsKey: true,
      path: `/api/v1/jobs/${encodeURIComponent(cancel.dataset.id)}/cancel`, payload: null,
      success: "Cancellation requested.",
    });
    const retry = $("#jobretry");
    if(retry) retry.onclick = () => runAction({
      title: "Retry this job",
      body: "Queues a new job with the same parameters. This one stays as it is, for the record.",
      confirm: "Retry", needsKey: true,
      path: `/api/v1/jobs/${encodeURIComponent(retry.dataset.id)}/retry`, payload: null,
      success: "Retry queued.",
      after: res => { location.hash = `#/jobs/${encodeURIComponent(res.job.id)}`; },
    });
    const auto = $("#autorun");
    if(auto) auto.onclick = async () => {
      const res = await runAction({
        title: "Run the automation checks now",
        body: "The same pass the schedule runs: per serving model, queue a drift scan if new traffic "
            + "arrived, queue retraining if its trigger fires, and check the SLOs. It only queues jobs.",
        confirm: "Run checks", needsKey: true,
        path: "/api/v1/automation/run", payload: null, success: "Checks ran.", after: () => {},
      });
      if(!res) return;
      $("#autoout").textContent = `${res.models} serving model(s) checked · ${res.drift_scans.length} drift scan(s) · `
        + `${res.retraining.length} retraining job(s) queued · ${res.watchdog.length} SLO breach(es)`;
      setTimeout(render, 1500);
    };
  },
  /* The list refreshes; a job's own page streams its log instead, and a
     re-render there would restart it. */
  get refresh(){ return routeParam() ? 0 : 10000; },
};
