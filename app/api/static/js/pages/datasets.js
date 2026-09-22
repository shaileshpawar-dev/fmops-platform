/* Datasets -- immutable, content-addressed versions, and what came of them.
 *
 *   #/datasets                every version, and the uploader
 *   #/datasets/v3-1a2b3c4d    one version: validation, profile, sample, lineage
 *
 * A version is a snapshot of exact bytes: its file name carries its hash, so a
 * later upload -- or a retraining run -- can never change what an earlier model
 * was trained on.
 */

PAGES.datasets = {
  title: "Datasets",
  intro: "Registered dataset versions. Versions are content-addressed and immutable: identical bytes "
       + "return the existing version, and nothing can change the bytes a model was trained on.",
  async render(){
    const version = routeParam();
    if(version) return await datasetDetail(version);
    const r = await loadAll({ list:"/api/v1/datasets", limits:"/api/v1/datasets/limits" }, 6000);
    const maxMb = r.limits.ok ? Math.round(r.limits.data.max_upload_bytes / 1048576) : 25;
    const uploader = card("Upload a dataset", `
      <p class="dim" style="margin:0 0 12px">CSV only, up to ${maxMb} MB. The file is parsed, registered
        as a new version and checked against its own columns — no schema needs declaring first. It is
        never executed, and the file name never chooses a path.</p>
      <div class="formrow">
        <input type="file" id="dsfile" accept=".csv,text/csv" aria-label="CSV file">
        <input type="text" id="dsname" placeholder="Name (defaults to the file name)" maxlength="128" aria-label="Dataset name">
        <input type="text" id="dsdesc" placeholder="Description (optional)" maxlength="500" aria-label="Description">
        <button class="btn pri" id="dsupload">Upload and validate</button>
      </div>
      <div id="dsresult" style="margin-top:12px"></div>`);

    const tbl = sect(r.list, d => {
      const versions = (d.versions || []).slice().reverse();
      return card("Registered versions", table([
        { label:"Version", sort:v => v.created_at, render:v =>
            `<a class="mono" href="#/datasets/${encodeURIComponent(v.version)}"><b>${esc(v.version)}</b></a>` },
        { label:"Dataset", sort:v => v.dataset_name, render:v => `<span class="mono">${esc(v.dataset_name||"-")}</span>` },
        { label:"Rows", num:true, sort:v => v.n_rows, render:v => int(v.n_rows) },
        { label:"Columns", num:true, render:v => int(v.n_columns) },
        { label:"Derived from", render:v => v.parent_version
            ? `<a class="mono dim" href="#/datasets/${encodeURIComponent(v.parent_version)}">${esc(v.parent_version)}</a>` : NA },
        { label:"Hash", render:v => v.content_hash ?
            `<span class="mono dim">${esc(String(v.content_hash).slice(0,12))}</span>` : NA },
        { label:"Description", render:v => `<span class="dim">${esc(String(v.description || "").slice(0, 70))}</span>` },
        { label:"Created", sort:v => v.created_at, render:v => when(v.created_at) },
      ], versions, { id:"datasets", sortKey:"Version", sortDir:"desc", filter:"Filter datasets",
        empty:"No datasets registered. Upload a CSV above to create the first version." }),
      { flush:true, sub:`${versions.length} version(s)` });
    }, "datasets");

    return uploader + tbl;
  },
  wire(){
    wireDatasetUpload();
    const tsel = $("#dstarget");
    if(tsel) tsel.onchange = async () => {
      const box = $("#dsvalidation");
      box.innerHTML = skeleton(3);
      try {
        const v = await api.get(`/api/v1/datasets/${encodeURIComponent(tsel.dataset.v)}/validation`
          + (tsel.value ? `?target=${encodeURIComponent(tsel.value)}` : ""), 0);
        box.innerHTML = renderValidation(v);
      } catch(e){ box.innerHTML = errorState(e.message, location.hash); }
    };
  },
};

async function datasetDetail(version){
  const enc = encodeURIComponent(version);
  const r = await loadAll({
    meta: `/api/v1/datasets/${enc}`,
    val:  `/api/v1/datasets/${enc}/validation`,
    prev: `/api/v1/datasets/${enc}/preview?rows=12`,
    lin:  `/api/v1/datasets/${enc}/lineage`,
  }, 10000);
  if(!r.meta.ok) return errorState(r.meta.error, "datasets");
  const m = r.meta.data;
  const cols = r.prev.ok ? (r.prev.data.column_profile || []).map(c => c.name) : (m.columns || []);

  const head = `<div class="mhead"><div class="top">
      <div style="min-width:0">
        <h1 class="mono" style="font-size:24px">${esc(version)}</h1>
        <div class="idl">
          <span>dataset <b>${esc(m.dataset_name || "-")}</b></span>
          <span><b>${int(m.n_rows)}</b> rows · <b>${int(m.n_columns)}</b> columns</span>
          <span>hash ${copyable(m.content_hash, String(m.content_hash || "").slice(0, 12), { label:"content hash" })}</span>
          ${m.parent_version ? `<span>derived from <b><a class="mono" href="#/datasets/${encodeURIComponent(m.parent_version)}">${esc(m.parent_version)}</a></b></span>` : ""}
          <span>registered <b>${esc(String(m.created_at || "").slice(0,19).replace("T"," "))}</b></span>
        </div>
        ${m.description ? `<p class="dim" style="margin:10px 0 0;font-size:12.5px">${esc(m.description)}</p>` : ""}
      </div>
      <span class="spacer"></span>
      <a class="btn pri" href="#/newproject">Train on a dataset</a>
      <a class="btn" href="#/datasets">All versions</a>
    </div></div>`;

  const validation = card("Validation", `
    <div class="pagebar" style="margin:0 0 12px"><label class="mpick" for="dstarget"><span>Check as a target</span>
      <select id="dstarget" data-v="${esc(version)}"><option value="">no target — dataset quality only</option>
        ${cols.map(c => `<option value="${esc(c)}">${esc(c)}</option>`).join("")}</select></label></div>
    <div id="dsvalidation">${r.val.ok ? renderValidation(r.val.data) : errorState(r.val.error, location.hash)}</div>
    <p class="dim" style="margin:10px 0 0;font-size:12px">With a target, the binary-label checks training
      enforces are included — so "passed" here means training will accept it.</p>`,
    { sub:"the same engine the training pipeline gates on" });

  const L = r.lin.ok ? r.lin.data : null;
  const lineage = card("Lineage", !L ? unavailable("Lineage could not be read.") : `
    <div class="eyebrow">Models trained on this version</div>
    ${L.models_trained.length ? table([
      { label:"Model", render:x => `<a href="#/models/${encodeURIComponent(x.name)}/${x.version}"><b>${esc(x.name)}</b> <span class="mono">v${int(x.version)}</span></a>` },
      { label:"Stage", render:x => stageBadge(x) },
      { label:"Algorithm", render:x => `<span class="mono">${esc(x.algorithm || "-")}</span>` },
      { label:"Registered", render:x => when(x.created_at) },
    ], L.models_trained, { empty:"" }) : `<p class="dim" style="margin:6px 0 14px">No model has trained on it yet.</p>`}
    <div class="eyebrow" style="margin-top:14px">Versions derived from it</div>
    ${L.next_versions.length ? `<ul class="plain">${L.next_versions.map(v => `<li><a class="mono"
      href="#/datasets/${encodeURIComponent(v.version)}">${esc(v.version)}</a> <span class="dim">${esc(v.description || "")}</span></li>`).join("")}</ul>`
      : `<p class="dim" style="margin:6px 0 0">None. A retraining run registers its training set as a version derived from this one.</p>`}`,
    { sub:"dataset → models → derived datasets" });

  const preview = r.prev.ok ? renderPreview(r.prev.data) : card("Preview", errorState(r.prev.error, location.hash));
  return head + `<div class="grid g2">${validation}${lineage}</div>` + preview;
}

function wireDatasetUpload(){
  const up = $("#dsupload");
  if(!up) return;
  up.onclick = async () => {
    const input = $("#dsfile");
    const file = input && input.files && input.files[0];
    if(!file){ toast("Choose a CSV file first.", "bad"); return; }
    if(!/\.csv$/i.test(file.name)){ toast("Only .csv files are accepted.", "bad"); return; }
    const needsKey = await authRequired();
    const proceed = await confirmAction({
      title: "Upload dataset",
      body: `Register ${file.name} (${(file.size/1024).toFixed(0)} KB) as a new dataset version and validate it.`,
      confirm: "Upload", needsKey,
    });
    if(!proceed) return;
    if(needsKey && !proceed.key){ toast("An API key is required to upload.", "bad"); return; }

    up.disabled = true; up.textContent = "Uploading…";
    const qs = new URLSearchParams({ filename: file.name, description: ($("#dsdesc").value || "") });
    const name = ($("#dsname").value || "").trim();
    if(name) qs.set("dataset_name", name);
    try {
      const res = await fetch(`/api/v1/datasets/upload?${qs}`, {
        method: "POST",
        headers: Object.assign({ "Content-Type": "text/csv" }, proceed.key ? { "X-API-Key": proceed.key } : {}),
        body: file,
      });
      const body = await res.json();
      if(!res.ok){
        const msg = (body.error && body.error.message) || `HTTP ${res.status}`;
        $("#dsresult").innerHTML = `<div class="note bad"><b>Upload rejected.</b><br>${esc(msg)}</div>`;
        toast("Upload rejected: " + msg, "bad");
        return;
      }
      api.bust();
      toast(`Registered ${body.version} (${body.rows} rows).`, "ok");
      $("#dsresult").innerHTML = `<div class="note"><b>Registered
        <a class="mono" href="#/datasets/${encodeURIComponent(body.version)}">${esc(body.version)}</a></b>
        as <b>${esc(body.dataset_name)}</b> — ${int(body.rows)} rows, ${int(body.columns)} columns.
        <a href="#/newproject">Train a model on it →</a></div>`
        + (body.validation ? renderValidation(body.validation) : "");
    } catch(e){
      toast("Upload failed: " + e.message, "bad");
    } finally {
      up.disabled = false; up.textContent = "Upload and validate";
    }
  };
}

function renderValidation(v){
  if(!v) return "";
  const head = `<div class="grid g4" style="margin-bottom:12px">
    ${kpi("Status", v.passed ? badge("PASSED","ok",true) : badge("FAILED","bad",true),
          `engine: ${esc(v.engine||"-")}`)}
    ${kpi("Expectations", int(v.expectations))}
    ${kpi("Passed", int(v.succeeded))}
    ${kpi("Failed", int(v.failed), has(v.warnings) ? `${v.warnings} warning(s)` : "")}
  </div>`;
  const fails = v.failures || [];
  const body = fails.length ? table([
    { label:"Expectation", render:f => `<span class="mono">${esc(f.expectation)}</span>` },
    { label:"Column", render:f => f.column ? `<span class="mono">${esc(f.column)}</span>` : NA },
    { label:"Severity", render:f => badge(f.severity, f.severity==="warning"?"warn":"bad") },
    { label:"Observed", render:f => f.observed ? `<span class="mono dim">${esc(f.observed)}</span>` : NA },
    { label:"Expected", render:f => f.expected ? `<span class="mono dim">${esc(f.expected)}</span>` : NA },
    { label:"Message", render:f => esc(f.message) },
  ], fails, { empty:"" }) : `<div class="state"><div class="big">All expectations passed</div>
      ${int(v.expectations)} checks, no failures.</div>`;
  return head + body + (v.truncated ? `<p class="dim" style="margin:10px 0 0">
    Only the first 50 failures are listed.</p>` : "");
}

function renderPreview(p){
  const cols = p.column_profile || [];
  const sample = p.sample || [];
  const names = cols.map(c => c.name);
  const rows = sample.length ? `<div class="scroll"><table><thead><tr>${
    names.map(n => `<th scope="col">${esc(n)}</th>`).join("")}</tr></thead><tbody>${
    sample.map(row => `<tr>${names.map(n =>
      `<td class="mono" style="font-size:11.5px">${row[n] === null || row[n] === undefined
        ? `<span class="dim">null</span>` : esc(String(row[n]).slice(0,40))}</td>`).join("")}</tr>`).join("")
  }</tbody></table></div>` : emptyState("No sample rows.");

  return card(`Profile and sample`, `
    <div class="grid g3" style="margin-bottom:12px">
      ${kpi("Rows", int(p.rows))}${kpi("Columns", int(p.columns))}
      ${kpi("Showing", int(p.preview_rows), "sample capped server-side")}
    </div>
    <div class="eyebrow" style="margin:14px 0 7px">Column profile</div>` +
    table([
      { label:"Column", render:c => `<span class="mono">${esc(c.name)}</span>` },
      { label:"Type", render:c => `<span class="mono dim">${esc(c.dtype)}</span>` },
      { label:"Missing", num:true, render:c => `${int(c.missing)} <span class="dim">(${c.missing_pct}%)</span>` },
      { label:"Unique", num:true, render:c => int(c.unique) },
      { label:"Example", render:c => c.example ? `<span class="mono dim">${esc(c.example)}</span>` : NA },
    ], cols, { empty:"No columns." }) +
    `<div class="eyebrow" style="margin:14px 0 7px">Sample rows</div>` + rows, { sub:esc(p.version) });
}
