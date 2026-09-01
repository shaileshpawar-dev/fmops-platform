"""Observability dashboard.

A single self-contained HTML page served by the API. It exists because a
Grafana dashboard answers "what are the numbers?" while this answers "what is
the platform doing right now?" -- which model is live, what the gates decided,
what drifted, what got rolled back and why.

It is deliberately dependency-free (no build step, no CDN) so it works in an
air-gapped container. Grafana remains the tool for time-series depth; this is
the control-plane view.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["dashboard"])


@router.get("/api/v1/dashboard", summary="Everything the dashboard renders")
def dashboard_data() -> dict[str, Any]:
    """Aggregated state for the dashboard, in one call.

    Each section degrades independently: a failure gathering LLM stats must not
    blank out the model section.
    """
    settings = get_settings()
    payload: dict[str, Any] = {
        "service": {
            "name": settings.service_name,
            "version": settings.version,
            "environment": settings.environment,
            "git_commit": settings.git_commit[:12],
            "deployment_provider": settings.deployment.provider,
            "registry_backend": settings.tracking.registry_backend,
            "llm_provider": settings.llm.provider,
            "aws_enabled": settings.aws.enabled,
        }
    }

    # Live model quality is the single most expensive thing on this page (a
    # join over the inference log plus sklearn metrics). Both the system panel
    # and the retraining trigger need it for the same model version at the same
    # instant, so compute it once here and hand the same value to both. This is
    # de-duplication within one request, not a cache: nothing is retained
    # between requests and the numbers are identical to computing it twice.
    live = _safe_value(_live_performance, "live_performance")

    payload["model"] = _safe(_model_section, "model")
    payload["deployment"] = _safe(_deployment_section, "deployment")
    payload["drift"] = _safe(_drift_section, "drift")
    payload["system"] = _safe(lambda: _system_section(live), "system")
    payload["llm"] = _safe(_llm_section, "llm")
    payload["retraining"] = _safe(lambda: _retraining_section(live), "retraining")
    payload["alerts"] = _safe(_alerts_section, "alerts")
    return payload


def _live_performance():
    """Live quality for the serving version, computed once per request."""
    from app.monitoring.service import get_monitoring_service

    return get_monitoring_service().live_performance()


def _safe_value(fn, name: str):
    """Like :func:`_safe` but for a value the sections share; None on failure."""
    try:
        return fn()
    except Exception as exc:
        logger.warning("dashboard.section_failed", extra={"section": name, "error": str(exc)})
        return None


def _safe(fn, name: str) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:
        logger.warning("dashboard.section_failed", extra={"section": name, "error": str(exc)})
        return {"error": str(exc), "available": False}


def _model_section() -> dict[str, Any]:
    from app.registry.factory import get_registry

    settings = get_settings()
    registry = get_registry()
    name = settings.tracking.registered_model_name

    production = registry.get_production(name)
    previous = registry.previous_production(name)
    versions = registry.list_versions(name)

    return {
        "available": production is not None,
        "model_name": name,
        "current_version": production.version if production else None,
        "current_stage": production.stage.value if production else None,
        "previous_version": previous.version if previous else None,
        "algorithm": production.algorithm if production else None,
        "dataset_version": production.dataset_version if production else None,
        "git_commit": (production.git_commit[:12] if production else None),
        "metrics": production.metrics if production else {},
        "total_versions": len(versions),
        "versions": [
            {
                "version": v.version,
                "stage": v.stage.value,
                "status": v.status.value,
                "roc_auc": round(float(v.metrics.get("roc_auc", 0.0)), 4),
                "f1": round(float(v.metrics.get("f1", 0.0)), 4),
                "created_at": v.created_at,
            }
            for v in versions[:10]
        ],
        "thresholds": settings.approval.model_dump(mode="json"),
    }


def _deployment_section() -> dict[str, Any]:
    from app.deployment.manager import get_deployment_manager

    manager = get_deployment_manager()
    deployment = manager.status()
    health = manager.health()
    recent = manager.list(limit=5)

    return {
        "available": deployment is not None,
        "endpoint": health.endpoint_name,
        "provider": manager.provider.name,
        "state": deployment.state.value if deployment else None,
        "strategy": deployment.strategy.value if deployment else None,
        "health": health.status.value,
        "checks": health.checks,
        "current_version": deployment.current_version if deployment else None,
        "previous_version": deployment.previous_version if deployment else None,
        "candidate_version": deployment.candidate_version if deployment else None,
        "shadow_version": deployment.shadow_version if deployment else None,
        "traffic": deployment.traffic if deployment else {},
        "rolled_back": (
            deployment.state.value in ("rolled_back", "rolling_back") if deployment else False
        ),
        "history": [
            {
                "id": d.id,
                "strategy": d.strategy.value,
                "state": d.state.value,
                "current_version": d.current_version,
                "created_at": d.created_at,
                "message": d.message,
            }
            for d in recent
        ],
    }


def _drift_section() -> dict[str, Any]:
    from app.monitoring.drift import recent_drift_reports

    settings = get_settings()
    reports = recent_drift_reports(settings.tracking.registered_model_name, limit=10)
    latest = reports[0] if reports else None
    detail = latest.get("report", {}) if latest else {}

    return {
        "available": latest is not None,
        "threshold": settings.drift.threshold,
        "engine": settings.drift.engine,
        "drift_detected": bool(latest["drift_detected"]) if latest else False,
        "dataset_drift_score": latest["dataset_drift_score"] if latest else None,
        "prediction_drift_score": latest["prediction_drift_score"] if latest else None,
        "concept_drift_status": latest["concept_drift_status"] if latest else "unavailable",
        "concept_drift_detail": detail.get("concept_drift_detail", ""),
        "drifted_features": latest["drifted_features"] if latest else [],
        "feature_drift": [
            {
                "feature": f["feature"],
                "score": f["score"],
                "drifted": f["drifted"],
                "test": f["test"],
                "statistic": f["statistic"],
            }
            for f in detail.get("feature_drift", [])
        ],
        "checked_at": latest["created_at"] if latest else None,
        "history": [
            {
                "created_at": r["created_at"],
                "score": r["dataset_drift_score"],
                "detected": bool(r["drift_detected"]),
            }
            for r in reports
        ],
    }


def _system_section(performance: Any = None) -> dict[str, Any]:
    from app.monitoring.resource_monitor import latest_resources
    from app.monitoring.service import get_monitoring_service

    # Call each collector exactly once. Going through summary() here would
    # recompute service_metrics and live_performance a second and third time,
    # which on a busy inference log is the most expensive thing on the page.
    service = get_monitoring_service()
    metrics = service.service_metrics(60)
    resources = latest_resources()
    if performance is None:
        performance = service.live_performance()

    return {
        "available": True,
        "requests": metrics.request_count,
        "errors": metrics.error_count,
        "error_rate": metrics.error_rate,
        "throughput_rpm": metrics.throughput_rpm,
        "latency_p50_ms": metrics.latency.p50_ms,
        "latency_p95_ms": metrics.latency.p95_ms,
        "latency_p99_ms": metrics.latency.p99_ms,
        "slo_latency_ms": metrics.slo_latency_ms,
        "slo_error_rate": metrics.slo_error_rate,
        "latency_slo_met": metrics.latency_slo_met,
        "error_slo_met": metrics.error_slo_met,
        "cpu_percent": resources.cpu_percent,
        "memory_percent": resources.memory_percent,
        "process_rss_mb": resources.process_rss_mb,
        "gpu_available": resources.gpu_available,
        "gpu": resources.gpu,
        "live_performance": performance.model_dump(mode="json"),
    }


def _llm_section() -> dict[str, Any]:
    from app.llmops.cost import get_cost_tracker
    from app.llmops.evaluation.runner import recent_evaluations
    from app.llmops.prompts.registry import get_prompt_registry
    from app.llmops.token_tracking import get_trace_store

    settings = get_settings()
    store = get_trace_store()
    totals = store.token_totals(30)
    cost = get_cost_tracker().summary()
    evaluations = recent_evaluations(5)
    registry = get_prompt_registry()

    prompts = {}
    for name in registry.list_names():
        versions = registry.list_versions(name)
        prompts[name] = {
            "latest": versions[-1].version,
            "versions": [v.version for v in versions],
        }

    return {
        "available": True,
        "provider": settings.llm.provider,
        "model": settings.llm.model,
        "is_mock": settings.llm.provider == "mock",
        "calls": totals["calls"],
        "error_rate": totals["error_rate"],
        "total_tokens": totals["total_tokens"],
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "avg_latency_ms": totals["avg_latency_ms"],
        "avg_tokens_per_call": totals["avg_tokens_per_call"],
        "today_cost_usd": cost.today_cost_usd,
        "month_cost_usd": cost.month_cost_usd,
        "daily_budget_usd": cost.daily_budget_usd,
        "daily_budget_used_pct": cost.daily_budget_used_pct,
        "monthly_budget_used_pct": cost.monthly_budget_used_pct,
        "by_model": {k: v.model_dump(mode="json") for k, v in cost.by_model.items()},
        "by_prompt_version": store.by_prompt_version(30),
        "prompts": prompts,
        "evaluations": [
            {
                "id": e["id"],
                "dataset": e["dataset"],
                "model": e["model"],
                "prompt_version": e["prompt_version"],
                "overall": round(float(e["aggregate"].get("overall", 0.0)), 4),
                "created_at": e["created_at"],
            }
            for e in evaluations
        ],
    }


def _retraining_section(live_performance: Any = None) -> dict[str, Any]:
    from app.retraining.trigger import evaluate_trigger, get_event_store

    events = get_event_store().recent(5)
    decision = evaluate_trigger(live_performance=live_performance)
    return {
        "available": True,
        "would_trigger": decision.should_retrain,
        "trigger": decision.trigger.value if decision.trigger else None,
        "reason": decision.reason,
        "suppressed_by_cooldown": decision.suppressed_by_cooldown,
        "checks": decision.checks,
        "events": [
            {
                "id": e.id,
                "trigger": e.trigger.value,
                "status": e.status.value,
                "decision": e.decision,
                "baseline_version": e.baseline_version,
                "candidate_version": e.candidate_version,
                "reason": e.reason,
                "created_at": e.created_at,
            }
            for e in events
        ],
    }


def _alerts_section() -> dict[str, Any]:
    from app.monitoring.alerts import get_alert_manager

    manager = get_alert_manager()
    alerts = manager.recent(15)
    return {
        "available": True,
        "open_count": manager.open_count(),
        "alerts": [
            {
                "id": a.id,
                "severity": a.severity.value,
                "category": a.category.value,
                "title": a.title,
                "message": a.message,
                "acknowledged": a.acknowledged,
                "created_at": a.created_at,
            }
            for a in alerts
        ],
    }


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard_page() -> HTMLResponse:
    return HTMLResponse(content=_PAGE)


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FMOps Platform</title>
<style>
  :root {
    --bg: #0f1218; --panel: #171b23; --panel-2: #1e232d; --line: #2a3140;
    --text: #e6e9ef; --muted: #8b94a7; --accent: #4f8cff;
    --ok: #35c07f; --warn: #e0a13c; --err: #e0574f; --dim: #5a6478;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #f5f6f8; --panel: #ffffff; --panel-2: #f0f2f5; --line: #dde1e8;
      --text: #1b1f27; --muted: #626b7d; --dim: #8a93a5;
    }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
    font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  header { padding:20px 24px; border-bottom:1px solid var(--line);
    display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }
  h1 { font-size:18px; margin:0; letter-spacing:-0.01em; }
  .env { font-size:12px; color:var(--muted); }
  .env b { color:var(--text); font-weight:600; }
  main { padding:20px 24px 60px; max-width:1500px; }
  .grid { display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
    padding:16px 18px; }
  .card h2 { font-size:12px; text-transform:uppercase; letter-spacing:0.08em;
    color:var(--muted); margin:0 0 14px; font-weight:600; }
  .kv { display:flex; justify-content:space-between; gap:12px; padding:5px 0;
    border-bottom:1px solid var(--line); font-variant-numeric:tabular-nums; }
  .kv:last-child { border-bottom:none; }
  .kv span:first-child { color:var(--muted); }
  .kv span:last-child { font-weight:600; text-align:right; }
  .big { font-size:30px; font-weight:650; letter-spacing:-0.02em; margin:2px 0 4px;
    font-variant-numeric:tabular-nums; }
  .sub { color:var(--muted); font-size:12px; margin-bottom:12px; }
  .pill { display:inline-block; padding:2px 9px; border-radius:99px; font-size:11px;
    font-weight:600; letter-spacing:0.02em; }
  .ok { background:rgba(53,192,127,.16); color:var(--ok); }
  .warn { background:rgba(224,161,60,.16); color:var(--warn); }
  .err { background:rgba(224,87,79,.16); color:var(--err); }
  .neutral { background:rgba(120,132,155,.16); color:var(--muted); }
  table { width:100%; border-collapse:collapse; font-size:12.5px;
    font-variant-numeric:tabular-nums; }
  th { text-align:left; color:var(--muted); font-weight:600; padding:5px 8px 5px 0;
    border-bottom:1px solid var(--line); font-size:11px; text-transform:uppercase;
    letter-spacing:0.05em; }
  td { padding:5px 8px 5px 0; border-bottom:1px solid var(--line); }
  tr:last-child td { border-bottom:none; }
  .bar { height:6px; background:var(--panel-2); border-radius:3px; overflow:hidden;
    margin-top:5px; }
  .bar > div { height:100%; border-radius:3px; }
  .note { font-size:11.5px; color:var(--dim); margin-top:10px; line-height:1.45;
    border-left:2px solid var(--line); padding-left:9px; }
  .wide { grid-column:1/-1; }
  code { background:var(--panel-2); padding:1px 5px; border-radius:4px; font-size:12px; }
  .err-box { color:var(--err); font-size:12px; }
  button { background:var(--panel-2); color:var(--text); border:1px solid var(--line);
    padding:5px 11px; border-radius:6px; cursor:pointer; font-size:12px; font-weight:500; }
  button:hover { border-color:var(--accent); }
  .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .banner { display:none; background:var(--panel); border:1px solid var(--err);
    border-left-width:3px; border-radius:8px; padding:10px 14px; margin-bottom:16px;
    font-size:13px; }
</style>
</head>
<body>
<header>
  <h1>FMOps Platform</h1>
  <div class="env" id="env">loading…</div>
  <div class="row" style="margin-left:auto">
    <span class="env" id="updated"></span>
    <button onclick="load()">Refresh</button>
  </div>
</header>
<main><div class="grid" id="grid"></div></main>
<script>
const $ = (h) => { const d=document.createElement('div'); d.innerHTML=h.trim(); return d.firstChild; };
const num = (v, d=4) => (v===null||v===undefined||Number.isNaN(v)) ? '—' : Number(v).toFixed(d);
const pct = (v) => (v===null||v===undefined) ? '—' : (Number(v)*100).toFixed(2)+'%';
const kv = (k,v) => `<div class="kv"><span>${k}</span><span>${v}</span></div>`;
const pill = (t,c) => `<span class="pill ${c}">${t}</span>`;

function card(title, body, wide=false) {
  return `<div class="card${wide?' wide':''}"><h2>${title}</h2>${body}</div>`;
}
function unavailable(s) {
  return s && s.error ? `<div class="err-box">unavailable: ${s.error}</div>` : null;
}

function modelCard(m) {
  const e = unavailable(m); if (e) return card('Model', e);
  if (!m.available) return card('Model',
    `<div class="sub">No model is in Production.</div>
     <div class="note">Run <code>make demo</code> or <code>make train &amp;&amp; make promote</code>.</div>`);
  const rows = (m.versions||[]).map(v => `<tr><td>v${v.version}</td>
     <td>${pill(v.stage, v.stage==='Production'?'ok':v.stage==='Archived'?'neutral':'warn')}</td>
     <td>${v.status==='rejected'?pill('rejected','err'):v.status}</td>
     <td>${num(v.roc_auc)}</td><td>${num(v.f1)}</td></tr>`).join('');
  return card('Model',
    `<div class="big">v${m.current_version}</div>
     <div class="sub">${m.model_name} · ${m.algorithm||'—'} · ${pill(m.current_stage,'ok')}</div>
     ${kv('ROC-AUC', num(m.metrics.roc_auc))}
     ${kv('F1', num(m.metrics.f1))}
     ${kv('Precision', num(m.metrics.precision))}
     ${kv('Recall', num(m.metrics.recall))}
     ${kv('Latency p95', num(m.metrics.inference_latency_p95_ms,2)+' ms')}
     ${kv('Previous version', m.previous_version ? 'v'+m.previous_version : '—')}
     ${kv('Dataset', '<code>'+(m.dataset_version||'—')+'</code>')}
     ${kv('Git commit', '<code>'+(m.git_commit||'—')+'</code>')}
     <table style="margin-top:12px"><tr><th>Ver</th><th>Stage</th><th>Status</th><th>AUC</th><th>F1</th></tr>${rows}</table>
     <div class="note">Gate: F1 ≥ ${m.thresholds.min_f1}, ROC-AUC ≥ ${m.thresholds.min_roc_auc},
       p95 ≤ ${m.thresholds.max_inference_latency_ms}ms, improvement ≥ ${m.thresholds.min_improvement}.</div>`);
}

function deployCard(d) {
  const e = unavailable(d); if (e) return card('Deployment', e);
  if (!d.available) return card('Deployment', `<div class="sub">No deployment yet.</div>`);
  const hc = d.health==='healthy'?'ok':d.health==='unhealthy'?'err':'warn';
  const traffic = Object.entries(d.traffic||{}).map(([v,p]) =>
    `<div class="kv"><span>v${v}</span><span>${Number(p).toFixed(1)}%</span></div>
     <div class="bar"><div style="width:${p}%;background:var(--accent)"></div></div>`).join('');
  const hist = (d.history||[]).map(h => `<tr><td>${h.strategy}</td>
     <td>${pill(h.state, h.state==='live'?'ok':h.state==='rolled_back'?'err':'warn')}</td>
     <td>${h.current_version?'v'+h.current_version:'—'}</td>
     <td style="color:var(--muted)">${(h.created_at||'').slice(0,19).replace('T',' ')}</td></tr>`).join('');
  return card('Deployment',
    `<div class="row" style="margin-bottom:10px">${pill(d.state,d.state==='live'?'ok':d.state==='rolled_back'?'err':'warn')}
       ${pill(d.health,hc)} ${pill(d.strategy,'neutral')} ${pill(d.provider,'neutral')}</div>
     ${kv('Endpoint','<code>'+d.endpoint+'</code>')}
     ${kv('Active version', d.current_version?'v'+d.current_version:'—')}
     ${kv('Previous', d.previous_version?'v'+d.previous_version:'—')}
     ${kv('Candidate', d.candidate_version?'v'+d.candidate_version:'—')}
     ${kv('Shadow', d.shadow_version?'v'+d.shadow_version:'—')}
     ${kv('Rolled back', d.rolled_back?pill('yes','err'):'no')}
     <div style="margin-top:12px"><h2 style="margin-bottom:8px">Traffic split</h2>${traffic||'<span class="sub">—</span>'}</div>
     <table style="margin-top:12px"><tr><th>Strategy</th><th>State</th><th>Ver</th><th>When</th></tr>${hist}</table>`);
}

function driftCard(d) {
  const e = unavailable(d); if (e) return card('Drift', e);
  if (!d.available) return card('Drift',
    `<div class="sub">No drift scan has run.</div>
     <div class="note">Send traffic, then <code>POST /api/v1/drift/scan</code>.</div>`);
  const feats = (d.feature_drift||[]).sort((a,b)=>b.score-a.score).slice(0,10).map(f =>
    `<tr><td>${f.feature}</td><td>${num(f.score)}</td><td>${num(f.statistic)}</td>
     <td>${f.drifted?pill('drifted','err'):pill('stable','ok')}</td></tr>`).join('');
  const ratio = Math.min(100, (d.dataset_drift_score/Math.max(d.threshold,0.001))*50);
  return card('Drift',
    `<div class="big">${num(d.dataset_drift_score)}</div>
     <div class="sub">dataset drift · threshold ${d.threshold} · engine ${d.engine}</div>
     <div class="bar"><div style="width:${ratio}%;background:${d.drift_detected?'var(--err)':'var(--ok)'}"></div></div>
     <div style="margin:12px 0">${d.drift_detected?pill('DRIFT DETECTED','err'):pill('stable','ok')}</div>
     ${kv('Prediction drift', num(d.prediction_drift_score))}
     ${kv('Concept drift', pill(d.concept_drift_status, d.concept_drift_status==='measured'?'ok':'neutral'))}
     ${kv('Drifted features', (d.drifted_features||[]).length)}
     ${kv('Checked at', (d.checked_at||'—').slice(0,19).replace('T',' '))}
     <table style="margin-top:12px"><tr><th>Feature</th><th>Score</th><th>PSI</th><th></th></tr>${feats}</table>
     <div class="note">${d.concept_drift_detail || 'Concept drift needs labels; data/prediction drift do not.'}</div>`);
}

function systemCard(s) {
  const e = unavailable(s); if (e) return card('System', e);
  return card('System',
    `<div class="big">${s.requests}</div><div class="sub">requests in the last 60 min</div>
     ${kv('Error rate', pct(s.error_rate)+' '+(s.error_slo_met?pill('SLO ok','ok'):pill('SLO breach','err')))}
     ${kv('Throughput', num(s.throughput_rpm,2)+' rpm')}
     ${kv('Latency p50', num(s.latency_p50_ms,2)+' ms')}
     ${kv('Latency p95', num(s.latency_p95_ms,2)+' ms '+(s.latency_slo_met?pill('SLO ok','ok'):pill('SLO breach','err')))}
     ${kv('Latency p99', num(s.latency_p99_ms,2)+' ms')}
     ${kv('CPU', num(s.cpu_percent,1)+'%')}
     ${kv('Memory', num(s.memory_percent,1)+'% · '+num(s.process_rss_mb,0)+' MB RSS')}
     ${kv('GPU', s.gpu_available?num(s.gpu[0]?.utilization_percent,1)+'%':pill('none detected','neutral'))}
     <div style="margin-top:12px"><h2 style="margin-bottom:8px">Live model quality</h2>
     ${s.live_performance.available
       ? kv('Accuracy',num(s.live_performance.accuracy))+kv('F1',num(s.live_performance.f1))
         +kv('ROC-AUC',num(s.live_performance.roc_auc))+kv('Labelled rows',s.live_performance.labelled_samples)
       : `<div class="note">${s.live_performance.detail}</div>`}</div>`);
}

function llmCard(l) {
  const e = unavailable(l); if (e) return card('LLMOps', e);
  const models = Object.entries(l.by_model||{}).map(([m,b]) =>
    `<tr><td>${m}</td><td>${b.requests}</td><td>${b.total_tokens}</td><td>$${num(b.total_cost_usd,4)}</td></tr>`).join('');
  const prompts = Object.entries(l.prompts||{}).map(([n,p]) =>
    `<tr><td>${n}</td><td><code>${p.latest}</code></td><td style="color:var(--muted)">${p.versions.join(', ')}</td></tr>`).join('');
  const evals = (l.evaluations||[]).map(ev =>
    `<tr><td>${ev.dataset}</td><td><code>${ev.prompt_version}</code></td><td>${num(ev.overall)}</td></tr>`).join('');
  return card('LLMOps',
    `<div class="big">${l.calls}</div><div class="sub">calls · ${l.provider} / ${l.model}
      ${l.is_mock?pill('MOCK PROVIDER','warn'):''}</div>
     ${kv('Total tokens', l.total_tokens.toLocaleString())}
     ${kv('In / out', l.input_tokens.toLocaleString()+' / '+l.output_tokens.toLocaleString())}
     ${kv('Avg tokens/call', num(l.avg_tokens_per_call,1))}
     ${kv('Avg latency', num(l.avg_latency_ms,1)+' ms')}
     ${kv('Error rate', pct(l.error_rate))}
     ${kv('Cost today', '$'+num(l.today_cost_usd,4))}
     ${kv('Cost this month', '$'+num(l.month_cost_usd,4))}
     ${kv('Daily budget used', num(l.daily_budget_used_pct,1)+'%')}
     <div class="bar"><div style="width:${Math.min(100,l.daily_budget_used_pct)}%;
       background:${l.daily_budget_used_pct>=100?'var(--err)':l.daily_budget_used_pct>=80?'var(--warn)':'var(--ok)'}"></div></div>
     ${models?`<table style="margin-top:12px"><tr><th>Model</th><th>Calls</th><th>Tokens</th><th>Cost</th></tr>${models}</table>`:''}
     ${prompts?`<table style="margin-top:12px"><tr><th>Prompt</th><th>Latest</th><th>Versions</th></tr>${prompts}</table>`:''}
     ${evals?`<table style="margin-top:12px"><tr><th>Eval dataset</th><th>Prompt</th><th>Score</th></tr>${evals}</table>`:''}
     ${l.is_mock?'<div class="note">The mock provider is deterministic and offline. Evaluation scores measure the harness, not model quality.</div>':''}`);
}

function retrainCard(r) {
  const e = unavailable(r); if (e) return card('Retraining', e);
  const rows = (r.events||[]).map(ev => `<tr><td>${ev.trigger}</td>
     <td>${pill(ev.status, ev.status==='succeeded'?'ok':ev.status==='rejected'?'warn':ev.status==='failed'?'err':'neutral')}</td>
     <td>${ev.baseline_version?'v'+ev.baseline_version:'—'} → ${ev.candidate_version?'v'+ev.candidate_version:'—'}</td>
     <td style="color:var(--muted)">${ev.decision||'—'}</td></tr>`).join('');
  const checks = (r.checks||[]).map(c =>
    `<div class="kv"><span>${c.name} ${c.fired?pill('fired','warn'):''}</span>
     <span style="font-weight:400;color:var(--muted);font-size:11.5px;max-width:60%">${c.detail}</span></div>`).join('');
  return card('Retraining',
    `<div style="margin-bottom:10px">${r.would_trigger?pill('WOULD TRIGGER: '+r.trigger,'warn'):pill('no trigger','ok')}
      ${r.suppressed_by_cooldown?pill('cooldown','neutral'):''}</div>
     <div class="sub">${r.reason}</div>
     ${checks}
     ${rows?`<table style="margin-top:12px"><tr><th>Trigger</th><th>Status</th><th>Versions</th><th>Decision</th></tr>${rows}</table>`:''}
     <div class="note">A candidate is promoted only if it clears the approval gate
       AND beats production by the configured minimum improvement.</div>`);
}

function alertCard(a) {
  const e = unavailable(a); if (e) return card('Alerts', e);
  const rows = (a.alerts||[]).map(al => `<tr>
     <td>${pill(al.severity, al.severity==='critical'?'err':al.severity==='warning'?'warn':'neutral')}</td>
     <td>${al.category}</td><td>${al.title}</td>
     <td style="color:var(--muted)">${(al.created_at||'').slice(0,19).replace('T',' ')}</td></tr>`).join('');
  return card('Alerts',
    `<div class="big">${a.open_count}</div><div class="sub">open (unacknowledged)</div>
     ${rows?`<table><tr><th>Sev</th><th>Category</th><th>Title</th><th>When</th></tr>${rows}</table>`
       :'<div class="sub">No alerts.</div>'}`, true);
}

// A failed poll must NOT blank the page. During a restart or a brief blip the
// last good render stays on screen and a banner says the data is stale --
// wiping the dashboard is both alarming and less useful than slightly old
// numbers you can still read.
let lastGoodAt = null;
let lastHtml = null;

function showStaleBanner(err) {
  let banner = document.getElementById('stale');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'stale';
    banner.className = 'banner';
    document.querySelector('main').prepend(banner);
  }
  const since = lastGoodAt
    ? `last successful update ${lastGoodAt.toLocaleTimeString()}`
    : 'no data has loaded yet';
  banner.innerHTML =
    `<span class="pill err">stale</span> could not reach the API (${err}) — ${since}. Retrying…`;
  banner.style.display = 'block';
}

function clearStaleBanner() {
  const banner = document.getElementById('stale');
  if (banner) banner.style.display = 'none';
}

async function load() {
  const grid = document.getElementById('grid');
  try {
    const r = await fetch('/api/v1/dashboard');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    const s = d.service;
    document.getElementById('env').innerHTML =
      `<b>${s.environment}</b> · v${s.version} · commit <code>${s.git_commit}</code>
       · deploy:${s.deployment_provider} · registry:${s.registry_backend}
       · llm:${s.llm_provider} · aws:${s.aws_enabled?'on':'off'}`;
    lastGoodAt = new Date();
    document.getElementById('updated').textContent =
      'updated ' + lastGoodAt.toLocaleTimeString();
    // Build the markup, then replace the DOM only if it actually changed.
    // A dashboard that rebuilds seven cards every 15 seconds flickers, loses
    // any text the reader had selected, and does a lot of layout work to show
    // identical numbers.
    const html = modelCard(d.model) + deployCard(d.deployment) + driftCard(d.drift)
      + systemCard(d.system) + llmCard(d.llm) + retrainCard(d.retraining) + alertCard(d.alerts);
    if (html !== lastHtml) {
      grid.innerHTML = html;
      lastHtml = html;
    }
    clearStaleBanner();
  } catch (err) {
    showStaleBanner(err);
    if (!lastGoodAt) {
      grid.innerHTML =
        `<div class="card"><h2>Cannot reach the API</h2>
         <div class="err-box">${err}</div>
         <div class="note">Is the server running? Try
         <code>fmops serve</code> or <code>make up</code>.</div></div>`;
    }
  }
}
load();
setInterval(load, 15000);
</script>
</body>
</html>
"""
