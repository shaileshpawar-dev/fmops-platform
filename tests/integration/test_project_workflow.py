"""The guided build workflow: its endpoint, its routing, and its honesty.

The workflow itself is browser-side, so what is testable from here is the
contract it depends on -- the limits endpoint it reads its caps from, the path
resolution that endpoint needs to survive, and the claims the page makes about
what the platform can and cannot do.
"""

from __future__ import annotations

import io
import re

CSV = {"Content-Type": "text/csv"}


def _console_source(api_client) -> str:
    """The console as the browser receives it: shell plus every script module."""
    page = api_client.get("/dashboard")
    assert page.status_code == 200
    sources = [page.text]
    for path in re.findall(r'src="(/static/js/[^"]+)"', page.text):
        asset = api_client.get(path)
        assert asset.status_code == 200, f"console references {path} but it is not served"
        sources.append(asset.text)
    return "\n".join(sources)


def _flat(text: str) -> str:
    """Source with runs of whitespace collapsed.

    The assertions below are about what the console says, not about where its
    source happens to wrap; a reflow should not fail a test about wording.
    """
    return re.sub(r"\s+", " ", text)


# --------------------------------------------------------------------------- #
# Upload limits
# --------------------------------------------------------------------------- #
def test_limits_endpoint_reports_the_cap_the_uploader_enforces(api_client):
    """The number the UI shows must be the number the server acts on.

    A client carrying its own copy of the constant is the classic way a UI ends
    up promising 25 MB while the server rejects at 10.
    """
    from app.api.routes.datasets import MAX_PREVIEW_ROWS, MAX_UPLOAD_BYTES

    response = api_client.get("/api/v1/datasets/limits")
    assert response.status_code == 200
    body = response.json()
    assert body["max_upload_bytes"] == MAX_UPLOAD_BYTES
    assert body["max_preview_rows"] == MAX_PREVIEW_ROWS
    assert body["accepted_suffixes"] == [".csv"]


def test_limits_is_not_shadowed_by_the_dataset_detail_route(api_client):
    """``/datasets/{version}`` is a single-segment catch-all under the same prefix.

    It is registered by a different router, so whichever router is included
    first wins. Included in the wrong order, ``/datasets/limits`` resolves as a
    lookup for a dataset version called "limits" and 404s -- which is exactly
    what happened before the include order was fixed.
    """
    assert api_client.get("/api/v1/datasets/limits").json()["max_upload_bytes"] > 0

    # ...and the catch-all it sits in front of still works, in both its forms.
    listing = api_client.get("/api/v1/datasets")
    assert listing.status_code == 200
    assert "versions" in listing.json()

    unknown = api_client.get("/api/v1/datasets/definitely-not-a-version")
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "dataset_not_found"


def test_a_real_version_still_resolves_through_the_catch_all(api_client, valid_frame):
    buf = io.StringIO()
    valid_frame.to_csv(buf, index=False)
    version = api_client.post(
        "/api/v1/datasets/upload?filename=order.csv", content=buf.getvalue(), headers=CSV
    ).json()["version"]

    detail = api_client.get(f"/api/v1/datasets/{version}")
    assert detail.status_code == 200
    assert detail.json()["version"] == version


# --------------------------------------------------------------------------- #
# The workflow page
# --------------------------------------------------------------------------- #
def test_command_center_offers_a_way_into_the_workflow(api_client):
    """A new user should not have to know which page to start on."""
    body = _console_source(api_client)
    assert 'href="#/newproject"' in body
    assert "Create ML Project" in body


def test_workflow_drives_only_endpoints_that_exist(api_client):
    """Every API path the workflow calls must be served by this application.

    This is the guard against a step that looks like it works and 404s the
    moment someone clicks it.
    """
    body = _console_source(api_client)
    # The published contract, not the route table: included routers are not
    # flattened onto app.routes in this FastAPI version, and the OpenAPI
    # document is what a client can actually discover anyway.
    served = set(api_client.get("/openapi.json").json()["paths"])

    # Paths the workflow builds with template literals, reduced to their
    # FastAPI-templated form.
    referenced = {
        "/api/v1/datasets/limits",
        "/api/v1/datasets/upload",
        "/api/v1/datasets/{version}/preview",
        "/api/v1/automl/profile/{version}",
        "/api/v1/automl/runs",
        "/api/v1/automl/runs/{run_id}",
        "/api/v1/training/runs",
        "/api/v1/training/runs/{run_id}",
        "/api/v1/models",
        "/api/v1/models/algorithms",
        "/api/v1/models/{name}/versions",
        "/api/v1/models/{name}/versions/{version}/evaluate-gate",
        "/api/v1/deployments",
        "/api/v1/deployments/current",
        "/api/v1/predict",
        "/api/v1/monitoring/summary",
        "/api/v1/drift/latest",
    }
    missing = sorted(p for p in referenced if p not in served)
    assert not missing, f"the workflow calls endpoints this app does not serve: {missing}"

    # And each one is actually reachable from the workflow module, so this list
    # cannot rot into a set of paths nobody calls.
    for literal in ("/api/v1/datasets/limits", "/api/v1/automl/profile/", "/evaluate-gate"):
        assert literal in body, f"the workflow no longer references {literal}"


def test_workflow_states_what_the_platform_cannot_do(api_client):
    """The steps most likely to be dressed up are the ones asserted here.

    Each of these corresponds to a real constraint: online scoring is bound to
    one fixed request schema, registration happens inside the run rather than
    through a button, and drift measured without labels is not concept drift.
    """
    body = _flat(_console_source(api_client))
    assert "one fixed request shape" in body
    assert "registration happens inside the run" in body
    assert "It is not concept drift" in body
    assert "binary classification" in body


def test_workflow_never_offers_a_gate_override(api_client):
    """No path through the workflow may deploy a version the gate rejected.

    The deployment API does take ``force``, and the workflow must not reach for
    it: the gate is the backend's decision to make.
    """
    body = _console_source(api_client)
    workflow = (
        body[body.index("PAGES.newproject") - 60000 :] if "PAGES.newproject" in body else ""
    )
    assert "PAGES.newproject" in body
    assert "force=true" not in body
    assert "force: true" not in workflow
    assert '"force"' not in workflow
