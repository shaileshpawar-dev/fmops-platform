# Datasets

## Versions are content, not events

A dataset version is identified by the hash of its bytes. Registering the same
file twice returns the version that already exists rather than creating a
second one, so a version id always names exactly one set of rows. This is what
makes `dataset_version` on a model record meaningful: it can be resolved back
to the data the model was actually fitted on, which is how the drift engine
finds its reference window.

```bash
curl localhost:8000/api/v1/datasets
curl localhost:8000/api/v1/datasets/v2-f4a45097
```

## Uploading

```bash
curl -X POST "localhost:8000/api/v1/datasets/upload?filename=loans.csv" \
  -H "Content-Type: text/csv" \
  --data-binary @loans.csv
```

The CSV is the raw request body; metadata goes in the query string. This is
deliberately **not** a multipart form: multipart would pull `python-multipart`
into the runtime image for one single-file endpoint, and the environment
already carries a conflicting `multipart` package that FastAPI refuses to use.
A raw body needs no parser and is trivial to send from a browser:

```js
fetch(url, { method: "POST", body: file, headers: { "Content-Type": "text/csv" } })
```

The console's Datasets page does exactly that.

Query parameters: `filename`, `dataset_name`, `description`, `validate`
(default `true`).

### What the endpoint refuses

| Condition | Response |
|---|---|
| Extension other than `.csv` | 422 |
| Body over 25 MB | 422, aborted mid-stream |
| Empty body | 422 |
| Unparseable CSV | 422, with the parser's message |
| Zero rows or zero columns | 422 |

The body is read incrementally and abandoned the moment it passes the cap, so
an oversized or endless upload is never fully buffered.

**The filename never chooses a path.** It is reduced to a bare stem —
only `[A-Za-z0-9._-]` survives, and only the last path component is considered
— so `../../../etc/passwd.csv` becomes `passwd`. Where the bytes land is
decided by the dataset registry, not by the client. The file is parsed with
pandas and nothing else: never executed, never passed to a shell.

## Validation

```bash
curl localhost:8000/api/v1/datasets/v2-f4a45097/validation
```

This is the same engine and the same expectations the training pipeline gates
on, so a dataset that passes here is one the pipeline will accept. A failing
report names the expectation, the column, what was observed and what was
expected:

```json
{
  "passed": false,
  "expectations": 58,
  "succeeded": 56,
  "failed": 2,
  "failures": [
    {
      "expectation": "column_values_in_set",
      "column": "employment_type",
      "severity": "error",
      "observed": "['full_time', 'part_time']",
      "expected": "['salaried', 'self_employed', 'contract', 'retired', 'unemployed']",
      "message": "employment_type contains unexpected categories: full_time, part_time"
    }
  ]
}
```

Only the first 50 failures are returned; `truncated` says when more exist.

## Preview

```bash
curl "localhost:8000/api/v1/datasets/v2-f4a45097/preview?rows=20"
```

Returns a bounded sample plus a per-column profile (dtype, missing count and
percentage, cardinality, an example value). Capped at 50 rows — a preview
exists to show the shape of the data, and shipping a training set to a browser
is neither useful nor safe. The full row count is still reported.

## What is not here

- **No delete.** Versions are immutable records that model versions point at;
  removing one would orphan the reference that makes drift comparison possible.
- **No in-place edit.** Fix the data and upload it; that is a new version.
- **No formats besides CSV.** Parquet and JSON are not accepted, because
  nothing in the platform currently needs them.
