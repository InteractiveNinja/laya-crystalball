# Laya Crystal Ball

A FastAPI service that wraps [Laya](https://pypi.org/project/laya/), a fast non-autoregressive decision engine. Send it a **question** and a list of **answer possibilities**, and it returns the option it chose — with a calibrated confidence score and per-option probabilities.

No text generation, no parsing, no hallucination. A single forward pass, ~1.7s on CPU.

## How it works

The API maps your request onto a Laya `choice` question and runs it through Laya's `Router`, which auto-selects the best checkpoint per request:

| Checkpoint | Use |
|---|---|
| `english` (ModernBERT-large) | English text |
| `multilingual` (mmBERT-base) | 100+ languages |

Laya's probabilities are calibrated, so the `confidence` value is meaningful and can be used for automated confidence gating (e.g. auto-route at ≥0.85, escalate to a human below that).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Models are downloaded from the Hugging Face Hub on first run and cached locally.

## Run

```bash
uvicorn app.main:app --reload
```

The server preloads all checkpoints at startup (cold load takes a moment). Then:

- API base: `http://127.0.0.1:8000`
- Interactive docs (Swagger): `http://127.0.0.1:8000/docs`

## Endpoints

### `POST /predict`

Returns the chosen option for a question.

**Request body**

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | string | yes | The input text to evaluate. |
| `options` | string[] | yes | Answer possibilities (at least 2). |
| `instructions` | string | no | Guidance for the decision. Defaults to *"Which option best matches the question?"*. |

**Response**

| Field | Type | Description |
|---|---|---|
| `choice` | string | The chosen option. |
| `confidence` | number | Calibrated confidence (0–1). |
| `probabilities` | object | Per-option probability map. |
| `model` | string | Which checkpoint answered (`english` or `multilingual`). |

**Example**

```bash
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
    "options": ["billing", "technical", "sales", "other"],
    "instructions": "Which department should handle this request?"
  }'
```

```json
{
  "choice": "billing",
  "confidence": 0.9478,
  "probabilities": {
    "billing": 0.989,
    "technical": 0.0039,
    "sales": 0.0047,
    "other": 0.0024
  },
  "model": "english"
}
```

### `GET /health`

Liveness check and currently loaded models.

```json
{ "status": "ok", "loaded": ["english", "multilingual", "typed-decisions"] }
```

## Notes & limitations

- **Startup cost:** `Router(preload=True)` loads all three checkpoints at boot so no request ever pays a model load. Cold load is slow; subsequent predictions are fast.
- **Option count:** Laya is well-suited to a handful of options. Accuracy degrades for very high-cardinality label sets (50+) due to the model's per-option token budget.
- **Confidence gating:** Because probabilities are calibrated, branch on `confidence` with confidence.

## Project layout

```
app/main.py        FastAPI app + /predict + /health
requirements.txt   fastapi, laya, uvicorn
```
