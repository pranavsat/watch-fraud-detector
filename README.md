# Watch Listing Anomaly Detector

An unsupervised anomaly-detection service for luxury watch listings. It scores how
unusual a listing is across three dimensions and returns a 0-100 risk score with a
plain-language breakdown. The score is a signal to investigate, not a fraud verdict.

Built on ~270k Chrono24 listings. Trained offline in the notebook, served as a
FastAPI endpoint with a Streamlit front end.

## What it scores

| Signal | Meaning |
|---|---|
| Underpriced | Price below the brand+model norm (the scam-relevant direction) |
| Spec anomaly | Movement / condition / age inconsistent with price (Isolation Forest) |
| Completeness | How many spec fields are missing |

Combined: `risk = 0.60 * underpriced + 0.25 * spec_anomaly + 0.15 * completeness`

## Key finding

On curated dealer data, statistical anomalies are mostly legitimate: overpriced
outliers are rare/vintage/luxury-quartz pieces, and the cheapest listings are largely
parts and accessories. There is no strong fraud signal in metadata alone, so the
service is framed honestly as a **listing-quality and mispricing detector** - which is
genuinely useful to marketplaces and buyers.

## Project layout

```
api/
  main.py       FastAPI app (/health, /score)
  scoring.py    feature recomputation + scoring (mirrors the notebook)
  schemas.py    Pydantic request/response models
dashboard/
  app.py        Streamlit client that calls the API
models/         saved artifacts (produced by the notebook save cell)
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Generate the model artifacts by running the training notebook end to end, including
the final save cell. It writes the `.pkl` files into `models/`.

## Run locally

API:

```bash
uvicorn api.main:app --reload --port 8000
```

Dashboard (in a second terminal):

```bash
API_URL=http://localhost:8000 streamlit run dashboard/app.py
```

## Example request

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"brand":"Rolex","model":"Submariner Date","price":3000,"mvmt":"Automatic","condition":"Unworn"}'
```

```json
{
  "risk_score": 96.6,
  "risk_band": "high",
  "reasons": [
    "Priced well below market for its brand/model (z=-5.0)",
    "Premium condition but unusually low price"
  ],
  "breakdown": { "...": "..." }
}
```

## Notes

- Only `brand` and `price` are required. Missing fields raise the completeness signal.
- The API is stateless; all learned reference data lives in `models/`.
- Interactive API docs are available at `/docs` when the server is running.
