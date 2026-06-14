# Watch Listing Anomaly Detector

**Live demo:** https://watch-fraud-detector-27bytjmxvdwgdfut3elsxq.streamlit.app/

An unsupervised machine learning service that scores luxury watch listings for how
unusual they are. Given a listing, it returns a 0-100 risk score with a plain-language
breakdown of what looks off. The score is a signal to investigate, not a verdict of fraud.

Built on roughly 270,000 real Chrono24 listings, trained offline in a Jupyter notebook,
and served through a FastAPI backend with a Streamlit front end.

## What problem it addresses

The counterfeit and mispricing problem in the secondary luxury watch market is large and
growing. Buyers on marketplaces have little protection against listings that are
mislabeled, miscategorised, or priced suspiciously. This project explores how far you can
get at flagging such listings using only the listing metadata (brand, model, price,
movement, condition, year, and so on), with no labelled examples of fraud.

## How it works

There are no "fake" labels in the data, so this is **unsupervised anomaly detection**. The
model learns what normal listings look like for each brand and model, then flags listings
that deviate. Three separate, interpretable signals are combined into the final score.

| Signal | What it measures |
|---|---|
| Underpriced | How far below the brand+model norm the price sits. This is the scam-relevant direction: a real-looking listing priced too low. |
| Spec anomaly | Movement, condition, and age inconsistent with price, scored by an Isolation Forest. |
| Completeness | How many key specification fields are missing from the listing. |

Final score:

```
risk = 0.60 * underpriced + 0.25 * spec_anomaly + 0.15 * completeness
```

The weighting leans on underpricing because overpricing usually means a rare or collectible
piece, not a scam.

## Key finding

On curated dealer data, statistical anomalies turn out to be mostly legitimate. The most
overpriced listings are rare, vintage, or luxury-quartz pieces. The cheapest listings are
largely watch parts and accessories listed under a full model name. There is no strong fraud
signal in metadata alone, because dealer platforms vet their listings.

For that reason the project is framed honestly as a **listing-quality and mispricing
detector**. It flags two things marketplaces and buyers genuinely care about: listings that
are mislabeled or miscategorised, and listings priced far from market norms. True
counterfeit detection would require labelled data or signals beyond metadata, such as image
verification or seller reputation.

## Project structure

```
api/
  main.py       FastAPI app with /health and /score endpoints
  scoring.py    feature recomputation and scoring (mirrors the training notebook)
  schemas.py    Pydantic request and response models
dashboard/
  app.py        Streamlit front end (calls the API locally, scores in-process on the cloud)
models/         saved model artifacts (.pkl files produced by the notebook)
requirements.txt
```

The Streamlit app runs in two modes. Locally, if the `API_URL` environment variable is set,
it calls the FastAPI service over HTTP. On Streamlit Cloud, where only the Streamlit process
runs, it imports the scoring module and scores in-process.

## Try it

Open the live demo and enter a listing. Here are a few examples that show each signal.

**1. A normal, fairly priced watch (expect a LOW score)**

```
Brand: Rolex
Model: Submariner Date
Price: 14000
Movement: Automatic
Condition: Very good
Year of production: 2021
Case size: 41 mm
Reference: 126610LN
```

**2. A suspiciously underpriced watch (expect a HIGH score)**

```
Brand: Rolex
Model: Submariner Date
Price: 3000
Movement: Automatic
Condition: Unworn
Year of production: 2022
Case size: 41 mm
```

This one is flagged because the price is several standard deviations below the market norm
for a Submariner Date, and it claims "Unworn" condition at a low price.

**3. A quartz watch priced like an automatic (expect a MEDIUM score)**

```
Brand: Rolex
Model: Datejust 41
Price: 20000
Movement: Quartz
Condition: New
```

**4. A bare-bones listing with little information (expect a MEDIUM score)**

```
Brand: Rolex
Price: 8000
```

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the model artifacts by running the training notebook end to end, including the
final save cell, which writes the `.pkl` files into `models/`.

Start the API:

```bash
uvicorn api.main:app --reload --port 8000
```

Start the dashboard in a second terminal:

```bash
API_URL=http://localhost:8000 streamlit run dashboard/app.py
```

Interactive API documentation is available at `http://localhost:8000/docs`.

## Example API request

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"brand":"Rolex","model":"Submariner Date","price":3000,"mvmt":"Automatic","condition":"Unworn"}'
```

```json
{
  "risk_score": 92.9,
  "risk_band": "high",
  "reasons": [
    "Priced well below market for its brand/model (z=-3.9)",
    "Premium condition but unusually low price"
  ],
  "breakdown": { "...": "..." }
}
```

## Tech stack

Python, pandas, scikit-learn (Isolation Forest, TF-IDF, TruncatedSVD), FastAPI, Pydantic,
Streamlit, joblib. Trained in Jupyter, deployed on Streamlit Community Cloud.

## Known limitations

- The model checks price and spec consistency, but not semantics. An impossible brand and
  model pairing (for example a "Longines Speedmaster", since Speedmaster is an Omega model)
  passes through, because no feature validates that a model name belongs to a brand.
- Reference numbers and years are not validated for plausibility. They only contribute to
  the completeness signal.
- Without labelled fraud data, the model is evaluated by inspecting top-ranked listings
  rather than by precision and recall.
