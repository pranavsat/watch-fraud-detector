"""FastAPI service for luxury watch listing anomaly scoring."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ListingRequest, ScoreResponse
from . import scoring

app = FastAPI(
    title="Watch Listing Anomaly Detector",
    description="Scores a luxury watch listing for how unusual it is "
                "(mispricing, spec inconsistency, incompleteness). "
                "Unsupervised - the score is a signal to investigate, not a fraud verdict.",
    version="1.0.0",
)

# allow the Streamlit dashboard (and local dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(listing: ListingRequest):
    return scoring.score_listing(listing.model_dump())
