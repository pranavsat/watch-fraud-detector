"""Typed request and response contracts for the scoring API."""
from typing import Optional, List
from pydantic import BaseModel, Field


class ListingRequest(BaseModel):
    """A single watch listing submitted for scoring.

    Only brand and price are required. Every other field is optional, because a
    missing field is itself a signal (it raises the completeness component of the
    risk score).
    """
    brand: str = Field(..., examples=["Rolex"])
    price: float = Field(..., gt=0, examples=[8500.0])
    model: Optional[str] = Field(None, examples=["Submariner Date"])
    ref: Optional[str] = Field(None, examples=["126610LN"])
    mvmt: Optional[str] = Field(None, examples=["Automatic"])
    casem: Optional[str] = Field(None, examples=["Steel"])
    bracem: Optional[str] = Field(None, examples=["Steel"])
    yop: Optional[str] = Field(None, examples=["2021"])
    condition: Optional[str] = Field(None, examples=["Very good"])
    size: Optional[str] = Field(None, examples=["41 mm"])


class ScoreBreakdown(BaseModel):
    """The individual signals that feed the final risk score."""
    price_zscore: float
    underpriced_score: float
    spec_anomaly_score: float
    completeness_score: float
    mvmt_price_flag: int
    cond_price_flag: int
    age_price_flag: int
    null_count: int
    price_pct_in_brand: float


class ScoreResponse(BaseModel):
    """The scoring result returned to the caller."""
    risk_score: float = Field(..., description="Combined 0-100 risk score")
    risk_band: str = Field(..., description="low / medium / high")
    reasons: List[str] = Field(..., description="Human-readable triggers")
    breakdown: ScoreBreakdown
