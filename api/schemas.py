"""Pydantic schemas for the FastAPI inference endpoint.

Defines the structure of the incoming HTTP request and the outgoing
prediction response, including SHAP-based reason codes.
"""

from typing import List
from pydantic import BaseModel, Field

# We can reuse the base transaction fields from ingestion
# but we shouldn't require ground truth 'isFraud' for prediction.
class PredictRequest(BaseModel):
    """Incoming transaction payload for inference."""
    step: int = Field(..., description="Time unit (1 step = 1 hour)")
    type: str = Field(..., description="Transaction type: TRANSFER, CASH_OUT, etc.")
    amount: float = Field(..., description="Transaction amount")
    nameOrig: str = Field(..., description="Origin account ID")
    oldbalanceOrg: float = Field(..., description="Origin balance before transaction")
    newbalanceOrig: float = Field(..., description="Origin balance after transaction")
    nameDest: str = Field(..., description="Destination account ID")
    oldbalanceDest: float = Field(..., description="Destination balance before transaction")
    newbalanceDest: float = Field(..., description="Destination balance after transaction")


class ReasonCode(BaseModel):
    """A single feature's SHAP contribution to the fraud score."""
    feature: str = Field(..., description="Name of the feature")
    value: float = Field(..., description="Actual value of the feature")
    contribution: float = Field(..., description="SHAP value (log-odds contribution)")


class PredictResponse(BaseModel):
    """Outgoing prediction result."""
    transaction_id: str = Field(..., description="Origin account ID (nameOrig)")
    is_fraud: bool = Field(..., description="Final ensemble decision based on optimal threshold")
    probability: float = Field(..., description="Ensemble fraud probability score [0.0 - 1.0]")
    threshold_used: float = Field(..., description="The probability threshold used for the decision")
    reasons: List[ReasonCode] = Field(default_factory=list, description="Top features driving the score")
