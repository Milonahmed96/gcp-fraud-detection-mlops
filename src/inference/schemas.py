"""Pydantic request/response schemas for the inference API.

Cloud Run was chosen over Vertex AI Prediction precisely so that this contract
is ours to define. Validation happens at the edge: a malformed transaction is
rejected with a 422 before it can reach the feature builder, where a bad type
would otherwise produce a confidently wrong prediction rather than an error.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

MAX_AMOUNT = 1_000_000.0
COUNTRY_CODE_LENGTH = 2


class TransactionRequest(BaseModel):
    """One transaction to score."""

    model_config = ConfigDict(extra="forbid")  # a typo'd field is a bug, not a default

    transaction_id: str = Field(min_length=1, description="Unique transaction identifier")
    customer_id: str = Field(min_length=1, description="Cardholder identifier; the entity key")
    timestamp: datetime = Field(description="Transaction event time, UTC")
    amount: float = Field(ge=0.0, le=MAX_AMOUNT, description="Transaction amount")
    merchant_id: str = Field(min_length=1)
    merchant_category: str = Field(min_length=1)
    country: str = Field(
        min_length=COUNTRY_CODE_LENGTH,
        max_length=COUNTRY_CODE_LENGTH,
        description="ISO-3166 alpha-2 country of the transaction",
    )
    customer_home_country: str = Field(
        min_length=COUNTRY_CODE_LENGTH,
        max_length=COUNTRY_CODE_LENGTH,
        description="ISO-3166 alpha-2 country of the customer",
    )
    card_present: bool = Field(description="False for card-not-present (online) transactions")


class AttributionResponse(BaseModel):
    """One feature's signed contribution, in log-odds space."""

    feature: str
    value: float
    shap_value: float
    direction: str


class PredictionResponse(BaseModel):
    """A scored transaction, with the explanation that justifies the decision."""

    transaction_id: str
    variant: str = Field(description="Which A/B model variant served this request")
    fraud_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(description="Decision threshold applied to the probability")
    is_flagged: bool = Field(description="True when fraud_probability >= threshold")
    base_value: float = Field(description="SHAP base value, log-odds space")
    top_features: list[AttributionResponse] = Field(
        description="Largest absolute SHAP contributions, incriminating and exculpatory alike"
    )
    latency_ms: float = Field(description="Server-side handler latency")
    new_customer: bool = Field(description="True when no prior history was found for this customer")


class HealthResponse(BaseModel):
    """Liveness and readiness signal for Cloud Run."""

    status: str
    variant: str
    model_loaded: bool
    explainer_loaded: bool
