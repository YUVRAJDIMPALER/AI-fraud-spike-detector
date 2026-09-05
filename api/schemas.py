from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    step: int = Field(..., ge=0)
    txn_count: float = Field(..., ge=0)
    unique_ratio: float = Field(..., ge=0)
    avg_amount: float = Field(..., ge=0)
    total_amount: float = Field(..., ge=0)


class ScoreResponse(BaseModel):
    alert: bool
    label: str
    confidence: float
    reason_codes: list[str]
