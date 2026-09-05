from fastapi import APIRouter

from api.schemas import ScoreRequest, ScoreResponse

router = APIRouter()


@router.post("/score", response_model=ScoreResponse)
def score(payload: ScoreRequest):
    return ScoreResponse(
        alert=(payload.txn_count > 0),
        label="normal",
        confidence=0.5,
        reason_codes=["baseline"],
    )
