from fastapi import APIRouter

router = APIRouter()


@router.post("/review")
def submit_review():
    return {"status": "accepted"}
