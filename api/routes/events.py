from fastapi import APIRouter

router = APIRouter()


@router.get("/events")
def list_events():
    return {"events": []}
