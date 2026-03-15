from typing import Any, Dict

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/items", tags=["items"])


@router.get("/{item_id}", response_model=Dict[str, Any])
async def read_item(item_id: int, q: str | None = None) -> Dict[str, Any]:
    if item_id <= 0:
        raise HTTPException(status_code=400, detail="item_id must be positive")
    return {"item_id": item_id, "q": q}

