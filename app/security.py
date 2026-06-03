from fastapi import Header, HTTPException, status

from app.settings import API_TOKEN


async def require_api_token(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    if not API_TOKEN:
        return

    bearer = f"Bearer {API_TOKEN}"
    if authorization == bearer or x_api_key == API_TOKEN:
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="invalid or missing API token",
    )
