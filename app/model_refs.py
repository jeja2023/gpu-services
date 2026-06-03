from fastapi import HTTPException, status


def cache_key(project_name: str, model_name: str) -> str:
    return f"{project_name}/{model_name}"


def validate_path_name(value: str) -> str:
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("path separators and relative path segments are not allowed")
    return value


def split_cache_key(value: str) -> tuple[str, str]:
    parts = value.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model must use 'project_name/model_name' format",
        )
    try:
        return validate_path_name(parts[0]), validate_path_name(parts[1])
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

