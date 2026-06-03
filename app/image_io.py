import asyncio

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.observability import now
from app.settings import MAX_IMAGE_BYTES


async def read_image_file(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"uploaded file '{file.filename}' is empty",
        )
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"uploaded file '{file.filename}' is too large: {len(data)} bytes, max {MAX_IMAGE_BYTES}",
        )
    return data


def decode_image(data: bytes, filename: str | None) -> Image.Image:
    try:
        from io import BytesIO

        with Image.open(BytesIO(data)) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"uploaded file '{filename or '<unnamed>'}' is not a valid image",
        ) from exc
async def load_images(files: list[UploadFile]) -> tuple[list[Image.Image], list[str | None], float]:
    decode_start = now()
    images: list[Image.Image] = []
    filenames: list[str | None] = []
    for file in files:
        data = await read_image_file(file)
        image = await asyncio.to_thread(decode_image, data, file.filename)
        images.append(image)
        filenames.append(file.filename)
    return images, filenames, now() - decode_start
