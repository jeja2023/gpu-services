import numpy as np
from PIL import Image

from app.schemas import LetterboxMeta


def letterbox_image(image: Image.Image, input_height: int, input_width: int) -> tuple[np.ndarray, LetterboxMeta]:
    original_width, original_height = image.size
    scale = min(input_width / original_width, input_height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    pad_left = (input_width - resized_width) / 2
    pad_top = (input_height - resized_height) / 2

    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (input_width, input_height), (114, 114, 114))
    canvas.paste(resized, (int(round(pad_left - 0.1)), int(round(pad_top - 0.1))))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = np.transpose(array, (2, 0, 1))
    meta: LetterboxMeta = {
        "original_width": original_width,
        "original_height": original_height,
        "input_width": input_width,
        "input_height": input_height,
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
    }
    return tensor, meta


def resize_image_tensor(image: Image.Image, input_height: int, input_width: int, normalize: str = "none") -> np.ndarray:
    resized = image.resize((input_width, input_height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    if normalize == "imagenet":
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        array = (array - mean) / std
    return np.transpose(array, (2, 0, 1))
