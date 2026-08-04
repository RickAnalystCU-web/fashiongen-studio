"""Small shared utilities for FashionGen Studio."""

import base64
import io

from PIL import Image


def image_to_base64_png(image: Image.Image) -> str:
    """Encode a Pillow image as a UTF-8 base64 PNG string."""

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# TODO: Add a tensor-to-Pillow conversion after the CVAE output range and
# normalization convention are finalized.
