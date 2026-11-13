from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image


IMAGE_SIZE = (224, 224)
HISTOGRAM_BINS = 16
EMBEDDING_SIZE = 512


def extract_face_features(image_path: Path, expected_features: list[str]) -> pd.DataFrame:
    """Extract the histogram and Facenet512 features used by the image notebook."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        from deepface import DeepFace
    except ImportError as error:
        raise RuntimeError(
            "Face inference requires DeepFace. Install the image dependencies with "
            "`python -m pip install -r requirements-demo.txt`."
        ) from error

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB").resize(IMAGE_SIZE)
    image_rgb = np.asarray(image)
    features: dict[str, float] = {}

    for channel_index, channel_name in enumerate(("r", "g", "b")):
        histogram = cv2.calcHist(
            [image_rgb], [channel_index], None, [HISTOGRAM_BINS], [0, 256]
        ).flatten()
        histogram /= histogram.sum() + 1e-7
        features.update(
            {
                f"hist_{channel_name}_{bin_index}": float(value)
                for bin_index, value in enumerate(histogram)
            }
        )

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    try:
        representations: list[dict[str, Any]] = DeepFace.represent(
            img_path=image_bgr,
            model_name="Facenet512",
            detector_backend="retinaface",
            enforce_detection=True,
        )
    except Exception as error:
        raise ValueError(f"Face detection or embedding failed for {image_path.name}: {error}") from error

    if len(representations) != 1:
        raise ValueError(
            f"Expected exactly one detected face in {image_path.name}; "
            f"found {len(representations)}."
        )

    embedding = np.asarray(representations[0]["embedding"], dtype=np.float32)
    if embedding.size != EMBEDDING_SIZE or not np.isfinite(embedding).all():
        raise ValueError(
            f"Expected a finite {EMBEDDING_SIZE}-value Facenet512 embedding, "
            f"got {embedding.size} values."
        )
    embedding /= np.linalg.norm(embedding) + 1e-8
    features.update({f"emb_{index}": float(value) for index, value in enumerate(embedding)})

    missing = [feature for feature in expected_features if feature not in features]
    extra = sorted(set(features) - set(expected_features))
    if missing or extra:
        raise ValueError(
            "Image feature schema does not match the saved face model "
            f"(missing={len(missing)}, extra={len(extra)})."
        )

    return pd.DataFrame([[features[name] for name in expected_features],], columns=expected_features)