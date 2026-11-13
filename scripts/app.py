from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw" / "audio"
PROCESSED_DIR = DATA_DIR / "processed"
VOICE_THRESHOLD = 0.60
FACE_THRESHOLD = 0.60


def load_artifact(name: str) -> Any:
    path = MODEL_DIR / f"{name}.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Required model artifact is missing: {path}")
    return joblib.load(path)


def load_models() -> dict[str, Any]:
    return {
        "face_model": load_artifact("face_model"),
        "face_encoder": load_artifact("face_encoder"),
        "face_features": load_artifact("face_features"),
        "voice_model": load_artifact("voice_model"),
        "voice_encoder": load_artifact("voice_encoder"),
        "voice_features": load_artifact("voice_features"),
        "product_model": load_artifact("product_recommender"),
        "product_encoder": load_artifact("target_encoder"),
        "product_features": load_artifact("model_features"),
    }


def extract_voice_features(audio_path: Path, expected_features: list[str]) -> pd.DataFrame:
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio recording not found: {audio_path}")

    from process_audio import extract_features

    waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    if waveform.size == 0:
        raise ValueError(f"Audio recording is empty: {audio_path}")

    values = {
        "sample_rate_hz": float(sample_rate),
        "duration_seconds": float(waveform.size / sample_rate),
        **extract_features(waveform.astype(np.float32), sample_rate),
    }
    missing = [feature for feature in expected_features if feature not in values]
    if missing:
        raise ValueError(f"Audio feature extractor is missing model inputs: {missing}")
    return pd.DataFrame([[values[name] for name in expected_features]], columns=expected_features)


def predict_identity(
    model: Any,
    encoder: Any,
    features: pd.DataFrame,
    threshold: float,
    modality: str,
) -> tuple[str | None, float, list[tuple[str, float]]]:
    probabilities = model.predict_proba(features)[0]
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    label = str(encoder.inverse_transform([best_index])[0])
    ranked = sorted(
        (
            (str(encoder.inverse_transform([index])[0]), float(probability))
            for index, probability in enumerate(probabilities)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if confidence < threshold:
        print(f"{modality} identity: REJECTED ({confidence:.3f} < {threshold:.2f})")
        return None, confidence, ranked
    print(f"{modality} identity: {label} ({confidence:.3f})")
    return label, confidence, ranked


def face_features_from_csv(csv_path: Path, image_path: Path, expected: list[str]) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Face feature CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    if "source_file" not in frame.columns:
        raise ValueError("Face feature CSV must include a source_file column.")
    matches = frame[frame["source_file"].astype(str).map(lambda value: Path(value).name) == image_path.name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one feature row for {image_path.name}, found {len(matches)}. "
            "Pass the CSV created by the project's image-processing pipeline."
        )
    missing = [feature for feature in expected if feature not in matches.columns]
    if missing:
        raise ValueError(f"Face feature CSV is missing model inputs: {missing}")
    return matches[expected].astype(float)


def recommend_product(models: dict[str, Any], customer_id: str | None) -> None:
    dataset_path = PROCESSED_DIR / "merged_customer_data.csv"
    if not dataset_path.is_file():
        print(f"Recommendation unavailable: dataset not found at {dataset_path}")
        return

    customers = pd.read_csv(dataset_path)
    if customer_id:
        if "customer_id" not in customers.columns:
            raise ValueError("Merged customer data has no customer_id column.")
        rows = customers[customers["customer_id"].astype(str) == customer_id]
        if rows.empty:
            print(f"Recommendation unavailable: customer {customer_id!r} is not in the dataset.")
            return
        customer = rows.iloc[[0]]
    else:
        customer = customers.iloc[[0]]

    missing = [feature for feature in models["product_features"] if feature not in customer.columns]
    if missing:
        raise ValueError(f"Merged customer data is missing product model inputs: {missing}")

    feature_row = customer[models["product_features"]].copy()
    probabilities = models["product_model"].predict_proba(feature_row)[0]
    ranking = sorted(
        (
            (str(models["product_encoder"].inverse_transform([index])[0]), float(probability))
            for index, probability in enumerate(probabilities)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    print(f"Product recommendations for {customer['customer_id'].iloc[0]}:")
    for product, probability in ranking[:3]:
        print(f"  {product}: {probability:.3f}")


def run_demo(
    models: dict[str, Any],
    image_path: Path,
    audio_path: Path,
    face_user: str | None,
    face_features_csv: Path | None,
    customer_id: str | None,
) -> int:
    print("Multimodal transaction demonstration")
    print(f"Face input:  {image_path}")
    print(f"Voice input: {audio_path}")

    if not image_path.is_file():
        print(f"ACCESS DENIED: image sample not found: {image_path}")
        return 2

    if face_user is not None:
        if face_user not in list(models["face_encoder"].classes_):
            print(f"ACCESS DENIED: {face_user!r} is not a known face identity.")
            return 2
        face_identity = face_user
        face_confidence = 1.0
        print(f"Face identity: {face_identity} (fixture, confidence 1.000)")
    elif face_features_csv is not None:
        try:
            image_features = face_features_from_csv(
                face_features_csv, image_path, models["face_features"]
            )
        except (FileNotFoundError, ValueError) as error:
            print(f"ACCESS DENIED: {error}")
            return 2
        face_identity, face_confidence, _ = predict_identity(
            models["face_model"],
            models["face_encoder"],
            image_features,
            FACE_THRESHOLD,
            "Face",
        )
        if face_identity is None:
            print("ACCESS DENIED: face identity confidence is below threshold.")
            return 2
    else:
        print(
            "ACCESS DENIED: live image inference is unavailable. "
            "This repository has no checked-in image feature extractor or image_features.csv. "
            "Use --face-user for the explicitly labeled demo fixture, or provide "
            "--face-features-csv from the image preprocessing pipeline."
        )
        return 2

    try:
        voice_row = extract_voice_features(audio_path, models["voice_features"])
    except (FileNotFoundError, ValueError) as error:
        print(f"ACCESS DENIED: {error}")
        return 2

    voice_identity, voice_confidence, _ = predict_identity(
        models["voice_model"],
        models["voice_encoder"],
        voice_row,
        VOICE_THRESHOLD,
        "Voice",
    )
    if voice_identity is None:
        print("ACCESS DENIED: voice identity confidence is below threshold.")
        return 2

    if face_identity != voice_identity:
        print(
            "ACCESS DENIED: face and voice identities do not match "
            f"({face_identity} != {voice_identity})."
        )
        return 2

    print(
        f"Authentication passed: {face_identity} "
        f"(face {face_confidence:.3f}, voice {voice_confidence:.3f})."
    )
    recommend_product(models, customer_id)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the assignment's sequential face/voice/product demo."
    )
    parser.add_argument("--image", type=Path, required=True, help="Image sample to authenticate.")
    parser.add_argument("--audio", type=Path, required=True, help="Spoken approval recording.")
    face_group = parser.add_mutually_exclusive_group()
    face_group.add_argument(
        "--face-user",
        help="Use a known identity as an explicitly labeled demo fixture, not image-model inference.",
    )
    face_group.add_argument(
        "--face-features-csv",
        type=Path,
        help="Image feature CSV produced by the image-processing pipeline, keyed by source_file.",
    )
    parser.add_argument("--customer-id", help="Customer row to recommend for (for example A151).")
    parser.add_argument("--voice-threshold", type=float, default=VOICE_THRESHOLD)
    parser.add_argument("--face-threshold", type=float, default=FACE_THRESHOLD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.voice_threshold <= 1 or not 0 <= args.face_threshold <= 1:
        print("Thresholds must be between 0 and 1.", file=sys.stderr)
        return 2
    global VOICE_THRESHOLD, FACE_THRESHOLD
    VOICE_THRESHOLD = args.voice_threshold
    FACE_THRESHOLD = args.face_threshold

    try:
        models = load_models()
        return run_demo(
            models=models,
            image_path=args.image.resolve(),
            audio_path=args.audio.resolve(),
            face_user=args.face_user,
            face_features_csv=args.face_features_csv.resolve() if args.face_features_csv else None,
            customer_id=args.customer_id,
        )
    except (FileNotFoundError, ValueError, ImportError) as error:
        print(f"Demo unavailable: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())