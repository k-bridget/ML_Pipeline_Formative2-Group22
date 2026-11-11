from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "data" / "raw" / "audio"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "reports" / "audio_visualizations"
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512


def audio_labels(audio_path: Path) -> tuple[str, str]:
    parts = audio_path.stem.split("_", maxsplit=1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Expected an audio filename in Member_phrase.mp3 format: {audio_path.name}"
        )
    return parts[0], parts[1]


def extract_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    rms = librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH
    )

    features = {
        f"mfcc_{index + 1}_{stat}": float(values)
        for index, coefficient in enumerate(mfcc)
        for stat, values in (("mean", np.mean(coefficient)), ("std", np.std(coefficient)))
    }
    features.update(
        {
            "spectral_rolloff_mean_hz": float(np.mean(rolloff)),
            "rms_energy_mean": float(np.mean(rms)),
            "spectral_centroid_mean_hz": float(np.mean(centroid)),
            "zero_crossing_rate_mean": float(np.mean(zero_crossing_rate)),
        }
    )
    return features


def make_noise_augmentation(audio: np.ndarray, seed: int) -> np.ndarray:
    signal_rms = float(np.sqrt(np.mean(np.square(audio))))
    noise_rms = signal_rms / (10 ** (30 / 20))
    noise = np.random.default_rng(seed).normal(0.0, noise_rms, size=audio.shape)
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


def plot_speaker_recordings(
    speaker: str,
    recordings: list[tuple[str, np.ndarray, int]],
    plot_dir: Path,
) -> Path:
    figure, axes = plt.subplots(len(recordings), 2, figsize=(13, 4 * len(recordings)))
    axes = np.atleast_2d(axes)

    for row, (phrase, audio, sample_rate) in enumerate(recordings):
        times = np.arange(audio.size) / sample_rate
        axes[row, 0].plot(times, audio, linewidth=0.65, color="#176b87")
        axes[row, 0].set(title=f"{phrase}: waveform", xlabel="Time (s)", ylabel="Amplitude")
        axes[row, 0].grid(alpha=0.2)

        spectrogram = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)), ref=np.max
        )
        image = librosa.display.specshow(
            spectrogram,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="hz",
            ax=axes[row, 1],
            cmap="magma",
        )
        axes[row, 1].set(title=f"{phrase}: spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)")
        figure.colorbar(image, ax=axes[row, 1], format="%+2.0f dB")

    figure.suptitle(f"Original voice samples: {speaker}", fontsize=14)
    figure.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"{speaker.lower()}_waveforms_spectrograms.png"
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return plot_path


def process_recordings(
    audio_dir: Path = DEFAULT_AUDIO_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    plot_dir: Path = DEFAULT_PLOT_DIR,
) -> pd.DataFrame:
    audio_paths = sorted(
        path for path in audio_dir.iterdir() if path.is_file() and path.suffix.lower() in {".mp3", ".wav"}
    )
    if not audio_paths:
        raise FileNotFoundError(f"No MP3 or WAV recordings found in {audio_dir}")

    feature_rows: list[dict[str, object]] = []
    plots_by_speaker: dict[str, list[tuple[str, np.ndarray, int]]] = {}
    augmentation_dir = output_dir / "audio_augmentations"

    for audio_path in audio_paths:
        speaker, phrase = audio_labels(audio_path)
        audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        if audio.size == 0:
            raise ValueError(f"Audio recording is empty: {audio_path}")
        audio = audio.astype(np.float32)
        plots_by_speaker.setdefault(speaker, []).append((phrase, audio, sample_rate))
        seed = zlib.crc32(audio_path.name.encode("utf-8"))

        variants = [
            ("original", audio, None),
            ("pitch_up_2_semitones", librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=2), "pitch_up_2_semitones"),
            ("noise_30db_snr", make_noise_augmentation(audio, seed), "noise_30db_snr"),
        ]
        for augmentation, variant, output_name in variants:
            audio_file = audio_path.relative_to(PROJECT_ROOT).as_posix()
            if output_name is not None:
                output_path = augmentation_dir / f"{audio_path.stem}_{output_name}.wav"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(output_path, variant, sample_rate)
                audio_file = output_path.relative_to(PROJECT_ROOT).as_posix()

            feature_rows.append(
                {
                    "sample_id": f"{audio_path.stem}__{augmentation}",
                    "speaker": speaker,
                    "phrase": phrase,
                    "augmentation": augmentation,
                    "audio_file": audio_file,
                    "sample_rate_hz": sample_rate,
                    "duration_seconds": float(variant.size / sample_rate),
                    **extract_features(variant, sample_rate),
                }
            )

    for speaker, recordings in sorted(plots_by_speaker.items()):
        plot_path = plot_speaker_recordings(speaker, recordings, plot_dir)
        print(f"Saved waveform/spectrogram plot: {plot_path.relative_to(PROJECT_ROOT)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    features = pd.DataFrame(feature_rows).sort_values(
        ["speaker", "phrase", "augmentation"], ignore_index=True
    )
    features_path = output_dir / "audio_features.csv"
    features.to_csv(features_path, index=False)
    print(f"Processed {len(audio_paths)} recordings into {len(features)} feature rows.")
    print(f"Saved features: {features_path.relative_to(PROJECT_ROOT)}")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment voice recordings and extract audio features.")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    args = parser.parse_args()
    process_recordings(args.audio_dir, args.output_dir, args.plot_dir)


if __name__ == "__main__":
    main()