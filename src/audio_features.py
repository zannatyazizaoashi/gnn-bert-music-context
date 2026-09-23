from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch


@dataclass(frozen=True)
class AudioFeatureConfig:
    sample_rate: int = 22050
    duration_seconds: float = 29.0
    segment_seconds: float = 5.0
    n_mels: int = 128
    n_mfcc: int = 20
    n_fft: int = 2048
    hop_length: int = 512


def load_audio(path: str | Path, config: AudioFeatureConfig) -> np.ndarray:
    target_length = int(config.sample_rate * config.duration_seconds)
    audio, _ = librosa.load(path, sr=config.sample_rate, mono=True, duration=config.duration_seconds)
    if audio.size == 0:
        raise ValueError(f"No audio decoded from {path}")
    audio = librosa.util.fix_length(audio, size=target_length)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


def log_mel_spectrogram(audio: np.ndarray, config: AudioFeatureConfig) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    mean = log_mel.mean()
    std = max(log_mel.std(), 1e-6)
    return ((log_mel - mean) / std).astype(np.float32)


def _summarize(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix.mean(axis=1), matrix.std(axis=1)], axis=0)


def segment_features(audio: np.ndarray, config: AudioFeatureConfig) -> np.ndarray:
    segment_samples = int(config.sample_rate * config.segment_seconds)
    features: list[np.ndarray] = []
    for start in range(0, len(audio), segment_samples):
        segment = audio[start : start + segment_samples]
        if len(segment) < config.sample_rate:
            continue
        mel = librosa.feature.melspectrogram(
            y=segment, sr=config.sample_rate, n_fft=config.n_fft,
            hop_length=config.hop_length, n_mels=config.n_mels, power=2.0
        )
        log_mel = librosa.power_to_db(mel + 1e-10, ref=np.max)
        chroma = librosa.feature.chroma_stft(
            y=segment, sr=config.sample_rate, n_fft=config.n_fft, hop_length=config.hop_length
        )
        mfcc = librosa.feature.mfcc(
            y=segment, sr=config.sample_rate, n_mfcc=config.n_mfcc,
            n_fft=config.n_fft, hop_length=config.hop_length
        )
        spectral = np.vstack(
            [
                librosa.feature.spectral_centroid(y=segment, sr=config.sample_rate, hop_length=config.hop_length),
                librosa.feature.spectral_bandwidth(y=segment, sr=config.sample_rate, hop_length=config.hop_length),
                librosa.feature.spectral_rolloff(y=segment, sr=config.sample_rate, hop_length=config.hop_length),
                librosa.feature.rms(y=segment, frame_length=config.n_fft, hop_length=config.hop_length),
                librosa.feature.zero_crossing_rate(segment, frame_length=config.n_fft, hop_length=config.hop_length),
            ]
        )
        vector = np.concatenate([_summarize(log_mel), _summarize(chroma), _summarize(mfcc), _summarize(spectral)])
        features.append(np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0))
    if not features:
        raise ValueError("Audio is too short to produce any segment")
    nodes = np.stack(features).astype(np.float32)
    mean = nodes.mean(axis=0, keepdims=True)
    std = np.maximum(nodes.std(axis=0, keepdims=True), 1e-6)
    return ((nodes - mean) / std).astype(np.float32)


def extract_track(path: str | Path, config: AudioFeatureConfig) -> tuple[torch.Tensor, torch.Tensor]:
    audio = load_audio(path, config)
    nodes = torch.from_numpy(segment_features(audio, config))
    mel = torch.from_numpy(log_mel_spectrogram(audio, config)).unsqueeze(0)
    return nodes, mel

