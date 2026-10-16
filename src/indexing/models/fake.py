from __future__ import annotations

import hashlib
import math

import numpy as np

from src.indexing.domain.entities import (
    ClassProbabilities,
    Detection,
    Frame,
    ModelDescriptor,
    SensitiveDetection,
)


def _descriptor(task: str) -> ModelDescriptor:
    return ModelDescriptor(
        task=task,
        provider="fake",
        requested_revision="fake-v1",
        resolved_revision="fake-v1",
        model_version="1.0.0",
        framework="fake",
        device="cpu",
    )


def _seed(parts: list[bytes]) -> int:
    digest = hashlib.sha256(b"\x1f".join(parts)).digest()
    return int.from_bytes(digest[:8], "big")


def _frame_bytes(frame: Frame) -> bytes:
    pixels = np.asarray(frame.pixels)
    sample = pixels[:: max(1, pixels.shape[0] // 16), :: max(1, pixels.shape[1] // 16)]
    return sample.tobytes() + str(frame.timestamp_ms).encode()


class FakeObjectDetector:
    descriptor = _descriptor("object_detection")

    def detect(self, frames: list[Frame]) -> list[list[Detection]]:
        results: list[list[Detection]] = []
        labels = ("car", "motorcycle", "person", "bus", "truck")
        for frame in frames:
            width, height = frame.source_size
            value = _seed([_frame_bytes(frame)])
            if value % 5 == 0:
                results.append([])
                continue
            label = labels[value % len(labels)]
            x1, y1 = width * 0.15, height * 0.2
            results.append([Detection(box_xyxy=(x1, y1, width * 0.65, height * 0.8), label=label, score=0.8)])
        return results


class FakePrivacyDetector:
    descriptor = _descriptor("privacy_detection")

    def detect(self, frames: list[Frame]) -> list[list[SensitiveDetection]]:
        results: list[list[SensitiveDetection]] = []
        for frame in frames:
            width, height = frame.source_size
            value = _seed([_frame_bytes(frame), b"privacy"])
            if value % 3:
                results.append([])
                continue
            kind = "face" if value % 2 else "plate"
            results.append(
                [
                    SensitiveDetection(
                        box_xyxy=(width * 0.4, height * 0.35, width * 0.6, height * 0.55),
                        label=kind,
                        score=0.9,
                    )
                ]
            )
        return results


class FakeWeatherClassifier:
    descriptor = _descriptor("weather_classification")

    def classify(self, frames: list[Frame]) -> list[ClassProbabilities]:
        output = []
        for frame in frames:
            mean = float(np.asarray(frame.pixels).mean()) / 255.0
            rain = min(0.8, abs(math.sin(frame.timestamp_ms / 1000)) * 0.35)
            clear = max(0.05, mean * (1 - rain))
            cloudy = max(0.05, 1 - clear - rain)
            total = clear + cloudy + rain
            output.append(
                ClassProbabilities(
                    probabilities={"clear": clear / total, "cloudy": cloudy / total, "rain": rain / total}
                )
            )
        return output
