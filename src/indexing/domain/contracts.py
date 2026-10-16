from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.indexing.domain.entities import (
    ClassProbabilities,
    Detection,
    Frame,
    ModelDescriptor,
    SensitiveDetection,
)


@runtime_checkable
class ObjectDetector(Protocol):
    @property
    def descriptor(self) -> ModelDescriptor: ...
    def detect(self, frames: list[Frame]) -> list[list[Detection]]: ...


@runtime_checkable
class PrivacyDetector(Protocol):
    @property
    def descriptor(self) -> ModelDescriptor: ...
    def detect(self, frames: list[Frame]) -> list[list[SensitiveDetection]]: ...


@runtime_checkable
class WeatherClassifier(Protocol):
    @property
    def descriptor(self) -> ModelDescriptor: ...
    def classify(self, frames: list[Frame]) -> list[ClassProbabilities]: ...
