from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from .metric_define import MetricDefine


@dataclass(frozen=True)
class EstimateCatalog:
    """Read-only view over configured estimates with lookup helpers."""

    entries: tuple[tuple[str, str], ...]
    _key_to_label: Mapping[str, str] = field(init=False, repr=False)
    _label_to_key: Mapping[str, str] = field(init=False, repr=False)
    _label_order: Mapping[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        key_to_label = dict(self.entries)
        label_to_key = {label: key for key, label in self.entries}
        label_order = {label: idx for idx, label in enumerate(label_to_key)}
        object.__setattr__(self, "_key_to_label", MappingProxyType(key_to_label))
        object.__setattr__(self, "_label_to_key", MappingProxyType(label_to_key))
        object.__setattr__(self, "_label_order", MappingProxyType(label_order))

    @classmethod
    def build(cls, mapping: Mapping[str, str]) -> "EstimateCatalog":
        return cls(tuple(mapping.items()))

    @property
    def key_to_label(self) -> Mapping[str, str]:
        return self._key_to_label

    @property
    def label_to_key(self) -> Mapping[str, str]:
        return self._label_to_key

    @property
    def label_order(self) -> Mapping[str, int]:
        return self._label_order

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(key for key, _ in self.entries)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(label for _, label in self.entries)

    def label_for(self, key: str) -> str:
        return self._key_to_label.get(key, key)


@dataclass(frozen=True)
class MetricCatalog:
    """Read-only view over configured metrics with ordering helpers."""

    metrics: tuple[MetricDefine, ...]
    _labels: tuple[str, ...] = field(init=False, repr=False)
    _names: tuple[str, ...] = field(init=False, repr=False)
    _name_order: Mapping[str, int] = field(init=False, repr=False)
    _label_order: Mapping[str, int] = field(init=False, repr=False)
    _name_set: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        labels = tuple(metric.label or metric.name for metric in self.metrics)
        names = tuple(metric.name for metric in self.metrics)
        name_order = {name: idx for idx, name in enumerate(names)}
        label_order = {label: idx for idx, label in enumerate(labels)}
        object.__setattr__(self, "_labels", labels)
        object.__setattr__(self, "_names", names)
        object.__setattr__(self, "_name_order", MappingProxyType(name_order))
        object.__setattr__(self, "_label_order", MappingProxyType(label_order))
        object.__setattr__(self, "_name_set", frozenset(names))

    @property
    def entries(self) -> tuple[MetricDefine, ...]:
        return self.metrics

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def name_order(self) -> Mapping[str, int]:
        return self._name_order

    @property
    def label_order(self) -> Mapping[str, int]:
        return self._label_order

    def contains(self, name: str) -> bool:
        return name in self._name_set


@dataclass(frozen=True)
class FormatterContext:
    """Lightweight bundle of formatting-related configuration."""

    group_by: Mapping[str, str]
    subgroup_by: Mapping[str, str]
    estimate_catalog: EstimateCatalog
    metric_catalog: MetricCatalog
    subgroup_categories: Sequence[str] = ()
