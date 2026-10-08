"""Композиция сценария из блоков: стабильные участки, дрейфы, возвраты.

Ответственный: Будаев К. В.

Поток описывается последовательностью блоков, как временная шкала:

* :class:`Stable` (длина L) — L объектов текущего концепта;
* :class:`Drift` — событие дрейфа; занимает ℓ объектов перехода (ℓ = 0 у
  внезапного дрейфа);
* :class:`Recur` — возврат к концепту с номером ``to``; тоже занимает ℓ
  объектов.

Позиции событий и длина потока вычисляются автоматически: курсор идёт по
блокам, событие с началом перехода в курсоре получает центр
τ = onset + ⌊ℓ/2⌋. Длина потока n равна сумме длин блоков.

Композиция не обходит ограничения: если стабильный участок короче окна
допуска Δ или первый дрейф попадает в разогрев, сценарий построится, а
нарушения (C4–C6) покажет валидатор. Так пользователь видит, какой блок
нужно удлинить.

Декларативная форма (YAML) — ключ ``blocks``::

    blocks:
      - stable: 6000
      - drift: {kind: real, magnitude: 0.2, form: gradual, width: 1000}
      - stable: 6000
      - recur: {to: 0}
      - stable: 5000
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from scendrift.scenario import io
from scendrift.scenario.schema import ScenarioSpec

__all__ = [
    "Stable",
    "Drift",
    "Recur",
    "Block",
    "compose",
    "expand_blocks",
    "blocks_from_data",
    "block_dicts",
]


@dataclass(frozen=True)
class Stable:
    """Стабильный участок длины ``length``."""

    length: int

    def __post_init__(self) -> None:
        if self.length < 0:
            raise ValueError("длина стабильного участка должна быть ≥ 0")


@dataclass(frozen=True)
class Drift:
    """Событие дрейфа (параметры — как у :class:`~scendrift.scenario.schema.DriftEvent`)."""

    kind: str = "real"
    magnitude: float | None = None
    form: str = "sudden"
    width: int = 0
    affected_share: float = 1.0
    shape: str = "linear"


@dataclass(frozen=True)
class Recur:
    """Возврат к концепту с номером ``to`` (0 — исходный концепт)."""

    to: int
    form: str = "sudden"
    width: int = 0
    shape: str = "linear"


Block = Stable | Drift | Recur


def _events_and_length(blocks: Sequence[Block]) -> tuple[list[dict[str, Any]], int]:
    """События канонической формы и длина потока."""
    cursor, events = 0, []
    for block in blocks:
        if isinstance(block, Stable):
            cursor += block.length
            continue
        width = block.width if block.form != "sudden" else 0
        event: dict[str, Any] = {
            "position": cursor + width // 2,
            "width": width,
            "form": block.form,
            "shape": block.shape,
        }
        if isinstance(block, Drift):
            event.update(kind=block.kind, affected_share=block.affected_share)
            if block.magnitude is not None:
                event["magnitude"] = block.magnitude
        elif isinstance(block, Recur):
            event["returns_to"] = block.to
        else:  # pragma: no cover - защищено типами
            raise TypeError(f"неизвестный блок {block!r}")
        events.append(event)
        cursor += width
    return events, cursor


def compose(
    blocks: Sequence[Block],
    *,
    stream: Mapping[str, Any] | None = None,
    evaluation: Mapping[str, Any] | None = None,
    seed: int = 0,
    name: str = "composed",
) -> ScenarioSpec:
    """Сценарий из последовательности блоков.

    Args:
        blocks: блоки временной шкалы.
        stream: остальные поля базовой конфигурации (n_samples вычисляется).
        evaluation: протокол оценки.
        seed: seed реализации.
        name: имя сценария.

    Raises:
        ValueError: если в ``stream`` задана длина, не равная сумме блоков.
    """
    events, length = _events_and_length(blocks)
    stream = dict(stream or {})
    if "n_samples" in stream and int(stream["n_samples"]) != length:
        raise ValueError(f"n_samples = {stream['n_samples']}, а сумма длин блоков {length}")
    stream["n_samples"] = length
    data: dict[str, Any] = {"name": name, "stream": stream, "events": events, "seed": seed}
    if evaluation:
        data["evaluation"] = dict(evaluation)
    return io.from_dict(data)


def blocks_from_data(items: Sequence[Mapping[str, Any]]) -> list[Block]:
    """Блоки из декларативной формы: ``{stable: L}``, ``{drift: {...}}``, ``{recur: {...}}``.

    Raises:
        ValueError: при неизвестном или неоднозначном блоке.
    """
    out: list[Block] = []
    for i, item in enumerate(items):
        if not isinstance(item, Mapping) or len(item) != 1:
            raise ValueError(f"blocks[{i}]: нужен ровно один ключ stable, drift или recur")
        ((key, value),) = item.items()
        if key == "stable":
            out.append(Stable(int(value)))
        elif key == "drift":
            out.append(Drift(**dict(value or {})))
        elif key == "recur":
            out.append(Recur(**dict(value)) if isinstance(value, Mapping) else Recur(int(value)))
        else:
            raise ValueError(f"blocks[{i}]: неизвестный блок {key!r}")
    return out


def expand_blocks(data: Mapping[str, Any]) -> dict[str, Any]:
    """Заменяет ``blocks`` в описании сценария на ``events`` и длину потока.

    Описание без ``blocks`` возвращается без изменений.
    """
    data = dict(data)
    if "blocks" not in data:
        return data
    events, length = _events_and_length(blocks_from_data(data.pop("blocks")))
    stream = dict(data.get("stream", {}))
    if "n_samples" in stream and int(stream["n_samples"]) != length:
        raise ValueError(f"n_samples = {stream['n_samples']}, а сумма длин блоков {length}")
    stream["n_samples"] = length
    data["stream"] = stream
    data["events"] = events
    return data


def block_dicts(blocks: Sequence[Block]) -> list[dict[str, Any]]:
    """Декларативная форма блоков (обратная к :func:`blocks_from_data`)."""
    out = []
    for block in blocks:
        if isinstance(block, Stable):
            out.append({"stable": block.length})
        else:
            key = "drift" if isinstance(block, Drift) else "recur"
            out.append({key: asdict(block)})
    return out
