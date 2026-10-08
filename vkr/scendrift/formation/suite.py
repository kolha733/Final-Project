"""Наборы сценариев: формирование по конфигурации, манифест, целостность, версии.

Ответственный: Будаев К. В.

Набор (suite) — упорядоченная совокупность сценариев с манифестом. Весь
путь от описания к набору детерминирован и задаётся одной конфигурацией
:class:`SuiteConfig`:

    конфигурация → пространство Ξ → план U → точки ξ → сценарии S(ξ)
                 → учёт ограничений → удаление повторов → набор + манифест.

Версионирование:

* ``schema_version`` — версия схемы сценария (несовместимое изменение полей);
* ``framework_version`` — версия пакета, которым сформирован набор;
* ``version`` — версия набора, которую назначают авторы (семантическая);
* ``suite_id`` — хэш содержимого: SHA-256 от отсортированного списка
  идентификаторов сценариев. Одинаковый ``suite_id`` означает одинаковые
  сценарии независимо от имени и версии; по нему связываются результаты.

При загрузке набор проверяется: идентификатор каждого сценария
пересчитывается по файлу и сравнивается с манифестом. Поэтому изменение
любого файла сценария вручную обнаруживается.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

import scendrift
from scendrift.formation.mapping import POLICIES, realize, summarize
from scendrift.formation.samplers import make_sampler
from scendrift.scenario import io
from scendrift.scenario.schema import SCHEMA_VERSION, ScenarioSpec
from scendrift.scenario.space import (
    DEFAULT_SPACE,
    Categorical,
    FloatDomain,
    IntDomain,
    ParameterDef,
    ParameterSpace,
    space_id,
)
from scendrift.scenario.taxonomy import signature

__all__ = [
    "DomainOverride",
    "DesignSpec",
    "SuiteConfig",
    "Suite",
    "SuiteIntegrityError",
    "build_space",
    "build_suite",
    "load_suite",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DomainOverride(_Frozen):
    """Новый домен параметра: отрезок [low, high] или список значений."""

    low: float | None = None
    high: float | None = None
    log: bool | None = None
    values: tuple[Any, ...] | None = None


class DesignSpec(_Frozen):
    """План эксперимента: метод, бюджет, seed."""

    method: Literal["grid", "random", "lhs", "sobol"] = "sobol"
    n: int = Field(64, ge=2, le=100_000, description="Бюджет: число точек плана.")
    seed: int = Field(0, ge=0)
    lhs_optimization: Literal["random-cd"] | None = None


class SuiteConfig(_Frozen):
    """Декларативное описание набора сценариев.

    Attributes:
        name: имя набора.
        version: версия набора (семантическая: MAJOR.MINOR.PATCH).
        description: назначение набора.
        template: шаблон сценария в компактной форме (может содержать ``extends``).
        vary: варьируемые параметры (по умолчанию — все свободные параметры
            пространства по умолчанию). Остальные берутся из шаблона.
        fix: фиксированные значения варьируемых параметров.
        domains: новые домены параметров.
        design: план эксперимента.
        constraints: способ учёта ограничений: reject, repair или adapt.
    """

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    version: str = Field("1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    description: str = ""
    template: dict[str, Any] = Field(default_factory=dict)
    vary: tuple[str, ...] | None = None
    fix: dict[str, Any] = Field(default_factory=dict)
    domains: dict[str, DomainOverride] = Field(default_factory=dict)
    design: DesignSpec = Field(default_factory=DesignSpec)
    constraints: Literal["reject", "repair", "adapt"] = "adapt"


def _override(p: ParameterDef, o: DomainOverride) -> ParameterDef:
    if o.values is not None:
        domain: Any = Categorical(tuple(o.values))
    else:
        d = p.domain
        if isinstance(d, Categorical):
            raise ValueError(f"{p.name}: для категориального параметра задайте values")
        low = d.low if o.low is None else o.low
        high = d.high if o.high is None else o.high
        log = d.log if o.log is None else o.log
        domain = (
            IntDomain(int(low), int(high), log)
            if isinstance(d, IntDomain)
            else (FloatDomain(float(low), float(high), log))
        )
    default = p.default if domain.contains(p.default) else domain.from_unit(0.5)
    return replace(p, domain=domain, default=default)


def build_space(config: SuiteConfig, base: ParameterSpace = DEFAULT_SPACE) -> ParameterSpace:
    """Пространство параметров набора: выбор, новые домены, фиксация.

    Условия активности, ссылающиеся на невыбранные параметры, снимаются:
    значение такого параметра берётся из шаблона и в плане не участвует.
    """
    names = list(config.vary) if config.vary is not None else base.free_names()
    unknown = [n for n in [*names, *config.fix, *config.domains] if n not in base]
    if unknown:
        raise KeyError(f"неизвестные параметры: {', '.join(unknown)}")
    keep = set(names) | set(config.fix)
    params = []
    for p in base:
        if p.name not in keep:
            continue
        p = replace(p, active_if=tuple(c for c in p.active_if if c[0] in keep))
        if p.name in config.domains:
            p = _override(p, config.domains[p.name])
        params.append(p)
    space = ParameterSpace(params)
    return space.fix(**config.fix) if config.fix else space


class SuiteIntegrityError(ValueError):
    """Содержимое набора не совпадает с манифестом."""


def _suite_id(ids: Sequence[str]) -> str:
    text = json.dumps(sorted(ids), separators=(",", ":"))
    return "ste-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Suite:
    """Набор сценариев с описанием происхождения.

    Attributes:
        name: имя набора.
        version: версия набора.
        scenarios: сценарии (без повторов).
        records: сведения о каждом сценарии: номер точки плана, статус и т. п.
        formation: сводка формирования (:func:`~scendrift.formation.mapping.summarize`).
        config: конфигурация, по которой набор сформирован (если есть).
        space_id: идентификатор пространства параметров.
        framework_version: версия пакета при формировании.
    """

    name: str
    version: str
    scenarios: tuple[ScenarioSpec, ...]
    records: tuple[dict[str, Any], ...] = ()
    formation: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] | None = None
    space_id: str | None = None
    framework_version: str = scendrift.__version__

    def __len__(self) -> int:
        return len(self.scenarios)

    @property
    def ids(self) -> list[str]:
        """Идентификаторы сценариев в порядке набора."""
        return [io.scenario_id(s) for s in self.scenarios]

    @property
    def suite_id(self) -> str:
        """Идентификатор набора по содержимому."""
        return _suite_id(self.ids)

    def table(self) -> list[dict[str, Any]]:
        """Строки для таблицы: параметры и класс таксономии каждого сценария."""
        rows = []
        for spec, rec in zip(self.scenarios, self.records or [{}] * len(self), strict=True):
            regular = [e for e in spec.events if e.returns_to is None]
            first = regular[0] if regular else None
            rows.append(
                {
                    "id": io.scenario_id(spec),
                    "name": spec.name,
                    "status": rec.get("status", ""),
                    "n": spec.stream.n_samples,
                    "d": spec.stream.n_features,
                    "π₀": spec.stream.minority_share,
                    "η": spec.stream.label_noise,
                    "K": spec.n_events,
                    "kind": first.kind.value if first and first.kind else None,
                    "form": spec.events[0].form.value if spec.events else None,
                    "ℓ": spec.events[0].width if spec.events else None,
                    "m": first.magnitude if first else None,
                    "α": first.affected_share if first else None,
                    "recurring": any(e.returns_to is not None for e in spec.events),
                    "signature": signature(spec),
                }
            )
        return rows

    def manifest(self) -> dict[str, Any]:
        """Манифест набора (то, что сохраняется в manifest.yaml)."""
        entries = []
        for spec, rec in zip(self.scenarios, self.records or [{}] * len(self), strict=True):
            entries.append(
                {
                    "id": io.scenario_id(spec),
                    "stream_id": io.stream_id(spec),
                    "name": spec.name,
                    "file": f"scenarios/{spec.name}.yaml",
                    "signature": signature(spec),
                    **{k: v for k, v in rec.items() if k in ("index", "status")},
                }
            )
        return {
            "suite": self.name,
            "version": self.version,
            "suite_id": self.suite_id,
            "schema_version": SCHEMA_VERSION,
            "framework_version": self.framework_version,
            "space_id": self.space_id,
            "size": len(self),
            "config": self.config,
            "formation": self.formation,
            "scenarios": entries,
        }

    def save(self, directory: str | Path) -> Path:
        """Сохраняет набор: ``manifest.yaml`` и ``scenarios/<name>.yaml``.

        Returns:
            Путь к манифесту.
        """
        directory = Path(directory)
        (directory / "scenarios").mkdir(parents=True, exist_ok=True)
        names = [s.name for s in self.scenarios]
        if len(set(names)) != len(names):
            raise ValueError("имена сценариев в наборе должны быть уникальны")
        for spec in self.scenarios:
            io.save(spec, directory / "scenarios" / f"{spec.name}.yaml")
        path = directory / "manifest.yaml"
        text = yaml.safe_dump(self.manifest(), allow_unicode=True, sort_keys=False)
        path.write_text(f"# suite_id: {self.suite_id}\n{text}", encoding="utf-8")
        return path

    def diff(self, other: Suite) -> dict[str, Any]:
        """Сравнение с другой версией: добавленные, удалённые и общие сценарии."""
        a, b = set(self.ids), set(other.ids)
        return {"added": sorted(b - a), "removed": sorted(a - b), "common": len(a & b)}


def load_suite(directory: str | Path) -> Suite:
    """Загружает набор и проверяет его целостность.

    Raises:
        SuiteIntegrityError: если файл сценария изменён (идентификатор не
            совпадает с манифестом), отсутствует, или не совпадает ``suite_id``.
        ValueError: при несовместимой версии схемы.
    """
    directory = Path(directory)
    manifest = yaml.safe_load((directory / "manifest.yaml").read_text(encoding="utf-8"))
    major = str(manifest["schema_version"]).split(".")[0]
    if major != SCHEMA_VERSION.split(".")[0]:
        raise ValueError(
            f"набор сформирован для схемы {manifest['schema_version']}, текущая {SCHEMA_VERSION}"
        )
    scenarios, records, problems = [], [], []
    for entry in manifest["scenarios"]:
        path = directory / entry["file"]
        if not path.exists():
            problems.append(f"нет файла {entry['file']}")
            continue
        spec = io.load(path)
        actual = io.scenario_id(spec)
        if actual != entry["id"]:
            problems.append(f"{entry['file']}: id {actual}, в манифесте {entry['id']}")
        scenarios.append(spec)
        records.append({k: entry[k] for k in ("index", "status") if k in entry})
    if problems:
        raise SuiteIntegrityError("набор не совпадает с манифестом:\n" + "\n".join(problems))
    suite = Suite(
        name=manifest["suite"],
        version=manifest["version"],
        scenarios=tuple(scenarios),
        records=tuple(records),
        formation=manifest.get("formation") or {},
        config=manifest.get("config"),
        space_id=manifest.get("space_id"),
        framework_version=manifest.get("framework_version", ""),
    )
    if suite.suite_id != manifest["suite_id"]:
        raise SuiteIntegrityError(
            f"suite_id {suite.suite_id} не совпадает с манифестом {manifest['suite_id']}"
        )
    return suite


def build_suite(config: SuiteConfig | Mapping[str, Any]) -> Suite:
    """Формирует набор сценариев по конфигурации (детерминированно).

    Повторяющиеся сценарии (одинаковый идентификатор, например узлы сетки,
    различающиеся только неактивными параметрами) удаляются; их число
    попадает в сводку формирования.
    """
    if not isinstance(config, SuiteConfig):
        config = SuiteConfig.model_validate(dict(config))
    if config.constraints not in POLICIES:  # pragma: no cover - защищено pydantic
        raise ValueError(config.constraints)
    space = build_space(config)
    options = (
        {"optimization": config.design.lhs_optimization} if config.design.method == "lhs" else {}
    )
    sampler = make_sampler(config.design.method, **options)
    U = sampler.design(space, config.design.n, config.design.seed)
    candidates = realize(
        space,
        U,
        template=config.template,
        policy=config.constraints,
        seed=config.design.seed,
        prefix=config.name,
    )
    scenarios, records, seen = [], [], set()
    for c in candidates:
        if c.spec is None:
            continue
        sid = io.scenario_id(c.spec)
        if sid in seen:
            continue
        seen.add(sid)
        scenarios.append(c.spec)
        records.append({"index": c.index, "status": c.status})
    formation = summarize(candidates)
    formation["duplicates"] = formation["accepted"] - len(scenarios)
    formation["design_points"] = len(U)
    return Suite(
        name=config.name,
        version=config.version,
        scenarios=tuple(scenarios),
        records=tuple(records),
        formation=formation,
        config=config.model_dump(mode="json"),
        space_id=space_id(space),
    )
