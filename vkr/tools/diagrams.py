"""Схемы архитектуры фреймворка: компоненты, поток данных, диаграмма классов.

Ответственные: Будаев К. В., Гарифзянов Т. Р.

Схемы рисуются средствами matplotlib, поэтому ноутбук и записка не зависят
от внешних программ вроде graphviz. Цвет блока обозначает ответственного
исполнителя: синий — Будаев К. В., зелёный — Гарифзянов Т. Р., серый —
совместная зона или внешние библиотеки.

Запуск: ``python tools/diagrams.py [каталог]`` (по умолчанию docs/figures).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

COLORS = {
    "K": ("#dbe7f5", "#2f5d8a"),  # Будаев К. В.
    "T": ("#dcefe2", "#2e7d4f"),  # Гарифзянов Т. Р.
    "J": ("#eeeeee", "#6b6b6b"),  # совместно
    "X": ("#ffffff", "#9a9a9a"),  # внешние библиотеки
}
TEXT = "#1f1f1f"


def _box(ax, x, y, w, h, title, lines=(), who="J", title_size=11, body_size=9):
    face, edge = COLORS[who]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.4,
            edgecolor=edge,
            facecolor=face,
            linestyle="--" if who == "X" else "-",
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.012,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=edge,
    )
    for i, line in enumerate(lines):
        ax.text(
            x + 0.012,
            y + h - 0.045 - i * 0.03,
            line,
            ha="left",
            va="top",
            fontsize=body_size,
            color=TEXT,
        )


def _arrow(ax, start, end, text="", rad=0.0, color="#444444", style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=13,
            linewidth=1.2,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if text:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        horizontal = abs(end[1] - start[1]) < 1e-9
        ax.text(
            mx if horizontal else mx + 0.008,
            my + (0.01 if horizontal else 0.0),
            text,
            fontsize=8,
            color=color,
            ha="center" if horizontal else "left",
            va="bottom" if horizontal else "center",
            style="italic",
        )


def _legend(ax, y=0.02):
    items = [
        ("K", "Будаев К. В."),
        ("T", "Гарифзянов Т. Р."),
        ("J", "совместно"),
        ("X", "внешние библиотеки"),
    ]
    x = 0.02
    for who, label in items:
        face, edge = COLORS[who]
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                0.02,
                0.022,
                boxstyle="round,pad=0.002",
                facecolor=face,
                edgecolor=edge,
                linestyle="--" if who == "X" else "-",
            )
        )
        ax.text(x + 0.027, y + 0.011, label, va="center", fontsize=9, color=TEXT)
        x += 0.17


def _save(fig, ax, path: Path) -> None:
    """Схема для ноутбука и копия без заголовка (``zapiska/``) для записки и презентации."""
    path = Path(path)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    clean = path.parent / "zapiska" / path.name
    clean.parent.mkdir(parents=True, exist_ok=True)
    # заголовок задан с loc="left": это отдельный объект, а не ax.title
    titles = [ax.title, ax._left_title, ax._right_title]  # noqa: SLF001
    for title in titles:
        title.set_visible(False)
    fig.savefig(clean, dpi=180, bbox_inches="tight", facecolor="white")
    for title in titles:
        title.set_visible(True)


def _canvas(title: str, size=(15, 9.5)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT, loc="left", pad=8)
    return fig, ax


def component_diagram(path: Path) -> Path:
    """Компонентная схема фреймворка scendrift."""
    fig, ax = _canvas("Рис. А. Компоненты фреймворка scendrift и зоны ответственности")
    xs, w = (0.02, 0.35, 0.68), 0.30
    _box(
        ax,
        0.30,
        0.86,
        0.40,
        0.09,
        "Вход: описание сценария",
        ["YAML / JSON (JSON Schema) или Python API", "компактная форма: расписание + defaults"],
        who="J",
    )
    _box(
        ax,
        xs[0],
        0.55,
        w,
        0.27,
        "scenario — формальная модель",
        [
            "schema: ScenarioSpec = ⟨B, E, Ω, s⟩",
            "semantics: цепочка концептов,",
            "   калибровка величины m (TV)",
            "validation: C1–C14, W1–W4, repair",
            "io: канонический JSON, id, схема",
            "space, taxonomy",
        ],
        who="K",
    )
    _box(
        ax,
        xs[1],
        0.55,
        w,
        0.27,
        "formation — формирование",
        [
            "samplers: grid, random, LHS, Sobol",
            "catalog: easy / medium / hard",
            "наследование extends + overrides",
            "композиция из блоков",
            "шкала сложности λ ∈ [0, 1]",
            "манифест набора (версия, хэши)",
        ],
        who="K",
    )
    _box(
        ax,
        xs[2],
        0.55,
        w,
        0.27,
        "generators — генерация",
        [
            "hyperplane_gauss (калиброванное)",
            "river:* — классические потоки",
            "real:* — внедрение в реальные данные",
            "переходы: linear / sigmoid",
            "ground truth: интервалы, величины",
            "StreamData(X, y, GT)",
        ],
        who="K",
    )
    _box(
        ax,
        xs[0],
        0.21,
        w,
        0.27,
        "detectors — детекторы",
        [
            "единый интерфейс DriftDetector",
            "ADWIN, KSWIN, PageHinkley (оба / up),",
            "DDM, EDDM, HDDM_A, HDDM_W, FHDDM",
            "базовые линии: NoDrift,",
            "Periodic, Oracle",
            "реестр + гиперпараметры",
        ],
        who="T",
    )
    _box(
        ax,
        xs[1],
        0.21,
        w,
        0.27,
        "evaluation — оценка",
        [
            "protocol: Ω = (h, a, W, Δ, r)",
            "runner: prequential, seeds,",
            "   joblib, кэш по id",
            "metrics: delay, MTD, MDR, MTFA,",
            "   MTR, P/R/F1, Δaccuracy",
            "stats: Friedman, Nemenyi, Wilcoxon",
        ],
        who="T",
    )
    _box(
        ax,
        xs[2],
        0.21,
        w,
        0.27,
        "reporting — отчёты",
        [
            "сводные таблицы и ранги",
            "тепловые карты, CD-диаграммы",
            "области отказа (дерево решений)",
            "экспорт HTML / CSV / Markdown",
        ],
        who="T",
    )
    _box(
        ax,
        0.02,
        0.06,
        0.96,
        0.11,
        "Внешние библиотеки (не являются вкладом авторов)",
        [
            "river 0.26: детекторы, GaussianNB, генераторы SEA/Agrawal/…   ·   "
            "scipy: qmc (LHS/Sobol), stats, special.owens_t",
            "pydantic   ·   PyYAML   ·   joblib   ·   numpy / pandas / matplotlib   ·   "
            "scikit-learn",
        ],
        who="X",
    )
    _arrow(ax, (0.36, 0.86), (0.17, 0.825))
    _arrow(ax, (0.50, 0.86), (0.50, 0.825))
    _arrow(ax, (xs[0] + w, 0.685), (xs[1], 0.685))
    _arrow(ax, (xs[1] + w, 0.685), (xs[2], 0.685))
    _arrow(ax, (0.83, 0.55), (0.60, 0.485), "StreamData", rad=-0.15)
    _arrow(ax, (0.17, 0.55), (0.40, 0.485), "ScenarioSpec, id", rad=0.15)
    _arrow(ax, (xs[0] + w, 0.345), (xs[1], 0.345))
    _arrow(ax, (xs[1] + w, 0.345), (xs[2], 0.345))
    _legend(ax, y=0.005)
    _save(fig, ax, path)
    plt.close(fig)
    return path


def dataflow_diagram(path: Path) -> Path:
    """Конвейер: от пространства параметров до отчёта (змейкой в две строки)."""
    fig, ax = _canvas(
        "Рис. Б. Конвейер автоматического формирования и исполнения бенчмарка", size=(15, 7.5)
    )
    row1: Sequence[tuple[str, list[str], str]] = [
        ("1. Пространство Ξ", ["домены параметров,", "условия активности"], "K"),
        ("2. План эксперимента", ["grid / random /", "LHS / Sobol / шкала λ"], "K"),
        ("3. Сборка S(ξ)", ["точка ξ → сценарий", "(шаблоны, композиция)"], "K"),
        ("4. Проверка", ["ограничения C1–C14:", "отклонить / исправить"], "K"),
        ("5. Набор сценариев", ["ScenarioSpec + id,", "манифест YAML"], "K"),
    ]
    row2: Sequence[tuple[str, list[str], str]] = [
        ("6. Генерация Γ(S, s)", ["StreamData:", "X, y, ground truth"], "K"),
        ("7. Раннер", ["prequential,", "seeds × детекторы"], "T"),
        ("8. Метрики", ["сопоставление", "срабатываний по окну Δ"], "T"),
        ("9. Анализ", ["ранги, тесты Фридмана", "и Неменьи, области отказа"], "T"),
        ("10. Отчёт", ["HTML / CSV / Markdown,", "графики"], "T"),
    ]
    w, h = 0.17, 0.20
    gap = (0.96 - 5 * w) / 4
    y1, y2 = 0.60, 0.22
    for i, (title, lines, who) in enumerate(row1):
        x = 0.02 + i * (w + gap)
        _box(ax, x, y1, w, h, title, lines, who=who, title_size=10.5, body_size=9)
        if i < 4:
            _arrow(ax, (x + w, y1 + h / 2), (x + w + gap, y1 + h / 2))
    for i, (title, lines, who) in enumerate(row2):
        x = 0.02 + (4 - i) * (w + gap)
        _box(ax, x, y2, w, h, title, lines, who=who, title_size=10.5, body_size=9)
        if i < 4:
            _arrow(ax, (x, y2 + h / 2), (x - gap, y2 + h / 2))
    x_last = 0.02 + 4 * (w + gap) + w / 2
    _arrow(ax, (x_last, y1), (x_last, y2 + h))
    ax.text(
        0.02,
        0.86,
        "Формирование бенчмарка (сценарии и данные)",
        fontsize=11,
        color=COLORS["K"][1],
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.47,
        "Исполнение и анализ (детекторы и метрики)",
        fontsize=11,
        color=COLORS["T"][1],
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.10,
        "Идентификатор сценария — SHA-256 канонического JSON ⟨B, E, Ω, s⟩: по нему "
        "кэшируются результаты.\nSeed реализации s (выборка X, y, шум) отделён "
        "от структурного seed c (геометрия концептов, расписание).",
        fontsize=9.5,
        color=TEXT,
        va="center",
    )
    _legend(ax, y=-0.03)
    _save(fig, ax, path)
    plt.close(fig)
    return path


def class_diagram(path: Path) -> Path:
    """Упрощённая UML-диаграмма основных классов и интерфейсов."""
    fig, ax = _canvas(
        "Рис. В. Основные классы и интерфейсы (упрощённая UML-диаграмма)", size=(15, 10)
    )
    _box(
        ax,
        0.36,
        0.72,
        0.28,
        0.22,
        "ScenarioSpec",
        [
            "schema_version, name, tags",
            "stream: StreamSpec",
            "events: tuple[DriftEvent]",
            "evaluation: EvaluationSpec",
            "seed: int",
            "with_seed(s)",
        ],
        who="K",
    )
    _box(
        ax,
        0.02,
        0.76,
        0.27,
        0.18,
        "StreamSpec  (B)",
        [
            "family, n_samples, n_features",
            "minority_share π, label_noise η",
            "concept_seed c, family_params",
        ],
        who="K",
    )
    _box(
        ax,
        0.71,
        0.74,
        0.27,
        0.20,
        "DriftEvent  (e_k)",
        [
            "position τ, width w, form φ",
            "kind κ, magnitude m",
            "affected_share α",
            "returns_to ρ, shape",
            "onset / end / is_recurring",
        ],
        who="K",
    )
    _box(
        ax,
        0.71,
        0.50,
        0.27,
        0.17,
        "EvaluationSpec  (Ω)",
        ["base_model h, on_detection a", "warmup W, acceptance_window Δ", "delay_reference r"],
        who="T",
    )
    _box(
        ax,
        0.02,
        0.48,
        0.27,
        0.22,
        "ConceptChain",
        [
            "states: list[ConceptState]",
            "  (w, μ, π, π₀, concept_id)",
            "geometry: list[EventGeometry]",
            "  (A, u/v, φ, θ, δ, realized)",
            "violations: list[Violation]",
        ],
        who="K",
    )
    _box(
        ax,
        0.36,
        0.45,
        0.28,
        0.20,
        "ValidationReport",
        [
            "violations: list[Violation]",
            "  (code, path, level, suggestion)",
            "ok, errors, warnings",
            "check(S) · repair(S) · ensure_valid(S)",
        ],
        who="K",
    )
    _box(
        ax,
        0.02,
        0.17,
        0.27,
        0.23,
        "«protocol» StreamGenerator",
        [
            "family: str",
            "generate(S, seed) → StreamData",
            "",
            "StreamData: X, y, y_clean,",
            "concept, progress, ground_truth",
        ],
        who="K",
    )
    _box(
        ax,
        0.36,
        0.17,
        0.28,
        0.21,
        "GroundTruth",
        [
            "intervals: tuple[DriftInterval]",
            "  (onset, center, end, form,",
            "   kind, magnitude, realized)",
            "drift_mask(), onsets, to_records()",
        ],
        who="K",
    )
    _box(
        ax,
        0.71,
        0.17,
        0.27,
        0.27,
        "«protocol» DriftDetector / Metric",
        [
            "DriftDetector:",
            "  update(x) → bool, reset(), clone()",
            "Metric:",
            "  (detections, GT, Ω) → float",
            "DetectionRun:",
            "  detections, accuracy, runtime",
        ],
        who="T",
    )
    _box(
        ax,
        0.36,
        0.02,
        0.28,
        0.10,
        "«protocol» ScenarioSampler",
        ["design(Ξ, n, seed) → U ⊂ [0, 1)ᴰ", "sample(Ξ, n, seed) → list[ξ]"],
        who="K",
    )
    _arrow(ax, (0.36, 0.88), (0.29, 0.88), "1", style="-|>")
    _arrow(ax, (0.64, 0.88), (0.71, 0.88), "0..K", style="-|>")
    _arrow(ax, (0.64, 0.75), (0.71, 0.62), "1", style="-|>")
    _arrow(ax, (0.36, 0.745), (0.16, 0.705), "build_chain(S)", rad=0.08)
    _arrow(ax, (0.50, 0.72), (0.50, 0.65), "check(S)")
    _arrow(ax, (0.15, 0.48), (0.15, 0.40), "используется Γ")
    _arrow(ax, (0.29, 0.28), (0.36, 0.28), "1")
    _arrow(ax, (0.64, 0.28), (0.71, 0.28), "вход")
    _legend(ax, y=-0.045)
    _save(fig, ax, path)
    plt.close(fig)
    return path


def make_all(out_dir: str | Path = "docs/figures") -> list[Path]:
    """Строит все схемы и возвращает пути к PNG."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [
        component_diagram(out / "arch_components.png"),
        dataflow_diagram(out / "arch_dataflow.png"),
        class_diagram(out / "arch_classes.png"),
    ]


if __name__ == "__main__":
    for p in make_all(sys.argv[1] if len(sys.argv) > 1 else "docs/figures"):
        print(p)
