"""Данные для презентации: числа и ряды диаграмм из результатов экспериментов.

Ответственные: Будаев К. В., Гарифзянов Т. Р.

Числа берутся из того же модуля, что и в пояснительной записке
(``docs/zapiska/facts.py``), поэтому презентация и записка не расходятся.
Ряды диаграмм читаются из ``results/*.csv``. Скрипт печатает JSON в stdout;
его вызывает ``build_pptx.js``.

Запуск: ``python docs/presentation/slide_data.py`` (из каталога ``vkr``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _facts_module():
    path = ROOT / "docs" / "zapiska" / "facts.py"
    spec = importlib.util.spec_from_file_location("zapiska_facts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def series() -> dict[str, object]:
    """Ряды для диаграмм; детекторы river упорядочены по среднему F1 на наборе Э2."""
    fm = _facts_module()
    e2 = fm.read("e2_runs")
    f1 = e2.groupby("detector")["f1"].mean()
    order = list(f1[fm.RIVER].sort_values(ascending=False).index)
    acc = fm.read("e8_accuracy").set_index("detector")["delta_accuracy:mean"]
    blind = fm.read("e3_virtual_blindness").set_index("detector")
    manual = fm.read("e6_f1").set_index("detector")
    periodic = "Periodic(2000)"

    def r(x: float, nd: int = 3) -> float:
        return round(float(x), nd)

    return {
        "order": order,
        "f1": [r(f1[d]) for d in order],
        "f1_periodic": r(f1[periodic]),
        "dacc": [r(acc[d], 4) for d in order],
        "dacc_periodic": r(acc[periodic], 4),
        "excess_real": [r(blind.loc[d, "превышение real"]) for d in order],
        "excess_virtual": [r(blind.loc[d, "превышение virtual"]) for d in order],
        "f1_manual": [r(manual.loc[d, "F1 (ручной)"]) for d in order],
        "f1_auto": [r(manual.loc[d, "F1 (Э2)"]) for d in order],
    }


def main() -> int:
    """Печатает JSON с числами (``facts``) и рядами диаграмм (``series``)."""
    fm = _facts_module()
    data = {"facts": fm.collect(), "series": series()}
    json.dump(data, sys.stdout, ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
