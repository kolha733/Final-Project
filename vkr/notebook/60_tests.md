## 6. Тесты и воспроизводимость [К+Т]

Корректность проверяется на трёх уровнях:

1. **Assert-ячейки** в ноутбуке:
   - численная проверка утверждений (п. 2.3);
   - точность калибровки (п. 4.1.10);
   - согласованность текста и кода (п. 4.1.12);
   - round-trip сериализации (п. 4.6).
2. **Модульные тесты pytest** проверяют:
   - $\Phi_2$ — против `scipy.stats.multivariate_normal`;
   - формулы величины — против Монте-Карло;
   - каждое ограничение C1–C13 — на сценариях, которые его нарушают;
   - исправление сценариев;
   - сериализацию;
   - «золотой» идентификатор: он ловит случайное изменение канонической формы.
3. **Воспроизводимость:**
   - версии библиотек зафиксированы в `requirements.txt`;
   - глобальный seed задан в п. 0.5;
   - генерация детерминирована по паре (структурный seed, seed реализации);
   - идентификатор сценария строится по содержимому.

Ячейки ниже создают файлы тестов и запускают pytest.
```writefile requirements.txt
```
```writefile tests/conftest.py
```
```writefile tests/test_semantics.py
```
```writefile tests/test_schema.py
```
```writefile tests/test_validation.py
```
```writefile tests/test_io.py
```
```writefile tests/test_space.py
```
```python
result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                        capture_output=True, text=True)
print(result.stdout[-1500:])
assert result.returncode == 0, result.stderr[-3000:]
```
