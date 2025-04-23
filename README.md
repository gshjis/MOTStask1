# Optimization with Gradient Descent

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Проект реализует метод градиентного спуска для задачи условной оптимизации с визуализацией.

## 📦 Установка

```bash
git clone https://github.com/gshjis/MOTStask1.git
cd MOTStask1
pip install -r requirements.txt
```

## 📝 Требования
- Python 3.8+
- Установленные зависимости:
  ```bash
  numpy sympy matplotlib pydantic
  ```

## 🚀 Быстрый старт

1. Отредактируйте коэффициенты в `main()`:
```python
# Коэффициенты целевой функции (ax₁² + bx₂² + cx₁x₂ + dx₁ + ex₂)
f_koefs = '3 1 -2 5 -9'.split()  

# Коэффициенты ограничений (a₁x₁ + b₁x₂ + c₁ ≤ 0)
g1_koefs = '-1 4 -12'.split()    
g2_koefs = '13 -8 -64'.split()   
```

2. Запустите программу:
```bash
python main.py
```

## 📊 Визуализация
Программа генерирует два типа графиков:

1. **Допустимая область и траектория**
   ![Feasible Region](images/feasible_region.png)

2. **Линии уровня целевой функции**
   ![Level Lines](images/level_lines.png)

## 🧮 Алгоритмы
| Метод | Описание |
|-------|----------|
| `gradient_descent()` | Реализация градиентного спуска с аналитическим шагом |
| `newton_method()` | Метод Ньютона (опционально) |
| `plot_veasible_region()` | Визуализация ОДР и ограничений |

## 📂 Структура проекта
```
project/
├── main.py                # Основной скрипт
├── optimization.py        # Класс Task1 с алгоритмами
├── requirements.txt       # Зависимости
├── README.md              # Этот файл

```

## 📄 Пример вывода
```text
Шаг 1:
Текущая точка (x_k): [1.00694129652690 5.49101128361580]
Градиент в точке (grad_x_k): [ 0.05963 -0.03186]
Следующая точка через alpha (x_k1_a_): [1.007 - 0.06*a 0.032*a + 5.491]
Градиент через alpha (grad_a_): [0.06 - 0.424*a 0.184*a - 0.032]
Скалярное произведение градиентов: 0.005 - 0.031*a
Оптимальное alpha: 0.147584513216108
Новая точка (x_k1): [0.998141538648834 5.49571333001782]
```

