import numpy as np
import matplotlib.pyplot as plt

# Создаем сетку точек
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Задаем функцию (пример: f(x,y) = x^2 + y^2)
Z = X**2 + Y**2

# Рисуем линии уровня
plt.axis('equal')  # Самый простой способ
plt.contour(X, Y, Z, levels=[2])  # Конкретные значения уровней
plt.xlabel('x')
plt.ylabel('y')
plt.title('Линии уровня функции f(x,y) = x² + y²')
plt.grid(True)
plt.savefig(f'lines/level{2}.png')