import numpy as np
import sympy as sm
from pydantic import BaseModel
from typing import List
import matplotlib.pyplot as plt


class Solution(BaseModel):
    x_k: np.array = None                   # Начальная точка
    grad_x_k: np.array = None              # Градиент в начальной точке
    grad_a_: np.array = None               # Градиент выраженный через alpha
    x_k1_a_: np.array = None               # Следующая точка выраженная через alpha
    alpha: float = None                      # alpha значение
    gradxT_dot_gradx: np.array = None        # Скалярное произведение векторов градиентов текущей и следующей точек
    x_k1: np.array =None                     # Конечная точка

    class Config:
        arbitrary_types_allowed = True

class Task1:
    RESOLUTION = 1000
    
    def __init__(self, function: str, g1: str, g2: str, x1: str, x2: str):
        self.x1, self.x2 = sm.symbols(f'{x1} {x2}')
        self.function = sm.sympify(function)
        self.g1 = sm.sympify(g1)
        self.g2 = sm.sympify(g2)
        self.gradient = np.array([sm.diff(self.function, var) for var in (self.x1, self.x2)])
        self.figure = plt.figure(figsize=(10, 8))
        self.hessian = np.array(sm.hessian(self.function, [self.x1, self.x2]),dtype=float)


    def axis_intersections(self):
        """Получение координат пересечения ограничениями g1 g2 осей координат"""
        # Пересечение g1 с осями
        g1_x1_intersect = sm.solve(self.g1.subs(self.x2, 0), self.x1)
        g1_x2_intersect = sm.solve(self.g1.subs(self.x1, 0), self.x2)
        
        # Пересечение g2 с осями
        g2_x1_intersect = sm.solve(self.g2.subs(self.x2, 0), self.x1)
        g2_x2_intersect = sm.solve(self.g2.subs(self.x1, 0), self.x2)
        
        # Собираем все точки пересечения
        intersections = []
        if g1_x1_intersect:
            intersections.append((float(g1_x1_intersect[0]), 0))
        if g1_x2_intersect:
            intersections.append((0, float(g1_x2_intersect[0])))
        if g2_x1_intersect:
            intersections.append((float(g2_x1_intersect[0]), 0))
        if g2_x2_intersect:
            intersections.append((0, float(g2_x2_intersect[0])))

        self.delta_x = np.abs(g1_x1_intersect[0] - g2_x1_intersect[0])
        self.delta_y = np.abs(g1_x2_intersect[0] - g2_x2_intersect[0])
        
        intersections = np.array(intersections)
        
        def round2absint(point):
            p_copy = np.array(point, copy=True)
            neg_mask = p_copy < 0
            pos_mask = ~neg_mask
            p_copy[neg_mask] = np.floor(p_copy[neg_mask])
            p_copy[pos_mask] = np.ceil(p_copy[pos_mask])
            return p_copy
        
        rounded_intersections = round2absint(intersections)
        return intersections, rounded_intersections
    
    def get_bool_mask(self):
        """Создает булеву маску для области, удовлетворяющей ограничениям g1, g2 и условиям неотрицательности"""
        _, rounded_intersections = self.axis_intersections()
        
        if len(rounded_intersections) == 0:
            raise ValueError("Нет пересечений с осями координат")
        
        x1_min = 0
        x1_max = 10
        x2_min = 0
        x2_max = 10
        
        x1_vals = np.linspace(x1_min, x1_max, self.RESOLUTION)
        x2_vals = np.linspace(x2_min, x2_max, self.RESOLUTION)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        
        g1_func = sm.lambdify((self.x1, self.x2), self.g1, 'numpy')
        g2_func = sm.lambdify((self.x1, self.x2), self.g2, 'numpy')
        
        g1_values = g1_func(X1, X2)
        g2_values = g2_func(X1, X2)
        
        mask = (g1_values <= 0) & (g2_values <= 0) & (X1 >= 0) & (X2 >= 0)
        
        self.X1 = X1
        self.X2 = X2
        self.mask = mask
        
        return mask
    
    def get_random_point(self, plot_all=False):
        """Находит все целые точки в ОДР и возвращает случайную"""
        if not hasattr(self, 'mask'):
            self.get_bool_mask()
        
        # Получаем границы области
        x1_min, x1_max = int(np.floor(self.X1.min())), int(np.ceil(self.X1.max()))
        x2_min, x2_max = int(np.floor(self.X2.min())), int(np.ceil(self.X2.max()))
        
        # Генерируем все возможные целые точки в этих границах
        x_vals = np.arange(x1_min, x1_max + 1)
        y_vals = np.arange(x2_min, x2_max + 1)
        X, Y = np.meshgrid(x_vals, y_vals)
        points = np.vstack([X.ravel(), Y.ravel()]).T
        
        # Фильтруем точки по условиям
        valid_points = []
        for x, y in points:
            if x >= 0 and y >= 0:  # Неотрицательность
                # Вычисляем значения ограничений
                g1_val = float(self.g1.subs({self.x1: x, self.x2: y}))
                g2_val = float(self.g2.subs({self.x1: x, self.x2: y}))
                if g1_val <= 1e-6 and g2_val <= 1e-6:
                    valid_points.append([x, y])
        
        if not valid_points:
            raise ValueError("Нет целых точек в допустимой области. Проверьте ограничения.")
        
        # Визуализация
        ax = self.figure.gca()
        valid_arr = np.array(valid_points)
        ax.scatter(valid_arr[:, 0], valid_arr[:, 1], color='blue', s=30, label='Целые точки ОДР')
        
        # Выбираем случайную точку
        random_point = valid_points[np.random.choice(len(valid_points))]
        ax.scatter(random_point[0], random_point[1], color='red', s=100, label='Выбранная точка')
        
        plt.draw()
        return np.array(random_point)

    def plot_veasible_region(self, trajectory=None):
        """Визуализация допустимой области (ОДР) и ограничений"""
        if not hasattr(self, 'mask'):
            self.get_bool_mask()
        
        # Допустимая область
        plt.contourf(self.X1, self.X2, self.mask, levels=[0.5, 1.5], 
                    colors=['lightgreen'], alpha=0.3)
        
        # Границы ограничений
        g1_vals = sm.lambdify((self.x1, self.x2), self.g1, 'numpy')(self.X1, self.X2)
        g2_vals = sm.lambdify((self.x1, self.x2), self.g2, 'numpy')(self.X1, self.X2)
        
        plt.contour(self.X1, self.X2, g1_vals, levels=[0], colors='red', linewidths=2)
        plt.contour(self.X1, self.X2, g2_vals, levels=[0], colors='blue', linewidths=2)
        
        # Точки пересечения с осями
        inters, _ = self.axis_intersections()
        plt.scatter(inters[:, 0], inters[:, 1], color='purple', s=100, label='Пересечения')

        # Оси
        plt.axvline(x=0, color='black', lw=2, label='Ось x')
        plt.axhline(y=0, color='black', lw=2, label='Ось y')

        # Разметка оси
        plt.xticks(np.arange(-50, 50, 1))
        plt.yticks(np.arange(-50, 50, 1))
        plt.legend(loc='best')     
       
        # Траектория
        if trajectory is not None:
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'o-', color='orange', markersize=4)
            plt.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100)
            plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=100)
        
        plt.xlabel('x1', fontsize=12)
        plt.ylabel('x2', fontsize=12)
        plt.title('Допустимая область и ограничения', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        plt.close()
    
    def gradient_descent(self, iterations: int, start_point: np.ndarray,min_:bool = False) -> np.ndarray:
        """Градиентный спуск с аналитическим подбором шага"""
        trajectory = [start_point.copy()]
        solutions = []
        fs = []
        x_k = start_point.copy()
        
        t = 1
        if min_:
            t = -1
        for _ in range(iterations):
            solution = Solution()
            solution.x_k = x_k.copy()
            fs1= self.function.subs({self.x1: x_k[0], self.x2: x_k[1]})
            fs.append(fs1)
            # Вычисляем градиент в текущей точке
            grad = t*np.array([
                float(self.gradient[0].subs({self.x1: x_k[0], self.x2: x_k[1]})),
                float(self.gradient[1].subs({self.x1: x_k[0], self.x2: x_k[1]}))
            ], dtype=np.float64)
            solution.grad_x_k = t*np.round(grad,5)

            if np.linalg.norm(grad) < 0.01:
                break

                
            # Оптимальный шаг
            alpha = sm.Symbol('a')
            x_new = x_k + alpha * grad
            # print(x_new)
            x_new = np.array([round_expr(item)for item in x_new], dtype=object)


            solution.x_k1_a_ = x_new

            grad_new = np.array([
                round_expr(self.gradient[0].subs({self.x1: x_new[0], self.x2: x_new[1]})),
                round_expr(self.gradient[1].subs({self.x1: x_new[0], self.x2: x_new[1]}))
            ])
            solution.grad_a_ = grad_new
            
        
            # Решаем уравнение для оптимального alpha
            scalar_dot_grad_xk_grad_xk1_T = grad @ grad_new
            solution.gradxT_dot_gradx = round_expr(scalar_dot_grad_xk_grad_xk1_T)

            try:
                alpha_val = sm.solve(grad @ grad_new, alpha)[0]
            except:
                break

            solution.alpha = alpha_val

            x_k = x_k + alpha_val * grad
            solution.x_k1 = x_k

            solutions.append(solution)
            trajectory.append(x_k.copy())
        
        fs = trajectory[::t]
        # self.draw_level_lines(trajectory)
        return np.array(trajectory), solutions, fs
    def draw_level_lines(self, points):
        """Рисует линии уровня функции f(x1, x2) на self.figure через значения в точках points."""
        if not hasattr(self, 'mask'):
            self.get_bool_mask()
        X1 = np.linspace(-10,10,1000)
        X2 = np.linspace(-10,10,1000)
        X1, X2 = np.meshgrid(X1,X2)
        # Преобразуем символьное выражение в числовую функцию
        f_lambd = sm.lambdify((self.x1, self.x2), self.function, 'numpy')
        
        # Вычисляем значения функции на сетке
        Z = f_lambd(X1, X2)
        
        # Вычисляем значения функции в точках points для задания уровней
        points = np.array(points)
        ps = f_lambd(points[:, 0], points[:, 1])
        
        # Получаем оси из self.figure
        ax = self.figure.gca()
        
        # Рисуем линии уровня
        contour = ax.contour(X1, X2, Z, levels=ps, cmap='viridis')
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
        
        # Отображаем точки
        ax.plot(points[:, 0], points[:, 1], 'o-', color='orange', label='Траектория')
        ax.scatter(points[0, 0], points[0, 1], color='green', s=100, label='Начало')
        ax.scatter(points[-1, 0], points[-1, 1], color='red', s=100, label='Конец')

        # Настройка графика
        ax.axhline(0, color='black', lw=1)
        ax.axvline(0, color='black', lw=1)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('Линии уровня функции f(x1, x2)')
        ax.grid(True, linestyle='-', alpha=0.5)
        ax.legend(loc='best')
        ax.set_aspect('equal')

        # Не вызываем plt.show(), чтобы график остался на self.figure


def print_solutions(solutions: list[Solution]):
    """Печатает информацию о каждом шаге градиентного спуска"""
    print("\n" + "="*50)
    print("Детализация шагов градиентного спуска")
    print("="*50 + "\n")
    
    for i, solution in enumerate(solutions, 1):
        print(f"Шаг {i}:")
        print(f"Текущая точка (x_k): {solution.x_k}")
        print(f"Градиент в точке (grad_x_k): {solution.grad_x_k}")
        print(f"Следующая точка через alpha (x_k1_a_): {solution.x_k1_a_}")
        print(f"Градиент через alpha (grad_a_): {solution.grad_a_}")
        print(f"Скалярное произведение градиентов: {solution.gradxT_dot_gradx}")
        print(f"Оптимальное alpha: {solution.alpha}")
        print(f"Новая точка (x_k1): {solution.x_k1}")
        print("-"*50 + "\n")


# Функция для округления чисел в выражении
def round_expr(expr, decimals=3):
    return expr.xreplace({
        n: round(n, decimals)
        for n in expr.atoms(sm.Number)
        if isinstance(n, (float, sm.Float))
    })
def main(f,g1,g2):
    # вводить сюда коефициенты
    f_koefs = f.split()
    g1_koefs = g1.split(' ')
    g2_koefs = g2.split(' ')

    a, b, c, d, e = map(float, f_koefs)
    function = f'{a}*x1**2 + {b}*x2**2 + {c}*x1*x2 + {d}*x1 + {e}*x2'
    # g1_koefs = input("Введите коэффициенты g1: ").split(' ')

    # g2_koefs = input("Введите коэффициенты g2: ").split(' ')

    a,b,c = map(float,g1_koefs)
    g1 = f'{a}*x1 + {b}*x2 + {c}'
    a,b,c = map(float,g2_koefs)
    g2 = f'{a}*x1 + {b}*x2 + {c}'
    
    task = Task1(function, g1, g2, 'x1', 'x2')
    
    # Начальная точка внутри ОДР
    start_point = task.get_random_point()
    eigenvalues = np.linalg.eigvals(task.hessian)
    min_ = False
    if eigenvalues.all() > 0:
        min_ = True
        
    trajectory,solutions,fs = task.gradient_descent(20,np.array([5.26,0]), min_= min_)
    task.draw_level_lines(np.array(fs[::-1]))
    


    #start info
    # 1 .F, g1,g2,grad(f), hessian, hessian^-1
    opt_task = 'минимум'
    if min_:
        opt_task = 'максимум'
    
    print(f"Найти {opt_task} функции")
    print("F(x) = ",task.function)
    print('g1 = ',task.g1)
    print('g2 = ', task.g2)
    print('H = ',task.hessian)
    print('H^-1 = ',np.linalg.inv(task.hessian))

    # 2.Newton method
    print()
    print()
    print("Метод ньютона")
    print("="*50 + "\n")
    grad = np.array([
                float(task.gradient[0].subs({task.x1: start_point[0], task.x2: start_point[1]})),
                float(task.gradient[1].subs({task.x1: start_point[0], task.x2: start_point[1]}))
            ])
    print('x* = ', start_point,' - ', np.linalg.inv(task.hessian),' * ', grad)
    new_point = start_point - np.linalg.inv(task.hessian)@grad
    print('x* = ', new_point)
    print("="*50 + "\n")

    #3. grad desc
    print("Градиентный спуск")
    print_solutions(solutions)

    # x = sm.symbols('a')
    # x1 = x*(-0.71)
    # x2 = x*(0.71)
    # result = task.function.subs({task.x1: x1, task.x2: x2})
    # expanded_result = sm.expand(result)
    # print('F(a) = ',expanded_result)
    # print('gradF = ',sm.diff(expanded_result,x))
    # a = sm.solve(sm.diff(expanded_result,x), x)[0]
    # # a = -5.26
    # print('a = ', a)
    # print('x1 = ', x1.subs(x,a))
    # print('x2 = ',x2.subs(x,a))


    task.plot_veasible_region(trajectory)

if __name__ == '__main__':
        # вводить сюда коефициенты
    f_koefs = '-1 -5 -1 -2 1'
    g1_koefs = '-8 3 0'
    g2_koefs = '1 1 -11'
    main(f_koefs,g1_koefs,g2_koefs)
