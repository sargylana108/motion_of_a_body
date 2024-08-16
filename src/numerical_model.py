import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

g = 9.8  # Ускорение свободного падения

# Функция для вычисления правой части системы ОДУ
def ode_system(t, y, a, b):
    X, Y, Vx, Vy = y
    dXdt = Vx
    dYdt = Vy
    dVxdt = -a * Vx
    dVydt = -g - b * Vy
    return [dXdt, dYdt, dVxdt, dVydt]

# Функция для решения системы ОДУ
def solve_ode(angle, v0, a, b, t_span=[0, 10], num_points=500):
    '''
    Параметры:
    angle : угол броска
    v0 : начальная скорость
    a : линейный коэффициент сопротивления
    b : квадратичный коэффициент сопротивления
    t_span : временной интервал
    num_points : количество точек
    Возвращает:
    t : временные отметки
    X : Горизонтальные координаты
    Y : Вертикальные координаты
    Vx : Горизонтальная компонента скорости
    Vy : Вертикальная компонента скорости
    '''
    angle_rad = np.radians(angle)
    Vx0 = v0 * np.cos(angle_rad)
    Vy0 = v0 * np.sin(angle_rad)
    initial_conditions = [0, 0, Vx0, Vy0]
    
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    solution = solve_ivp(ode_system, t_span, initial_conditions, args=(a, b), t_eval=t_eval)
    
    return solution.t, solution.y[0], solution.y[1], solution.y[2], solution.y[3]

# Функция для построения траектории решения системы ОДУ
def plot_trajectory_ode(t, X, Y, v0, angle, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(X, Y)
    plt.title(f'Траектория тела (v0 = {v0} m/s, angle = {angle}°)')
    plt.xlabel('Дальность полёта (X)')
    plt.ylabel('Высота подъёма (Y)')
    plt.ylim(bottom=0)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# Функция для расчета ключевых характеристик траектории
def calculate_trajectory_characteristics(t, X, Y, Vx, Vy):
    """
    Вычисляет ключевые характеристики траектории:
    - Дальность полёта
    - Максимальная высота подъёма
    - Время полёта
    - Время подъёма на максимальную высоту
    - Модуль скорости в момент падения

    Параметры:
    t : Временные отметки
    X : Горизонтальные координаты
    Y : Вертикальные координаты
    Vx : Горизонтальная компонента скорости
    Vy : Вертикальная компонента скорости
    Возвращает:
    results : Словарь с ключевыми характеристиками траектории
    """
    flight_range = X[-1]
    max_height = np.max(Y)
    flight_time = t[-1]
    time_to_max_height = t[np.argmax(Y)]
    final_speed = np.sqrt(Vx[-1]**2 + Vy[-1]**2)
    
    results = {
        "Дальность полёта (м)": flight_range,
        "Максимальная высота подъёма (м)": max_height,
        "Время полёта (сек)": flight_time,
        "Время подъёма на максимальную высоту (сек)": time_to_max_height,
        "Модуль скорости в момент падения (м/с)": final_speed
    }
    
    return results


if __name__ == "__main__":
    # Пример использования
    angle = 30  # Угол в градусах
    speed = 100  # Скорость в м/с
    
    # Решение с учётом сопротивления (например, a=0.1, b=0.01)
    t, X, Y, Vx, Vy = solve_ode(angle, speed, a=0.1, b=0.01)
    plot_trajectory_ode(t, X, Y, speed, angle, save_path='results/trajectory_with_resistance.png')
    
    # Вычисление ключевых характеристик траектории
    characteristics = calculate_trajectory_characteristics(t, X, Y, Vx, Vy)
    for key, value in characteristics.items():
        print(f"{key}: {value}")