import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

g = 9.8  # Ускорение свободного падения

# Функция для расчёта высоты и дистанции
def trajectory(v0, angle, t_span):
    '''
    Параметры:
    v0 : начальная скорость
    angle : угол броска
    t_span : временной интервал
    Возвращает:
    y : высота подъёма
    x : дальность полёта
    '''
    angle_rad = np.radians(angle)
    x = v0 * t_span * np.cos(angle_rad)  
    y = v0 * t_span * np.sin(angle_rad) - 0.5 * g * t_span**2  
    return x, y 

# Функция для расчёта времени полёта
def calculate_time_of_flight(v0, angle):
    return 2 * v0 * np.sin(np.radians(angle)) / g

# Функция для графика траектории тела
def plot_trajectory(v0, angle):
    t_max = calculate_time_of_flight(v0, angle)  # время полёта
    t_span = np.linspace(0, t_max, num=500)  # временной интервал
    x, y = trajectory(v0, angle, t_span)
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.title(f'Траектория тела (v0 = {v0} m/s, angle = {angle}°)')
    plt.xlabel('Дальность полёта (м)')
    plt.ylabel('Высота подъёма (м)')
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.show()

# Интерактивный график
def interactive_plot():
    interact(plot_trajectory, 
             v0=FloatSlider(min=1, max=300, step=1, value=50),  # Начальная скорость (1-300 m/s)
             angle=FloatSlider(min=10, max=80, step=1, value=45))  # Угол броска (10-80°)

if __name__ == "__main__":
    # Пример использования
    interactive_plot()