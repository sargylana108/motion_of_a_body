import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Функция для загрузки датасета
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    data = data.rename(columns={'velocity': 'v'})  # В заданиии 'v' вместо 'velocity' 
    return data

# Функция для определения 1-го, 2-го и 3-го квартилей 'v'
def calculate_quartiles(data):
    quartiles = data['v'].quantile([0.25, 0.5, 0.75])
    return quartiles

# Функция для вычисления параметров линейной регрессии для 'a' 
def linear_regression_a(data):
    slope, intercept, r_value, p_value, std_err = stats.linregress(data['v'], data['a'])
    return slope, intercept

# Функция для построения графика регрессии для 'a'
def plot_regression_a(data, slope, intercept, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.scatter(data['v'], data['a'], label='Data Points')
    plt.plot(data['v'], slope * data['v'] + intercept, color='red', label='Linear Regression')
    plt.xlabel('Скорость (v)')
    plt.ylabel('Параметр a')
    plt.title('Линейная регрессия для параметра a')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# Функция для предсказания значения 'a' по 'v'
def predict_a(v, slope, intercept):
    return slope * v + intercept

# Функция для вычисления параметров регрессии для 'b'
def regression_b(data):
    slope, intercept, r_value, p_value, std_err = stats.linregress(data['v']**2, data['b'])
    return slope, intercept

# Функция для построения графика регрессии для 'b'
def plot_regression_b(data, slope, intercept, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.scatter(data['v']**2, data['b'], label='Data Points')
    plt.plot(data['v']**2, slope * (data['v']**2) + intercept, color='red', label='Linear Regression')
    plt.xlabel('Скорость (v^2)')
    plt.ylabel('Параметр b')
    plt.title('Регрессия для параметра b')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# Функция для предсказания значения 'b' по 'v'
def predict_b(v, slope, intercept):
    return slope * (v ** 2) + intercept

if __name__ == "__main__":
    # Пример использования
    data = load_dataset('data/dataset.csv')
    
    # Вычисление квартилей
    quartiles = calculate_quartiles(data)
    print(f"Квартильные значения скорости::\n{quartiles}")
    
    # Линейная регрессия для 'a'
    slope_a, intercept_a = linear_regression_a(data)
    plot_regression_a(data, slope_a, intercept_a, save_path='results/regression_a.png')
    
    # Регрессия для 'b'
    slope_b, intercept_b = regression_b(data)
    plot_regression_b(data, slope_b, intercept_b, save_path='results/regression_b.png')
    
    # Примеры предсказания
    v_example = 100
    a_pred = predict_a(v_example, slope_a, intercept_a)
    b_pred = predict_b(v_example, slope_b, intercept_b)
    print(f"Предсказанное значение a для скорости={v_example}: {a_pred}")
    print(f"Предсказанное значение b для скорости={v_example}: {b_pred}")