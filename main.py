import os
from src.analytical_model import interactive_plot
from src.data_analysis import (
    load_dataset,
    calculate_quartiles,
    linear_regression_a,
    plot_regression_a,
    predict_a,
    regression_b,
    plot_regression_b,
    predict_b,
)
from src.numerical_model import solve_ode, plot_trajectory_ode, calculate_trajectory_characteristics

def main():
    # Проверка наличия директории для сохранения результатов
    if not os.path.exists('results'):
        os.makedirs('results')
        
    # Аналитическая модель без учёта сопротивления воздуха
    interactive_plot()  # Интерактивный график
    
    # Загрузка данных
    data = load_dataset('data/dataset.csv')
    
    # Расчёт квартилей
    quartiles = calculate_quartiles(data)
    print(f"Квартильные значения скорости:\n{quartiles}")
    
    # Линейная регрессия (параметр 'a')
    slope_a, intercept_a = linear_regression_a(data)
    plot_regression_a(data, slope_a, intercept_a, save_path='results/regression_a.png')
    
    # Линейная регрессия (параметр 'b')
    slope_b, intercept_b = regression_b(data)
    plot_regression_b(data, slope_b, intercept_b, save_path='results/regression_b.png')
    
    # Примеры предсказаний
    speed_example = quartiles[0.75] 
    a_pred = predict_a(speed_example, slope_a, intercept_a)
    b_pred = predict_b(speed_example, slope_b, intercept_b)
    print(f"Предсказанное значение 'a' при v={speed_example}: {a_pred}")
    print(f"Предсказанное значение 'b' при v={speed_example}: {b_pred}")
    
    # Численная модель с учётом сопротивления воздуха
    t, X, Y, Vx, Vy = solve_ode(30, speed_example, a_pred, b_pred)
    plot_trajectory_ode(t, X, Y, speed_example, 30, save_path='results/trajectory_with_resistance.png')
    
    # Вычисление и вывод ключевых характеристик траектории
    characteristics = calculate_trajectory_characteristics(t, X, Y, Vx, Vy)
    for key, value in characteristics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()