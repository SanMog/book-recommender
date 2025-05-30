# Рекомендательная система на основе матричной факторизации и градиентного спуска (SGD)
# Работает на первых 2000 строках файла ratings_S.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# === ШАГ 1: Загрузка данных ===
print("Загрузка данных...")
df = pd.read_csv("ratings_S.csv")
df = df.head(2000)  # Используем первые 2000 записей
print(f"Загружено {len(df)} записей")

# === ШАГ 2: Подготовка данных ===
print("\nПодготовка данных...")
# Преобразуем user_id и book_id в индексы для удобства работы с матрицами
user_ids = df['user_id'].unique()
book_ids = df['book_id'].unique()

user_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
book_to_index = {bid: idx for idx, bid in enumerate(book_ids)}

df['user_index'] = df['user_id'].map(user_to_index)
df['book_index'] = df['book_id'].map(book_to_index)

print(f"Количество уникальных пользователей: {len(user_ids)}")
print(f"Количество уникальных книг: {len(book_ids)}")

# === ШАГ 3: Разделение на обучение и тест ===
print("\nРазделение данных на обучающую и тестовую выборки...")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Размер обучающей выборки: {len(train_df)}")
print(f"Размер тестовой выборки: {len(test_df)}")

# === ШАГ 4: Инициализация скрытых матриц ===
print("\nИнициализация параметров модели...")
num_users = len(user_ids)
num_items = len(book_ids)
latent_factors = 20  # Количество скрытых факторов

# Инициализируем матрицы P (пользователи) и Q (предметы) случайными значениями
P = np.random.normal(scale=1./latent_factors, size=(num_users, latent_factors))
Q = np.random.normal(scale=1./latent_factors, size=(num_items, latent_factors))

# === ШАГ 5: Обучение с помощью градиентного спуска ===
print("\nНачало обучения модели...")
lr = 0.01  # Скорость обучения
epochs = 30  # Количество эпох
lambda_reg = 0.1  # Параметр регуляризации

train_errors = []

for epoch in range(epochs):
    # Обучение на каждой записи обучающей выборки
    for _, row in train_df.iterrows():
        u = int(row['user_index'])
        i = int(row['book_index'])
        r_ui = row['rating']

        # Предсказание и вычисление ошибки
        prediction = np.dot(P[u], Q[i])
        error = r_ui - prediction

        # Обновление параметров с учетом регуляризации
        P[u] += lr * (error * Q[i] - lambda_reg * P[u])
        Q[i] += lr * (error * P[u] - lambda_reg * Q[i])

    # Вычисление RMSE на обучающей выборке
    train_preds = [np.dot(P[int(row['user_index'])], Q[int(row['book_index'])]) for _, row in train_df.iterrows()]
    train_truth = train_df['rating'].values
    rmse = np.sqrt(mean_squared_error(train_truth, train_preds))
    train_errors.append(rmse)
    print(f"Эпоха {epoch+1}: RMSE на обучающей выборке = {rmse:.4f}")

# === ШАГ 6: Оценка на тестовой выборке ===
print("\nОценка модели на тестовой выборке...")
test_preds = [np.dot(P[int(row['user_index'])], Q[int(row['book_index'])]) for _, row in test_df.iterrows()]
test_truth = test_df['rating'].values
test_rmse = np.sqrt(mean_squared_error(test_truth, test_preds))
print(f"RMSE на тестовой выборке: {test_rmse:.4f}")

# === ШАГ 7: График обучения ===
print("\nПостроение графика обучения...")
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), train_errors, marker='o')
plt.xlabel('Эпоха')
plt.ylabel('RMSE на обучающей выборке')
plt.title(f'Обучение модели (RMSE на тестовой выборке: {test_rmse:.4f})')
plt.grid(True)
plt.tight_layout()
plt.savefig("train_rmse_plot.png")
print("График сохранен в файл train_rmse_plot.png")

# === ШАГ 8: Пример рекомендаций ===
print("\nПример рекомендаций для первого пользователя:")
user_idx = 0
user_ratings = np.dot(P[user_idx], Q.T)
top_5_books = np.argsort(user_ratings)[-5:][::-1]
print("Топ-5 рекомендуемых книг:")
for i, book_idx in enumerate(top_5_books, 1):
    book_id = book_ids[book_idx]
    predicted_rating = user_ratings[book_idx]
    print(f"{i}. Книга {book_id}: предсказанный рейтинг {predicted_rating:.2f}")
