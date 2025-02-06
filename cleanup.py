import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка датасета (пример с Yelp Reviews, замените путь на свой)
df = pd.read_csv('yelp_reviews_sample.csv')  # Предполагается, что в датасете есть поля: review_id, user_id, text, stars, date

# Первичный осмотр данных
print(df.head())
print(df.info())
print(df.describe())

# Искусственно добавим дубликаты для демонстрации очистки
df = df.append(df.iloc[0:5], ignore_index=True)

# Поиск и удаление дубликатов
print("Дубликатов до очистки:", df.duplicated().sum())
df_clean = df.drop_duplicates()
print("Дубликатов после очистки:", df_clean.duplicated().sum())

# Приведение столбца date к типу datetime
df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')

# Обнаружение пропусков
print("Пропуски в каждом столбце:\n", df_clean.isnull().sum())

# Заполнение или удаление пропусков (например, удалим строки с пропущенными датами)
df_clean = df_clean.dropna(subset=['date'])

# Простейшая визуализация распределения оценок
sns.countplot(data=df_clean, x='stars')
plt.title('Распределение оценок')
plt.show()

# Формирование гипотезы: допустим, чем длиннее текст, тем выше оценка?
df_clean['text_length'] = df_clean['text'].apply(lambda x: len(str(x)))
sns.boxplot(x='stars', y='text_length', data=df_clean)
plt.title('Длина текста в зависимости от оценки')
plt.show()

# Можно вынести данные по авторам в отдельную таблицу
authors = df_clean[['user_id']].drop_duplicates().reset_index(drop=True)
print("Количество уникальных авторов:", len(authors))

# Далее можно сохранить результат:
df_clean.to_csv('yelp_reviews_clean.csv', index=False)
authors.to_csv('yelp_authors.csv', index=False)
