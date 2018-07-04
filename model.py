import pandas as pd
import numpy as np
import re
from sklearn.linear_model import RidgeCV
# do 'pip install shap'
import shap


df = pd.read_csv('NPost2.csv')
df.columns = ['link', 'text', 'date', 'likes_count', 'comments_count', 'reposts_count',
              'visits_count', 'wiki_views', 'views', 'likes_per_views', 'comments_per_views',
              'reposts_per_views', 'visits_pre_views', 'wiki_per_views',]
df.text = df.text.fillna(' ').astype(str)
df = df.loc[:, ['link', 'text', 'date', 'likes_count', 'views', 'likes_per_views',]]

# Число символов
df['symbols_count'] = df.text.str.len()
# Число символов капсом
df['caps'] = df.text.map(lambda x: sum([len(str(x)) for x in x if x.isupper()]))
# Число абзацев
df['lines'] = df.text.map(lambda x: len(x.split('\n')))
# Число цифр
df['nums'] = df.text.map(lambda x: len(re.findall('[0-9]', x)))
# Число восклицательных знаков
df['exclamations'] = df.text.map(lambda x: x.count('!'))

# Убираем проценты, перегоняем данные в дробный числовой формат
df.likes_per_views = df.likes_per_views.map(lambda x: re.sub('%', '', x)).astype(np.float32)

# Проверяем таблицу
df.tail(30)

# Создаем обучающие данные и ответы к ним
X = df.drop(['link', 'text', 'date', 'likes_count', 'views', 'likes_per_views'], axis=1)
y = df.likes_per_views

# Выбираем линейную модель со встроенной кросс-валидацией для обучения
model = RidgeCV(cv=5)

# Обучаем модель
model.fit(X, y)

# Смотрим, как близко эта модель на этих данных смогла объяснить результаты
# (здесь считается среднее квадратичное отклонение)
# Стоит пробовать поднять точность, отдавая модели другие данные помимо наших 5 колонок
print(f'Точность модели: {model.score(X, y)}') # 0.0196

# Коэффициенты к параметрам, которые подобрала модель
print(f'Веса параметров: {model.coef_}')

# Вывод коэффициентов по колонкам
print('Влияние значений по колонкам на результат:')
for i, col in enumerate(X.columns):
    print(f'{col}: {round(model.coef_[i], 8)}')

# Отображение части данных (первая 1000) с зависимостью от значения фичи
X_summary = shap.kmeans(X, 2)
shap_values = shap.KernelExplainer(model.predict, X_summary).shap_values(X.iloc[:1000, :])
shap.summary_plot(shap_values, X.iloc[:1000, :])
