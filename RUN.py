# DWork
# Импортируем Pandas и Numpy 
import pandas as pd
import numpy as np

# Обращаемся к файлу
df = pd.read_csv('NPost2.csv')

# Вводим новый столбец "Кол-во символов"
df['Кол-во символов'] = df['Текст'].fillna('').str.len()

# Вводим новую переменную "Капс"
df['Капс'] = df.textcolumn.map(lambda x: len([df['Текст'] for df['Текст'] in x if df['Текст'].isupper()])

# Создаем отношение длины текста к колличесву просмотров

# Создаем отношение количества капс символов к просмотрам
  
# Проверяем как все получается                                           
df.sort_values(by=['Комментариев', 'Капс'], ascending=[False, True]).head()

#Сохраняем как новый файл
df.to_csv('MNPostTable.csv')
