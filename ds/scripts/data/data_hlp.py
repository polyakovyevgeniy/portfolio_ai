import numpy as np
from typing import List, Dict
import string
import os
import re
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm, tqdm_notebook
tqdm.pandas(tqdm, leave=False)
from IPython.display import clear_output
import pandas as pd
import pickle
from bs4 import BeautifulSoup
from collections import OrderedDict
from nltk.stem import WordNetLemmatizer

def load_data_set(path: str, sub_path: List[str]) -> (List[str], [List[int]]):
    """Загрузка данных из файлов"""
    texts = [] # текст
    labels = [] # метки
    count_iter = len(sub_path) # количество итераций
    current_iter = 0 # текущая итерация
    for label_type in sub_path:
        current_iter = current_iter + 1
        clear_output()
        print("iter: {}/{}".format(current_iter, count_iter))
        dir_name = os.path.join(path, label_type) # текущая директория
        for fname in tqdm_notebook(os.listdir(dir_name)):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname), encoding='utf-8')
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    return (texts, labels)

def save_data(df: pd.DataFrame, path: str)-> None:
    """Сохраняет данные на диск"""
    with open(path, 'wb') as f:
        pickle.dump(df, f)
    print('saving success!')

def safe_to_file(file:any, path:str):
    """Сохраняет лб на диск"""
    with open(path, 'wb') as f:
        pickle.dump(file, f)
    print('saving success!')

def load_data(path: str)-> pd.DataFrame:
    """Восстенавливает данные с диска"""
    with open(path, 'rb') as f:
            df = pickle.load(f)
    return df

def load_from_file(path: str)-> any:
    """Восстенавливает данные с файла"""
    with open(path, 'rb') as f:
            df = pickle.load(f)
    return df

def clean_stop_words(series: pd.Series, stop_words: List[str]) -> List[str]:
    """Удаляет стоп-слова"""
    result = []
    for text in tqdm_notebook(series):
        result.append(' '.join([w for w in text.strip().split() if not w in stop_words]))
    return result

def _strip_html(text:str) -> str:
    """Удаляет из текста HTML-теги"""
    return BeautifulSoup(text, "lxml").text

def _clean_symbols_data(text:str) -> str:
    """Приведение к нижнему регистру и очистка от лишних символов"""
    text = _strip_html(text) # Удалим HTML-теги из текста
    text = text.lower() # приводим к нижнему регистру
    review_text = re.sub("[^а-яА-Яa-zA-Z]", " ", text)
    return review_text.strip() # Удаляем лишние пробемы в начале и конце строки

def clean_symbols_data(series: pd.Series):
    """Приведение к нижнему регистру и очистка от лишних символов"""
    new_series = series.progress_apply(_clean_symbols_data)
    return new_series

def create_vocab(series: List[str]):
    """Создает словарь"""
    popular = {}
    for x in tqdm_notebook(series):
        splitted = x.split()
        for z in splitted:
            if z not in popular:
                popular[z] = 1
            else:
                popular[z] = popular[z] + 1

    # Сортируем по частоте встречаемости
    sorted_popular = dict(OrderedDict(sorted(
        popular.items(), key=lambda t: t[1], reverse=True)))

    order_by_popular = {}
    count = 1
    for pop in sorted_popular:
        order_by_popular[pop] = count
        count = count + 1
    return sorted_popular, order_by_popular

def create_vectors(arr: List[str], dic:Dict[str, int], num_words=None) -> List[List[int]]:
    """Создает векторы на основе словаря и текста"""
    result = []
    for txt in arr:
        tmp_text = []
        splitted = txt.split()
        for word in splitted:
            if word in dic:
                # Проверяем на максимальное количество слов
                # если не привышено, записываем
                if dic[word] < num_words - 1:
                    tmp_text.append(dic[word])
                else: # Иначе пишем максимальное по количеству
                    tmp_text.append(num_words - 1)
            else:
                tmp_text.append(num_words - 1) # Добавляем 0, если не найден элемент в словаре
        result.append(tmp_text)
    return result


def lemmatization(series: pd.Series):
    wordnet_lemmatizer = WordNetLemmatizer()
    """Лемматизация текса"""
    result = []
    for text in tqdm_notebook(series):
        texts =  str(text).split() # разбиваем слова по пробелу
        lemm_word = []
        for txt in texts:
            lemm_word.append(wordnet_lemmatizer.lemmatize(txt))
        result.append(' '.join(lemm_word))
    return result


def load_glove(path:str):
    embedding_index = {}
    texts = []
    f = open(path, encoding='utf-8')
    texts.append(f.readlines())
    f.close()
    for line in tqdm_notebook(texts[0]):
        values = line.split()
        word = values[0]
        coefs = np.asanyarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    return embedding_index

def tokenizer(seties:pd.Series):
    result = []
    for text in seties:
        words = text.split()
        tmp = []
        for word in words:
            tmp.append(word)
            result.append(tmp)
    return result