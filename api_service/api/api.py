from celery import Celery
import pickle
import re
from bs4 import BeautifulSoup
import nltk
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from flask import request
import json


configuration = {
    'tokenizer': '../tokenizer.pickle',
    'keras_model': '../model_best_neuro.json',
    'keras_model_weights': '../model_best_neuro.h5'
}



def make_celery(app):
    celery = Celery(
        # app.import_name,
        'api',
        backend=app.config['result_backend'],
        broker=app.config['broker_url']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


from flask import Flask

flask_app = Flask(__name__)
flask_app.config.update(
    broker_url='redis://127.0.0.1:6379',
    result_backend='redis://127.0.0.1:6379'
)
celery = make_celery(flask_app)


@celery.task()
def add_together(self, a, b):
    return a + b


@flask_app.route('/api/imdb_sa/<text>', methods=['POST', 'GET'])
def imdb_sa(text=None):
    result = add_together.delay(23, 42)
    result.wait()  # 65
    return str(result.result)

@flask_app.route('/api/imdb_sa_s/<text_data>', methods=['POST', 'GET'])
def imdb_sa_s(text_data=None):
    d = json.loads(request.data)
    txt = d['Text']
    text = clean_data(txt)
    seq =  tokenizer.texts_to_sequences([text])
    prepared_vector = pad_sequences(seq, 100)
    predict = loaded_model.predict_classes(prepared_vector)
    proba = loaded_model.predict_proba(prepared_vector)
    return str(predict[0][0]) + ";" + str(proba[0][0])

loaded_model = None
tokenizer = None
en_sw = None


def _clean_stop_words(txt:str) -> str:
    """Удаляет стоп-слова"""
    result = ' '.join([w for w in txt.strip().split() if not w in en_sw])
    return result

def _strip_html(text:str) -> str:
    """Удаляет из текста HTML-теги"""
    return BeautifulSoup(text, "lxml").text

def clean_data(text:str) -> str:
    """Приведение к нижнему регистру и очистка от лишних символов"""
    text = _strip_html(text) # Удалим HTML-теги из текста
    text = text.lower() # приводим к нижнему регистру
    review_text = re.sub("[^а-яА-Яa-zA-Z]", " ", text)
    review_text = _clean_stop_words(review_text)
    return review_text.strip() # Удаляем лишние пробемы в начале и конце строки

def load():
    # Загрузка модели и весов
    global loaded_model
    json_file = open('../ds/model_best_neuro.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("../ds/model_best_neuro.h5")

    # Загрузка токенайзера
    global tokenizer
    with open('../ds/tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    # Загрузка стоп-слов
    global en_sw
    nltk.download('stopwords')  # Загружаем стоп-слова
    en_sw = nltk.corpus.stopwords.words('english')


if __name__ == '__main__':
    load()
    flask_app.run()

