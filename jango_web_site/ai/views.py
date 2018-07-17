from django.shortcuts import render
from django.http import HttpResponse
import requests

def index(request):
    ''' Форма по умолчанию '''
    if request.method == "GET":
        return render(request, 'ai/sa.html',
                   {'source': '', 'result': '?'})
    if request.method == "POST":
        data = ({"Text": request.POST['q']})
        r = requests.post("http://127.0.0.1:5000/api/imdb_sa_s/hello", json=data)
        result = r.text
        res = result.split(';')
        pred = res[0]
        prob = res[1]

        answer_pred = ''
        if pred == '1':
            answer_pred = '+'
        else:
            answer_pred = '-'

        return render(request, 'ai/sa.html',
                      {'source': request.POST['q'], 'result': answer_pred + " (" + prob + ")"})