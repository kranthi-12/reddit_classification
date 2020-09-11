import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.linear_model import LogisticRegression

filename = 'model.pkl'
logmodel = pickle.load(open(filename, 'rb'))
tfidf = pickle.load(open('tranform.pkl','rb'))
app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message=request.form["message"]
        data=[message]
        vect=tfidf.transform(data).toarray()
        my_prediction=logmodel.predict(vect)
    return render_template('result.html',prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
        
    