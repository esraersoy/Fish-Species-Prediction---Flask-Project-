# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:16:16 2022

@author: eeers
"""
from flask import Flask, redirect, url_for, render_template, request
import pickle
import numpy as np
import heapq


app = Flask(__name__)

model = pickle.load(open('classifier.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_features = [x for x in request.form.values()]
    inputs_array = [np.array(input_features)]
    prediction = model.predict(inputs_array)
    return render_template("index.html", content = "This fish belongs to species " + str(prediction))


if __name__=='__main__':
    app.run()
    