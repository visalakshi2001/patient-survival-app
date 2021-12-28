from flask import Flask, render_template
import flask
import pandas as pd
import numpy as np
from model import predict_survival
import os
import joblib
from flask_restful import Api, Resource, reqparse
import ast


curr_path = os.path.dirname(os.path.realpath(__file__))
FEATS = joblib.load(curr_path + "/dataset/feats.pkl" )
DTYPES = joblib.load(curr_path + "/dataset/dtypes.pkl" )


app = Flask(__name__)
parser = reqparse.RequestParser()
parser.add_argument('list', type=list)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', features=FEATS, dtypes=DTYPES, zip = zip, len = len, range=range, list=list,)


@app.route('/predict', methods=['POST'])
def predict():
    csv_data = flask.request.files['csv_file']
    test_data = pd.read_csv(csv_data)

    y_pred = predict_survival(test_data)
    print(len(y_pred))

    

    return render_template('index.html', range=range, list=list, len = len, pred = y_pred.tolist(), length = len(y_pred), zip = zip, features=FEATS, dtypes=DTYPES,)

@app.route('/api/predict', methods=['POST'])
def api_route():
    data = flask.request.args('input')
    i = [n.strip() for n in ast.literal_eval(data)]

    y_pred = predict_survival(np.array(i))

    return y_pred



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9874)