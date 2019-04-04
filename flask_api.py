import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request
from keras.models import model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("keras_model.h5")

df = pd.read_csv('data.csv')

app = Flask(__name__)

@app.route('/earth', methods = ['POST'])

def make_prediction ():
    data = request.get_json(force = True)
    
    #ADD THE PARAMETERS WHICH HAVE TO BE REQUESTED FROM THE WEB SERVER
    
    predict_request = [df.loc[df['building_id'] == data]]
    predict_request = np.array([predict_request])
    
    #ORGANISING THE RECIEVED DATA INTO THE RIGHT FORM

    y_pred = loaded_model(predict_request)
    output = [y_pred[0]]
    return jsonify(results = output)

if __name__ = '__main__':
    app.run(port = 9000, debug = True)
