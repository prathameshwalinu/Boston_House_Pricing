import pickle 
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

#Loading the model

model = pickle.load(open('reg_model.pkl','rb'))
scalar = pickle.load(open('standardising.pkl','rb'))

@app.route('/')
def home():
        return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
        data = request.json['data']
        print(data)
        #print(np.array(list(data.values)).reshape(1,-1))
        new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
        prediction = model.predict(new_data) 
        print(prediction[0])
        return jsonify(prediction[0]) 

@app.route('/predict_api/kn',methods=['POST'])
def predict_api1():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == "__main__":
        app.run(debug=True)

