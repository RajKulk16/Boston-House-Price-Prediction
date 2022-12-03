import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd
import numpy as np

app = Flask(__name__) #starting point of application
model = pickle.load(open('regmodel.pkl','rb'))
initial_scalar = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html') #index page


@app.route('/predict_api', methods=['POST']) #using APIs for production
def predict_api():
    data = request.json['data'] #key-value dictionary format
    print(data)
    del data['TAX']
    print(np.array(list(data.values())).reshape(1,-1))
    # Need not "compulsory" do the scaling values. This is because we are using RandomForest model.
    new_data = np.array(list(data.values())).reshape(1,-1)
    #new_data = initial_scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])


if __name__=="__main__":
    app.run(debug=True)