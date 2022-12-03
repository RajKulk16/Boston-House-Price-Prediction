import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd
import numpy as np

app = Flask(__name__) #starting point of application
model = pickle.load(open('regmodel.pkl','rb'))
initial_scalar = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html') #index page


@app.route('/predict_api', methods=['POST']) #using APIs for getting values
def predict_api():
    data = request.json['data'] #key-value dictionary format
    print(data)
    del data['TAX'] #removing tax as it is highly correlated to RAD
    print(np.array(list(data.values())).reshape(1,-1))
    # Need not "compulsory" do the scaling values. This is because we are using RandomForest model.
    new_data = np.array(list(data.values())).reshape(1,-1)
    #new_data = initial_scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST']) # to predict by getting input from user in the html page
def predict():
    data = [float(x) for x in request.form.values()]
    del data[9] #removing tax as it is highly correlated to RAD
    final_data = np.array((data)).reshape(1,-1) #reshaped data
    print(final_data)
    output = model.predict(final_data)
    for i in output:
        final_output = round(i,3)
    return render_template("index.html",prediction_text ="Price of the house is approximately = {}".format(final_output))






if __name__=="__main__":
    app.run(debug=True)