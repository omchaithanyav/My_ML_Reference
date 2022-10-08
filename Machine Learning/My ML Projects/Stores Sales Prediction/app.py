from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

        Item_Weight= float(request.form['Item_Weight'])
        Item_Fat_Content= int(request.form['Item_Fat_Content'])
        Item_Visibility= float(request.form['Item_Visibility'])
        Item_Type= int(request.form['Item_Type'])
        Item_MRP= float(request.form['Item_MRP'])
        Outlet_Size= int(request.form['Outlet_Size'])
        Outlet_Location_Type= int(request.form['Outlet_Location_Type'])
        Outlet_Type= int(request.form['Outlet_Type'])



        inputs = np.array([Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, Outlet_Size, Outlet_Location_Type, Outlet_Type]).reshape(1, -1)

        # prediction
        model = pickle.load(open('models/gb.pkl', 'rb'))
        prediction = model.predict(inputs)
        prediction = float(prediction)
        return render_template('result.html', results='The outlet sales for the given item should be: {}'.format(prediction))



if __name__ == '__main__':
    app.run(debug=True, port = int(os.environ.get('PORT',5000)))