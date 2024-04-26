from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd  # Import pandas library
import joblib

app = Flask(__name__)

mainmodel = joblib.load("model.pkl")
scalemodel = joblib.load("preprocessing.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    year = request.args.get('year')
    odometer = request.args.get('odometer')
    condition = request.args.get('condition')
    mmr = request.args.get('mmr')
    make = request.args.get('make')
    body = request.args.get('body')
    color = request.args.get('color')
    interior = request.args.get('interior')
    return redirect(url_for('result', year=year, odometer=odometer, condition=condition, mmr=mmr, make=make, body=body, color=color, interior=interior))

@app.route('/result')
def result():
    year = request.args.get('year')
    odometer = request.args.get('odometer')
    condition = request.args.get('condition')
    mmr = request.args.get('mmr')
    make = request.args.get('make')
    body = request.args.get('body')
    color = request.args.get('color')
    interior = request.args.get('interior')
    
    # Convert input data into a DataFrame
    input_data = pd.DataFrame([[year, condition, odometer, mmr, make, body, color, interior]],
                              columns=['year', 'condition', 'odometer', 'mmr', 'make', 'body', 'color', 'interior'])

    # Scale the input data
    input_data_scaled = scalemodel.transform(input_data)
    
    # Predict using the main model
    pred = mainmodel.predict(input_data_scaled)
    
    return render_template('result.html', p1=pred)

@app.route('/visual')
def visualization_page():
    return render_template('visual.html')

if __name__ == '__main__':
    app.run(debug=True)

