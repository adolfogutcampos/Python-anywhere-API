import os
import pandas as pd
import pickle
import sqlite3
from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

# Cargar el modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(DATA_DIR, 'advertising_model')
DATABASE = os.path.join(DATA_DIR, 'advertising.db')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Conectar a la base de datos
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# 1- Endpoint para predecir

@app.route('/v2/predict', methods=['GET'])
def predict():
    tv = request.args.get('tv', type=float)
    radio = request.args.get('radio', type=float)
    newspaper = request.args.get('newspaper', type=float)

    if tv is None or radio is None or newspaper is None:
        return "Missing parameters", 400

    prediction = model.predict([[tv, radio, newspaper]])
    return jsonify({'prediction': prediction[0]})

# 2- Endpoint para ingestar datos

@app.route('/v2/ingest_data', methods=['POST'])
def ingest_data():
    tv = request.args.get('tv', type=float)
    radio = request.args.get('radio', type=float)
    newspaper = request.args.get('newspaper', type=float)
    sales = request.args.get('sales', type=float)

    conn = get_db_connection()
    cursor = conn.cursor()

    query = '''INSERT INTO campañas VALUES(?, ?, ?, ?)'''
    query_2 = '''SELECT * FROM campañas'''

    cursor.execute(query, (tv, radio, newspaper, sales))
    result = cursor.execute(query_2).fetchall()

    conn.commit()
    conn.close()
    return "Data ingested", 201

# 3- Endpoint para reentrenar el modelo

@app.route('/v2/retrain', methods=['PUT'])
def retrain():

    connection = sqlite3.connect('data/advertising.db')
    cursor = connection .cursor()

    query = '''SELECT * FROM campañas'''
    data = cursor.execute(query).fetchall()

    df = pd.DataFrame(data, columns= ['TV', 'radio', 'newspaper', 'sales'])
    df.dropna(inplace= True)

    X = df.drop(columns= 'sales')
    Y = df['sales']

    with open(MODEL_PATH, 'rb') as f:
        current_model = pickle.load(f)

    current_predictions = current_model.predict(X)
    current_mae = ((current_predictions - Y) ** 2).mean()

    new_model = LinearRegression()
    new_model.fit(X, Y)

    new_predictions = new_model.predict(X)
    new_mae = ((new_predictions - Y) ** 2).mean()

    if current_mae > new_mae:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(new_model, f)
        return jsonify({'message': 'Model updated', 'old_mae': current_mae, 'new_mae': new_mae})
    else:
        return jsonify({'message': 'Keep old model', 'old_mae': current_mae, 'new_mae': new_mae})

#if __name__ == '__main__':
 #   app.run()