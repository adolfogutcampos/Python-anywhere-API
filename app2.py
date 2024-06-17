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

# Endpoint para predecir
@app.route('/v2/predict', methods=['GET'])
def predict():
    tv = request.args.get('tv', type=float)
    radio = request.args.get('radio', type=float)
    newspaper = request.args.get('newspaper', type=float)
    
    if tv is None or radio is None or newspaper is None:
        return "Missing parameters", 400
    
    prediction = model.predict([[tv, radio, newspaper]])
    return jsonify({'prediction': prediction[0]})

# Endpoint para ingestar datos
@app.route('/v2/ingest_data', methods=['POST'])
def ingest_data():
    data = request.get_json()
    if not data:
        return "Invalid data", 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO advertising (tv, radio, newspaper, sales) VALUES (?, ?, ?, ?)',
                   (data['tv'], data['radio'], data['newspaper'], data['sales']))
    conn.commit()
    conn.close()
    return "Data ingested", 201

# Endpoint para reentrenar el modelo
@app.route('/v2/retrain', methods=['PUT'])
def retrain():
    conn = get_db_connection()
    df = pd.read_sql_query('SELECT * FROM advertising', conn)
    conn.close()

    X = df[['tv', 'radio', 'newspaper']]
    y = df['sales']

    global model
    model = LinearRegression()
    model.fit(X, y)

    # Guardar el nuevo modelo
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    return "Model retrained", 200

#if __name__ == '__main__':
 #   app.run()