from flask import Flask, render_template, request
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from datetime import datetime
import io
import base64

app = Flask(__name__)

# Mendapatkan API Key dari Tiingo
api_key = '60b72460f66eed7ec99a062003fe6a87bb8e94bb'

def get_historical_data(ticker, start_date, end_date):
    url = f'https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&endDate={end_date}&token={api_key}'
    response = requests.get(url)
    data = response.json()
    return data

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        historical_data = get_historical_data(ticker, start_date, end_date)
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        df['timestamp'] = df.index.map(datetime.timestamp)
        X = df['timestamp'].values.reshape(-1, 1)
        y = df['close'].values

        model = LinearRegression()
        model.fit(X, y)
        df['trend'] = model.predict(X)

        # Menghitung metrik evaluasi
        mae = mean_absolute_error(y, df['trend'])
        mse = mean_squared_error(y, df['trend'])
        r2 = r2_score(y, df['trend'])

        img = io.BytesIO()
        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df['close'], label='Harga Penutupan')
        plt.plot(df.index, df['trend'], label='Tren', linestyle='--')
        plt.xlabel('Tanggal')
        plt.ylabel('Harga Penutupan (USD)')
        plt.title(f'Analisis Tren Harga Saham {ticker}')
        plt.legend()
        plt.grid(True)
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('index.html', plot_url=plot_url, mae=mae, mse=mse, r2=r2)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
