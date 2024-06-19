import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 주식 데이터 가져오기
ticker = 'AAPL'  # 예시로 Apple 주식 데이터를 사용합니다
start_date = '2022-06-01'
end_date = '2024-06-01'
data = yf.download(ticker, start=start_date, end=end_date)

# 종가(Close) 데이터만 사용합니다
prices = data['Close'].values.reshape(-1, 1)

# 데이터 정규화 (0과 1 사이의 값으로 스케일링)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

# 입력 데이터와 타겟 데이터 생성 함수
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# 과거 60일치 데이터를 사용해 예측
time_step = 60
X, y = create_dataset(scaled_data, time_step)

# 데이터를 훈련(train) 데이터와 테스트(test) 데이터로 나눕니다
X_train, X_test = X[:len(X)-30], X[len(X)-30:]
y_train, y_test = y[:len(y)-30], y[len(y)-30:]

# 선형 회귀 모델 사용 예시
model = LinearRegression()
model.fit(X_train, y_train)

# 향후 30일간의 예측
future_predictions = []
last_data = X[-1].reshape(1, -1)  # 가장 최근의 데이터를 시작점으로 설정
for _ in range(30):
    pred = model.predict(last_data)[0]
    future_predictions.append(pred)
    last_data = np.append(last_data[:, 1:], pred).reshape(1, -1)

# 예측값을 실제 주가 단위로 역변환
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

# 향후 30일간의 예측 값을 데이터프레임으로 정리
future_dates = pd.date_range(start=data.index[-1], periods=30, freq='B')
future_data = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
future_data.set_index('Date', inplace=True)

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(data.index[-100:], prices[-100:], label='Actual Prices')
plt.plot(future_data.index, future_data['Predicted Price'], label='Predicted Prices', marker='o')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Predicted Stock Prices for Next 30 Days')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
