import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 데이터 가져오기
ticker = 'AAPL'  # 종목 티커를 입력하세요
data = yf.download(ticker, start='2022-06-01', end='2024-06-01')
prices = data['Close'].values.reshape(-1, 1)
print(prices)