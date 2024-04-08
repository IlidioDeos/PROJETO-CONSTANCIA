import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Definição da classe LSTMModel
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=4, hidden_size=50, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(50, 50, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.dense = nn.Linear(50, 1)

    def forward(self, x):
        x, (hidden, cell) = self.lstm1(x)
        x = self.dropout1(x)
        x, (hidden, cell) = self.lstm2(x)
        x = self.dropout2(x)
        x = self.dense(x[:, -1, :])
        return x

# Carregar modelo
model_path = 'modelo_constancia_apple_stock_training2.pth'
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Função para baixar e preparar dados
def download_prepare_data(stock, interval, start_date, end_date):
    data = yf.download(stock, start=start_date, end=end_date, interval=interval)
    data['MA_1'] = data['Close'].rolling(window=1).mean()
    data['MA_2'] = data['Close'].rolling(window=2).mean()
    data['MA_7'] = data['Close'].rolling(window=7).mean()
    data['Avg_Open_7min'] = data['Open'].rolling(window=7).mean()
    data['Avg_Open_7min'] = data['Avg_Open_7min'].shift(1)
    data.dropna(inplace=True)
    return data

# Função para fazer previsões
def make_prediction(model, data, scaler):
    if data.empty:
        return None  # Retorna None ou uma mensagem apropriada
    try:
        data_scaled = scaler.transform(data[['Close', 'MA_1', 'MA_7', 'Avg_Open_7min']])
    except ValueError as e:
        print(e)  # Ou use st.error("Mensagem de erro") para exibir no Streamlit
        return None
    x, _ = create_dataset(data_scaled, 7)
    if len(x) == 0:  # Verifica se após a criação do dataset ainda há dados
        return None
    x_tensor = torch.tensor(x).float()
    with torch.no_grad():
        predictions = model(x_tensor).numpy().flatten()
    predictions = scaler.inverse_transform(np.column_stack((predictions, np.zeros((predictions.shape[0], 3)))))[:, 0]
    return predictions

# Função para criar o dataset
def create_dataset(dataset, time_step=7):
    x, y = [], []
    for i in range(time_step, len(dataset)):
        x.append(dataset[i-time_step:i, :])
        y.append(dataset[i, 0])
    return np.array(x), np.array(y)

def evaluate_predictions(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, rmse, r2, mae


# Interface do usuário
st.title('Previsão do Preço de Ações com LSTM')

# Definição de variáveis comuns
interval = '1m'
stock_symbol = 'AAPL'

# Criação das abas
tab1, tab2 = st.tabs(["Comparação valores Históricos", "Prever Futuro"])

with tab1:
    st.subheader("Comparação Valores Reais e Previsões")
    stock = st.text_input('Sigla da Ação (Verificar no YahooFinance)', value=stock_symbol, key='stock1_hist')
    start_date = st.date_input('Data Inicial', key='start1_hist')
    end_date = st.date_input('Data Final', key='end1_hist')
    if st.button('Prever Valores Históricos', key='predict1_hist'):
        data = download_prepare_data(stock, interval, start_date, end_date)
        # Certifique-se de que data está ordenado corretamente aqui
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data[['Close', 'MA_1', 'MA_7', 'Avg_Open_7min']])
        predictions = make_prediction(model, data, scaler)
        actuals = data['Close'].values[-len(predictions):]
        # Calcula a margem de erro (valor absoluto da diferença)
        error_margin = np.abs(actuals - predictions)
        # Preparando o DataFrame para exibição
        df_predictions = pd.DataFrame({
            'horario': data.index[-len(predictions):],  # Índices temporais dos dados
            'valor real': actuals,
            'valor previsto': predictions,
            'margem do erro da previsao': error_margin
        })
        mse, rmse, r2, mae = evaluate_predictions(actuals, predictions)
        
        # Assegure que os dados estejam alinhados corretamente para a plotagem
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(actuals, color='red', label='Real Stock Price')
        ax.plot(predictions, color='blue', label='Predicted Stock Price')
        ax.set_title(f'Real vs Prediction | R²={r2:.4f} | MSE={mse:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Stock Price')
        ax.legend()
        st.pyplot(fig)
        st.dataframe(df_predictions)

with tab2:
    st.subheader("Prever Ações no Futuro")
    stock = st.text_input('Sigla da Ação (Verificar no YahooFinance)', value=stock_symbol, key='stock2_future')
    chosen_date = st.date_input('Escolha a data para prevermos', key='chosen_date')

    def predict_next_minute(stock, chosen_date):
        # Busca os dados dos últimos 7 dias, incluindo o dia escolhido
        start_date = (chosen_date - timedelta(days=7)).strftime('%Y-%m-%d')
        end_date = (chosen_date + timedelta(days=1)).strftime('%Y-%m-%d')  # +1 para incluir o dia inteiro escolhido
        data = yf.download(stock, start=start_date, end=end_date, interval="1m")

        if data.empty:
            st.write("Sem dados disponíveis para os últimos 7 dias.")
            return

        # Preparação dos dados
        data['MA_1'] = data['Close'].rolling(window=1).mean()
        data['MA_7'] = data['Close'].rolling(window=7).mean()
        data['Avg_Open_7min'] = data['Open'].rolling(window=7).mean()
        data.dropna(inplace=True)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data[['Close', 'MA_1', 'MA_7', 'Avg_Open_7min']])
        x, _ = create_dataset(data_scaled, 7)
        x_tensor = torch.tensor(x).float()

        if len(x_tensor) == 0:
            st.write("Não há dados suficientes para realizar a previsão.")
            return

        with torch.no_grad():
            predictions = model(x_tensor).numpy().flatten()
            predictions = scaler.inverse_transform(np.column_stack((predictions, np.zeros((predictions.shape[0], 3)))))[:, 0]

        if len(predictions) > 0:
            # st.write("Valor previsto para o próximo minuto:", predictions[-1])

            # Geração de horários fictícios para as previsões
            prediction_times = pd.date_range(start=datetime.now(), periods=len(predictions), freq='T')

            # Preparação dos dados para plotagem
            df_predictions = pd.DataFrame({
                'Horário': prediction_times,
                'Valor previsto': predictions
            })

            # Plotagem
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_predictions['Horário'], df_predictions['Valor previsto'], color='blue', label='Preço Previsto')
            ax.set_title('Previsões de Preço de Ações para o Próximo Minuto')
            ax.set_xlabel('Horário')
            ax.set_ylabel('Preço da Ação')
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            st.dataframe(df_predictions)

        else:
            st.write("Não foi possível realizar a previsão com os dados fornecidos.")

    if st.button('Prever próximo minuto', key='predict_next_minute'):
        predict_next_minute(stock, chosen_date)

