# Relatório Projeto Concorde

---

### Objetivo : Desenvolver um modelo preditivo para antecipar os preços das ações

---

### Processo


1. **Pesquisa e Análise**: Focamos em compreender os modelos de previsão financeira, com especial atenção às RNNs e LSTMs. Esta fase foi essencial para selecionarmos a abordagem mais promissora.
2. **Coleta e Preparação de Dados**: Os dados históricos, fundamentais para o treinamento e teste do modelo, foram cuidadosamente coletados e preparados. Este processo envolveu a normalização dos dados, o tratamento de valores ausentes e a segmentação em conjuntos de dados de treinamento e teste.
3. **Implementação do Modelo LSTM**: Empregamos Python e PyTorch para desenvolver a arquitetura do modelo, que incluiu camadas LSTM e dropout para minimizar o risco de overfitting. O modelo foi treinado com os dados preparados, seguido de ajustes nos hiperparâmetros para otimizar seu desempenho.
4. **Avaliação e Ajustes**: O modelo foi avaliado usando um conjunto de teste. Com base nessa avaliação, fizemos ajustes iterativos no modelo para aprimorar a precisão das previsões.


# **Protótipo de nosso modelo**

---

[**PROJETO CONCORDE - CONSTANCIA**](https://projeto-constancia-alpha-v1.streamlit.app/)

---

O código em questão desenvolve uma aplicação web utilizando a biblioteca Streamlit, destinada a prever preços de ações utilizando um modelo de aprendizado de máquina baseado em redes neurais recorrentes, especificamente o modelo Long Short-Term Memory (LSTM). Vamos destrinchar cada parte do código para que possa ser compreendido por um público não técnico e profissionais iniciantes na área de Ciência de Dados e Inteligência Artificial.

## Importações

```python
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

```

Aqui importamos as bibliotecas necessárias para executar nossa aplicação. Cada uma tem um propósito específico:

- `streamlit`: Cria interfaces web facilmente.
- `torch` e `torch.nn`: Usadas para definir e treinar modelos de redes neurais.
- `numpy` e `pandas`: Manipulação e análise de dados.
- `yfinance`: Baixar dados financeiros do Yahoo Finance.
- `matplotlib.pyplot`: Visualização de dados.
- `sklearn.preprocessing.MinMaxScaler`: Normalização de dados.
- `datetime` e `timedelta`: Manipulação de datas e tempos.
- `sklearn.metrics`: Métricas de avaliação do modelo.

## Definição da Classe LSTMModel

```python
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        ...

```

Esta seção define a arquitetura do modelo LSTM. A rede é composta por duas camadas LSTM seguidas de camadas de dropout para regularização (redução de overfitting) e uma camada densa no final para previsão do preço. A escolha do LSTM deve-se à sua capacidade de capturar dependências temporais em séries temporais, essencial para prever preços de ações.

## Carregamento do Modelo

```python
model_path = 'modelo_constancia_apple_stock_training2.pth'
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

```

Carrega um modelo previamente treinado e o coloca em modo de avaliação, preparando-o para fazer previsões sem alterar seus pesos.

## Funções Auxiliares

- `download_prepare_data`: Baixa e prepara os dados para serem utilizados pelo modelo.
- `make_prediction`: Faz previsões com o modelo a partir dos dados preparados.
- `create_dataset`: Transforma os dados em um formato adequado para entrada no modelo LSTM.
- `evaluate_predictions`: Calcula e retorna métricas de avaliação das previsões.

Essas funções tratam do processamento de dados e da interação com o modelo, desde a obtenção dos dados até a avaliação das previsões.

## Interface do Usuário com Streamlit

```python
st.title('Previsão do Preço de Ações com LSTM')
...

```

Utilizando a biblioteca Streamlit, cria-se uma interface de usuário interativa para a aplicação. A interface permite ao usuário inserir símbolos de ações, selecionar datas e fazer previsões de preços históricos ou futuros. A interface é dividida em abas para organizar as funcionalidades.

---

# Próximos Passos

Para levar nosso projeto ao próximo nível e garantir previsões ainda mais precisas e confiáveis, identificamos várias áreas para futuras melhorias:

- **Validação Cruzada Temporal**: Aplicaremos técnicas de validação cruzada específicas para séries temporais, para uma avaliação mais rigorosa do modelo.
- **Ajuste de Hiperparâmetros**: Experimentaremos diferentes configurações de hiperparâmetros para otimizar o desempenho do nosso modelo LSTM.
- **Engenharia de Recursos**: Incluir outros indicadores técnicos e dados externos, como notícias do mercado e indicadores econômicos, para enriquecer nosso modelo.
- **Diversificação de Dados**: Expandir nossa base de dados para incluir informações de múltiplas fontes, visando construir um modelo mais robusto.