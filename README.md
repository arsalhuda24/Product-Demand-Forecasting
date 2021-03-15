# Product Sale Forecasting 
The goal of this proect is to build a Realtime multivariate sales forecasting model using deep learning techniques which can be deployed into production using MLOps infrastructure. This practice will help C-Level executives to initiate business strategies. 

![Model](https://github.com/arsalhuda24/Product-Demand-Forecasting/blob/main/model_deployment.png)

## Objective
In this repo we will build a multivariate time series model using different machine/deep learning techniques to forecast multiple products in different stores. 

## Time Series Analysis
The data used here is taken from Kaggle's Store Item Demand Forecasting Challenge. Below you see a snap shot example of sales of item2 from store 1. 

![Model](https://github.com/arsalhuda24/Product-Demand-Forecasting/blob/main/Trend.png)

## Exploratory Data Analysis (EDA) 
### Yearly growth of sales per store

![Model](https://github.com/arsalhuda24/Product-Demand-Forecasting/blob/main/yearly_growth_store.png)


## Data Preperation 
### Sliding Window Method

![Model](https://github.com/arsalhuda24/Product-Demand-Forecasting/blob/main/sliding_window.png)



## Modeling 
### 1) LSTM-Autoencoder 
This is a self-supervised learning technique that can learn a compact representation of data. In this case LSTM network is organized into an encoder-decoder architecture which takes an input sequnce and encoded into a context vector (hidden and cell states). The decoder then takes this context vector as an input and produces an output sequence

![Model](https://github.com/arsalhuda24/Product-Demand-Forecasting/blob/main/lstm_autoencoder.png)
