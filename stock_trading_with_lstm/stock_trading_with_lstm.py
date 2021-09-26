import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


class StockTrader:
    def __init__(self, cash=5000):
        self.cash = cash
        self.n_stocks = 0

        self.history = []
    
    def take_action(self, current_price, predicted_price, next_price):
        if predicted_price > current_price:
            self.buy(next_price)
        elif current_price > predicted_price:
            self.sell(next_price)
        else:
            pass

        self.history.append(self.total_net(next_price))

    def buy(self, next_price):
        if self.cash >= next_price:
            self.n_stocks += self.cash // next_price
            self.cash = self.cash % next_price

    def sell(self, next_price):
        if self.n_stocks > 0:
            self.cash += self.n_stocks * next_price
            self.n_stocks = 0 

    def total_net(self, current_price):
        return self.cash + self.n_stocks*current_price


def load_data(path):
    dataset = pd.read_csv(path)
    # get open price of the stock
    dataset = dataset['Open'].values
    dataset = dataset.reshape(-1,1)

    return dataset

def arange_data_for_lstm(dataset, look_back=1):
    n_samples = len(dataset)-look_back
    X = np.empty((n_samples, look_back))
    Y = np.empty((n_samples, 1))

    for i in range(len(dataset)-look_back):
        X[i,:] = dataset[i:i+look_back].flatten()
        Y[i] = dataset[i+look_back]

    return X, Y

def preprocess_data(dataset, look_back=1):
    # normalize data
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)

    # split for train and test sets
    n_samples = len(dataset)
    train_size = int(0.7*n_samples)
    train = dataset[:train_size]
    test = dataset[train_size:]

    # rearange data to be suitable for LSTM network
    X_train, y_train = arange_data_for_lstm(train, look_back)
    X_test, y_test = arange_data_for_lstm(test, look_back)

    # reshape input to be suitable for LSTM network
    # (samples, time_steps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, y_train, X_test, y_test, scaler

def create_model(look_back=1):
    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def plot_loss(history):
    plt.title('Loss graph')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(history['loss'], label='train_loss')
    plt.plot(history['val_loss'], label='validation_loss')
    plt.legend()
    plt.show()

def plot_results(dataset, y_hat_train, y_hat_test):
    plt.title('Real data VS Predicted data')
    plt.xlabel('Days from 12/12/1980 to 9/23/2021')
    plt.ylabel('Price in $')
    plt.plot(dataset, label='real_data')
    plt.plot(y_hat_train, label='predicted_train')
    plt.plot(y_hat_test, label='predicted_test')
    plt.legend()
    plt.show()

def test_with_an_agent(y_hat_test, y_test):
    agent = StockTrader()
    
    for t in range(len(y_test)-1):
        agent.take_action(y_test[t], y_hat_test[t+1], y_test[t+1])
        #print(f'Total net in epsiode {t}: ${round(agent.total_net(y_test[t])[0], 2)}')

    scaler = MinMaxScaler(feature_range=(min(y_test),max(y_test)))
    history = scaler.fit_transform(agent.history)

    buy_n_hold_yield = int((y_test[-1] / y_test[0] - 1) * 100)
    agent_yield = int((agent.history[-1] / agent.history[0] - 1) * 100)

    plt.title('Agent performance scaled to stock price'
             f'\nBuy and Hold  yield: {buy_n_hold_yield}%, Agent yield: {agent_yield}%')
    plt.fill_between(range(len(history)), 0, history.flatten(), color='green', label='agent_net')
    plt.plot(y_test, label='real price')
    plt.plot(y_hat_test, label='predicted price')
    plt.xlabel('Days')
    plt.ylabel('Price in $')
    plt.legend()
    plt.show()
    
def main():
    look_back = 1

    # load data
    dataset = load_data('Assets\AAPL.csv')

    # preprocess data
    X_train, y_train, X_test, y_test, scaler = preprocess_data(dataset, look_back)

    # create model
    model = create_model(look_back)
    # or load model
    #model = load_model('Assets\model.h5')

    # fit model
    history = model.fit(X_train, y_train, batch_size=50, epochs=20, verbose=1, validation_split=0.1)

    # save model
    model.save('Assets\model.h5')

    # evaluate model
    # loss function
    plot_loss(history.history)

    # make predictions
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    # inverse normalization
    y_test = scaler.inverse_transform(y_test)
    y_hat_train = scaler.inverse_transform(y_hat_train)
    y_hat_test_shifted = scaler.inverse_transform(y_hat_test)
    y_hat_test = np.empty(dataset.shape)
    y_hat_test[:] = np.nan
    y_hat_test[len(y_hat_train)+2*look_back:] = y_hat_test_shifted

    # plot results
    plot_results(dataset, y_hat_train, y_hat_test)

    # test with an agent
    test_with_an_agent(y_hat_test_shifted, y_test)

if __name__ == '__main__':
    main()