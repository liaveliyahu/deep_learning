import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def create_data():
    N = 1000
    x = 6 * np.random.random((N,2)) - 3 # uniformly distributed between (-3, +3)
    y = np.cos(2*x[:,0]) + np.cos(3*x[:,1])
    y = y.reshape(-1, 1)

    x_train = x[:int(0.7*N)]
    x_test = x[int(0.7*N):]
    y_train = y[:int(0.7*N)]
    y_test = y[int(0.7*N):]

    return x_train, x_test, y_train, y_test

def plot_data(x, y):
    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(x[:,0], x[:,1], y, label='data', marker='o')

    plt.title('3D Data Plot')
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_zlabel('y')
    plt.legend()
    plt.show()

def build_model(output_shape):
    model = Sequential()
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer=Adam(0.01), loss='mse')
    return model

def plot_loss(r):
    plt.title('Model loss')
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_prediction(x, y, model):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:,0], x[:,1], y)

    line = np.linspace(-3, 3, 50)
    xx, yy = np.meshgrid(line, line)
    x_grid = np.vstack((xx.flatten(), yy.flatten())).T
    y_hat = model.predict(x_grid).flatten()

    ax.plot_trisurf(x_grid[:,0], x_grid[:,1], y_hat, linewidth=0.1, antialiased=True, color='red')

    blue_patch = mpatches.Patch(label='data')
    red_patch = mpatches.Patch(color='red', label='prediction')
    
    plt.title('3D Data and Prediction Plot')
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_zlabel('y')
    plt.legend(handles=[blue_patch, red_patch])

    plt.show()


def main():
    # create data
    x_train, x_test, y_train, y_test = create_data()

    # plot data 
    plot_data(x_train, y_train)

    # build model
    model = build_model(y_train.shape[1])

    # fit model
    r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=200, verbose=1)

    # plot loss
    plot_loss(r)

    # evaluate model
    plot_prediction(x_test, y_test, model)


if __name__ == '__main__':
    main()