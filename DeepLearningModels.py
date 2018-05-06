from keras.layers import Dense, Flatten, Input, Dropout, LSTM, SimpleRNN
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from IPython.display import clear_output
from matplotlib import pyplot as plt
import keras
import time
import numpy as np


def sine_wave(nsamples=50, repeat=1, amplitude=1, periods=1, phase=0):
    wave = []
    for _ in range(repeat):
        for x in range(nsamples):
            step = amplitude * np.sin(2 * np.pi * periods / nsamples * x + phase)
            wave.append(step)
    return np.array(wave)


class PlotLosses(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.logs = []
        self.fig = plt.figure()
        self.val_losses = []
        self.losses = []
        self.x = []
        self.i = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()


class KerasModels:

    def __init__(self, batch_size, x_shape, y_shape, callback_info=True, learning_rate=0.01, val_split=0.1,
                 option='loss',
                 save_model=False, realtime_plot=False):
        self.val_split = val_split
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.X_shape = x_shape
        self.Y_shape = y_shape
        self.option = option
        self.save_model = save_model
        self.model_name = ''
        self.realtime_plot = realtime_plot

        self.callback = None
        if callback_info:
            self.callback = self.callback_list()
        self.model = None
        self.fit_time = 0
        self.nr_layers = 0
        self.units = 0

    def plot_training(self):

        plt.subplot(211)
        plt.title('Loss ')
        plt.plot(self.model.history.history['val_loss'], label='val_loss')
        plt.plot(self.model.history.history['loss'], label='loss')
        plt.legend()

        plt.subplot(212)
        plt.title('Acc ')
        plt.plot(self.model.history.history['val_acc'], label='val_acc')
        plt.plot(self.model.history.history['acc'], label='acc')

        plt.legend()
        plt.tight_layout()
        plt.show()

    def fit(self, data_x, data_y, epochs=300):
        start = time.time()
        self.model.fit(data_x, data_y
                       , epochs=epochs
                       , batch_size=self.batch_size
                       , verbose=0
                       , callbacks=self.callback
                       , validation_split=self.val_split)
        end = time.time()
        self.fit_time = int(end - start)

    def predict(self, data_x):
        return self.model.predict(data_x, batch_size=1)

    def load_model(self):
        pass

    def callback_list(self):
        if self.option == 'loss':
            monitor = 'val_loss'
            mode = 'min'
        else:
            monitor = 'val_acc'
            mode = 'max'

        early_stop = EarlyStopping(monitor=monitor, patience=70, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=20, factor=0.2, min_lr=0.0001)

        callback_list = [early_stop, reduce_lr]

        if self.save_model:
            check_point = ModelCheckpoint(self.model_name, monitor=monitor, verbose=0, save_best_only=True, mode=mode)
            callback_list.extend([check_point])

        if self.realtime_plot:
            plot_losses = PlotLosses()
            callback_list.extend([plot_losses])
        return callback_list

    def plot_predict(self, first_prediction, time_forward, len_range=100):
        plot_model = []
        predicted = first_prediction

        for i in range(len_range):
            predicted = predicted.reshape(1, 1, time_forward)
            predicted = self.model.predict(predicted)
            plot_model.extend(predicted[0])
        return plot_model

    def model_information(self):
        model_info = '{0}\n{1} layers with {2} units (total {3} params) \nTraining took {4}s '
        model_info = model_info.format(self.model_name, self.nr_layers
                                       , self.units, self.model.count_params(), self.fit_time, sep='')
        return model_info


class ModelFullyConnected(KerasModels):

    def create_model(self, nr_layers=3, units=20, model_name='Fully_connected'):
        self.nr_layers = nr_layers
        self.units = units
        self.model_name = model_name.format(nr_layers)

        # Input
        inputs = Input(batch_shape=(self.batch_size, self.X_shape[1], self.X_shape[2]))

        x = inputs
        for _ in range(nr_layers):
            x = Dense(units)(x)
            x = Dropout(0.1)(x)
        x = Flatten()(x)

        # Output
        predictions = Dense(self.Y_shape[1])(x)

        # Model
        model = Model(inputs=[inputs], outputs=predictions)
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse', metrics=['accuracy'])
        self.model = model


class ModelRNN(KerasModels):

    def create_model(self, nr_layers=3, units=20, model_name='RNN'):
        self.nr_layers = nr_layers
        self.units = units
        self.model_name = model_name.format(nr_layers)

        # Input
        inputs = Input(batch_shape=(self.batch_size, self.X_shape[1], self.X_shape[2]))

        x = inputs
        for _ in range(nr_layers - 1):
            x = SimpleRNN(units, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)(x)
        x = SimpleRNN(units, dropout=0.1, recurrent_dropout=0.1)(x)

        # Output
        predictions = Dense(self.Y_shape[1])(x)

        # Model
        model = Model(inputs=[inputs], outputs=predictions)
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse', metrics=['accuracy'])

        self.model = model


class ModelLSTM(KerasModels):

    def create_model(self, nr_layers=3, units=20, model_name='LSTM'):
        self.nr_layers = nr_layers
        self.units = units
        self.model_name = model_name.format(nr_layers)

        # Input
        inputs = Input(batch_shape=(self.batch_size, self.X_shape[1], self.X_shape[2]))

        x = inputs
        for _ in range(nr_layers - 1):
            x = LSTM(units, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)(x)
        x = LSTM(units, dropout=0.1, recurrent_dropout=0.1)(x)

        # Output
        predictions = Dense(self.Y_shape[1])(x)

        # Model
        model = Model(inputs=[inputs], outputs=predictions)
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse', metrics=['accuracy'])

        self.model = model
