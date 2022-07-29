# ML part based on NeuralNine tutorials

# For ML
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM

# For App and Tkinter/Matplotlib graph
from tkinter import *
from pandas import DataFrame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class App():
    def __init__(self):
        self.root = Tk()
        self.root.title('Predicting stock prices')

        self.root.minsize(800, 800)

        self.logo = Label(self.root, text='Predicting stock prices', font=("Poppins", 22, "bold"))
        self.logo.pack(pady=5)

        self.descriptionLabel = Label(self.root, text="Program is predicting tomorrow's stock price of any company based on last given number of days", font=("Poppins", 15))
        self.descriptionLabel.pack(pady=0)

        self.spacer1 = Label(self.root, text="", font=("Poppins", 5))
        self.spacer1.pack(pady=2)

        # input
        self.typecompanyStockSymbol = Label(self.root, text="Type any company stock symbol like FB/GOOG", font=("Poppins", 15, "bold"))
        self.typecompanyStockSymbol.pack(pady=6)

        self.companyStockSymbol = Entry(self.root, font=("Poppins", 15, "bold"))
        self.companyStockSymbol.pack(pady=0)

        self.typeHowManyDays = Label(self.root, text="Type on how many days from the past you want to predict (60 by default)", font=("Poppins", 14, "bold"))
        self.typeHowManyDays.pack(pady=6)

        self.howManyDays = Entry(self.root, font=("Poppins", 14, "bold"))
        self.howManyDays.pack(pady=0)
        # end of input

        self.getcompanyStockSymbol = Button(self.root, text="Predict", command=self.loadData, font=("Poppins"))
        self.getcompanyStockSymbol.pack(pady=5)

        self.spacer2 = Label(self.root, text="", font=("Poppins", 5))
        self.spacer2.pack(pady=2)

        self.info = Label(self.root, text="Wait a while for model to train and make prediction", font=("Poppins", 14))
        self.info.pack(pady=5)

        self.f = plt.Figure(figsize=(5,4), dpi=100)
        self.line = FigureCanvasTkAgg(self.f, self.root)
        self.line.get_tk_widget().pack(expand=True)

        self.predictedPrice = Label(self.root, text="Predicted price for tomorrow: type needed informations to run training", font=("Poppins", 18, "bold"))
        self.predictedPrice.pack(pady=10)

        self.disclaimer = Label(self.root, text="This is a simple model which can be often incorrect so I don't recommend to use it for real trading purposes", font=("Poppins", 14))
        self.disclaimer.pack(pady=10)
        
        self.typecompanyStockSymbol.focus()

        self.root.mainloop()

    def updateGraph(self):
        day_delta = dt.timedelta(days=1)
        end_date = dt.date.today()
        start_date = end_date - len(self.actual_prices)*day_delta

        data = {
         'X': [start_date + i*day_delta for i in range((end_date - start_date).days)],
         'Y': self.actual_prices
         }

        self.f.clear()

        self.ax=self.f.add_subplot(111)
        df=DataFrame(data,columns=["X", "Y"])
        df = df[['X','Y']].groupby('X').sum()
        df.plot(kind='line', ax=self.ax, color='blue', fontsize=7)
        self.ax.set_title(self.company + ' stock price graph from last ' + str(self.prediction_days) + ' days')

        self.line.draw_idle()

    def loadData(self):
        self.company = self.companyStockSymbol.get()
        if self.company!="":
            try:
                try:
                    self.prediction_days = int(self.howManyDays.get())
                except:
                    self.prediction_days = 60

                # Preparing data for training purposes
                start = dt.datetime(2012,1,1)
                end = dt.datetime.today() - timedelta(days=self.prediction_days)

                data = web.DataReader(self.company, 'yahoo', start, end)

                scaler = MinMaxScaler(feature_range=(0,1))
                scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

                x_train = []
                y_train = []

                for x in range(self.prediction_days,len(scaled_data)):
                   x_train.append(scaled_data[x-self.prediction_days:x,0]) # Adding different arrays of len of self.prediction_days
                   y_train.append(scaled_data[x,0]) # Adding next item as a proper answer

                x_train, y_train = np.array(x_train), np.array(y_train)
                x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

                # Building model
                model = Sequential()
                model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
                model.add(Dropout(0.2))
                model.add(LSTM(units=50, return_sequences=True))
                model.add(Dropout(0.2))
                model.add(LSTM(units=50))
                model.add(Dropout(0.2))
                model.add(Dense(units=1)) # 1 neuron for the output - prediction of the next day's closing value
                model.compile(optimizer='adam', loss='mean_squared_error')

                model.fit(x_train,y_train, epochs=25, batch_size=32)

                # Preparing data for testing - data which model did not see during the training process
                test_end = dt.datetime.today()
                test_start = dt.datetime.today() - timedelta(days=self.prediction_days)

                test_data = web.DataReader(self.company, 'yahoo', test_start, test_end)

                self.actual_prices = test_data['Close'].values

                total_dataset = pd.concat((data['Close'],test_data['Close']),axis=0)

                model_inputs = total_dataset[len(total_dataset)-len(test_data)-self.prediction_days:].values
                model_inputs = model_inputs.reshape(-1,1)
                model_inputs = scaler.transform(model_inputs)

                # Predicting next day
                real_data = [model_inputs[len(model_inputs) - self.prediction_days : len(model_inputs)+1, 0]]
                real_data = np.array(real_data)
                real_data = np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))

                prediction = model.predict(real_data)
                prediction = scaler.inverse_transform(prediction)

                self.predictedPrice.configure(text='Predicted price for tomorrow: ' + str(round(prediction[0][0], 2)))
                self.updateGraph()

            except Exception as e:
                print(e)

App()