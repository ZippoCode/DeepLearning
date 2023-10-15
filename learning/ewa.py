import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd


def load_data(year: int) -> pandas.DataFrame:
    filename = os.path.join('../datasets', "FRPARIS.txt")
    dataframe = pd.read_fwf(filename, names=['Month', 'Day', 'Year', 'Temperature'])

    dataframe = dataframe[dataframe.Year == year].copy()
    dataframe[dataframe.Temperature < 0] = 58.0
    dataframe["Temperature"] = (dataframe["Temperature"] - 32) * 5 / 9
    dataframe.reset_index(inplace=True)

    days = [i for i in range(1, dataframe.shape[0] + 1)]
    dataframe.drop(columns=["Days"], axis=1, inplace=True, errors="ignore")
    dataframe.insert(0, "Days", days)
    dataframe["Date"] = pd.date_range(start="1/1/2019", end="12/31/2019")

    return dataframe


def ewa(values: np.ndarray, beta=0.5):
    length_ewa = values.shape[0]
    w_t = np.zeros((length_ewa, 1))
    for i in range(1, length_ewa):
        w_t[i] = beta * w_t[i - 1] + (1 - beta) * values[i]
    return w_t


data = load_data(year=2019)
days_vector = np.array(range(0, 91))

beta = [0.98, 0.9, 0.5]
exponential_vector_one = np.power(beta[0], days_vector)
exponential_vector_two = np.power(beta[1], days_vector)
exponential_vector_three = np.power(beta[2], days_vector)

fig, (axis1, axis2, axis3) = plt.subplots(3, figsize=(16, 16))

axis1.title.set_text("Exponential Weights")
axis1.invert_xaxis()
axis1.plot(days_vector, exponential_vector_two, 'o', color='limegreen', label=r"$\beta = 0.9$")
axis1.plot(days_vector, exponential_vector_one, 'o', color='red', label=r"$\beta = 0.98$")
axis1.plot(days_vector, exponential_vector_three, 'o', color='blue', label=r"$\beta = 0.5$")
axis1.legend()

axis2.title.set_text("Temperature")
date = data.Date[0:91]
temperature = data.Temperature[0:91]

axis2.set_xlim(date.min(), date.max())
axis2.plot(date, temperature, '.', alpha=0.8, color='g', label="Temperature $^\circ$C")
axis2.legend()

axis3.title.set_text(r'Exponentially Weighted Average ($\beta = 0.9, 0.98, 0.5$)')
ewa_0_9 = ewa(temperature, beta=0.9)
ewa_0_98 = ewa(temperature, beta=0.98)
ewa_0_5 = ewa(temperature, beta=0.5)
axis3.plot(days_vector, temperature, 'o', color='black', alpha=0.3, label="Temperature $^\circ$C")
axis3.plot(days_vector, ewa_0_9, color='limegreen', linewidth=2, label="EWA with Beta = 0.9")
axis3.plot(days_vector, ewa_0_98, color='red', linewidth=2, label="EWA with Beta = 0.98")
axis3.plot(days_vector, ewa_0_5, color='blue', linewidth=2, label="EWA with Beta = 0.5")

axis3.legend()

axis3.set(xlabel='day', ylabel=r'$W_t$')
plt.show()
