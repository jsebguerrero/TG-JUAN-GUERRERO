from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def find_vel_and_pos(df):
    """
    Reads a DataFrame, and calculates velocity and position for X, Y and Z
    :param df: DataFrame
    :return: DataFrame
    """
    # velocity
    df["vX [m/s]"] = numerical_integration(df[["aX [m/s^2]", "Timestamp [ns]"]])
    df["vY [m/s]"] = numerical_integration(df[["aY [m/s^2]", "Timestamp [ns]"]])
    df["vZ [m/s]"] = numerical_integration(df[["aZ [m/s^2", "Timestamp [ns]"]])
    # position
    df["rX [m]"] = numerical_integration(df[["vX [m/s]", "Timestamp [ns]"]])
    df["rY [m]"] = numerical_integration(df[["vY [m/s]", "Timestamp [ns]"]])
    df["rZ [m]"] = numerical_integration(df[["vZ [m/s]", "Timestamp [ns]"]])

    return df


def numerical_integration(data_frame):
    """
    Reads a two column DataFrame, then computes the product of column 1 with the difference between column 2
    and column 2 shifted backwards
    :param data_frame:
    :return: Series
    """
    integrated = data_frame.iloc[:, 0] * (data_frame.iloc[:, 1].shift(-1, fill_value=0) - data_frame.iloc[:, 1])
    return integrated.shift(-1, fill_value=0) + integrated


def resample(data_frame, fs, columns):
    # Fill NaNs with the mean
    data_frame = data_frame.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
    # Set parameters and create X axis arrays
    lower_limit = len(data_frame.index)
    x = np.arange(0, lower_limit)
    xnew = np.arange(0, lower_limit, 1 / fs)
    kind = 'quadratic'
    fill_value = "extrapolate"
    new_data = []
    for column in data_frame.columns:
        f_column = interpolate.interp1d(x, data_frame[column], kind=kind, fill_value=fill_value)
        # plots for the article
        # plt.style.use('seaborn-pastel')
        # plt.plot(x[0:200], data_frame[column][0:200], 'o', xnew[0:10000], f_column[0:10000], '-')
        # plt.xlabel('samples')
        # plt.ylabel('raw amplitude')
        # Resample and transform
        new_data.append(f_column(xnew))
        # Reset dataframe and put new values in a new one
    return pd.DataFrame(np.array(new_data).reshape(len(xnew), len(new_data)), columns=columns)


def filter_gravity(data_frame):
    return (data_frame - data_frame.mean(axis=0)) / data_frame.max(axis=0)
