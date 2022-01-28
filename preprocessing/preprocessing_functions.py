from scipy import signal


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


def resample(data_frame, fs):
    return data_frame.apply()


def filter_gravity(data_frame):
    return (data_frame - data_frame.mean(axis=0)) / data_frame.max(axis=0)
