import pandas as pd
import numpy as np
import os
from preprocessing.preprocessing_functions import segmentate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

overlap = 62
window = 125


def get_channeled_data_from_dir(directory, overlap, window, usecols=None, sep=','):
    d = [[], [], []]
    for file in os.listdir(directory):
        df = pd.read_csv(os.path.join(directory, file), sep=sep, usecols=usecols)
        try:
            result = segmentate(df, overlap, window)
            for i in range(len(d)):
                d[i].extend(result[i])
        except:
            print('not parsed ', file)
            pass
    return np.array(d, dtype=np.float16)


if __name__ == "__main__":
    main_df = pd.DataFrame()
    hc_dir_l = os.path.join(os.getcwd(), 'hc', 'l')
    hc_dir_r = os.path.join(os.getcwd(), 'hc', 'r')
    pd_dir = os.path.join(os.getcwd(), 'pd')
    hclfoot = get_channeled_data_from_dir(hc_dir_l, overlap, window)
    hcrfoot = get_channeled_data_from_dir(hc_dir_r, overlap, window)
    pdlfoot = get_channeled_data_from_dir(pd_dir, overlap, window, sep=' ',
                                          usecols=['accXLeft', 'accYLeft', 'accZLeft'])
    pdrfoot = get_channeled_data_from_dir(pd_dir, overlap, window, sep=' ',
                                          usecols=['accXRight', 'accYRight', 'accZRight'])
    channels = np.hstack((hclfoot, hcrfoot, pdlfoot, pdrfoot))
    channels = np.moveaxis(channels, [0, 1, 2], [1, 0, 2])
    # Reshape for loading into Pytorch models
    X_train, X_test = train_test_split(channels, test_size=0.33, random_state=42)
    scaler1 = MinMaxScaler((-1,1))
    scaler2 = MinMaxScaler((-1,1))
    scaler3 = MinMaxScaler((-1,1))
    X_train = np.moveaxis(X_train, [0, 1, 2], [1, 0, 2])
    X_test = np.moveaxis(X_test, [0, 1, 2], [1, 0, 2])
    for i in range(len(X_train)):
        if i == 0:
            X_train[i] = scaler1.fit_transform(X_train[i])
        elif i == 1:
            X_train[i] = scaler2.fit_transform(X_train[i])
        else:
            X_train[i] = scaler3.fit_transform(X_train[i])
    for i in range(len(X_test)):
        if i == 0:
            X_test[i] = scaler1.transform(X_test[i])
        elif i == 1:
            X_test[i] = scaler2.transform(X_test[i])
        else:
            X_test[i] = scaler3.transform(X_test[i])
    X_train = np.moveaxis(X_train, [0, 1, 2], [1, 0, 2])
    X_test = np.moveaxis(X_test, [0, 1, 2], [1, 0, 2])
    np.save('../train.pickle', X_train, allow_pickle=True)
    np.save('../test.pickle', X_test, allow_pickle=True)
