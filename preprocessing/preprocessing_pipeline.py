import pandas as pd
import os
from preprocessing import preprocessing_functions
from sklearn.preprocessing import MinMaxScaler
import skimage

def pipeline_csv(input_dir, output_dir, columns, skiprows=0, sep=',', fs=50):
    # Check whether the processed dir exists or not
    isExist = os.path.exists(output_dir)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(output_dir)
        print("Created directory for processing data!")
    # Iterate over each file
    for file in os.listdir(input_dir):
        df = pd.read_csv(os.path.join(input_dir, file),
                         usecols=columns,
                         sep=sep,
                         skiprows=skiprows
                         )
        # Subsample to 50Hz
        df = preprocessing_functions.resample(data_frame=df, fs=fs, columns=columns)
        # Filter for gravity
        df = preprocessing_functions.filter_gravity(data_frame=df)
        # Scale down everything to -1 to 1, and assign class label
        scaler = MinMaxScaler((-1, 1))
        df[df.columns] = scaler.fit_transform(df[df.columns])
        # Save again
        df.to_csv(os.path.join(output_dir, file))


def pipeline_split_data(directory, window_size, overlap):
    X = [[], [], []]
    y = [[], [], []]
    for file in os.listdir(directory):
        df = pd.read_csv(os.path.join(directory, file))
        X[0].append(skimage.util.view_as_windows(df.iloc[:, 0].values, window_size, step=overlap))
        y[0].append([1 if 'PD' in file else 0])
        X[1].append(skimage.util.view_as_windows(df.iloc[:, 1].values, window_size, step=overlap))
        y[1].append([1 if 'PD' in file else 0])
        X[2].append(skimage.util.view_as_windows(df.iloc[:, 2].values, window_size, step=overlap))
        y[2].append([1 if 'PD' in file else 0])
    return X, y
