import pandas as pd
import os
from preprocessing import preprocessing_functions


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
        # Save again
        df.to_csv(os.path.join(output_dir, file))
