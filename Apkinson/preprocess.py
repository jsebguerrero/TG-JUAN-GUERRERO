import pandas as pd
import os
from preprocessing import preprocessing_functions

if __name__ == "__main__":
    # Read local directories for both feet
    apkinson_dir = os.path.join(os.getcwd(), 'data')
    # Check wether the processed dir exists or not
    isExist = os.path.exists(os.path.join(os.getcwd(), 'processed'))
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(os.path.join(os.getcwd(), 'processed'))
        print("Created directory for processing data!")
    # Iterate over each file
    for file in os.listdir(apkinson_dir):
        df = pd.read_csv(os.path.join(apkinson_dir, file),
                         usecols=["Timestamp [ns]", "aX [m/s^2]", "aY [m/s^2]", "aZ [m/s^2]"],
                         sep=';',
                         skiprows=5
                         )
        # Subsample to 50Hz
        df = preprocessing_functions.resample(df, 50)
        # Filter for gravity
        df = preprocessing_functions.filter_gravity(df)
        # Save again
        pd.to_csv(os.path.join(apkinson_dir, file), df)
