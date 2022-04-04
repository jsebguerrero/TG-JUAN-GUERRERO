import os
from preprocessing import preprocessing_pipeline

if __name__ == "__main__":
    # Read local directories for both feet
    data_dir = 'data'
    processed_dir = 'processed'
    apkinson_dir = os.path.join(os.getcwd(), data_dir)
    processed_dir = os.path.join(os.getcwd(), processed_dir)
    columns = ["aX [m/s^2]", "aY [m/s^2]", "aZ [m/s^2]"]
    # Send directories through pipeline
    preprocessing_pipeline.pipeline_csv(apkinson_dir, processed_dir, columns, skiprows=5, sep=';')
