import os
from preprocessing import preprocessing_pipeline

if __name__ == "__main__":
    # Read local directories for both feet
    processed_dir = 'processed'
    # left foot dir
    egait_dir_l = os.path.join(os.getcwd(), 'l')
    processed_dir_l = os.path.join(os.getcwd(), processed_dir, 'l')
    # right foot dir
    egait_dir_r = os.path.join(os.getcwd(), 'r')
    processed_dir_r = os.path.join(os.getcwd(), processed_dir, 'r')
    columns = [1, 2]
    # Send directories through pipeline
    preprocessing_pipeline.pipeline_csv(egait_dir_l, processed_dir_l, columns, skiprows=5, sep=',', fs=50)
    preprocessing_pipeline.pipeline_csv(egait_dir_r, processed_dir_r, columns, skiprows=5, sep=',', fs=50)
