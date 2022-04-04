import pandas as pd
import os


def parse_df(directory, file):
    """
    Opens a DataFrame for a csv file in directory, then maps the three columns into another DataFrame
    where they are preceeded with column names for the axis
    :param file: str
    :param directory: str
    :return: DataFrame
    """
    # Empty DF
    df = pd.DataFrame()
    temp = pd.read_csv(os.path.join(directory, file))
    df["aX [m/s^2]"] = temp.iloc[:, 0].values
    df["aY [m/s^2]"] = temp.iloc[:, 1].values
    df["aZ [m/s^2]"] = temp.iloc[:, 2].values
    df.to_csv(os.path.join(directory, file))


if __name__ == "__main__":
    # Read local directories for both feet
    left_foot_dir = os.path.join(os.getcwd(), 'l')
    right_foot_dir = os.path.join(os.getcwd(), 'r')
    # Iterate over each directory and process files
    for file in os.listdir(left_foot_dir):
        parse_df(left_foot_dir, file)
    for file in os.listdir(right_foot_dir):
        parse_df(right_foot_dir, file)
