import pandas as pd
import os


def parse_df(directory, file):
    """
    Opens a DataFrame for a csv file in directory, then maps the three first rows into df
    where they will be columns
    :param file: str
    :param directory: str
    :return: DataFrame
    """
    # Empty DF
    df = pd.DataFrame(columns=["1", "2", "3"])
    temp = pd.read_csv(os.path.join(directory, file))
    df["1"] = temp.columns
    df["2"] = temp.iloc[0].values
    df["2"] = temp.iloc[1].values
    df.to_csv(os.path.join(left_foot_dir, file))


if __name__ == "__main__":
    # Read local directories for both feet
    left_foot_dir = os.path.join(os.getcwd(), 'l')
    right_foot_dir = os.path.join(os.getcwd(), 'r')
    # Iterate over each directory and process files
    for file in os.listdir(left_foot_dir):
        parse_df(left_foot_dir, file)
    for file in os.listdir(right_foot_dir):
        parse_df(right_foot_dir, file)
