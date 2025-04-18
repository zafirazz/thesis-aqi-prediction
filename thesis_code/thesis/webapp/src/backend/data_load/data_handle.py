import pandas as pd


class DataLoader:
    """Class for loading dataset"""
    def __init__(self):
        self.file_path = "/Users/zafiraibraeva/Code/uni coding/thesis/thesis_code/thesis/webapp/dataset/final_data.csv"

    def load_data(self):
        """
        Creates a dataframe of dataset and deletes unnecessary columns

        :return: Cleaned dataframe
        """
        df = pd.read_csv(self.file_path)

        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)
        print(df.head(2))
        return df

    def get_data(self):
        """
        Loads data and checks whether it was found in file path or not.

        :return: DataFrame that will be used by models
        """
        df = self.load_data()
        if df is None:
            raise ValueError("Dataset was not found.")
        return df
