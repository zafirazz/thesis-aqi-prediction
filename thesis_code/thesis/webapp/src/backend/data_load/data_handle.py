import pandas as pd


class DataLoader:
    def __init__(self):
        self.file_path = "/Users/zafiraibraeva/Code/uni coding/thesis/thesis_code/thesis/webapp/dataset/final_data.csv"

    def load_data(self):
        df = pd.read_csv(self.file_path)

        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)
        print(df.head(2))
        return df

    def get_data(self):
        df = self.load_data()
        if df is None:
            raise ValueError("Dataset was not found.")
        return df