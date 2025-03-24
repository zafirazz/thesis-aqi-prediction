import pandas as pd

PATH = "/Users/zafiraibraeva/Code/uni coding/thesis/thesis_code/thesis/data/final_data/result_data.csv"

class DataHandler:
    def __init__(self):
        self.data = pd.read_csv(PATH)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)

    def filter_by_date(self, start_date, end_date):
        return self.data.loc[start_date:end_date]

    def get_features(self, filtered_df):
        features = [col for col in filtered_df.columns if col != "Station2_PM10"]
        return features

    def get_target(self, filtered_df):
        return filtered_df['Station2_PM10']
