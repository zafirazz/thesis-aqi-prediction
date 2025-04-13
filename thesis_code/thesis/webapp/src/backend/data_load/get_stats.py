from backend.data_load.data_handle import DataLoader

class GetStats:
    """Class for getting statistics numbers of models"""
    def __init__(self):
        self.df = DataLoader().get_data()

    def stats_data(self):
        """
        Calculates average values of yearly historical data

        :return: dictionary with mean values of PM10
        """
        yearly_avg = self.df.groupby('year').agg({
            'Station1_PM10': 'mean',
            'Station2_PM10': 'mean'
        }).round(2)

        yearly_avg.reset_index(inplace=True)
        return yearly_avg.to_dict(orient='records')
