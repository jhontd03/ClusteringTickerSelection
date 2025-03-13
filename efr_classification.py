import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt

from data_loader import DataLoader
from function_cluster import KMeansClustering

class EfficiencyRatioAnalysis:
    """
    A class for analyzing efficiency ratios of financial instruments.
    
    This class calculates efficiency ratios for multiple tickers across different
    time periods, clusters them using K-means, and provides visualization tools
    for analysis.
    
    Attributes:
        tickers (list): List of ticker symbols to analyze.
        start_date (str): Start date for the data in 'YYYY-MM-DD' format.
        end_date (str): End date for the data in 'YYYY-MM-DD' format.
        time_frame (str): Time frame for the data (e.g., '1h', '1d'). Default is '1h'.
        init_period (int): Initial period for efficiency ratio calculation. Default is 6.
        end_period (int): End period for efficiency ratio calculation. Default is 30.
        n_clusters (int): Number of clusters for K-means clustering. Default is 3.
        df_tickers (DataFrame): DataFrame containing price data for all tickers.
        df_er (DataFrame): DataFrame containing efficiency ratios for all tickers.
        clusters_er (DataFrame): DataFrame containing clustering results.
    """
    def __init__(self, tickers, start_date, end_date, 
                 time_frame='1h', init_period=6, 
                 end_period=30, n_clusters=3):
        """
        Initialize the EfficiencyRatioAnalysis class.
        
        Args:
            tickers (list): List of ticker symbols to analyze.
            start_date (str): Start date for the data in 'YYYY-MM-DD' format.
            end_date (str): End date for the data in 'YYYY-MM-DD' format.
            time_frame (str, optional): Time frame for the data. Default is '1h'.
            init_period (int, optional): Initial period for efficiency ratio calculation. Default is 6.
            end_period (int, optional): End period for efficiency ratio calculation. Default is 30.
            n_clusters (int, optional): Number of clusters for K-means clustering. Default is 3.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.time_frame = time_frame
        self.init_period = init_period
        self.end_period = end_period
        self.n_clusters = n_clusters
        self.df_tickers = self.load_data()
        self.df_er = self.calculate_er()
        self.clusters_er = self.tickers_clusters_er()

    def load_data(self):
        """
        Load price data for all tickers from MT5.
        
        Returns:
            DataFrame: DataFrame containing close prices for all tickers.
        """
        data_loader = DataLoader()
        df_tickers = []

        for symbol in self.tickers:            
            data_loader.load_from_MT5(
                symbol, self.start_date, 
                self.end_date, self.time_frame)                                                         
            df_tickers.append(data_loader.get_data()[['Close']])

        df_tickers = pd.concat(df_tickers, axis=1)
        df_tickers.columns = self.tickers
        df_tickers.dropna(inplace=True)
        return df_tickers

    def calculate_er(self):
        """
        Calculate efficiency ratios for all tickers across multiple periods.
        
        The efficiency ratio measures the directional movement of price relative to
        the total price movement over a specified period.
        
        Returns:
            DataFrame: DataFrame containing efficiency ratios for all tickers and periods.
        """
        er_join = dict()
        for length in range(self.init_period, self.end_period):
            df_er = pd.DataFrame()
            for k, v in self.df_tickers.items():
                df_er[f'{k}'] = ta.er(v, length=length)
            er_join[f'{length}'] = df_er

        er_mean = [
            pd.DataFrame(value.mean().round(5), columns=[f'period_{key}'])
            for key, value in er_join.items()
        ]

        df_er = pd.concat(er_mean, axis=1)
        return df_er

    def tickers_clusters_er(self):
        """
        Cluster tickers based on their efficiency ratios using K-means.
        
        This method normalizes the efficiency ratios, applies K-means clustering,
        and calculates the mean efficiency ratio for each ticker.
        
        Returns:
            DataFrame: DataFrame containing symbols, mean efficiency ratios, and cluster labels.
        """
        df_er_norm = (self.df_er - self.df_er.min()) \
            / (self.df_er.max() - self.df_er.min())

        kmeans = KMeansClustering(n_clusters=self.n_clusters)
        cluster_labels = kmeans.fit_predict(df_er_norm)
        self.df_er["cluster_label"] = cluster_labels

        self.df_er["mean_er"] = self.df_er.iloc[:, :-1].mean(axis=1)
        self.df_er.reset_index(inplace=True)
        self.df_er.columns.values[0] = 'symbol'

        df_er = self.df_er[['symbol', 'mean_er', 'cluster_label']]
        return df_er

    def graph_clusters_er(self):
        """
        Create a bar chart visualization of the clustered efficiency ratios.
        
        The chart displays the mean efficiency ratio for each ticker, with bars
        colored according to their cluster assignment.
        
        Returns:
            None: Displays the plot but does not return any value.
        """
        # Get unique colors for each cluster
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Add more colors if needed
        cluster_colors = {i: colors[i] for i in range(self.n_clusters)}
        
        # Prepare data
        clusters_er = self.df_er.groupby(["cluster_label", "symbol"]).mean()
        sorted_data = clusters_er['mean_er'].sort_values()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar plot with modified index to show only symbols
        sorted_data.index = [idx[1] for idx in sorted_data.index]  # Extract only symbols from index
        sorted_data.plot(
            kind='bar',
            ax=ax,
            legend=None
        )
        
        # Color bars according to cluster
        for idx, bar in enumerate(ax.patches):
            symbol = sorted_data.index[idx]
            cluster = self.df_er[self.df_er['symbol'] == symbol]['cluster_label'].iloc[0]
            bar.set_color(cluster_colors[cluster])
        
        # Customize plot
        plt.title('Symbol Cluster Efficiency Ratio', fontsize=12)
        plt.xlabel('Symbols', fontsize=10)
        plt.ylabel('Mean Efficiency Ratio', fontsize=10)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        plt.show()
        return
