from datetime import datetime as dt

from efr_classification import EfficiencyRatioAnalysis

if __name__ == "__main__":

    tickers = [
        'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD',
        'CADCHF', 'CADJPY', 'CHFJPY', 'EURAUD', 'EURCAD', 
        'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 'EURUSD',
        'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD',
        'GBPUSD', 'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD', 
        'USDCAD', 'USDCHF', 'USDJPY',
 
        'AUS200', 'FCHI40', 'GDAXI', 'NDX', 'NI225',
        'SP500', 'STOXX50E', 'UK100', 'WS30',

        'XAGUSD', 'XAUUSD', 'XNGUSD', 'XTIUSD',
    ]
        
    start_date = "2015-01-01"
    end_date = dt.strftime(dt.now(), "%Y-%m-%d")
    time_frame = '1h'

    # Create an instance of the EfficiencyRatioAnalysis class with the specified parameters
    analysis = EfficiencyRatioAnalysis(
        tickers=tickers, start_date=start_date, 
        end_date=end_date, 
        time_frame=time_frame,
        init_period=6, end_period=48, 
        n_clusters=3
        )

    # Generate and display a graph showing the clustered efficiency ratios
    analysis.graph_clusters_er()    
    