import numpy as np
from backtesting import Backtest, Strategy

class SimulatedBackTestStrategy(Strategy):

    def init(self):        
        # Prepare empty, all-NaN forecast indicator
        self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')

    def next(self):
        # Proceed only with out-of-sample data. Prepare some variables
        action = self.data.actions[0][0]

        forecast = action

        # Update the plotted "forecast" indicator
        self.forecasts[-1] = forecast

        if forecast > 0:
            self.buy(size=forecast)
        elif forecast < 0:
            self.sell(size=forecast)