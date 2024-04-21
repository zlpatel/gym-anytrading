import numpy as np
from .adv_trading_env import AdvTradingEnv, AdvActions, AdvPositions


class AdvStocksEnv(AdvTradingEnv):

    def __init__(self, df, window_size, frame_bound, render_mode=None):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size, render_mode)

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)

        features_columns = [col for col in self.df.columns if "feature" in col]
        all_features_columns = ["Close", "Open", "High", "Low", "Volume"] + list(features_columns)

        #self.df['Diff'] = diff
        #all_features = {k: self.df.loc[:, k].to_numpy() for k in features_columns}
        #print(all_features)
        all_features = np.array(self.df[all_features_columns], dtype= np.float32)
        all_features[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        all_features = all_features[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        #all_features['Diff'] = diff
        #signal_features = np.column_stack((prices, diff))
        
        #signal_features = np.column_stack([all_features[k] for k in range(all_features.shape[0])])
        signal_features = all_features
        
        return prices.astype(np.float32), signal_features.astype(np.float32)

        #TODO: handle additional features
        '''
        # ====== build feature map ========
        all_feature_name = ['Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume']
        all_feature = {k: self.df.loc[:, k].to_numpy() for k in all_feature_name}
        # add feature "Diff"
        prices = self.df.loc[:, 'Close'].to_numpy()
        diff = np.insert(np.diff(prices), 0, 0)
        all_feature_name.append('Diff')
        all_feature['Diff'] = diff
        # =================================

        # you can select features you want
        selected_feature_name = ['Close', 'Diff', 'Volume']
        selected_feature = np.column_stack([all_feature[k] for k in selected_feature_name])
        feature_dim_len = len(selected_feature_name)




        df = df.copy()
        self._features_columns = [col for col in df.columns if "feature" in col]
        self._info_columns = list(set(list(df.columns) + ["close"]) - set(self._features_columns))
        
        self.df = df
        self._obs_array = np.array(self.df[self._features_columns], dtype= np.float32)
        self._info_array = np.array(self.df[self._info_columns])
        self._price_array = np.array(self.df["close"])
        '''

    def _calculate_reward(self, action):
        '''
        step_reward = 0

        trade = False
        if (
            (action == AdvActions.Buy.value and self._position == AdvPositions.Short) or
            (action == AdvActions.Sell.value and self._position == AdvPositions.Long)
        ):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == AdvPositions.Long:
                step_reward += price_diff

        return step_reward
        '''
        step_reward = 0.
        current_price = (self.prices[self._current_tick])
        last_trade_price = (self.prices[self._last_trade_tick])
        ratio = current_price / last_trade_price
        cost = np.log((1 - self.trade_fee_ask_percent) * (1 - self.trade_fee_bid_percent))

        if action == AdvActions.BUY.value and self._position == AdvPositions.SHORT:
            step_reward = np.log(2 - ratio) + cost

        if action == AdvActions.SELL.value and self._position == AdvPositions.LONG:
            step_reward = np.log(ratio) + cost

        if action == AdvActions.DOUBLE_SELL.value and self._position == AdvPositions.LONG:
            step_reward = np.log(ratio) + cost

        if action == AdvActions.DOUBLE_BUY.value and self._position == AdvPositions.SHORT:
            step_reward = np.log(2 - ratio) + cost

        step_reward = float(step_reward)

        return step_reward

    
    def _update_profit(self, action):
        trade = False
        if (
            (action == AdvActions.SELL.value and self._position == AdvPositions.FLAT) or
            (action == AdvActions.BUY.value and self._position == AdvPositions.FLAT) or
            (action == AdvActions.DOUBLE_SELL.value and (self._position == AdvPositions.LONG or self._position == AdvPositions.FLAT)) or
            (action == AdvActions.DOUBLE_BUY.value and (self._position == AdvPositions.SHORT or self._position == AdvPositions.FLAT))):
                trade = True
        if trade or self._truncated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == AdvPositions.LONG:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:

            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick
                       and self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1

                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                tmp_profit = profit * (2 - (current_price / last_trade_price)) * (1 - self.trade_fee_ask_percent
                                                                                  ) * (1 - self.trade_fee_bid_percent)
                profit = max(profit, tmp_profit)
            else:
                while (current_tick <= self._end_tick
                       and self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1

                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                tmp_profit = profit * (current_price / last_trade_price) * (1 - self.trade_fee_ask_percent
                                                                            ) * (1 - self.trade_fee_bid_percent)
                profit = max(profit, tmp_profit)
            last_trade_tick = current_tick - 1

        return profit

