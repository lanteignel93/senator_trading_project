#!/usr/bin/env python
# coding: utf-8

from pandas.tseries.offsets import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import scipy as sp
import seaborn as sns
from tqdm.auto import tqdm,trange
plt.style.use('seaborn')


test_data = pd.read_pickle('test_data.pkl')
test_data_sectoral = pd.read_pickle('test_data_sectoral.pkl')
equity_data = pd.read_csv('test_equity_data.csv')
equity_data.set_index('Date',inplace = True)
equity_data.index = pd.to_datetime(equity_data.index)
ret_df = pd.read_csv('df_beta_spy_fwd_rate.csv')
merge_df = ret_df.drop_duplicates().reset_index().drop('index', axis=1)
## trading strategy class
class TS:
    def __init__(self,data,equity_data,K=1000000000,stoploss_ratio = 0.005,force_exit_days=100, leverage = -1,market_impact = .005, funding_rate = 0,short_bias = 1):
        self.data = data
        self.equity_data = equity_data
        self.K = K
        self.stoploss_ratio = stoploss_ratio
        self.leverage = leverage
        self.market_impact = market_impact
        self.funding_rate = funding_rate
        self.short_bias = short_bias
        ## if there are NAs, discard the observations
        self.valid_tickers = equity_data.isna().sum(axis = 0)[(equity_data.isna().sum(axis = 0) == 0)]
        self.tick_symbols = list(set(self.data['ticker'].unique()).intersection(set(self.valid_tickers.index)))
        self.data = self.data.loc[self.data.ticker.apply(lambda x: True if x in self.tick_symbols else False)]
        
        self.trading_dates = pd.date_range(data.loc[0,'file_date'], data.loc[len(data)-1,'file_date'], freq=BDay()).strftime('%Y-%m-%d')
        self.file_dates = self.data['file_date'].unique()
        self.outstanding_position_value = 0;
        
        self.force_exit_days = force_exit_days
        self.position_tracker = dict()
        self.hold_time_tracker = dict()
        for tick in self.tick_symbols:
            self.position_tracker[tick] = pd.DataFrame(0,index = self.trading_dates,columns = ['position','price','daily_pnl','cum_pnl'])
            self.hold_time_tracker[tick] = pd.DataFrame(columns = ['position','hold_time'])
        self.starting_K = K
        self.pnl_tracker =pd.Series(index  = self.trading_dates)
        self.cash_tracker= pd.Series(index  = self.trading_dates)
        self.total_outstanding_positions_tracker = pd.Series(index  = self.trading_dates)
        self.date_tracker  = []
        self.pnl = 0
        
   
    def get_price(self,date,tick):
        date = pd.to_datetime(date)
        try:
            price = self.equity_data.loc[date,tick]
        except:
            price = None
        try:
            if np.isnan(price):
                price = None
        except TypeError:
            price = None
        return price
    
    def enter_position(self,date,tick,side,size):
        price = self.get_price(date,tick)
        date = pd.to_datetime(date)
        if self.leverage == 0:
            position = size//price
        elif self.leverage == -1:
            position = 10000//price
        elif self.leverage > 0:
            position = (self.outstanding_position_value + self.K)*self.leverage//price
        if side==0:
            position = -position*self.short_bias
            
        self.K -= position*price
        self.K-= abs(position*price*self.market_impact)
        self.outstanding_position_value += position*price
    
        if self.K < 0:
            return None
        return (position,price)
    
    def update_pnl(self,date):
        ix = self.trading_dates.get_loc(date)
        if ix==0: return
        prev_date = self.trading_dates[ix-1]
        for key,df in self.position_tracker.items():
            curr_price = self.get_price(date,key)
            df.loc[date,'position'] = df.loc[prev_date,'position']
            if curr_price:
                df.loc[date,'price'] = curr_price
                if df.loc[prev_date,'position'] == 0:
                    continue
                df.loc[date,'daily_pnl'] = df.loc[date,'position'] * (curr_price - df.loc[prev_date,'price'])
                self.outstanding_position_value +=df.loc[date,'daily_pnl']
            else:
                df.loc[date,'price'] = df.loc[prev_date,'price']

                df.loc[date,'daily_pnl'] = 0    
            df['cum_pnl'] = df['daily_pnl'].cumsum()
        self.pnl = self.K + self.outstanding_position_value
        self.pnl_tracker.iloc[ix] = self.pnl
        self.cash_tracker.iloc[ix] = self.K
        self.total_outstanding_positions_tracker.iloc[ix] = self.outstanding_position_value
        
    def update_hold_time(self,date):
        for key,df in self.hold_time_tracker.items():
            if not df.empty: 
                df['hold_time'] += 1
    
    def exit_position(self,date,tick,force_exit=True):
        price = self.position_tracker[tick].loc[date,'price']
        if force_exit:
            force_date = self.hold_time_tracker[tick].index[0]
            pos = self.hold_time_tracker[tick].loc[force_date,'position']
            self.hold_time_tracker[tick].drop(index = force_date,inplace = True)
            self.position_tracker[tick].loc[date,'position'] -= pos
        else:
            pos = self.position_tracker[tick].loc[date,'position']
            self.position_tracker[tick].loc[date,'position']= 0
        
        self.K += pos * price
        self.K-= abs(pos*price*self.market_impact)
        self.outstanding_position_value -=pos * price
        
    def check_exit(self,date):
        for tick in self.position_tracker.keys():
            if self.position_tracker[tick].loc[date,'daily_pnl'] < - self.stoploss_ratio * (self.K+self.outstanding_position_value):
                self.exit_position(date,tick,False)
            if not self.hold_time_tracker[tick].empty and self.hold_time_tracker[tick].iloc[0,1] > self.force_exit_days:
                self.exit_position(date,tick,True)
                
    def simulation(self):
        for date in tqdm(self.trading_dates):
            self.date_tracker.append(date)
            if self.K < 0: ## pay interest on the debt you might owe
                self.K -= abs(self.K*self.funding_rate/252)
                self.pnl -=abs(self.K*self.funding_rate/252)
            if self.K > 0:
                interest = min(0, self.funding_rate - .0150)
                self.K += abs(self.K*interest/252)
                self.pnl +=abs(self.K*interest/252)
            if self.K + self.outstanding_position_value > 0:
                self.update_pnl(date)
                self.update_hold_time(date)
                if date in self.file_dates:
                    sub_df = self.data[(self.data['file_date']==date)&(self.data['win/loss']==1)]
                    for index, row in sub_df.iterrows():
                        entry = self.enter_position(row['file_date'],row['ticker'],row['order_type'],row['size'])
                        if entry:
                            new_pos,price = entry
                        else:
                            break
                        self.hold_time_tracker[row['ticker']].loc[date,'position'] = new_pos
                        self.hold_time_tracker[row['ticker']].loc[date,'hold_time'] = 0
                        self.position_tracker[row['ticker']].loc[date,'position'] += new_pos
                        self.position_tracker[row['ticker']].loc[date,'price'] = price
                        self.position_tracker[row['ticker']].loc[date,'daily_pnl'] += 0
                self.check_exit(date)
            else:
                print("Capital goes negative!")
                break
        out_df =  pd.DataFrame([self.pnl_tracker,self.cash_tracker,self.total_outstanding_positions_tracker],index = ['PNL','CASH',"POS_VALUE"]).T
        out_df.index = self.date_tracker
        return out_df
    


# In[40]:

def plot_sample_pnl_path(figsize,pathname):
    try:
        df = pd.read_pickle(pathname)
    except:
        df = TS(test_data,equity_data,leverage = .001,funding_rate = .015,market_impact = .005,\
                short_bias = 2.5, stoploss_ratio = .0003).simulation()
        df.to_pickle(pathname)
    fig, ax = plt.subplots(1,1,figsize = figsize)    
    ax.plot(df.PNL)
    ax.set_title('Sample Path PnL')
    fig.autofmt_xdate()
    plt.show()
    
    


def check_robustness_to_market_impact(figsize,pathname):
    fig, ax = plt.subplots(1,1,figsize = figsize)
    try:
        market_impact_df = pd.read_pickle(pathname)   
    except:
        market_impact_df = pd.DataFrame()
        for market_impact in [0,.005,.01,.015,.02,.025,.03,.04,.05,.07]:
            temp = TS(test_data,equity_data,leverage = .02,funding_rate = .03,\
                      market_impact = market_impact, short_bias = 2).simulation().PNL
            market_impact_df[market_impact] = temp
        market_impact_df.to_pickle(pathname)
    for market_impact,color in zip([0,.005,.01,.015,.02,.025,.03,.04,.05,.07],['indianred', 'black','aqua','darkblue','fuchsia','darkorange','lime']):   
        ax.plot(market_impact_df[market_impact],label = 'Market Impact = ' + str(market_impact),c = color)
    plt.legend()
    plt.show()


# In[24]:


def check_robustness_to_funding_costs(figsize,pathname):
    fig, ax = plt.subplots(1,1,figsize = figsize)
    try:
        funding_costs_df = pd.read_pickle(pathname)   
    except:
        funding_costs_df = pd.DataFrame()
        for funding_costs in [0,.005,.01,.015,.02,.025,.03,.04,.05,.07]:
            temp = TS(test_data,equity_data,leverage = .02,funding_rate = funding_costs,                    market_impact = .005, short_bias = 2).simulation().PNL
            funding_costs_df[funding_costs] = temp
        funding_costs_df.to_pickle(pathname)
    for funding_cost in [0,.005,.01,.015,.02,.025,.03,.04,.05,.07]:
        ax.plot(funding_costs_df[funding_cost],label = 'Funding Cost = ' + str(funding_cost))
    ax.legend()
    plt.show()


# In[25]:


def plot_leverage_effects (figsize,pathname):
    fig, ax = plt.subplots(1,1,figsize = figsize)
    try:
        leverage_effects_df = pd.read_pickle(pathname)   
    except:
        leverage_effects_df = pd.DataFrame()
        for leverage in [.0001,.005,.01,.02,.03,.04,.05]:
            temp = TS(test_data,equity_data,leverage = leverage,funding_rate = .03, \
                      market_impact = .005, short_bias = 2.5,stoploss_ratio = leverage/2).simulation().PNL
            leverage_effects_df[leverage] = temp
        leverage_effects_df.to_pickle(pathname)
    for leverage in [.0001,.005,.01,.02,.03,.05]:
        ax.plot(leverage_effects_df[leverage],label = 'Leverage = ' + str(leverage))
    ax.legend()
    plt.show()


# In[26]:


def plot_short_bias (figsize,pathname, leverage = .001):
    fig, ax = plt.subplots(1,1,figsize = figsize)
    try:
        short_bias_df = pd.read_pickle(pathname)   
    except:
        short_bias_df = pd.DataFrame()
        for short_bias in [1,1.5,2.0,2.5,3.0,3.5,4.0]:
            temp = TS(test_data,equity_data,leverage = leverage,funding_rate = .03,                    market_impact = .005, short_bias = short_bias).simulation().PNL
            short_bias_df[short_bias] = temp
        short_bias_df.to_pickle(pathname)
    for short_bias,color in zip([1,1.5,2.0,2.5,3.0,3.5,4.0],['indianred', 'black','aqua','darkblue','fuchsia','darkorange','lime','silver']):   
        ax.plot(short_bias_df[short_bias],label = 'Short Bias =' + str(short_bias),c = color)
    ax.legend()
    plt.show()

    
def plot_stoploss (figsize,pathname, leverage = .015):
    fig, ax = plt.subplots(1,1,figsize = figsize)
    try:
        stop_loss_df = pd.read_pickle(pathname)   
    except:
        stop_loss_df = pd.DataFrame()
        for stop_loss in [.01,.006,.003,.002,.001]:
            temp = TS(test_data,equity_data,leverage = leverage,funding_rate = .03,\
                      market_impact = .005, short_bias = 2.5, stoploss_ratio = stop_loss).simulation().PNL
            stop_loss_df[stop_loss] = temp
        stop_loss_df.to_pickle(pathname)
    for stop_loss,color in zip([.01,.006,.003,.002,.001],['indianred', 'black','aqua','fuchsia']):   
        ax.plot(stop_loss_df[stop_loss],label = 'Stop Loss =' + str(stop_loss),c = color)
    ax.legend()
    plt.show()


# In[27]:




# In[28]:


def strategy_ret_versus_normal(figsize,pathname):
    fig, ax = plt.subplots(1,1,figsize = figsize)
    try:
        df = pd.read_pickle(pathname)   
    except:
        df = TS(test_data,equity_data,leverage = .02,funding_rate = .01,market_impact = .005, short_bias = 2.5)
        df.to_pickle(pathname)
    
    df['PNL'].fillna(1e9)
    ret = df['PNL'].apply(np.log).diff().dropna()
    
    x = np.arange(-8,8,0.001)
    y = pd.Series(norm.pdf(x,0,1), index = x, name = 'Normal Distribution')
    
    df['Strat_Normalized'] = (ret - ret.mean())/ret.std()
    
    ax.plot(x,y, label = 'Normal Distribution')
    df['Strat_Normalized'].plot.kde()
    ax.set_xlabel('# of Standard Deviations Outside the Mean')
    ax.set_title('PDF Normalized Returns Strategy vs Normal Distribution', fontsize = 18)
    plt.legend(loc='upper right', title='Legend', fontsize = 16)
    print('P-value of the D\'Agostino and Pearson\'s test = {}'.format(sp.stats.normaltest(df['Strat_Normalized'].dropna())[1]))
    plt.show()




# In[17]:


def max_drawdown(figsize,pathname):
    fig, ax = plt.subplots(1,1,figsize = figsize)
    try:
        df = pd.read_pickle(pathname)   
    except:
        df = TS(test_data,equity_data,leverage = .02,funding_rate = .01,market_impact = .005, short_bias = 2.5)
        df.to_pickle(pathname)
    df['PNL'] = df['PNL'].fillna(1e9)
    k = np.argmax(np.maximum.accumulate(df['PNL']) - df['PNL']) # end of the period
    j = np.argmax(df['PNL'].iloc[:k]) # start of period

    plt.plot(df['PNL'])

    drawdown_start = df['PNL'].index[j]
    drawdown_end = df['PNL'].index[k]

    drawdown_peak = df['PNL'].iloc[j]
    drawdown_min = df['PNL'].iloc[k]

    drawdown = drawdown_peak - drawdown_min

    plt.scatter(drawdown_start,drawdown_peak, marker='o',color='red',label = 'Peak')
    plt.scatter(drawdown_end,drawdown_min, marker='x',color='red',label = 'Min')

    date_range = [drawdown_start, drawdown_end]
    data_range = [drawdown_peak, drawdown_min]

    plt.plot(date_range, data_range, '--', color = 'r',label = 'Max Drawdown: ' + str(round(drawdown,5)))

    k = np.argmax(df['PNL'] - np.minimum.accumulate(df['PNL'])) # end of the period
    j = np.argmin(df['PNL'].iloc[:k]) # start of period

    upside_start = df['PNL'].index[j]
    upside_end = df['PNL'].index[k]

    upside_peak = df['PNL'].iloc[k]
    upside_min = df['PNL'].iloc[j]

    upside = upside_peak - upside_min
    plt.scatter(upside_start,upside_min, marker='o',color='green',label = 'Min')
    plt.scatter(upside_end,upside_peak, marker='x',color='green',label = 'Peak')

    date_range = [upside_start, upside_end]
    data_range = [upside_min, upside_peak]

    plt.plot(date_range, data_range, '--', color ='green', label = 'Max Upside: ' + str(round(upside,5)))

    plt.title('Max Drawdown and Upside PnL', size = 18)
    plt.ylabel('Cumulative PL', size = 16)
    plt.xlabel('Date', size = 16)
    plt.legend(fontsize = 'large')
    plt.show()


# In[36]:


def pacf_adf_stats(figsize,pathname):
    fig, ax = plt.subplots(1,1,figsize = figsize)
    try:
        df = pd.read_pickle(pathname)   
    except:
        df = TS(test_data,equity_data,leverage = .02,funding_rate = .01,market_impact = .005, short_bias = 2.5)
        df.to_pickle(pathname)
    
    df['PNL'].fillna(1e9)
    ret = df['PNL'].apply(np.log).diff().dropna()
    
    df['Strat_Normalized'] = (ret - ret.mean())/ret.std()
    
    fig = sm.graphics.tsa.plot_pacf(df['Strat_Normalized'].dropna(), ax = ax, lags = 50)
    ADF_spread = sm.tsa.stattools.adfuller(df['Strat_Normalized'].dropna().values)
    print("ADF Stat: {:.4f} \n p-value: {:.4f}".format(ADF_spread[0], ADF_spread[1]))
    plt.show()



# In[86]:


def seasonality(figsize,pathname, qq_plot = False):
    try:
        df = pd.read_pickle(pathname)   
    except:
        df = TS(test_data,equity_data,leverage = .02,funding_rate = .01,market_impact = .005, short_bias = 2.5)
        df.to_pickle(pathname)
    
    df['PNL'].fillna(1e9)
    ret = df['PNL'].apply(np.log).diff().dropna()
    
    df['Strat_Normalized'] = (ret - ret.mean())/ret.std()
    
    seasonality = sm.tsa.seasonal_decompose(df['Strat_Normalized'].dropna().values, model='additive', period=20)
    plt.rcParams['figure.figsize'] = [figsize[0], figsize[1]]
    
    if qq_plot:
        res = seasonality.resid
        res = pd.Series(res)
        res = res.fillna(res.mean())
        fig = sm.qqplot(res, sp.stats.t, fit=True, line="45")
        plt.show()
    else:
        seasonality.plot()
        plt.show()


# In[153]:


def ff_correlation(figsize,pathname):
    fig, axes = plt.subplots(1,3,figsize = figsize)
    try:
        df = pd.read_pickle(pathname)   
    except:
        df = TS(test_data,equity_data,leverage = .02,funding_rate = .01,market_impact = .005, short_bias = 2.5)
        df.to_pickle(pathname)
    
    df['PNL'].fillna(1e9)
    ret = df['PNL'].apply(np.log).diff().dropna()
    ret = ret.to_frame()
    ff_df = pd.read_csv('F-F_Research_Data_Factors_daily.CSV')
    ff_df.rename(columns={'Unnamed: 0':'Date'}, inplace = True)
    ff_df['Date'] = pd.to_datetime(ff_df['Date'], format='%Y%m%d')
    ff_df = ff_df.set_index('Date')
    ff_df = ff_df[ff_df.index.isin(ret.index)]
    ret.index = pd.to_datetime(ret.index)
    ret = ret[ret.index.isin(ff_df.index)]
    ff_df['Strat_Ret'] = ret.values
    
    ff_df['Mkt'] = ff_df['Mkt-RF'] + ff_df['RF']
    for i in range(3):
        sns.regplot(data = ff_df,y = 'Strat_Ret',x=ff_df.columns[i], ax = axes[i], color= np.random.rand(3,))
    plt.suptitle("Correlation with FF Factors", fontsize = 20)
    
    display(ff_df.corrwith(ff_df['Strat_Ret']).to_frame().rename_axis("Factors").rename(columns={0:'Beta'}))
    plt.show()


# In[154]:

def downside_beta(pathname):
    try:
        df = pd.read_pickle(pathname)   
    except:
        df = TS(test_data,equity_data,leverage = .02,funding_rate = .01,market_impact = .005, short_bias = 2.5)
        df.to_pickle(pathname)
    
    df['PNL'].fillna(1e9)
    ret = df['PNL'].apply(np.log).diff().dropna()
    ret = ret.to_frame()
    ff_df = pd.read_csv('F-F_Research_Data_Factors_daily.CSV')
    ff_df.rename(columns={'Unnamed: 0':'Date'}, inplace = True)
    ff_df['Date'] = pd.to_datetime(ff_df['Date'], format='%Y%m%d')
    ff_df = ff_df.set_index('Date')
    ff_df = ff_df[ff_df.index.isin(ret.index)]
    ret.index = pd.to_datetime(ret.index)
    ret = ret[ret.index.isin(ff_df.index)]
    ff_df['Ret'] = ret.values
    
    ff_df["RF"] /= 100
    ff_df['Mkt-RF'] /= 100
    ff_df['Mkt'] = ff_df['Mkt-RF'] + ff_df['RF']
    ff_df = ff_df.fillna(0)

    Sharpe_Ratio = np.sqrt(52)*(ff_df['Ret'] - ff_df["RF"]).mean()/ff_df['Ret'].std()
    Sortino_Ratio  = np.sqrt(52)*(ff_df['Ret'] - ff_df["RF"]).mean()/ (np.sqrt(ff_df['Ret'][ff_df['Ret']<0]**2).sum()/len(ff_df['Ret']))
    Treynor_Ratio = 52*(ff_df['Ret'] - ff_df["RF"]).mean()/np.corrcoef(ff_df['Ret'],ff_df['Mkt-RF'])[0][1]
    metric_df = pd.Series({'Sharpe_Ratio':Sharpe_Ratio,'Sortino_Ratio':Sortino_Ratio,'Treynor_Ratio':Treynor_Ratio}).to_frame().rename_axis("Metrics")

    downmkt_df = ff_df[ff_df['Mkt-RF'] + ff_df['RF'] < 0]
    y = downmkt_df['Ret']
    x = sm.add_constant(downmkt_df['Mkt-RF'])

    downside_beta = sm.OLS(y, x).fit().params.to_frame('Downside Beta Regression')
    downside_beta.loc['R${^2}$'] = sm.OLS(y, x).fit().rsquared
    display(downside_beta)


# In[163]:



# In[168]:


def performance_metric(pathname):
    try:
        df = pd.read_pickle(pathname)   
    except:
        df = TS(test_data,equity_data,leverage = .02,funding_rate = .01,market_impact = .005, short_bias = 2.5)
        df.to_pickle(pathname)
    
    df['PNL'].fillna(1e9)
    ret = df['PNL'].apply(np.log).diff().dropna()
    ret = ret.to_frame()
    ff_df = pd.read_csv('F-F_Research_Data_Factors_daily.CSV')
    ff_df.rename(columns={'Unnamed: 0':'Date'}, inplace = True)
    ff_df['Date'] = pd.to_datetime(ff_df['Date'], format='%Y%m%d')
    ff_df = ff_df.set_index('Date')
    ff_df = ff_df[ff_df.index.isin(ret.index)]
    ret.index = pd.to_datetime(ret.index)
    ret = ret[ret.index.isin(ff_df.index)]
    ff_df['Ret'] = ret.values
    
    ff_df["RF"] /= 100
    ff_df['Mkt-RF'] /= 100
    ff_df['Mkt'] = ff_df['Mkt-RF'] + ff_df['RF']
    ff_df = ff_df.fillna(0)

    Sharpe_Ratio = np.sqrt(52)*(ff_df['Ret'] - ff_df["RF"]).mean()/ff_df['Ret'].std()
    Sortino_Ratio  = np.sqrt(52)*(ff_df['Ret'] - ff_df["RF"]).mean()/ (np.sqrt(ff_df['Ret'][ff_df['Ret']<0]**2).sum()/len(ff_df['Ret']))
    Treynor_Ratio = 52*(ff_df['Ret'] - ff_df["RF"]).mean()/np.corrcoef(ff_df['Ret'],ff_df['Mkt-RF'])[0][1]
    metric_df = pd.Series({'Sharpe_Ratio':Sharpe_Ratio,'Sortino_Ratio':Sortino_Ratio,'Treynor_Ratio':Treynor_Ratio}).to_frame().rename_axis("Metrics").rename(columns={0:'Score'})

    display(metric_df)




# In[176]:


def returns_stats(pathname):
    try:
        df = pd.read_pickle(pathname)   
    except:
        df = TS(test_data,equity_data,leverage = .02,funding_rate = .01,market_impact = .005, short_bias = 2.5)
        df.to_pickle(pathname)
    
    df['PNL'].fillna(1e9)
    df['Strat_Ret'] = df['PNL'].apply(np.log).diff().dropna()
    
    data1 = df['Strat_Ret']
    stats = pd.DataFrame(data = None, columns =['Returns Statistics'], index = ['Mean', 'Median', 'Std Dev','1st Quartile', '3rd Quartile', 'Skew', 'Kurtosis'])
    stats.loc['Mean'] = data1.mean()
    stats.loc['Median'] = data1.median()
    stats.loc['Std Dev'] = data1.std()
    stats.loc['1st Quartile'] = data1.quantile(0.25)
    stats.loc['3rd Quartile'] = data1.quantile(0.75)
    stats.loc['Skew'] = sp.stats.skew(data1.dropna())
    stats.loc['Kurtosis'] = sp.stats.kurtosis(data1.dropna())
    display(stats)

def get_net_long(leverage, short_bias):
    
    TS1 = TS(test_data,equity_data,leverage = leverage,funding_rate = .03,\
                    market_impact = .005, short_bias = short_bias)
    df1 = TS1.simulation()
    net_long= pd.DataFrame(0,index = TS1.trading_dates,columns = ['net_long'])
    for tick,df in TS1.position_tracker.items():
        df['Beta'] = merge_df.Beta.iloc[np.where(merge_df.ticker == tick)[0][0]]
        if df['Beta'].isna().sum() ==0:
            net_long['net_long'] += df['position'].astype(float)*df['price'].astype(float)*df['Beta'].astype(float)
            
    plt.plot(df1.index, net_long.values/df1.PNL.values.reshape(-1,1))
    plt.title('Net Long or Short Exposure vs Capital')
    plt.show()
    


def plot_shortbias_differences(figsize):
    fig, ax = plt.subplots(1,2,figsize = figsize,sharey = True)
    fig.suptitle('COMPARING MODEL DIFFERENCES -- SHORT BIASES')
    try:
        short_bias_df = pd.read_pickle('short_bias_sectoral.pkl')   
    except:
        short_bias_df = pd.DataFrame()
        for short_bias in [1,1.5,2.0,2.5,3.0,4.0]:
            temp = TS(test_data_sectoral,equity_data,leverage = .02,funding_rate = .03,\
                    market_impact = .005, short_bias = short_bias).simulation().PNL
            short_bias_df[short_bias] = temp
        short_bias_df.to_pickle('short_bias_sectoral.pkl')
    for short_bias,color in zip([1,1.5,2.0,2.5,3.0,4.0],['indianred', 'black','aqua','fuchsia','darkorange','silver']):   
        ax[1].plot(short_bias_df[short_bias],label = 'Short Bias =' + str(short_bias),c = color)
    ax[1].legend()
    ax[1].set_title('Sectoral Model')

    try:
        short_bias_df = pd.read_pickle('short_bias_example.pkl')   
    except:
        short_bias_df = pd.DataFrame()
        for short_bias in [1,1.5,2.0,2.5,3.0,4.0]:
            temp = TS(test_data,equity_data,leverage = .02,funding_rate = .03,\
                      market_impact = .005, short_bias = short_bias).simulation().PNL
            short_bias_df[short_bias] = temp
        short_bias_df.to_pickle('short_bias_example.pkl')
    for short_bias,color in zip([1,1.5,2.0,2.5,3.0,4.0],['indianred', 'black','aqua','fuchsia','darkorange','silver']):   
        ax[0].plot(short_bias_df[short_bias],label = 'Short Bias =' + str(short_bias),c = color)
    ax[0].legend()
    ax[0].set_title('Non Sectoral Model')
    plt.show()

def plot_leverage_differences(figsize):
    
    fig, ax = plt.subplots(1,2,figsize = figsize, sharey = True)
    fig.suptitle('COMPARING MANY SAMPLE PATHS WITH LEVERAGE VARIED')
    try:
        leverage_effects_df = pd.read_pickle('leverage_effects_example.pkl')   
    except:
        leverage_effects_df = pd.DataFrame()
        for leverage in [.0001,.005,.01,.02,.03,.05]:
            temp = TS(test_data,equity_data,leverage = leverage,funding_rate = .03,\
                    market_impact = .005, short_bias = 2).simulation().PNL
            leverage_effects_df[leverage] = temp
        leverage_effects_df.to_pickle('leverage_effects_example.pkl')
        
    for leverage in [.0001,.005,.01,.02,.03,.05]:
        ax[0].plot(leverage_effects_df[leverage],label = 'Leverage = ' + str(leverage))
    ax[0].legend()
    ax[0].set_title('Non Sectoral Model')
 
    try:
        leverage_effects_df = pd.read_pickle('leverage_effects_sectoral.pkl')   
    except:
        leverage_effects_df = pd.DataFrame()
        for leverage in [.0001,.005,.01,.02,.03,.05]:
            temp = TS(test_data_sectoral,equity_data,leverage = leverage,funding_rate = .03,\
                    market_impact = .005, short_bias = 2).simulation().PNL
            leverage_effects_df[leverage] = temp
        leverage_effects_df.to_pickle('leverage_effects_sectoral.pkl')
    for leverage in [.0001,.005,.01,.02,.03,.05]:
        ax[1].plot(leverage_effects_df[leverage],label = 'Leverage = ' + str(leverage))
    ax[1].legend()
    ax[1].set_title('Sectoral Model')
    