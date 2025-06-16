import numpy as np
import pandas as pd
file_path = 'C:/Users/sahil/Desktop/Raghav project/btc us data/btc.xlsx'
data = pd.read_excel(file_path)

# setting date time format , indexed open time for easier ploting
data['open time'] = pd.to_datetime(data['open time'])
data = data.sort_values('open time')
data.set_index('open time', inplace=True )

print(data.head())
print(data.dtypes)
print(data.describe()) # basic data features
print(data.isnull().sum()) # check for missing values

import matplotlib.pyplot as plt
plt.figure(figsize=(15,7))
data['close'].plot()
plt.title('close values')
plt.xlabel('time')
plt.ylabel('close price')
plt.show()


plt.figure(figsize=(15,7))
data['volume'].plot()
plt.title('volume traded')
plt.xlabel('time')
plt.ylabel('volume')
plt.show()

data['SMA_5'] = data['close'].rolling(window=5).mean() #5 day MA
data['SMA_20'] = data['close'].rolling(window=20).mean() #20 days MA
data['EMA_5'] = data['close'].ewm(span=5, adjust=False).mean() #5 days EMA
data['EMA_20'] = data['close'].ewm(span=20, adjust=False).mean() #20 days EMA


plt.figure(figsize=(15,7))
plt.plot(data['close'], label='close price', color='blue')
plt.plot(data['SMA_5'], label='ma5', color='red')
plt.plot(data['SMA_20'], label='ma20', color='green')
plt.title('close with moving averages') #is short MA5 is more than long MA20 then buy else sell
plt.xlabel('time')
plt.ylabel('price')
plt.legend
plt.show()


plt.figure(figsize=(15,7))
plt.plot(data['close'], label='close price', color='blue')
plt.plot(data['EMA_5'], label='ema5', color='red')
plt.plot(data['EMA_20'], label='ema20', color='green')
plt.title('close with exponential moving averages') #is short MA5 is more than long MA20 then buy else sell
plt.xlabel('time')
plt.ylabel('price')
plt.legend
plt.show()



plt.figure(figsize=(15,7))
plt.plot(data['close'], label='close price', color='blue')
plt.plot(data['EMA_5'], label='ma5', color='red')
plt.plot(data['SMA_20'], label='ema20', color='green')
plt.title('close with exponential moving averages') #is short MA5 is more than long MA20 then buy else sell
plt.xlabel('time')
plt.ylabel('price')
plt.legend
plt.show()

#If Close > SMA and rising → bullish signal.
#If SMA rising steadily → long-term uptrend.
#If EMA crosses above SMA → buy signal.
#If EMA crosses below SMA → sell signal.
#If Close far above SMA/EMA → price may revert.
#EMA reacts faster → catches short-term moves
#SMA gives smoother long-term trend.
# we use for short EMA and for long SMA and use this as signal as mentioned above

data['perc_change'] = ((data['close'] - data['open'])/data['open']) * 100
plt.figure(figsize=(15,6))
plt.plot(data['perc_change'])
plt.xlabel('time')
plt.ylabel('percentage change')
plt.show()

#here change>0% means candle is bullish and change<0% means bearish close to 0 is low momentum
#basically large +change implies strong upward trend and buying pressure and opp for -ve change

data['hl_range'] = data['high'] - data['low']
plt.figure(figsize=(15,7))
plt.plot(data['hl_range'], color='red')
plt.title('high-low range')
plt.xlabel('time')
plt.ylabel('price range(high-low)')
plt.show()
#large range --> high volatility small range -->low volatility small series then large--> potential breakout


#true range is how much price moved including any gaps 3 moments normal intraday movement , high - prev close, low - prev close
#then we take rolling mean of 14,20 whatever because we want not just volatility of one candle as it could have more noise
# rolling average if too smal then more noise and if too large then can miss changes

data['previous close'] =data['close'].shift(1)
data['tr1']= data['high']- data['low']
data['tr2']= data['high']- data['previous close']
data['tr3']= data['low']-data['previous close']
#true range
data['true_range']=data[['tr1', 'tr2', 'tr3']].max(axis=1)
#atr
data['atr14'] = data['true_range'].rolling(window=14).mean()


plt.figure(figsize=(15,7))
plt.plot(data['atr14'])
plt.title('atr14--volatality')
plt.ylabel('ATR')
plt.xlabel('time')
plt.show()
#When ATR is rising → volatility increasing → bigger price moves.
#When ATR is falling → volatility decreasing → smaller price moves → possible breakout coming.

#RSI--> relative strength index how strong recent gains are vs recent losses--"is price moving up too fast or down too fast
#rsi>70 asset may be overbought --> price moved up too fast(may fall)
#rsi<30 asset may be oversold --> price fell too fast(may bounce up)
#can be used for entry exit choice
#calculation price change in two closes this is 'delta' then we define gain where delta >0 and loss where delta <0
#take average rolling mean of this as avg gain and loss and dividing these two we get relative strength

delta = data['close'].diff()
gain = delta.where(delta>0,0)
loss = -delta.where(delta<0,0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs= avg_gain / avg_loss

data['RSI_14']= 100 - (100 / (1+rs))

print (gain.head(20))
print(avg_gain.head(20))
print(rs.head(20))
plt.figure(figsize=(15,7))
plt.plot(data['RSI_14'])
plt.title('rsi_14 (relative strength index')
plt.xlabel('time')
plt.ylabel('RSI')
plt.axhline(70, color='red',  linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.show()

#bollinger bands --  price moves around a mean(avg) now when it moves way far from mean it means volatility has increased
#core idea is of normal distribution, when data is distributed normally 68%values lie in +- one std dev , 95% in +- 2 std devs
#and 99.7% in 3 std devs so in bollinger bands we use 2 std devs to as we assume that under normal circumstances 95% of prices
# should stay within the bands so we take a middle band which could be SMA 20 (usual because 20 day trading month) and then
# upper band 2 std devs from middle and lower -2 std devs from middle
# narrow band imply that std dev is low --> stable price --> low volatility
# wide bands --> high std dev --> breakout or trend (volatile)
#price hitting upper band --> relatively high price and may break out or will revert
# price hitting lower band --> relatively low pricee and may rise or break down further

data['middle_band'] = data['close'].rolling(window=15).mean()
data['std_dev']= data['close'].rolling(window=15).std()
data['upper_band'] = data['middle_band'] + (2 * data['std_dev'])
data['lower_band'] = data['middle_band'] - (2 * data['std_dev'])

plt.figure(figsize=(15,7))
plt.plot(data['close'],label='close price', color= 'blue')
plt.plot(data['middle_band'], label='middle band (sma 15)', color= 'black')
plt.plot(data['lower_band'], label='lower band', color='red')
plt.plot(data['upper_band'], label='upper band', color='green')
plt.fill_between(data.index, data['lower_band'], data['upper_band'], color='grey', alpha=0.1)
plt.title('bollering bands')
plt.xlabel('time')
plt.ylabel('price')
plt.show()

# interpretation is done with rsi combined RSI<30 and price near lower band --> high buy probability setup
# RSI > 70 price is near upper band --> high probability sell setup


#volume moving average --> more than normal -> confirms breakout
#lower than normal -> signals weak moves

data['volume_ma20']= data ['volume'].rolling(window=20).mean()

plt.figure(figsize=(15,7))
plt.plot(data['volume'],label='volume', color='orange')
plt.plot(data['volume_ma20'],label='volume MA20', color='blue')
plt.title('volume and molume MA20')
plt.ylabel('volume')
plt.xlabel('time')
plt.legend()
plt.show()
import matplotlib.pyplot as plt
#pattern and signal detection
# Buy Signal → RSI < 30 and Close < Lower Band
data['buy_signal'] = ((data['RSI_14']<30) & (data['close'] < data['lower_band'])).astype(int)
data['sell_signal'] = ((data['RSI_14']>70) & (data['close'] > data['upper_band'])).astype(int)
#1 in buy is buy 0 dont buy 1 in sell is sell and 0 dont sell
plt.figure(figsize=(15,7))
plt.plot(data['close'], label='Close Price', color='blue')
#base plot^
# Plot buy signals
plt.scatter(data.index, data['close'].where(data['buy_signal'] == 1), label='Buy Signal', marker='^', color='green', s=100)

# Plot sell signals
plt.scatter(data.index, data['close'].where(data['sell_signal'] == 1), label='Sell Signal', marker='v', color='red', s=100)
plt.title('buy sell signals')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()


#creating filters
data['atr_MA_50'] = data['atr14'].rolling(window=50).mean()
data['EMA_5'] = data['close'].ewm(span=5,adjust=False).mean()
data['SMA_20'] = data['close'].rolling(window=20).mean()
data['recent_returns'] = data['close'].pct_change(periods=20)#percentage cange in 20 candles
data['final_buy_signal'] = (
    (data['RSI_14']<30)&
    (data['close']<data['lower_band'])&
    (data['atr14']>0.9 * data['atr_MA_50'])&
    (data['EMA_5']>0.98 * data['SMA_20'])&
    (data['volume']>0.95 * data['volume_ma20'])&
    (data['recent_returns']>-0.04) #new condition
).astype(int)

data['final_sell_signal'] = (
    (data['RSI_14']>70)&
    (data['close']>data['upper_band'])&
    (data['atr14']>0.9 * data['atr_MA_50'])&
    (data['EMA_5']<1.02 * data['SMA_20'])&
    (data['volume']>0.95 * data['volume_ma20'])
).astype(int)

plt.figure(figsize=(15,7))
plt.plot(data['close'], label='close price', color='blue')
plt.scatter(data.index,data['close'].where(data['final_buy_signal']==1), label='buy signal', marker='^',color='green',s=100)
plt.scatter(data.index,data['close'].where(data['final_sell_signal']==1), label='sell signal', marker='v',color='red',s=100)
plt.title('final buy sell signal')
plt.xlabel('time')
plt.ylabel('price')
plt.show()


#backtesting
#initializing variables
position = 0 # 0 = no position, 1 = long
entry_price = 0
pnl=[]
position_value=[]
buy_marker = []
sell_marker=[]
stop_loss_pct = 0.03
#loop through data
for index, row in data.iterrows():
    if position == 0 and row['final_buy_signal']==1 and row['recent_returns'] > -0.04:
        position = 1
        entry_price=row['close']
        entry_index = index #for future ploting
        buy_marker.append((index,row['close'])) #stores buy
    elif position == 1:
        price_change = (row['close']- entry_price)/entry_price
        #while in position
        if price_change < -stop_loss_pct:
            position = 0
            pnl.append(price_change) #record loss
            position_value.append(row['close'])
            sell_marker.append((index, row['close']))



        elif row['final_sell_signal'] == 1:
            position = 0
            pnl.append(price_change)
            position_value.append(row['close'])
            sell_marker.append((index,row['close']))

if pnl:
    cumulative_return = (1 + pd.Series(pnl)).prod() - 1
else:
    cumulative_return = 0

print(f"number of trades: {len(pnl)}")
print(f"cumulative return: {cumulative_return:.2%}")

equity_curve = (pd.Series(pnl).add(1).cumprod())*100
plt.figure(figsize=(15,7))
plt.plot(data['close'], label= 'close price', color = 'blue')
if buy_marker:
    buy_x,buy_y = zip(*buy_marker)
    plt.scatter(buy_x, buy_y, marker='^', color='green', label='Buy', s=100)

    # Plot sells
if sell_marker:
    sell_x, sell_y = zip(*sell_marker)
    plt.scatter(sell_x, sell_y, marker='v', color='red', label='Sell', s=100)

plt.title("Trade Execution Plot")
plt.xlabel("Index")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(15,7))
plt.plot(equity_curve,label='equity curve')
plt.title('backtest equity curve')
plt.xlabel('trade number')
plt.ylabel('equity')
plt.legend()
plt.show()
#win rate
returns = np.array(pnl)
transaction_cost= 0.001
win_rate = np.sum(returns>0)/len(returns)
avg_return =  np.mean(returns)
max_gain = np.max(returns)
max_loss = np.min(returns)
sharpe_ratio = np.mean(returns)/np.std(returns)
returns_after_cost = returns - transaction_cost
cum_return_after_cost= np.prod(1+returns_after_cost)-1

print(f"total trades: {len(returns)}")
print(f"win rate: {win_rate:.2%}")
print(f"average return per trade:{avg_return:.2%}")
print(f"max gain: {max_gain: .2%}")
print(f"max loss: {max_loss: .2%}")
print(f"sharpe ratio : {sharpe_ratio:.2f}")
print(f"cumulative return after cost: {cum_return_after_cost: .2%}")















































