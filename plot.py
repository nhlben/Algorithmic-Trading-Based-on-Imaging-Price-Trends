from __init__ import *


def set_img_size(time_interval, indicator):
    # hidden
    pass

def heikin_ashi(df):
    '''
    Given a DataFrame `df` with a Pandas.DatetimeIndex and (at least the following)
    columns "Open", "High", "Low", and "Close", this function will construct and return a
    DataFrame of Heikin Ashi candles  w/columns "HAOpen", "HAHigh", "HALow", and "HAClose" 
    '''

    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close'])/4

    ha_df = pd.DataFrame(dict(HAClose=ha_close))

    ha_df['HAOpen']  = [0.0]*len(df)

    # "seed" the first open:
    prekey = df.index[0]
    ha_df.at[prekey,'HAOpen'] = df.at[prekey,'Open']

    # calculate the rest  
    for key in df.index[1:]:
        ha_df.at[key,'HAOpen'] = (ha_df.at[prekey,'HAOpen'] + ha_df.at[prekey,'HAClose']) / 2.0
        prekey = key

    ha_df['HAHigh']  = pd.concat([ha_df.HAOpen,df.High],axis=1).max(axis=1)
    ha_df['HALow']   = pd.concat([ha_df.HAOpen,df.Low ],axis=1).min(axis=1)
    ha_df["Volume"] = df["Volume"]
    return ha_df

def MACD(df, window_slow, window_fast, window_signal):

    macd = df

    macd['ema_slow'] = df['Close'].ewm(span=window_slow).mean()
    macd['ema_fast'] = df['Close'].ewm(span=window_fast).mean()
    macd['macd'] = macd['ema_slow'] - macd['ema_fast']
    macd['signal'] = macd['macd'].ewm(span=window_signal).mean()
    macd['diff'] = macd['macd'] - macd['signal']
    macd['bar_positive'] = macd['diff'].map(lambda x: x if x > 0 else 0)
    macd['bar_negative'] = macd['diff'].map(lambda x: x if x < 0 else 0)

    return macd

def RSI(df, time_interval):

    df['rsi'] = ta.RSI(df['Close'], timeperiod=time_interval)

    return df

def MFI(df, time_interval):

    df['mfi'] = ta.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=time_interval)

    return df

def BBANDS(df, time_interval):

    df['upper'], df['middle'], df['lower'] = ta.BBANDS(df['Close'], timeperiod=time_interval)

    return df

    
def get_add_plot(data, indicator):
    # hidden
    pass

def draw_pic(stock_dataframe, chart_type, img_size, indicator):
    # hidden
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--I', type=int, default=5, help='Time Interval of chart time')
    parser.add_argument('--R', type=int, default=5, help='Return Interval')
    parser.add_argument('--data_path', type=str, default='./data/I5R5/train', help='path for storing data')
    parser.add_argument('--type', type=str, default='ohlc', help='chart type')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--start_date', type=str, default="1993-01-01", help='Start date of stock data')
    parser.add_argument('--end_date', type=str, default="2005-12-31", help='End date of stock data')
    parser.add_argument('--indicator', type=str, default="MA", help='Add indicator')
    parser.add_argument('--testing', type=bool, default=False, help='Testing')
    parser.add_argument('--stock', type=str, default="SP500", help='Stock list')
    opt = parser.parse_args()
    print(opt, flush=True)
    
    # set parameters
    R = opt.R # set return time interval same as image
    i = opt.I - 1
    time_interval = opt.I
    mav = opt.I # moving average same as time_interval

    if opt.num_classes==2:
        folder1 = f"{opt.data_path}/Positive"
        folder2 = f"{opt.data_path}/Negative"
        os.makedirs(folder1, exist_ok=True)
        os.makedirs(folder2, exist_ok=True)

    # set image size
    img_size = set_img_size(time_interval, opt.indicator)

    if opt.stock=="SP500":
        ticker_list = pd.read_csv("SP500_constituents.csv")["Symbol"].to_list()
    # start and end date of data
    start = opt.start_date
    end = opt.end_date
    # set ticker and download data
    count_data = 0
    
    if opt.testing:
        print("testing", flush=True)
        ticker_list = ["AEP"]
    

    nyse = mcal.get_calendar('NYSE')
    period = nyse.schedule(start_date=opt.start_date, end_date=opt.end_date)
    num_trading_days = len(mcal.date_range(period, frequency='1D'))
    for ticker in ticker_list:
        try:
            data = yf.download(ticker, auto_adjust=True, start=start, end=datetime.strftime(datetime.strptime(opt.end_date, "%Y-%m-%d") + timedelta(days=1), "%Y-%m-%d"))
            if len(data) != num_trading_days:
                continue
        except:
            continue
        #print(data)
        print("\n", flush=True)
        # compute all indicators
        data["MA"] = data["Close"].rolling(window=time_interval).mean()
        data = MACD(data, 12, 26, 9)
        data = RSI(data, time_interval)
        data = MFI(data, time_interval)
        data = BBANDS(data, time_interval)
        close_price = data["Close"].to_numpy()
        # reset i
        i = opt.I - 1
        # plot data
        while (i + R) < (data.shape[0] - R):
            if (((i+1) % (R*10))==0):
                print(f"{ticker}:{(i+1) // R} / {(data.shape[0]-2*R) // (R)}" , flush=True)
            #print(data)
            plotting_data = data.iloc[i:i + time_interval]
            if opt.num_classes == 2:
                if  close_price[i + R - 1] > close_price[i + R - 1 + R]: # compare close price between last day and R days later
                    # negative return
                    save_name = f"{opt.data_path}/Negative/{ticker}_{i+1}.png"
                else:
                    # positive return
                    save_name = f"{opt.data_path}/Positive/{ticker}_{i+1}.png"
            elif opt.num_classes == 3:
                pct_change = close_price[i + R - 1 + R] / close_price[i + R - 1] 
                if (pct_change >= (0.99)) and (pct_change <= 1.01):  # 1% threshold
                    save_name = f"{opt.data_path}/Non-significant/{ticker}_{i+1}.png"
                elif  close_price[i + time_interval - 1] > close_price[i + time_interval - 1 + R]: # compare close price between last day and R days later
                    # negative return
                    save_name = f"{opt.data_path}/Negative/{ticker}_{i+1}.png"
                else:
                    # positive return
                    save_name = f"{opt.data_path}/Positive/{ticker}_{i+1}.png"                
            if os.path.exists(save_name):
                i = i + R
                count_data += 1
                continue
                    
            plotting_data.iloc[:, -2:-1] = plotting_data.iloc[:, -2:-1].apply(lambda x: x / x.std()) 
            image = draw_pic(plotting_data , opt.type, img_size, opt.indicator)
            image.save(save_name)
            count_data += 1
            i = i + R

        print(f"{ticker} Completed", flush=True)
    print(f"Number of images plotted: {count_data}")





