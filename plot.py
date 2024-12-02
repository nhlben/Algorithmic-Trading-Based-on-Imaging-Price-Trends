from __init__ import *


def set_img_size(time_interval, indicator):
    if time_interval==5:
        px = 1/plt.rcParams['figure.dpi'] 
        img_size = [30*px, 64*px]
    elif time_interval==20:
        px = 1/plt.rcParams['figure.dpi'] 
        img_size = [128*px, 128*px]

    indicators = indicator.split(",")
    num_panels = len(indicators)
    if "MA" in indicators:
        num_panels -= 1
    if "BB" in indicators:
        num_panels -= 1
    # TODO
    if indicator=="MA,MACD":
        px = 1/plt.rcParams['figure.dpi'] 
        img_size = [60*px, 100*px]
        
    if indicator=="MA,BB":
        px = 1/plt.rcParams['figure.dpi'] 
        img_size = [60*px, 80*px]
    return img_size

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
    # TODO
    indicators = indicator.split(",")
    num_panels = len(indicators)
    apdict = []
    current_panel = 1
    for ind in indicators:
        if ind=="MA":
            apdict.append(mpf.make_addplot(data["MA"], color="white", width=1, panel=0))
        if ind=="MACD":
            apdict.append(mpf.make_addplot((data['diff']), color='white', width=1, secondary_y=True, panel=current_panel+1))
            apdict.append(mpf.make_addplot((np.zeros(len(data['diff']))), color='white', panel=2, width=1, secondary_y=True))
            current_panel += 1
        if ind=="BB":
            apdict.append(mpf.make_addplot((data['upper']), color='white', width=1, secondary_y=False, panel=0))
            apdict.append(mpf.make_addplot((data['lower']), color='white', width=1, secondary_y=False, panel=0))
        if ind=="RSI":
            apdict.append(mpf.make_addplot((data['rsi']), color='white', width=1, secondary_y=False, panel=current_panel+1))
            current_panel += 1
        if ind=="MFI":
            apdict.append(mpf.make_addplot((data['mfi']), color='white', width=1, secondary_y=False, panel=current_panel+1))
            current_panel += 1    
        # TODO
    return apdict, current_panel

def draw_pic(stock_dataframe, chart_type, img_size, indicator):
        mc = mpf.make_marketcolors(up='white',down='white',edge="white", ohlc="white", volume="white", inherit=True)
        ms = mpf.make_mpf_style(
            y_on_right=True,
            marketcolors=mc,
            edgecolor='black',
            figcolor='black',
            facecolor='black', 
            gridcolor='black',
            )
        # MACD only

        if indicator == "MACD_only":
            apdict =[ mpf.make_addplot((stock_dataframe['diff']), color='white', width=1, secondary_y=False),
                      mpf.make_addplot((np.zeros(len(stock_dataframe['diff']))), color='white', panel=0, width=1, secondary_y=False)
                      ]
            length = len(stock_dataframe)
            o = [0.0]*length
            h = [0.0]*length
            l = [0.0]*length
            c = [0.0]*length

            df = pd.DataFrame(dict(Open=o,High=h,Low=l,Close=c))
            df.index = stock_dataframe.index
            img_buf = io.BytesIO()
            mpf.plot(df, type='line',volume=False, figsize=(img_size[0], img_size[1]),
                    style=ms, linecolor='black', addplot=apdict,axisoff=True, savefig=img_buf)
            plt.savefig(img_buf, format='png')
            # open image from buffer and convert into B&W directly
            bw_im = Image.open(img_buf).convert("1", dither=Image.NONE)#.point(lambda x : 255 if x > 0 else 0, mode='1')
            plt.close()
            return bw_im
        if chart_type=="ohlc": #ohlc chart
            if indicator == "MA": # MA
                fig = mpf.figure(style=ms ,figsize=(img_size[0], img_size[1]))

                ax1 = fig.add_axes([0, 0.2, 1, 0.80])
                ax2 = fig.add_axes([0, 0, 1, 0.2], sharex=ax1)
                apdict = mpf.make_addplot(stock_dataframe["MA"], color="white", width=1, ax=ax1)
                img_buf = io.BytesIO()
                mpf.plot(stock_dataframe,
                    ax=ax1,
                    volume=ax2, tight_layout=True,
                    style=ms, linecolor='white', addplot=apdict, type=chart_type, axisoff=True, update_width_config=dict(volume_width=0.3), savefig=img_buf)
                plt.savefig(img_buf, format='png')

            elif indicator=="NONE":
                fig = mpf.figure(style=ms ,figsize=(img_size[0], img_size[1]))
                ax1 = fig.add_axes([0, 0.2, 1, 0.80])
                ax2 = fig.add_axes([0, 0, 1, 0.2], sharex=ax1)
                mpf.plot(stock_dataframe,
                    ax=ax1,
                    volume=ax2, tight_layout=True,
                    style=ms, linecolor='white', type=chart_type, axisoff=True, update_width_config=dict(volume_width=0.3)) 
            #ax2.set_ylim(0, 0.01)
            else: # MA5 and MACD
                """
                apdict =[mpf.make_addplot(stock_dataframe["MA5"], color="white", width=1),
                         mpf.make_addplot((stock_dataframe['diff']), color='white', panel=2, width=1, secondary_y=True),
                         mpf.make_addplot((np.zeros(len(stock_dataframe['diff']))), color='white', panel=2, width=1, secondary_y=True)]
                """
                apdict, num_panel = get_add_plot(stock_dataframe, indicator)
                #print(num_panel)
                #print(apdict)
                img_buf = io.BytesIO()
                if num_panel==1:
                    mpf.plot(stock_dataframe, volume=True, tight_layout=False, figsize=(img_size[0], img_size[1]),
                        style=ms, linecolor='white', addplot=apdict, type=chart_type, axisoff=True, update_width_config=dict(volume_width=0.3), savefig=img_buf)
                elif num_panel==2:
                    mpf.plot(stock_dataframe, volume=True,tight_layout=True, figsize=(img_size[0], img_size[1]), panel_ratios=(3,1,3),
                        style=ms, linecolor='white', addplot=apdict, type=chart_type, axisoff=True, update_width_config=dict(volume_width=0.3), savefig=img_buf)
                #plt.show()
                #exit()
        elif chart_type=="ha": # Heikin Ashi
            fig = mpf.figure(style=ms ,figsize=(img_size[0], img_size[1]))

            ax1 = fig.add_axes([0, 0.2, 1, 0.80])
            ax2 = fig.add_axes([0, 0, 1, 0.2], sharex=ax1)
            apdict = mpf.make_addplot(stock_dataframe["MA"], color="white", width=1, ax=ax1)
            ha_df = heikin_ashi(stock_dataframe)
            mpf.plot(ha_df, ax=ax1, volume=ax2, tight_layout=True,
                    style=ms, linecolor='white', addplot=apdict, type="candle", axisoff=True, update_width_config=dict(volume_width=0.3), 
                    columns=['HAOpen','HAHigh','HALow','HAClose','Volume'])
            #plt.show()
        elif chart_type=="candle": # candle stick
            fig = mpf.figure(style=ms ,figsize=(img_size[0], img_size[1]))

            ax1 = fig.add_axes([0, 0.2, 1, 0.80])
            ax2 = fig.add_axes([0, 0, 1, 0.2], sharex=ax1)
            apdict = mpf.make_addplot(stock_dataframe["MA"], color="white", width=1, ax=ax1)
            mpf.plot(stock_dataframe,
                    ax=ax1,
                    volume=ax2, tight_layout=True,
                    style=ms, linecolor='white', addplot=apdict, type=chart_type, axisoff=True, update_width_config=dict(volume_width=0.3))
            #plt.show()
            
        # save plt figure to buffer
        #if indicator!="MA,MACD":
        #    img_buf = io.BytesIO()
        #    plt.savefig(img_buf, format='png')

        # open image from buffer and convert into B&W directly
        bw_im = Image.open(img_buf).convert("1", dither=Image.NONE)#.point(lambda x : 255 if x > 0 else 0, mode='1')
        plt.close()
        return bw_im

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
    elif opt.num_classes==3:
        folder1 = f"{opt.data_path}/Positive"
        folder2 = f"{opt.data_path}/Negative"
        folder3 = f"{opt.data_path}/Non-significant"
        os.makedirs(folder1, exist_ok=True)
        os.makedirs(folder2, exist_ok=True)
        os.makedirs(folder3, exist_ok=True)

    # set image size
    img_size = set_img_size(time_interval, opt.indicator)

    if opt.stock=="SP500":
        ticker_list = pd.read_csv("SP500_constituents.csv")["Symbol"].to_list()
    elif opt.stock=="NASDAQ":
        ticker_list = pd.read_csv("filtered_NASDAQ.csv")["wind_code"].apply(lambda x: x.split(".")[0]).to_list()
    elif opt.stock=="NYSE":
        ticker_list = pd.read_csv("filtered_NYSE.csv")["wind_code"].apply(lambda x: x.split(".")[0]).to_list()
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
            #plotting_data = plotting_data.apply(lambda x:(x-x.mean())/x.std()) #标准化
            # 看R天后增减
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
            #画图
            if os.path.exists(save_name):
                i = i + R
                count_data += 1
                continue
                    
            plotting_data.iloc[:, -2:-1] = plotting_data.iloc[:, -2:-1].apply(lambda x: x / x.std())  # Vol部分标准化
            image = draw_pic(plotting_data , opt.type, img_size, opt.indicator)
            image.save(save_name)
            count_data += 1
            i = i + R
            #if count_data == 10000:
            #    exit()
            
            # for debugging
            #exit()
        print(f"{ticker} Completed", flush=True)
    print(f"Number of images plotted: {count_data}")





