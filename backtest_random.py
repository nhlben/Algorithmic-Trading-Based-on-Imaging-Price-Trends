from __init__ import *
from plot import *
from evaluate import plot_return, sharpe_ratio, annualized_return

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(7600)
np.random.seed(7600)


def backtest(stock_data, strategy, time_interval, rolling_window, trade_interval, num_trade, index):
    # time interval = I
    # rolling window  = R
    # trade interval = T
    # num_trade = N
    initial_capital = 100000.0 # USD
    if index=="HSI":
        initial_capital = initial_capital * 7.8 # to HKD
    elif index=="SHA":
        initial_capital = initial_capital * 7.2 # to CNY
    elif index=="NIKKEI":
        initial_capital = initial_capital * 160 # to JPY

    trades = []  # saves information of trades
    buy_date, sell_date, buy_prices, sell_prices, return_rate, return_value, total_return = [],[],[],[],[],[],[]
    i = 0
    log = ""
    data_len = len(list(stock_data.values())[0].index)

    # hidden

    # Calcuate the return
    total_return = 0 # keep track of total return ($)
    total_returns = [] # list to store total returns
    total_return_rate = 0 # keetp track of total return rate (%)
    total_return_rates = [] # list to store total return rates

    # initial capital = I
    # capital used for trading a stock is I / (R / T) / N = I / ((R/T) * N)
    # capital dividing factor F = (R / T) * N
    # capital used for trading a stock becomes I / (R / T) / N = I / F

    f = (rolling_window / trade_interval) * num_trade

    # hidden


    results = ({
            'buy_date': buy_date,
            'sell_date': sell_date,
            'buy_price': buy_prices,
            'sell_price': sell_prices,
            'trade_return_rate': [i * f for i in return_rate],
            'portfolio_return_rate': return_rate,
            'trade_return': return_value,
            'total_return': total_returns,
            'total_return_rate': total_return_rates,
            'stock': index
    })

    #print(log)
    return results, log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--I', type=int, default=5, help='Time Interval of chart time')
    parser.add_argument('--R', type=int, default=5, help='Return Interval')
    parser.add_argument('--save_path', type=str, default='./backtest/I5R5', help='path for storing backtest result')
    parser.add_argument('--start_date', type=str, default="2006-01-01", help='Start date of stock data')
    parser.add_argument('--end_date', type=str, default="2015-12-31", help='End date of stock data')
    parser.add_argument('--strategy', type=str, default="longonly", help='strategy used')
    parser.add_argument('--N', type=int, default=10, help='number of stocks to buy each time')
    parser.add_argument('--testing', type=bool, default=False, help='Testing')
    parser.add_argument('--T', type=int, default=0, help='Time Interval of trading')
    parser.add_argument('--stock', type=str, default="SP500", help='stock used to backtest')
    opt = parser.parse_args()
    print(opt, flush=True)

    os.makedirs(opt.save_path, exist_ok=True)
    time_interval = opt.I

    stock_data = dict()

    if opt.stock=="SP500":
        ex = mcal.get_calendar('NYSE')
        period = ex.schedule(start_date=opt.start_date, end_date=opt.end_date)
        num_trading_days = len(mcal.date_range(period, frequency='1D'))
        ticker_list = pd.read_csv("SP500_constituents.csv")["Symbol"].to_list()
    elif opt.stock=="HSI":
        ex = xcals.get_calendar("XHKG")
        num_trading_days = len(ex.sessions_in_range(opt.start_date, opt.end_date))
        ticker_list = pd.read_csv("HSI_2006.csv")["wind_code"].to_list()
    elif opt.stock=="SHA":
        ex = xcals.get_calendar('XSHG')
        num_trading_days = len(ex.sessions_in_range(opt.start_date, opt.end_date))
        ticker_list = pd.read_csv("SHA_2006.csv")["wind_code"].apply(lambda x: x.replace("SH", "SS")).to_list()
    elif opt.stock=="NIKKEI":
        ex = xcals.get_calendar("XTKS")
        num_trading_days = len(ex.sessions_in_range(opt.start_date, opt.end_date)) + 10
        ticker_list = pd.read_csv("NIKKEI_2006.csv")["wind_code"].to_list()

    if opt.testing:
        print("testing", flush=True)
        ticker_list = ticker_list[:20]
    
    if opt.T==0: # not set
        trade_interval = opt.I # set to I directly
    else:
        trade_interval = opt.T

    for ticker in ticker_list:
        try:
            data = yf.download(ticker, start=opt.start_date, end=datetime.strftime(datetime.strptime(opt.end_date, "%Y-%m-%d") + timedelta(days=1), "%Y-%m-%d")
                               , auto_adjust=True)  # last day not inclusive, manually add one
            if len(data) >= num_trading_days:
                data[f"MA"] = data["Close"].rolling(window=time_interval).mean()
                stock_data[f"{ticker}"] = data
                #print(stock_data)
        except:
            continue
    
    # backtest randomly
    results, log = backtest(stock_data,strategy=opt.strategy, time_interval=opt.I, 
                            rolling_window=opt.R, trade_interval=trade_interval, num_trade=opt.N, index=opt.stock)
    
    # sva results
    result_df = pd.DataFrame.from_dict(results)
    result_df.to_csv(f"{opt.save_path}/result.csv", index=False)
    num_trading_days = len(result_df["buy_date"].to_numpy()) / opt.N * opt.I
    log += f"Total Return Rate: { result_df.iloc[-1]['total_return_rate']}\n"
    print(f"Total Return Rate: { result_df.iloc[-1]['total_return_rate']}", flush=True)
    log += f"Annualized Return Rate: { annualized_return(result_df.iloc[-1]['total_return_rate'], num_trading_days)}\n"
    print(f"Annualized Return Rate: { annualized_return(result_df.iloc[-1]['total_return_rate'], num_trading_days)}", flush=True)
    log += f"Annualized Sharpe Ratio: {sharpe_ratio(result_df['total_return'], opt.N, trade_interval)}"
    print(f"Annualized Sharpe Ratio: {sharpe_ratio(result_df['total_return'], opt.N, trade_interval)}", flush=True)
    f = open(f"{opt.save_path}/log.txt", "w")
    f.write(log)
    f.close()

    # plot return rate
    result_df = pd.read_csv(f"{opt.save_path}/result.csv")
    plot_return(result_df, opt.save_path, trade_interval, opt.N)
