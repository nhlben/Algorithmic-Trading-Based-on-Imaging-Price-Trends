from __init__ import *

# functions to calculate different metrics

def plot_return(result_df, save_path, trade_interval, num_trade):
    interval = num_trade
    total_return_rates = [] # return rate
    index_return_rates = [] # return rate of the stock index used

    # get first and last trading date
    initial_date = result_df["buy_date"].to_numpy()[0].replace("/", "-") # replace / with - in the dates
    last_date = result_df["sell_date"].to_numpy()[-1].replace("/", "-") # replace / with - in the dates

    try:
        stock = result_df["stock"].to_numpy()[0]
        if stock=="SP500":
            index_ticker = "^GSPC"
        elif stock=="HSI":
            index_ticker = "^HSI"
        elif stock=="SHA":
            index_ticker = "000001.SS"
        elif stock=="NIKKEI":
            index_ticker = "^N225"
    except:
        # if stock column is not available
        stock = "SP500"
        index_ticker = "^GSPC"


    # get the value of the index
    index = yf.download(index_ticker, initial_date, datetime.strftime(datetime.strptime(last_date, "%Y-%m-%d")  + timedelta(days=1), "%Y-%m-%d"), interval="1d")

    initial_index_value = index["Adj Close"].to_numpy()[0]

    # calculate total return and index return
    for i in range(interval, len(result_df["total_return_rate"])+1, interval): # combine return within each interval
        total_return_rates.append(result_df.iloc[i-1]["total_return_rate"])
        date = result_df.iloc[i-1]["sell_date"].replace("/", "-") # replace / with - in the dates
        try:
            index_return_rates.append((index.loc[date]["Adj Close"]-initial_index_value) / initial_index_value * 100)
        except KeyError:
            date_before = result_df.iloc[i-1-interval]["sell_date"].replace("/", "-")

            index_return_rates.append((index.loc[date_before]["Adj Close"] - initial_index_value) / initial_index_value * 100)

    # plot portfolio return
    plt.style.use('default')
    plt.plot(np.arange(trade_interval, len(total_return_rates)*trade_interval+1, trade_interval), total_return_rates)
    plt.title("Total Return Rate Over Time")
    plt.xlabel("Number of Trading Days")
    plt.ylabel("Total Return Rate (%)")
    plt.savefig(f"{save_path}/return_rate.png")
    #plt.show()
    plt.close()

    # plot return + index
    plt.style.use('default')
    plt.plot(np.arange(trade_interval, len(total_return_rates)*trade_interval+1, trade_interval), total_return_rates, label="Our Method")
    plt.plot(np.arange(trade_interval, len(total_return_rates)*trade_interval+1, trade_interval), index_return_rates, label=f"{stock}")
    plt.legend()
    plt.title("Total Return Rate Over Time")
    plt.xlabel("Number of Trading Days")
    plt.ylabel("Total Return Rate (%)")
    plt.savefig(f"{save_path}/return_rate_withindex.png")

    plt.close()

    return

# calculate sharpe ratio
def sharpe_ratio(total_return, num_trade, T):
    total_returns = []
    for i in range(num_trade, len(total_return)+1, num_trade): # add up num_trade of rows
        total_returns.append(total_return[i-1])
    total_returns_diff = np.diff(total_returns)

    # assume an average annual risk-free rate over the period of 5%
    total_returns_diff  = total_returns_diff - 0.05/(252 / T)

    sr = np.sqrt(252 / T) * (np.mean(total_returns_diff) / np.std(total_returns_diff))  # 252 trading days
    return sr

def annualized_return(total_return_rate, num_trading_days):
    ar = total_return_rate / num_trading_days * 252
    return ar

def maximum_drawdown(total_return, num_trade):
    total_returns = []
    for i in range(num_trade, len(total_return)+1, num_trade): # add up num_trade of rows
        total_returns.append(total_return[i-1])
    result_df = pd.DataFrame({"total_return":total_returns})
    dr = result_df.pct_change(1)
    r=dr.add(1).cumprod()
    dd=r.div(r.cummax()).sub(1)
    mdd=dd.min()
    return -mdd[0]


def profit_factor(result_df): # gross profit divided by the gross loss
    gross_profit = result_df[result_df['trade_return'] > 0]['trade_return'].sum()
    gross_loss = - result_df[result_df['trade_return'] < 0]['trade_return'].sum()
    return gross_profit / gross_loss

def win_rate(result_df): # gross profit divided by the gross loss
    winning_trades = result_df[result_df['trade_return'] > 0]['trade_return'].count()
    return winning_trades / len(result_df) * 100

def average_win(result_df):
    winning_trades = result_df[result_df['trade_return_rate'] > 0]['trade_return_rate'].mean()
    return winning_trades

def average_loss(result_df):
    losing_trades = result_df[result_df['trade_return_rate'] < 0]['trade_return_rate'].mean()
    return -losing_trades

def excess_return(result_df):

    # get first and last trading date
    initial_date = result_df["buy_date"].to_numpy()[0].replace("/", "-") # replace / with - in the dates
    last_date = result_df["sell_date"].to_numpy()[-1].replace("/", "-") # replace / with - in the dates
    try:
        stock = result_df["stock"].to_numpy()[0]
        if stock=="SP500":
            index_ticker = "^GSPC"
        elif stock=="HSI":
            index_ticker = "^HSI"
        elif stock=="SHA":
            index_ticker = "000001.SS"
        elif stock=="NIKKEI":
            index_ticker = "^N225"
    except:
        stock = "SP500"
        index_ticker = "^GSPC"

    index = yf.download(index_ticker, initial_date, datetime.strftime(datetime.strptime(last_date, "%Y-%m-%d")  + timedelta(days=1), "%Y-%m-%d"), interval="1d")
    index_value = index["Adj Close"].to_numpy()
    index_return = (index_value[-1] - index_value[0]) / index_value[0] * 100
    trade_return = result_df.iloc[-1]['total_return_rate']
    #print(f"index return {index_return}")
    #print(f"trade return {trade_return}")


    return trade_return - index_return

# compute the trading interval (I)
def get_trade_interval(result_df):
    
    buy_date = result_df['buy_date'].to_numpy()
    first_date = buy_date[0]
    for date in buy_date:
        # get the second trading date
        if date!=first_date:
            second_date = date
            break
    try:
        stock = result_df["stock"].to_numpy()[0]
        if stock=="SP500":
            ex = mcal.get_calendar('NYSE')
            period = ex.schedule(start_date=first_date, end_date=second_date)
            trade_interval = len(mcal.date_range(period, frequency='1D'))
        elif stock=="HSI":
            ex = xcals.get_calendar("XHKG")
            trade_interval = len(ex.sessions_in_range(first_date, second_date))
        elif stock=="SHA":
            ex = xcals.get_calendar('XSHG')
            trade_interval = len(ex.sessions_in_range(first_date, second_date))
        elif stock=="NIKKEI":
            ex = xcals.get_calendar("XTKS")
            trade_interval = len(ex.sessions_in_range(first_date, second_date))
    except:
        stock = "SP500"
        ex = mcal.get_calendar('NYSE')
        period = ex.schedule(start_date=first_date, end_date=second_date)
        trade_interval = len(mcal.date_range(period, frequency='1D'))

    #print("trade_interval",trade_interval)
    return trade_interval - 1

def get_num_trading_days(result_df):

    initial_date = result_df["buy_date"].to_numpy()[0].replace("/", "-") # replace / with - in the dates
    last_date = result_df["sell_date"].to_numpy()[-1].replace("/", "-") # replace / with - in the dates

    try:
        stock = result_df["stock"].to_numpy()[0]
        if stock=="SP500":
            ex = mcal.get_calendar('NYSE')
            period = ex.schedule(start_date=initial_date, end_date=last_date)
            num_trading_days = len(mcal.date_range(period, frequency='1D'))
        elif stock=="HSI":
            ex = xcals.get_calendar("XHKG")
            num_trading_days = len(ex.sessions_in_range(initial_date, last_date))
        elif stock=="SHA":
            ex = xcals.get_calendar('XSHG')
            num_trading_days = len(ex.sessions_in_range(initial_date, last_date))
        elif stock=="NIKKEI":
            ex = xcals.get_calendar("XTKS")
            num_trading_days = len(ex.sessions_in_range(initial_date, last_date))
    except:
        stock = "SP500"
        ex = mcal.get_calendar('NYSE')
        period = ex.schedule(start_date=initial_date, end_date=last_date)
        num_trading_days = len(mcal.date_range(period, frequency='1D'))

    return num_trading_days

def get_num_trade(result_df):
    num_trade = 0
    buy_date = result_df['buy_date'].to_numpy()
    first_date = buy_date[0]
    for date in buy_date:
        if date!=first_date:
            break
        num_trade += 1
    return num_trade

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./backtest/I5R5', help='path for storing backtest result')
    parser.add_argument('--mode', type=str, default="single", help='End date of stock data')
    opt = parser.parse_args()
    print(opt, flush=True)

    # calculate number of trading days

    if opt.mode=="single": # evaluate single result.csv
        result_df = pd.read_csv(f"{opt.save_path}/result.csv")
        num_trade = get_num_trade(result_df)
        trade_interval = get_trade_interval(result_df)
        num_trading_days = get_num_trading_days(result_df)
        print(f"Number of trading days: {num_trading_days}")
        print(f"T: {get_trade_interval(result_df)}")
        print(f"N: {get_num_trade(result_df)}")
        print(f"Annualized Return Rate (%): { annualized_return(result_df.iloc[-1]['total_return_rate'], num_trading_days)}", flush=True)
        print(f"Maximum Total Loss (%) : {-result_df['total_return_rate'].min()}")
        print(f"Maximum Drawdown (%) : {maximum_drawdown(result_df['total_return'], num_trade)}")
        print(f"Annualized Sharpe Ratio: {sharpe_ratio(result_df['total_return'], num_trade, trade_interval)}", flush=True)
        print(f"Average Trade Return Win (%): {average_win(result_df)}", flush=True)
        print(f"Average Trade Return Loss (%): {average_loss(result_df)}", flush=True)
        print(f"Profit Factor: {profit_factor(result_df)}", flush=True)
        print(f"Win Rate (%): {win_rate(result_df)}", flush=True)
        print(f"Excess Return Rate (%): {excess_return(result_df)}", flush=True)
        plot_return(result_df, opt.save_path, trade_interval, num_trade)
        
    elif opt.mode=="all":
        opt.save_path = "./backtest"
        results = []
        for dir in os.listdir(opt.save_path):
            if os.path.isdir(f"{opt.save_path}/{dir}"):
                csv_path = f"{opt.save_path}/{dir}/result.csv"
                print(f"Reading {csv_path}")
                result_df = pd.read_csv(csv_path)
                #print(result_df.head())
                if "stock" not in result_df.columns:
                    stock = "SP500"
                else:
                    stock = result_df["stock"].iloc[0]
                start_date = result_df["buy_date"].iloc[0]
                end_date = result_df["sell_date"].iloc[len(result_df) - 1]

                num_trading_days = get_num_trading_days(result_df)
                print(f"Number of trading days:{num_trading_days}")
                result_dict = {}

                num_trade = get_num_trade(result_df)
                trade_interval = get_trade_interval(result_df)

                # plot return 
                # compute results
                result_dict["Model"] = dir
                result_dict["Start Date"] = start_date
                result_dict["End Date"] = end_date
                result_dict["Number of Trading Days"] = num_trading_days
                result_dict["T"] = trade_interval
                result_dict["N"] = num_trade
                result_dict["Annualized Return Rate (%)"] = annualized_return(result_df.iloc[-1]['total_return_rate'], num_trading_days)
                result_dict["Maximum Total Loss (%)"] = -result_df['total_return_rate'].min()
                result_dict["Maximum Drawdown (%)"] = maximum_drawdown(result_df['total_return'], num_trade)
                result_dict["Annualized Sharpe Ratio"] = sharpe_ratio(result_df['total_return'], num_trade, trade_interval)
                result_dict["Average Trade Return Win (%)"] = average_win(result_df)
                result_dict["Average Trade Return Loss (%)"] = average_loss(result_df)
                result_dict["Profit Factor"] = profit_factor(result_df)
                result_dict["Win Rate (%)"] = win_rate(result_df)
                result_dict["Excess Return (%)"] = excess_return(result_df)
                result_dict["Stock"] = stock
                results.append(result_dict)
                plot_return(result_df, f"{opt.save_path}/{dir}", trade_interval, num_trade)

        result_df = pd.DataFrame(results)
        result_df.to_csv(f"{opt.save_path}/all_results.csv", index=False)
