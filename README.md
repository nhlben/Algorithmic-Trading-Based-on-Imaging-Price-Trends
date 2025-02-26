# Algorithmic Trading Based on Imaging Price Trends
This is the project repository for DASC7600 Data Science Project: Algorithmic Trading Based on Imaging Price Trends. 
The model achieves a high annualized return rate and Sharpe Ratio, and can be used in different markets.

The project is greatly inspired by the paper: 

<b>J. JIANG, B. KELLY, and D. XIU, “(re-)imag(in)ing price trends,”
The Journal of Finance, vol. 78, no. 6, pp. 3193–3249, 2023.</b>

<b>Available: https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13268</b>

## Introduction

The foundational inspiration for this study stems from
the seminal work, (Re-)Imag(in)ing Price Trends by Jing-
wen Jiang, Bryan Kelly, and Dacheng Xiu. Their research
introduced an innovative methodology of treating stock price
charts as images, allowing convolutional neural networks
(CNNs) to process this visual information directly. By lever-
aging CNNs, their study uncovered intricate return-predictive
patterns that extended beyond traditional market phenomena,
such as momentum and reversal effects.

<p align="center">
  <img src="./figures/paper_result.png" width="800"><br>
  <em>Result of J. JIANG, B. KELLY, and D. XIU, “(re-)imag(in)ing price trends; MOM,STR,WSTR,TREND are traditional methods</em>
</p>

## Main Idea
The project advances the exploration of image-based stock
price prediction by leveraging cutting-edge deep learning
architectures and introducing enhancements to data repre-
sentation. Building upon the work of Jiang et al., we incor-
porate additional technical indicators and adopt optimized
visualization techniques to enrich the feature space available
for predictive modeling. Our approach seeks to maximize the
utility of CNNs and other advanced architectures in capturing
nuanced price patterns for improved return forecasting

## Methodology
In this repository, due to confidentiality reasons, some of the codes are hidden intentionally. All the model weights are not uploaded also.

### Plotting
The following parameters are used for plotting:
- <b>I</b> : Number of days of stock data used / time interval

- <b>R</b> : Number of days after for calculating return / time interval of return

Example: I=5, R=5: 5-days stock chart plot, compare the stock price the in last day with that of 5 days later.
Example: I=5, R=20: 5-days stock chart plot, compare the stock price the in last day with that of 20 days later.

The default setting used $I=5, R=5$.

The stocks are represented in OHLC chart in black and white, along with volume bars and a moving average line, as shown below
<p align="center">
  <img src="./examples/AEP_30.png" width="60"><br>
  <em>Example of a 5-day stock image generated</em>
</p>

Different chart type and indicators are also added and available.
<div align="center">

|<img src="./examples/AEP_40.png" width="60">|<img src="./examples/MMM_1120.png" width="80">|<img src="./examples/rsi.png" width="80">|
|:-:|:-:|:-:|
|Heikin Ashi Chart|MACD|RSI|

</div>
After plotting, all the images will be classified as Positive and Negative, according to their return R days later.

To plot and classify the images, run

`python plot.py`

There are multiple argument parameters available.
For example, to plot SSEC index stocks from 2001-01-01 to 2005-01-01 with I=20, R=20 and RSI indicator, run

`python plot.py --I 20 --R 20 --start_date 2001-01-01 --end_date 2005-01-01 --indicator RSI`


### Training

The image dataset is splitted with 70% training, 30% validation by default. The data are from 1993-01-01 to 2005-12-31.
The default CNN, DenseNet, EfficientNet and ViT are available for training. Notice that ViT resizes the image into square dimensions. The best model will be saved according to the save path.

To train the images, run
`python train.py`

Similarly, there are multiple argument parameters available.
For example, to train a DenseNet model with batch size of 256, learning rate of 1e-5 for 100 epochs, run

`python plot.py --model DENSENET --batch_size 256 --learning_rate 1e-5 --num_epoch 100`

Retraining a model from a certain `.pt` checkpoint is possible, one will have to specify the `model_path` parameter.

The training result are available below:

<p align="center">
  <img src="./figures/model_trained.png" width="500"><br>
  <img src="./figures/model_performance.png" width="500"><br>
</p>

### Backtesting
The overall flow for the backtesting is as follow:
<p align="center">
  <img src="./figures/flow.png" width="500"><br>
</p>

The following parameters are used for backtesting:
- <b>N</b> : Number of stocks traded in one operation

- <b>T</b> : Number of days between each trade / time interval of trading

- <b>C</b> : Initial size of capital

The default setting used $N=10, T=I=5, C=100000$ USD
After training, the models are backtested, with the following two trading strategies :

All the trades are equal weighted, each stock trade uses $\frac{C}{N}$ amount of money (equal weighted)

- Long Only: Buy the top $N$ stocks with the highest probability, sell them after $R$ days. The trades happens every $T$ days.
- Long and Short: Buy the top $\frac{N}{2}$ stocks with the highest probability, sell them after $R$ days. Short-selling the bottom $\frac{N}{2}$ stocks with the lowest probability, buy to cover the shorts after $R$ days. The trades happens every $T$ days.
- Random: Performs the Long Only or Long and Short strategy with randomly selected stocks.

The backtesting are performed within the period 2006-01-01 to 2015-12-31. Also, the capital used for trading is <b>fixed</b> in the following results.

## Results

The following shows the initial resuls to test the best backbone models:
<p align="center">
  <img src="https://github.com/user-attachments/assets/9ab4b26d-a7f0-4d1a-a309-59c1ef226d6a" width="500"><br>
</p>

With the above result, EffcientNet was chosen to be the backbone.

All the settings are experimented to find the best combination:
<p align="center">
  <img src="https://github.com/user-attachments/assets/8babb7d2-93e4-412f-8eaf-bff08f04916e" width="500"><br>
</p>

The best result for Long Only and Long and Short are both the I5R5 EfficientNet model, achieving a decent annualized return rate and Sharpe Ratio which surpasses the result of the paper.

The following are the graph of return to better show the result.
<p align="center">
  <img src="https://github.com/user-attachments/assets/55a28d7c-dd1e-4c47-a7a3-82a880366f0b" width="500"><br>
  <em>Result of I5R5 EfficientNet</em>
</p>

To ensure the fairness of the model, the random strategy are computed as well, with the following result:
<p align="center">
  <img src="https://github.com/user-attachments/assets/f7c8d401-890a-4e86-9113-02744e8f6edb" width="500"><br>
  <em>Comparison of I5R5 EfficientNet with Random</em>
</p>

Apart from the US market, the model is also backtested with the HSI, SSEC index, Nikkei 225 index, further proving its robustness.
<p align="center">
  <img src="https://github.com/user-attachments/assets/f406d81c-22af-47d7-9aae-85929ab4a834" width="500"><br>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/301f19f8-4c5b-40fb-9f23-5e5bf9809fb0" width="500"><br>
  <em>Result of I5R5 EffcientNet Long and Short on other markets</em>
</p>

## Further Improvement

The capital are compounded, i.e. the return are accumulated as capital as well. The results are shown in below:
<p align="center">
  <img src="https://github.com/user-attachments/assets/be0cda0d-59b7-4071-aa97-0af0d13ebab7" width="500"><br>
  <em>Result of I5R5 EffcientNet Long and Short from 2006-2015 with compounded capital</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/807d5861-1e23-4702-8fa9-6be2c6207249" width="500"><br>
  <em>Result of I5R5 EffcientNet Long and Short  from 2006-2023 with compounded capital</em>
</p>

The model achieves an excellent result with a 24.6% annualized return and 1.36 Sharpe Ratio from 2006-2015, while it obtains 27.79% and 0.89 from 2006-2023 respectively.






