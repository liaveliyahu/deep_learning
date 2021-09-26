# Stock Trading with LSTM network
<p align="left">
  <img src="Assets\lstm.png" width="450">
</p>

## The Data
The data for this project is daily stock prices of Apple between 12/12/1980 to 9/23/2021 (AAPL).
Time series data need to be rearanged to be suitable for the LSTM network.

First, we need to prepare X and y so X will have all the stock prices and y (the target) will have the next day stock price depends the last few days price.
For example (y depends on the last 2 days): Data = [145, 150, 155, 160], X = [[145, 150],[150, 155]], y = [155, 160].
Second, need to reshape X with the following dimensions: (samples, time_steps, features).

After this "special" preprocessing, we need to split the data and scale it in order to get better results and evaluation.

## The Model
The model is pretty simple - includes only one LSTM layer and one Dense layer.

| Layer (type)  | Output Shape | Param # |
| ------------- | ------------ | ------- |
| lstm (LSTM)   | (None, 4)    | 96      |
| dense (Dense) | (None, 1)    | 5       |

Total params: 101                           
Trainable params: 101                            
Non-trainable params: 0                                                       

## Results
### Loss
<p align="left">
  <img src="Assets\loss.png" width="450">
</p>
The loss funcion converged immediately on the first epoch with th train set, and after only 20 epochs of the validation set.

## Predictions
<p align="left">
  <img src="Assets\real_vs_predicted.png" width="450">
</p>
We got prefect fit with the train set, and pretty good fit with the test set.

## Test predictions with an agent
<p align="left">
  <img src="Assets\agent_performance.png" width="450">
</p>
I created an agent that each day decide to buy or sell stocks depeneds the next day prediction.
The agent buy or sell the maximum stocks each time.
The agent got 5,000$ at the begging and finished with ~147,000$.
Actually it better to Buy and Hold the stock than use the agent.

<p align="left">
  <img src="Assets\lagging.png" width="450">
</p>
The main reason it happened is because the lagging in the predictions -
means the predictions lagging 1 day after the real prices and makes it difficult to predict the real price.

## References
* Data obtained from: https://finance.yahoo.com/
