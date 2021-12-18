######### Libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
import matplotlib.pylab as plt
import statsmodels.api as sm
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

######### Reading Data from Excel
data = pd.read_excel("data.xlsx", index_col=0)

######### Splitting Data
dataframe = pd.DataFrame(data)
split_length = int((60/100)*len(dataframe))
train = dataframe.iloc[:split_length+1,]
test = dataframe.iloc[split_length:,]

########## Creating Time Series object
timeseries = pd.Series(train['Close'])  
log_timeseries = np.log(timeseries)

########## Looking at Data
fig = go.Figure()
fig.add_trace(go.Scatter(x=dataframe.index, y=train['Close'],
        mode='lines',
        name='Training Values'))
fig.add_trace(go.Scatter(x=test.index, y=test['Close'],
        mode='lines',
        name='Testing Values'))
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price($)"
)
fig.update_layout(plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color="white",
            margin=dict(l=50, r=50, b=100, t=100,pad=4))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
# fig.show()

########## Looking at Data Statistics
dftest = adfuller(timeseries, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic',
                                'P-value',
                                'Number of Lags',
                                'Number of Observations'])
for key, value in dftest[4].items():
    dfoutput['Critical Value {%s}'%key] =value
dfoutput = dfoutput.to_frame()
# print(dfoutput)

########## Rolling Mean and Standard Deviation
rolling1 = dataframe.copy()
rolling1['Rolling Mean']= rolling1['Close'].rolling(12).mean()
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=rolling1.index, y=rolling1['Close'],
        mode='lines',
        name='Closing Price'))
fig1.add_trace(go.Scatter(x=rolling1.index, y=rolling1['Rolling Mean'],
        mode='lines',
        name='Rolling Mean'))
fig1.update_layout(
    xaxis_title="Date",
    yaxis_title="Price($)"
)
fig1.update_layout(plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color="white",
            margin=dict(l=50, r=50, b=100, t=100,pad=4))
fig1.update_xaxes(showgrid=False)
fig1.update_yaxes(showgrid=False)
rolling2 = dataframe.copy()
rolling2['Rolling Standard Deviation']= rolling2['Close'].rolling(12).std()
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=rolling2.index, 
                        y=rolling2['Rolling Standard Deviation'], 
                        fill='tozeroy',
                        mode='lines'))
fig2.update_layout(
    xaxis_title="Date",
    yaxis_title="Standard Deviation"
)
fig2.update_layout(plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color="white",
            margin=dict(l=50, r=50, b=100, t=100,pad=4))
fig2.update_xaxes(showgrid=False)
fig2.update_yaxes(showgrid=False)
# fig1.show()
# fig2.show()

######### Auto Arima
model = pm.auto_arima(log_timeseries, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(log_timeseries)
forecast = model.predict(len(test))

######## Rolling Prediction
history = list(y for y in train['Close'])
predictions = []
for t in range(len(test)):
    model = ARIMA(history, order=(0,1,2))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test['Close'][t]
    history.append(obs)
results = test.copy()
results['Predicted Values']=predictions
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=dataframe.index, y=train['Close'],
        mode='lines',
        name='Training Values'))
fig3.add_trace(go.Scatter(x=test.index, y=results['Close'],
        mode='lines',
        name='Testing Values'))
fig3.add_trace(go.Scatter(x=test.index, y=results['Predicted Values'],
        mode='lines',
        name='Predicted Values'))
fig3.update_layout(
    xaxis_title="Date",
    yaxis_title="Price($)"
)
fig3.update_layout(plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color="white",
            margin=dict(l=50, r=50, b=100, t=100,pad=4))
fig3.update_xaxes(showgrid=False)
fig3.update_yaxes(showgrid=False)
# fig3.show()

###### Calculating R-squared
correlation_matrix = np.corrcoef(results['Close'],results['Predicted Values'])
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print(r_squared)