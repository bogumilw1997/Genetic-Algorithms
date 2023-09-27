import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from rich.console import Console
import rich.traceback
from rich.progress import track
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import random
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothingResults
from sklearn.metrics import r2_score
import matplotlib.dates as mdates

df = pd.read_csv('../data/owid-covid-data.csv', index_col = 'date', parse_dates= True)
df_poland = df.loc[df['location'] == 'Poland']
df_poland_tests = df_poland.loc[:, ['new_tests_per_thousand']]
df_poland_tests.index.freq = 'D'
df_poland_tests_ = df_poland_tests.loc[df_poland_tests.index >= '2021-09-01']
df_poland_nc = df_poland.loc[:, ['new_cases_per_million']]
df_poland_nc.index.freq = 'D'
df_poland_nc_cut_ = df_poland_nc.loc[df_poland_nc.index >= '2021-09-01']

#df_poland_nc_cut = df_poland_nc_cut_.loc[df_poland_nc_cut_.index <= '2022-01-04']

df_poland_nc_cut = df_poland_nc_cut_

plt.rcParams["figure.figsize"] = [15, 8]
sns.set_theme(style="white", font_scale=1.5)

test_range = 14

train = df_poland_nc_cut.iloc[:-test_range]
test = df_poland_nc_cut.iloc[-test_range:]
test_months = [test.index[0], test.index[-1]]

# sns.lineplot(data=df_poland_nc_cut, x = 'date', y = 'new_cases_per_million').set_title('Dzienna liczba zachorowań na COVID-19 w Polsce na 1mln mieszkańców')

#sns.lineplot(data=df_poland_tests_, x = 'date', y = 'new_tests_per_thousand').set_title('Dzienna liczba nowych testów na 1 tys. mieszkańców')
# results = seasonal_decompose(df_poland_nc_cut['new_cases_per_million'])
# results.plot();
#plt.show()

# sns.lineplot(data=train, x = 'date', y = 'new_cases_per_million', label = 'train')
# sns.lineplot(data=test, x = 'date', y = 'new_cases_per_million', label = 'test')
# sns.lineplot(data=df_poland_nc_cut, x = 'date', y = 'new_cases_per_million')
# plt.xlabel('data')
# plt.ylabel('dzienne zachorowania')
# plt.axvspan(*test_months, facecolor='grey', alpha=0.25)
# plt.title('Dzienna liczba zachorowań na COVID-19 w Polsce na 1mln mieszkańców')
# plt.show()

model_load = ExponentialSmoothingResults.load('models/model_temp')
print(model_load.summary())
predictions = model_load.predict(start = 0, end = len(df_poland_nc_cut)-1)

rmse=mean_squared_error(df_poland_nc_cut[-test_range:],predictions[-test_range:], squared=False)
train_rmse = np.sqrt(model_load.sse/len(train))
r2 = r2_score(train, predictions[:-test_range])
print(f'{rmse = }, {train_rmse = }, {r2 = }')

g = sns.lineplot(data=df_poland_nc_cut, x = 'date', y = 'new_cases_per_million', label = 'dane')
sns.lineplot(data = predictions, label= 'model').set_title('Dzienna liczba zachorowań na COVID-19 w Polsce na 1mln mieszkańców')
plt.axvspan(*test_months, facecolor='grey', alpha=0.25)
plt.xlabel('data')
plt.ylabel('dzienne zachorowania')

g.xaxis.set_major_locator(mdates.DayLocator(interval=15))
#g.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.show()
