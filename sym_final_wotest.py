from cgi import print_directory
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
from json import load
import matplotlib.dates as mdates

def rysuj_model(df_dane, model, test_months):
    predictions = model.predict(start = 0, end = len(df_dane)-1)
    g = sns.lineplot(data=df_dane, x = 'date', y = 'new_cases_per_million', label = 'dane')
    sns.lineplot(data = predictions, label= 'model', ax= g)
    plt.axvspan(*test_months, facecolor='grey', alpha=0.25, axes = g)
    #return g

warnings.simplefilter('ignore', ConvergenceWarning)

with open("../data/parameters.json") as f:
    parameters = load(f)

path = parameters['data_path']

df = pd.read_csv(path, index_col = 'date', parse_dates= True)
df_poland = df.loc[df['location'] == 'Poland']
df_poland_nc = df_poland.loc[:, ['new_cases_per_million']]
df_poland_nc.index.freq = 'D'
df_poland_nc_cut_ = df_poland_nc.loc[df_poland_nc.index >= '2021-09-01']

#df_poland_nc_cut = df_poland_nc_cut_.loc[df_poland_nc_cut_.index <= '2022-01-04']

df_poland_nc_cut = df_poland_nc_cut_

plt.rcParams["figure.figsize"] = [15, 8]
sns.set_theme(style="white")
# sns.lineplot(data=df_poland_nc_cut, x = 'date', y = 'new_cases_per_million')

# results = seasonal_decompose(df_poland_nc_cut['new_cases_per_million'])
# results.plot();
# plt.show()

test_range = parameters['test_range']

train = df_poland_nc_cut
#test = df_poland_nc_cut.iloc[-test_range:]

#test_months = [test.index[0], test.index[-1]]

# sns.lineplot(data=train, x = 'date', y = 'new_cases_per_million', label = 'train')
# sns.lineplot(data=test, x = 'date', y = 'new_cases_per_million', label = 'test')
# plt.axvspan(*test_months, facecolor='grey', alpha=0.25)
# plt.show()

population_list = []

sigma_0 = parameters['sigma_0']
sigma_0_list = np.array([sigma_0 for _ in range(3)])

mi = parameters['mi']
lamb = parameters['lambda']

L = 3

tau_0 = 1/np.sqrt(2 * L)
tau_1 = 1/np.sqrt(2* np.sqrt(L))

c_d = parameters['c_d']
c_e = parameters['c_e']

k = parameters['k']

max_sigma  = parameters['max_sigma']
min_sigma = parameters['min_sigma']
predictions = parameters['predictions']

for _ in range(mi):
    
    alpha, beta, gamma = np.random.random(3)
    
    model = ExponentialSmoothing(endog = train['new_cases_per_million'],dates= train.index, trend = 'add', seasonal = 'mul', seasonal_periods = 7, freq = 'D').fit(smoothing_level = alpha, smoothing_trend = beta, smoothing_seasonal= gamma, optimized = True)
    
    osobnik = (model.sse, [alpha, beta, gamma], sigma_0_list)
    population_list.append(osobnik)

random_numbers = np.arange(0, mi, 1)
potomkowie_list = []

ilosc_mutacji = 0
ilosc_dobrych_mutacji = 0

#population_list = sorted(population_list, key=lambda x: x[0])
generations = parameters['generations']

for i in track(range(generations)):

    if (i + 1) % k == 0:
        
        fi_k = ilosc_dobrych_mutacji/ilosc_mutacji
        
        if fi_k > 0.2:
            for n in range(mi):
                population_list[n][2] = population_list[n][2] * c_e
                for j in range(L):
                    if population_list[n][2][j] > max_sigma:
                        population_list[n][2][j] = max_sigma
        elif fi_k < 0.2:
            for n in range(mi):
                population_list[n][2] = population_list[n][2] * c_d
                for j in range(L):
                    if population_list[n][2][j] < min_sigma:
                        population_list[n][2][j] = min_sigma
            
        ilosc_mutacji = 0
        ilosc_dobrych_mutacji = 0
    
    potomkowie_list.clear()

    for i in range(lamb):
        
        random.shuffle(random_numbers)
        a,b = random_numbers[0:2]
        
        rodzic1 = population_list[a]
        rodzic2 = population_list[b]
        
        alpha = random.choice([rodzic1[1][0], rodzic2[1][0]])
        beta = random.choice([rodzic1[1][1], rodzic2[1][1]])
        gamma = random.choice([rodzic1[1][2], rodzic2[1][2]])
        
        std_alpha = np.mean([rodzic1[2][0], rodzic2[2][0]])
        std_beta = np.mean([rodzic1[2][1], rodzic2[2][1]])
        std_gamma = np.mean([rodzic1[2][2], rodzic2[2][2]])
        
        parameters_list = np.array([alpha , beta, gamma])
        stdev_list = np.array([std_alpha, std_beta, std_gamma])
        
        N1 = np.random.normal(0,1)
        
        stdev_list = np.array([sigma*np.exp(tau_0 * N1 + tau_1 * np.random.normal(0,1)) for sigma in stdev_list])
        
        for j in range(L):
            
            if stdev_list[j] > max_sigma:
                stdev_list[j] = max_sigma
            elif stdev_list[j] < min_sigma:
                stdev_list[j] = min_sigma
                
        model = ExponentialSmoothing(endog = train['new_cases_per_million'],dates= train.index, trend = 'add', seasonal = 'mul', seasonal_periods = 7, freq = 'D').fit(smoothing_level = alpha, smoothing_trend = beta, smoothing_seasonal= gamma, optimized = True)
        
        sse_pre = model.sse
        
        alpha = parameters_list[0] + np.random.normal(0, stdev_list[0])
        beta = parameters_list[1] + np.random.normal(0, stdev_list[1])
        gamma = parameters_list[2] + np.random.normal(0, stdev_list[2])
        
        while not (0 < alpha < 1):
            alpha = parameters_list[0] + np.random.normal(0, stdev_list[0])
            
        while not (0 < beta < 1):
            beta = parameters_list[1] + np.random.normal(0, stdev_list[1])
            
        while not (0 < gamma < 1):
            gamma = parameters_list[2] + np.random.normal(0, stdev_list[2])
            
        model = ExponentialSmoothing(endog = train['new_cases_per_million'],dates= train.index, trend = 'add', seasonal = 'mul', seasonal_periods = 7, freq = 'D').fit(smoothing_level = parameters_list[0], smoothing_trend = parameters_list[1], smoothing_seasonal= parameters_list[2], optimized = True)
        
        sse_aft = model.sse
        
        if sse_aft < sse_pre:
            ilosc_dobrych_mutacji += 1
        
        potomek = [sse_aft, parameters_list, stdev_list]
        potomkowie_list.append(potomek)
        
        ilosc_mutacji += 1
        
    potomkowie_list = sorted(potomkowie_list, key=lambda x: x[0])
    population_list = potomkowie_list[:mi]

alpha = population_list[0][1][0]
beta = population_list[0][1][1]
gamma = population_list[0][1][2]

model = ExponentialSmoothing(endog = train['new_cases_per_million'],dates= train.index, trend = 'add', seasonal = 'mul', seasonal_periods = 7, freq = 'D').fit(smoothing_level = alpha, smoothing_trend = beta, smoothing_seasonal= gamma, optimized = True)

model.save('models/model_temp')
print(f'SSE = {model.sse}')
print(f'{alpha =}, {beta =}, {gamma =}')
predictions = model.predict(start = 0, end = len(df_poland_nc_cut) + predictions)
g = sns.lineplot(data=df_poland_nc_cut, x = 'date', y = 'new_cases_per_million', label = 'dane')
sns.lineplot(data = predictions, label= 'model', ax= g)
plt.title('Dzienna liczba zachorowań na COVID-19 w Polsce na 1mln mieszkańców')
plt.xlabel('data')
plt.ylabel('dzienne zachorowania')
g.xaxis.set_major_locator(mdates.DayLocator(interval=15))

plt.show()

