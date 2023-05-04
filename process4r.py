# Purpose: Process data for R
import pandas as pd
import numpy as np

aud = pd.read_csv('data/AUD_v2.csv')
aud['tri'] = aud.tri.values + 50
aud['log_return'] = aud['tri'].apply(np.log).diff()
aud.dropna(how='any', inplace=True)

log_return = aud['log_return'].values
train_test_split = aud[aud.timestamp.str.contains('2019')].index[-1] + 1
train = log_return[:train_test_split]
test = log_return[train_test_split:]

np.savetxt('data/train.csv', train, delimiter=',')
np.savetxt('data/test.csv', test, delimiter=',')