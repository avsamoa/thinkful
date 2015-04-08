__author__ = 'avaai'

import pandas as pd
import statsmodels.api as sm
import numpy as np

loansData=pd.read_csv('loansData_clean.csv')
# Annual Income is Based on Monthly Income
loansData['Annual.Income']=loansData['Monthly.Income'].map(lambda x: x*12)

x1=np.matrix(loansData['Annual.Income']).transpose()
y=np.matrix(loansData['Interest.Rate']).transpose()

X = sm.add_constant(x1)
est = sm.OLS(y, X, missing='drop').fit()

est.summary()

