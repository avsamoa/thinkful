import pandas as pd
import statsmodels.api as sm

df=pd.read_csv('loansData_clean.csv')

df['Int.Rate.Less.12']=df['Interest.Rate'].map(lambda x: x<=.12)

