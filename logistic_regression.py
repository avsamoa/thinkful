import pandas as pd
import statsmodels.api as sm
import math

loansData=pd.read_csv('loansData_clean.csv')
loansData['IR_TF']=loansData['Interest.Rate']<.12
loansData['Intercept']

ind_vars=['FICO.Score','Amount.Requested','Intercept']
logit=sm.Logit(loansData['IR_TF'],loansData[ind_vars])
result=logit.fit()

coeff=result.params
print coeff

def logistic_function(fico,loanamt,coeff):

    z = coeff[0]*fico + coeff[1]*loanamt + coeff[2]

    return 1/(1+math.exp(-1*z))
