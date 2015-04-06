import numpy as np
import statsmodels.api as sm
import pandas as pd

loansData=pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
# convert to Decimal Percentage
loansData['Interest.Rate']=loansData['Interest.Rate'].map(lambda x: round(float(x.strip('%'))/100,4))
# remove months from Loan.Length records
loansData['Loan.Length'] = loansData['Loan.Length'].map(lambda x: int(x.rstrip(' months')))
# split FICO.Range into a list of low and high
loansData['FICO.Range']=loansData['FICO.Range'].map(lambda x: x.split('-'))
# select minimum value in list as the FICO.Score and create new column FICO.Score and fill with records.
loansData['FICO.Score']=loansData['FICO.Range'].map(lambda x: int(min(x)))

intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

y = np.matrix(intrate).transpose()
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

x = np.column_stack([x1,x2])

X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

print 'Coefficients: ', f.params[0:2]
print 'Intercept: ', f.params[2]
print 'P-Values: ', f.pvalues
print 'R-Squared: ', f.rsquared