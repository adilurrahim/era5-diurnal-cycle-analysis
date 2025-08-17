import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.stattools import durbin_watson
#perform Durbin-Watson test
years = [i for i in range(1981,2021)]
months = ['01','02','03','04','05','06','07','08','09','10','11','12']
df_final = pd.DataFrame()
list1 = []
for yr in years:
    for month in months:
        path = 'Analysis/Data/Monthly'+'/'+month
        df = pd.read_csv(path+'/'+str(yr)+'.csv')
        df_above = df[['Above']]
        df_final = pd.concat([df_final,df_above],axis=1)
        list1.append([yr,int(month)])
def run_rowwise_loop(row):
    row1 = pd.Series(row,name='Above').reset_index()
    X = np.arange(len(row1[['Above']]))
    X1 = pd.DataFrame(X)
    X1.columns = ['X']
    df_dw = pd.concat([X1,row1['Above']],axis=1)
    model = ols('Above ~ X', data=df_dw).fit()
    dw_stat = durbin_watson(model.resid)
    return pd.Series({'C':dw_stat})

df1 = pd.DataFrame()
df1[['DW test statistics']] = df_final.apply(run_rowwise_loop, axis=1)
df2 = pd.concat([df[['x','y']],df1],axis=1)
df2.to_csv('Analysis/Data/DW Test'+'/Overall.csv',index=False)