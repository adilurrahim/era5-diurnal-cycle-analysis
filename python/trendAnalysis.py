from scipy import stats
import pymannkendall as mk
import pandas as pd

def run_rowwise_loop(row):
    rho,pval = stats.spearmanr(row,years)
    trend = mk.original_test(row)
    #return ([rho,pval,trend.slope,trend.p])
    return pd.Series({'C':rho,'D':pval,'E':trend.slope,'F':trend.p})

years = [i for i in range(1981,2021)]
months = ['01','02','03','04','05','06','07','08','09','10','11','12']

input_path = 'Data/Monthly'
output_path = 'Data/monthlyTrend'

for month in months:
    path = input_path+'/'+month

    df = pd.read_csv(path+'/'+str(years[0])+'.csv')
    df_above = df[['Above']]
    df_below = df[['Below']]
    
    for year in years[1:]:
        df1 = pd.read_csv(path+'/'+str(year)+'.csv')
        df_above = pd.concat([df_above,df1['Above']],axis=1)
        df_below = pd.concat([df_below,df1['Below']],axis=1)
            
    df2 = pd.DataFrame()
    df2[['Spearman rho','Spearman pval','Slope','MK pval']] = df_above.apply(run_rowwise_loop, axis=1)
    df4 = pd.concat([df1[['x','y']],df2],axis=1)
    
    df3 = pd.DataFrame()
    df3[['Spearman rho','Spearman pval','Slope','MK pval']] = df_below.apply(run_rowwise_loop, axis=1)
    df5 = pd.concat([df1[['x','y']],df3],axis=1)

    df4.to_csv(output_path+'/TrendAnalysisAbove_'+ month +'.csv',index=False)
    df5.to_csv(output_path+'/TrendAnalysisBelow_'+ month +'.csv',index=False)