import pandas as pd
import os

years = [i for i in range(1981,2021)]
path = "Processed_Data/"
for year in years:
    dates = os.listdir(path+str(year))
    dates_avg = [i for i in dates if 'average' in i]
    
    if len(dates_avg)==365:
        months = [dates_avg[0:31],dates_avg[31:59],dates_avg[59:90],dates_avg[90:120],dates_avg[120:151],dates_avg[151:181],dates_avg[181:212],dates_avg[212:243],dates_avg[243:273],dates_avg[273:304],dates_avg[304:334],dates_avg[334:]]

    if len(dates_avg)==366:
        months = [dates_avg[0:31],dates_avg[31:60],dates_avg[60:91],dates_avg[91:121],dates_avg[121:152],dates_avg[152:182],dates_avg[182:213],dates_avg[213:244],dates_avg[244:274],dates_avg[274:305],dates_avg[305:335],dates_avg[335:]]
    
    for month in months:     
        df = pd.read_csv(path+str(year)+'/'+month[0])
        df_above = df[['Above']]
        df_below = df[['Below']]
        
        for mnt in month[1:]:
            df1 = pd.read_csv(path+str(year)+'/'+mnt)
            df_above = pd.concat([df_above,df1['Above']],axis=1)
            df_below = pd.concat([df_below,df1['Below']],axis=1)
            
        df2 = df[['x','y']]
        df2['Above'] = df_above.mean(axis=1)
        df2['Below'] = df_below.mean(axis=1)
        df2.to_csv('Analysis/Data/Monthly'+'/'+mnt[5:7]+'/'+str(year)+'.csv',index=False)        