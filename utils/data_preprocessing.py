import pandas as pd

def load_and_clean_rainfall(file_path):
    df = pd.read_csv(file_path)
    for month in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']:
        df[month] = df.groupby('SUBDIVISION')[month].transform(lambda x: x.fillna(x.mean()))
    return df

def melt_rainfall(df):
    df_long = df.melt(id_vars=['YEAR','SUBDIVISION'],
                      value_vars=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'],
                      var_name='Month', value_name='Rainfall')
    month_map = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
                 'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
    df_long['Month_Num'] = df_long['Month'].map(month_map)
    return df_long
