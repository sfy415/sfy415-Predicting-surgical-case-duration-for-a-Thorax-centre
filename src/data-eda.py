# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings(action='ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('max_columns',50)
pd.set_option('max_colwidth',150)


data_dir='../data/'
tmp_dir='../tmp/'


def is_number(s):
    s=str(s)
    try:
        float(s)
        return True
    except ValueError:
        return False

# comma convert to dot(decimal)
def comma_to_dot(x):
    if ',' in str(x):
        x=str(x).replace(',','.')
    if 'Onbekend' in str(x): # Onbekend means 'NULL'
        x=np.nan
    return x


# plot miss ratio on samples
def plot_col_scatter(df,col,img_name='miss_ratio'):
    plt.figure(figsize=(8,6))
    plt.scatter(range(df.shape[0]), np.sort(df[col].values))
    plt.xlabel('index', fontsize=12)
    plt.ylabel('%s'%img_name, fontsize=12)
    plt.savefig(tmp_dir+'%s.png'%img_name)
    # plt.show()
    plt.close()


# plot the frequency histogram of miss ration on columns
def plot_col_hist(df,col,img_name='miss_ratio_hist'):
    plt.figure(figsize=(12,8))
    sns.distplot(df[col].values, bins=50, kde=False, color="red")
    plt.title("Histogram of miss_ratio")
    plt.xlabel('miss_ratio', fontsize=12)
    plt.grid(True,ls=':')
    plt.savefig(tmp_dir+'%s.png'%img_name)
    # plt.show()


def map_cate2_feats(x):
    if x == 'V':
        return 0
    elif x == 'M':
        return 1

    if x == 'N':
        return 0
    elif x == 'J':
        return 1
    else:
        return np.nan


Surgery_type_values=['AVR','AVR + MVP','Bentall procedure','CABG','CABG + AVR','CABG + MVP',
                     'Epicardiale LV-lead','Lobectomie of segmentresectie','Mediastinoscopie',
                     'MVP','TVP','MVP + TVP','MVR','Nuss-procedure','Nuss bar','Refixatie sternum',
                     'Rethoracotomie','Staaldraden verwijderen','VATS Boxlaesie','wondtoilet','other types']


# map the Surgery_type into the specific fields
def clean_Surgery_type(x):
    if x in Surgery_type_values:
        return x
    else:
        return Surgery_type_values[-1]


def map_Time_of_day(x):
    if x=='Ochtend': #Ochtend:Morning
        return 1
    elif x=='Middag': #Middag:Afternoon
        return 2
    elif x=='Avond': #Avond:Evening
        return 3
    elif x=='Nacht': #Nacht:Night
        return 3
    else:
        return np.nan


def map_Urgency(x):
    if x=='Electief':
        return 0
    elif x=='Spoed < 24 uur':
        return 1
    elif x=='Acuut < 30 minuten':
        return 2
    elif x=='Spoed':
        return 3
    elif x=='Spoed < 5 uur':
        return 4
    else:
        return -1


def map_pulmonary_hypertension(x):
    if x=='Normaal': # Normal
        return 0
    elif x=='Matig': # Moderate
        return 1
    elif x=='Ernstig': # Severe
        return 2
    else: # Null
        return -1


# ETL(Extract-Transform-Load) the source data
def data_ETL(df):
    # rename the columns' name
    cols_dicts={'Operatietype':'Surgery_type','Benadering':'Surgical_approach','Chirurg':'Surgeon','Anesthesioloog':'Anesthesiologist',
                'OK':'Operation_room','Casustype':'Urgency','Dagdeel':'Time_of_day','Aantal anastomosen':'Amount_of_bypasses',
                'HLM':'Cardiopulmonary_bypass_use','Leeftijd':'Age','Geslacht':'Gender','AF':'Atrial_Fibrillation',
                'Chronische longziekte':'chronic_lung_disease','Extracardiale vaatpathie':'extracardial_arteriopathy',
                'Actieve endocarditis':'active_endocarditis','Hypertensie':'hypertension','Pulmonale hypertensie':'pulmonary_hypertension',
                'Slechte mobiliteit':'poor_mobility','Hypercholesterolemie':'hypercholesterolemia',
                'Perifeer vaatlijden':'peripherial_vascular_disease','Linker ventrikel':'Left_ventricle',
                'Nierfunctie':'Renal_function','DM':'diabetes_mellitus',
                'Eerdere hartchirurgie':'Previous_heart_surgery','Kritische preoperatieve status':'pre-OR_state',
                'Myocard infact <90 dagen':'Mycordial_infarct_before_surgery','Aorta chirurgie':'Aortic_surgery',
                'Euroscore1':'Euroscore1','Euroscore2':'Euroscore2','CCS':'CCS_score',
                'NYHA':'NYHA_score','Geplande operatieduur':'Planned_surgery_duration',
                'Operatieduur':'Surgery_duration','Ziekenhuis ligduur':'Hospital_days','IC ligduur':'Intensive_care_days'}
    df.rename(columns=cols_dicts,inplace=True)

    # count the sample miss ratio
    rows=df.shape[0]
    cols=df.shape[1]
    rows_miss_series=df.isnull().sum(axis=1)
    sample_miss_df=pd.DataFrame(rows_miss_series.values,index=df.index,columns=['miss_nums'])
    sample_miss_df['miss_ratio']=sample_miss_df['miss_nums'].apply(lambda x:float('%.3f'%(x/cols)) if x>0 else x)
    print(sample_miss_df.head())
    sample_miss_df.to_csv(tmp_dir+'sample_miss_ratio.csv')
    plot_col_scatter(sample_miss_df,col='miss_ratio',img_name='sample_miss_ratio')
    plot_col_hist(sample_miss_df,col='miss_ratio',img_name='sample_miss_ratio_hist')

    # count the miss ratio on columns
    cols_miss_series=df.isnull().sum(axis=0)
    cols_miss_df=pd.DataFrame(cols_miss_series.values,index=df.columns,columns=['miss_nums'])
    cols_miss_df['miss_ratio']=cols_miss_df['miss_nums'].apply(lambda x:float('%.3f'%(x/rows)) if x>0 else x)
    print(cols_miss_df.head())
    cols_miss_df.to_csv(tmp_dir+'columns_miss_ratio.csv')
    plot_col_scatter(cols_miss_df,col='miss_ratio',img_name='cols_miss_ratio')
    plot_col_hist(cols_miss_df,col='miss_ratio',img_name='cols_miss_ratio_hist')

    # delete the samples of the miss_ratio over the thresh=0.65
    df.drop(labels=sample_miss_df[sample_miss_df['miss_ratio']>0.65].index,axis=0,inplace=True)

    # delete the columns of miss_ratio over the thresh=0.7:[Renal_function、Left_ventricle、Euroscore2]
    df.drop(labels=cols_miss_df[cols_miss_df['miss_ratio']>0.75].index,axis=1,inplace=True)
    print('df.shape:',df.shape)
    df.to_excel(data_dir+'new_Data.xlsx',index=False)

    # Type conversion learning
    for col in ['Surgeon','Anesthesiologist','Euroscore1','BMI','Hospital_days','Intensive_care_days']:
        df[col]=df[col].apply(comma_to_dot)
    surgeon_max_val=np.max(df['Surgeon'].apply(lambda x: float(x) if is_number(x) else np.nan))
    df['Surgeon']=df['Surgeon'].apply(lambda x: surgeon_max_val+1 if x=='Ander specialisme' else x)

    # need convert the data type:int Surgeon/Anesthesiologist/Hospital_days/Intensive_care_days
    for col in ['Surgeon','Anesthesiologist','Hospital_days','Intensive_care_days']:
        df[col]=df[col].apply(float)
        df[col]=df[col].astype(np.int8,errors='ignore')

    # conver the data type float:Euroscore1/Euroscore2/BMI
    for col in ['Euroscore1','BMI']:
        df[col]=df[col].apply(float)
        df[col]=df[col].astype(np.float32)

    # clean these columns of category(2 unique values)
    cate2_lists=['Gender','Atrial_Fibrillation','chronic_lung_disease','extracardial_arteriopathy','Previous_heart_surgery',
                 'active_endocarditis','pre-OR_state','Mycordial_infarct_before_surgery','Aortic_surgery',
                 'poor_mobility','diabetes_mellitus','hypercholesterolemia','hypertension','peripherial_vascular_disease',
                 'Cardiopulmonary_bypass_use']
    for col in cate2_lists:
        print('---->',col)
        print(df[col].value_counts())
        df[col]=df[col].apply(map_cate2_feats)

    # clean the column of Surgery_type
    df['Surgery_type'].fillna('other types',inplace=True)
    df['Surgery_type']=df['Surgery_type'].apply(clean_Surgery_type)
    print('after cleaned:\n',df['Surgery_type'].value_counts())
    df.to_excel(data_dir+'new_Data2.xlsx',index=False)

    # Label encode the categorical columns(>2 unique values)
    df['Time_of_day']=df['Time_of_day'].apply(map_Time_of_day)
    df['Urgency']=df['Urgency'].apply(map_Urgency)
    df['pulmonary_hypertension']=df['pulmonary_hypertension'].apply(map_pulmonary_hypertension)
    multi_cates=['Surgery_type','Surgical_approach','Operation_room']
    df[multi_cates].fillna('NULL',inplace=True)

    for col in multi_cates:
        print('---->',col)
        if df[col].nunique()>30:
            ser=df[col].value_counts()
            ser.to_csv(tmp_dir+'%s.csv'%col)
        print(df[col].value_counts())
        enc=LabelEncoder()
        df[col]=enc.fit_transform(list(df[col]))

    df.reset_index(drop=True)
    df.to_excel(data_dir+'new_Data3.xlsx',index=False)

    # fill the little miss data
    print('before fill：\n',df.isnull().sum(axis=0))
    df['BMI'].fillna(df['BMI'].median(),inplace=True)
    df['Amount_of_bypasses'].fillna(df['BMI'].median(),inplace=True)
    df['CCS_score'].fillna(-1,inplace=True)
    df['NYHA_score'].fillna(-1,inplace=True)
    df['Anesthesiologist'].fillna(-1,inplace=True)
    df['Age'].fillna(df['Age'].median(),inplace=True)
    # delete the samples including  missing the Surgery_duration data
    df.dropna(axis=0,how='any',subset=['Surgery_duration'],inplace=True)
    print('after filled:\n',df.isnull().sum(axis=0))
    df.to_excel(data_dir+'new_Data4.xlsx',index=False)

    return df


def base_analysis(df,df_name='data'):
    stats = []
    for col in df.columns:
        stats.append((col, df[col].nunique(), df[col].isnull().sum()/ df.shape[0],
                      df[col].value_counts(normalize=True, dropna=False).values[0], df[col].dtype))

    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                            'Percentage of values in the biggest category', 'dtype'])
    stats_df.to_csv(tmp_dir+'%s_stats.csv'%df_name,index=False)
    print(stats_df.sort_values('Percentage of missing values', ascending=False)[:10])


if __name__=="__main__":
    data=pd.read_csv(data_dir+'surgical_case_durations.csv',sep=';',encoding='gbk')
    data2=data_ETL(data)
    base_analysis(data)
