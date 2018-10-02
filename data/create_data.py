import numpy as np
import pandas as pd
import pickle



def read_csv(med_path='./PRESCRIPTIONS.csv', diag_path='./DIAGNOSES_ICD.csv', patient_path='./gather_firstday.csv'):
    """ 
    med_pd: subject_id, hadm_id, icustay_id, drug, ndc, STARTDATE
    diag_pd: subject_id, hadm_id, icd9
    side_pd: subject_id, hadm_id, icustay_id, ***
    """
    # load data
    med_pd = pd.read_csv(med_path, dtype={'NDC':'category'})
    diag_pd = pd.read_csv(diag_path)
    side_pd = pd.read_csv(patient_path)

    # filter
    med_pd.drop(columns=['ROW_ID','DRUG_TYPE','DRUG_NAME_POE','DRUG_NAME_GENERIC',
                     'FORMULARY_DRUG_CD','GSN','PROD_STRENGTH','DOSE_VAL_RX',
                     'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP','FORM_UNIT_DISP',
                      'ROUTE','ENDDATE','DRUG'], axis=1, inplace=True)
    med_pd.drop(index = med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')
    med_pd = med_pd.reset_index(drop=True)
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)

    #process diagnosis
    diag_pd.dropna(inplace=True)
    # def icd9_tree(x):
    #     if x[0]=='E':
    #         return x[:4] 
    #     return x[:3]
    # diag_pd['ICD9_CODE'] = diag_pd['ICD9_CODE'].map(icd9_tree)
    diag_pd.drop(columns=['SEQ_NUM','ROW_ID'],inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)

    #process side_information
    side_pd = side_pd.dropna(thresh=60)
    side_pd.fillna(side_pd.mean(), inplace=True)
    side_pd = pd.concat([side_pd,pd.get_dummies(side_pd['ethnicity'])],axis=1)
    side_pd.drop(columns=['ethnicity'],inplace=True)  

    return med_pd, diag_pd, side_pd

def ndc2atc4(med_pd):
    with open('./ndc2rxnorm_mapping.txt', 'r') as f:
        ndc2rxnorm = eval(f.read())
    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)

    rxnorm2atc = pd.read_csv('./ndc2atc_level4.csv')
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR','MONTH','NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(index = med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)
    
    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)
    med_pd = med_pd.rename(columns={'ATC4':'NDC'})
    # med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:4])
    med_pd = med_pd.drop_duplicates()    
    med_pd = med_pd.reset_index(drop=True)
    # print(med_pd['NDC'].unique().shape)
    return med_pd

def filter_first24hour_med(med_pd):
    med_pd_new = med_pd.drop(columns=['NDC'])
    med_pd_new = med_pd_new.groupby(by=['SUBJECT_ID','HADM_ID','ICUSTAY_ID']).head([1]).reset_index(drop=True)
    med_pd_new = pd.merge(med_pd_new, med_pd, on=['SUBJECT_ID','HADM_ID','ICUSTAY_ID','STARTDATE'])
    med_pd_new = med_pd_new.drop(columns=['STARTDATE'])
    return med_pd_new

def filter_2000_most_diag(diag_pd):
    diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:1999, 'ICD9_CODE'])]
    
    return diag_pd.reset_index(drop=True)

def filter_1000_most_med(pres_pd):
    pres_count = pres_pd.groupby(by=['NDC']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    pres_pd = pres_pd[pres_pd['NDC'].isin(pres_count.loc[:999, 'NDC'])]
    
    return pres_pd.reset_index(drop=True)


# reduce number of diag and medications
def filter_by_count(pres_pd, diag_pd):
    print('before process shape ----- pres_pd:{},diag_pd:{}'.format(pres_pd.shape, diag_pd.shape))
    diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    count_filter = 1100
    print('diagnois ratio: ',diag_count[diag_count['count']>count_filter]['count'].sum() / diag_count['count'].sum())
    print('diagnois number:',diag_count[diag_count['count']>count_filter].shape[0])
    diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count[diag_count['count']>count_filter]['ICD9_CODE'])]
    
    ## medicine count filter
    pres_count = pres_pd.groupby(by=['NDC']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    count_filter = 5000
    print('medications ratio: ',pres_count[pres_count['count']>count_filter]['count'].sum() / pres_count['count'].sum())
    print('medications number: ',pres_count[pres_count['count']>count_filter].shape[0])
    pres_pd = pres_pd[pres_pd['NDC'].isin(pres_count[pres_count['count']>count_filter]['NDC'])]
    
    print('after process shape ----- pres_pd:{},diag_pd:{}'.format(pres_pd.shape, diag_pd.shape))
    return pres_pd.reset_index(drop=True), diag_pd.reset_index(drop=True)

def merge_three_part(side_pd, pres_pd, diag_pd):
    intersection = side_pd[side_pd['subject_id'].isin(pres_pd['SUBJECT_ID'])]['subject_id']
    intersection = diag_pd[diag_pd['SUBJECT_ID'].isin(intersection)]['SUBJECT_ID'].drop_duplicates()
    
    side_pd = side_pd[side_pd['subject_id'].isin(intersection)].reset_index(drop=True)
    pres_pd = pres_pd[pres_pd['SUBJECT_ID'].isin(intersection)].reset_index(drop=True)
    diag_pd = diag_pd[diag_pd['SUBJECT_ID'].isin(intersection)].reset_index(drop=True)
    
    side_data = []
    pres_data = []
    diag_data = []
    unique_pres = []
    unique_diag = []
    for idx, row in side_pd.iterrows():
        if idx % 1000 ==0:
            print('{} of {}'.format(idx, side_pd.shape[0]))
        subject_id, hadm_id, icustay_id = row['subject_id'], row['hadm_id'], row['icustay_id']
        
        pres_rows = pres_pd[(pres_pd['SUBJECT_ID']==subject_id)
                            &(pres_pd['HADM_ID']==hadm_id)
                            &(pres_pd['ICUSTAY_ID']==icustay_id)]['NDC'].tolist()
        if len(pres_rows) == 0:
            continue
        
        diag_rows = diag_pd[(diag_pd['SUBJECT_ID']==subject_id)
                            &(diag_pd['HADM_ID']==hadm_id)]['ICD9_CODE'].tolist()
        if len(diag_rows) == 0:
            continue
        
        unique_diag.extend(diag_rows)
        unique_pres.extend(pres_rows)

        diag_data.append(diag_rows)
        pres_data.append(pres_rows)
        side_data.append(side_pd.iloc[idx,3:].values)
    
    unique_diag = list(set(unique_diag))
    unique_pres = list(set(unique_pres))
    with open('unique_diag.pkl', 'wb') as f:
        pickle.dump(unique_diag, f)
    with open('unique_pres.pkl', 'wb') as f:
        pickle.dump(unique_pres, f)

    with open('side_data.pkl','wb') as f:
        pickle.dump(side_data,f)
    with open('diag_data.pkl','wb') as f:
        pickle.dump(diag_data,f)    
    with open('pres_data.pkl','wb') as f:
        pickle.dump(pres_data,f)

    print('patient num:', len(diag_data))
    print('unique_diag:', len(unique_diag))
    print('unique_med:',len(unique_pres))
        
    print('save complete')

def split(path, ratio=0.8):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    total_len = len(data)
    
    with open(os.path.join('train', path), 'wb') as f:
        pickle.dump(data[: int(total_len*ratio)], f)
    with open(os.path.join('eval', path), 'wb') as f:
        pickle.dump(data[int(total_len*ratio): int(total_len*(ratio+0.1))], f)
    with open(os.path.join('test', path), 'wb') as f:
        pickle.dump(data[int(total_len*(ratio+0.1)):], f)


if __name__ == '__main__':
    med_pd, diag_pd, side_pd = read_csv()  
    med_pd = filter_first24hour_med(med_pd)
    med_pd = ndc2atc4(med_pd)

#     med_pd = filter_1000_most_med(med_pd)
    diag_pd = filter_2000_most_diag(diag_pd)

    merge_three_part(side_pd, med_pd, diag_pd)

    eval_path = 'eval'
    train_path = 'train'
    test_path = 'test'
    for i in [eval_path, train_path, test_path]:
        if not os.path.exists(i):
            os.mkdir(i)

    for i in ['diag_data.pkl', 'pres_data.pkl', 'side_data.pkl']:
        split(i)