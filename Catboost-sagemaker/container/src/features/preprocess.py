import pandas as pd
import numpy as np
import yaml
from numpy import percentile
import sys
sys.path.insert(0, '/opt/program/src')

import matplotlib.pyplot as plt
from features import raw
from utils.utils import store_attribute
from utils.config import TRAIN_PATH_CLICK,TRAIN_PATH_SMS,TEST_PATH,PROCESSED_TRAIN_PATH,PROCESSED_TEST_PATH

def load_attributes(data):
    num_attributes = data.select_dtypes(include=[np.number]).columns.tolist()
    cat_attributes = data.select_dtypes(exclude=[np.number]).columns.tolist()
    try:
        num_attributes.remove("event_type")
        num_attributes.remove('customer_class')
        cat_attributes.append('customer_class')
        data['customer_class'] = self.data['customer_class'].astype(str)
    except:
        pass
    return num_attributes, cat_attributes

def identify_columns(data, high_dim=100, verbose=True, save_output=True):
    
    """
        
        This funtion takes in the data, identify the numerical and categorical
        attributes and stores them in a list
        
    """
    num_attributes, cat_attributes = load_attributes(data)
        
    low_cat = []
    hash_features = []
    dict_file = {}
    input_columns = [cols for cols in data.columns]
    input_columns.remove('event_type')
    input_columns.remove('msisdn.1')
    for item in cat_attributes:
        if data[item].nunique() > high_dim:
            if verbose:
                print('\n {} has a high cardinality. It has {} unique attributes'.format(item, data[item].nunique()))
            hash_features.append(item)
        else:
            low_cat.append(item)
    if save_output:
        dict_file['num_feat'] = num_attributes
        dict_file['cat_feat'] = cat_attributes
        dict_file['hash_feat'] = hash_features
        dict_file['lower_cat'] = low_cat
        dict_file['input_columns'] = input_columns
        store_attribute(dict_file)
        print('\nDone!. Data columns successfully identified and attributes are stored in /data/')
        
def remove_outliers(data):
        
    '''
    This function takes in the numerical data and removes outliers
    
    '''
    num_attributes, cat_attributes = load_attributes(data)
    for column in num_attributes:
        
        data.loc[:,column] = abs(data[column])
        mean = data[column].mean()

        #calculate the interquartlie range
        q25, q75 = np.percentile(data[column].dropna(), 25), np.percentile(data[column].dropna(), 75)
        iqr = q75 - q25

        #calculate the outlier cutoff
        cut_off = iqr * 1.5
        lower,upper = q25 - cut_off, q75 + cut_off

        #identify outliers
        outliers = [x for x in data[column] if x < lower or x > upper]
        
        derived_columns = ['investment_score', 'ctr_score', 'interactions_sms', 
                            'interactions_click', 'interaction_conversion', 
                            'loan_propensity']
        if column in derived_columns:
            pass
        else:
            data.loc[:,column] = data[column].apply(lambda x : mean 
                                                        if x < lower or x > upper else x)
            
    return data

def replace_region(data):
    
    North_Central = ['Benue','Kogi','Kwara','Nasarawa','Niger','Plateau','Abuja','Makurdi','Lokoja','Asokoro']
    North_East = ['Adamawa','Bauchi','Borno','Gombe','Taraba','Yobe','Jos','Minna','Maiduguri','Yola']
    North_West =  ['Jigawa','Kaduna','Kano','Katsina','Kebbi','Sokoto','Zamfara','Zaria','Dutse'] 
    South_West = ['Ekiti','Lagos','Ogun','Ondo','Osun','Oyo','Ikeja','Ikire','Badagri','Ikorodu','Ibadan',
                'Suleja','Ilorin','Abeokuta','Osogbo','Akure','Ede','Ikotun','Lekki','Ikoyi','Ota','Ojota',
                'Sagamu','Ogudu','Mowe','Agege','Omu-Aran','Aponri','Iponri','Ile-Ife','Ogbomoso']
    South_South = ['Akwa Ibom','Cross River','Bayelsa','Rivers','Delta','Edo','Benin City','Port Harcourt','Asaba',
                'Warri','Nsukka','Calabar','Uyo','Yenagoa','Eket','Sagbama','Bonny','Effurun']
    South_East = ['Abia','Anambra','Ebonyi','Enugu','Imo','Abakaliki','Owerri','Umuahia']
    africa = ['Nairobi', 'Eldoret', 'Pietermaritzburg', 'Pretoria', 'Midrand', 'Mitchells Plain', 'Thohoyandou']
    
    nc = data.location_state.isin(North_Central)
    ne = data.location_state.isin(North_East)
    nw = data.location_state.isin(North_West)
    sw = data.location_state.isin(South_West)
    ss = data.location_state.isin(South_South)
    se = data.location_state.isin(South_East)
    others = data.location_state.isin(africa)

    data.loc[nc, 'location_region'] = 'North_Central'
    data.loc[ne, 'location_region'] = 'North_East'
    data.loc[sw, 'location_region'] = 'South_West'
    data.loc[nw, 'location_region'] = 'North_West'
    data.loc[ss, 'location_region'] = 'South_South'
    data.loc[se, 'location_region'] = 'South_East'
    data.loc[others, 'location_region'] = 'others'

def _age(age):
    if  20 <= age <= 30:
        column = 'young'
    elif 30 < age <=50:
        column = 'middle_age'
    elif age>50:
        column = 'elder'
    else:
        column = 'missing'
        #raise ValueError(f'Invalid hour: {age}')
    return column

# concatenating name and version to form a new single column
def concat_feat(data):
    data['gender_location'] = data['gender'] + data['location_state']
    data['os_name_version'] = data['os_name'] + data['os_version'].astype(str)
    data['age_bucket'] = data.apply(lambda x: _age(x['age']), axis=1)
#         data['interactions'] = data['gender'] + data['age_bucket']
    
def check_nan(data):
    
    """
    
    Function checks if NaN values are present in the dataset for both categorical
    and numerical variables

    
    """
    missing_values = data.isnull().sum()
    count = missing_values[missing_values>1]
    print('\n Features       Count of missing value')
    print('{}'.format(count))

def handle_nan(data,strategy='mean',fillna='mode',drop_outliers=False):
    
    """
    
    Function handles NaN values in a dataset for both categorical
    and numerical variables

    Args:
        strategy: Method of filling numerical features
        fillna: Method of filling categorical features
    """
    
#     identify_columns(data)
    if drop_outliers: remove_outliers(data)
    num_attributes, cat_attributes = load_attributes(data)
    
    if strategy=='mean':
        for item in num_attributes:
            data.loc[:,item] = data[item].fillna(data[item].mean())
    if fillna == 'mode':
        for item in cat_attributes:
            data.loc[:,item] = data[item].fillna(data[item].mode())
    else:
        for item in data[cat_attributes]:
            if item == 'customer_class':
                data.loc[:,item] = data[item].fillna(data[item].mean())
            else:
                data.loc[:,item] = data[item].fillna(fillna)
    check_nan(data)
            
    return data

def drop_cols(data,columns):
    data = data.drop(columns,axis=1)
    return data
    
def map_target(data,column):
    data.loc[:,column] = data[column].map({'sms':0,'click':1})
    
    
def preprocess(dataframe=None,train=True,validation_path=None):
    if train:
        dataframe = dataframe[~dataframe['customer_class'].isnull()]
#         data = data.dropna(thresh=data.shape[1]*0.8)
        map_target(dataframe,'event_type')
        dataframe = handle_nan(dataframe,fillna='missing',drop_outliers=True)
        dataframe = drop_cols(dataframe,columns=['msisdn.1'])
        dataframe.to_pickle(PROCESSED_TRAIN_PATH)
        print(f'\nDone!. Input data has been preprocessed successfully and stored in {PROCESSED_TRAIN_PATH}')
        
    else:
        data = raw.read_data(path=validation_path)
        map_target(data,'event_type')
        data = handle_nan(data,fillna='missing',drop_outliers=True)
        data = drop_cols(data,columns=['msisdn.1'])

        data.to_pickle(PROCESSED_TEST_PATH)

        print(f'\nDone!. Input data has been preprocessed successfully and stored in {PROCESSED_TEST_PATH}')
    
    
if __name__ == '__main__':
    preprocess()