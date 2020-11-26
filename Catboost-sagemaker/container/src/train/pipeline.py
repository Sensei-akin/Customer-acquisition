# import sys
# sys.path.insert(0, '/home/ec2-user/SageMaker/Accessbank CTR/src')
from features.preprocess import identify_columns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from utils.utils import numerical_attribute,categorical_attribute,hash_features


def pipeline(hash_size):
        
    """
    
        Function contains the pipeline methods to be used.
        It is broken down into numerical, categorical and hash pipeline
            
    """
    num_pipeline = Pipeline(steps= [('imputer', SimpleImputer(strategy='mean')), ('std_scaler', MinMaxScaler())])
    cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                                    ('one_hot_encoding', OneHotEncoder(handle_unknown = "ignore", sparse = False))])
    hash_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                                ('hasher', FeatureHasher(n_features=hash_size, input_type='string'))])
    
    return num_pipeline,cat_pipeline,hash_pipeline

def train_test(data,hash_size,test_size):
    identify_columns(data,high_dim=hash_size, verbose=True)
    y = data['event_type']
    X = data.drop(['event_type'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y,stratify = y,test_size=test_size)
    return X_train, X_test, y_train, y_test

def fit_transform(data, hash_size, test_size):
    
    """
    
    Function that builds the pipeline and returns the 
    pipeline object and the data to be used for modeling
            
    Args:
        hash_bucket size
    
    Returns:
        pipeline object
        data to be used for training after being transformed by the pipeline
    
    """

    num_pipeline,cat_pipeline,hash_pipeline = pipeline(hash_size)
    full_pipeline = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numerical_attribute),
        ('cat', cat_pipeline, categorical_attribute),
        ('hash', hash_pipeline, hash_features)
    ])
    X_train, X_test, y_train, y_test = train_test(data,hash_size,test_size)
    
    full_pipeline.fit(X_train)
    
    X_train = full_pipeline.transform(X_train)
    X_test = full_pipeline.transform(X_test)
    
    print(X_train.shape)
    return X_train, X_test, y_train, y_test, full_pipeline