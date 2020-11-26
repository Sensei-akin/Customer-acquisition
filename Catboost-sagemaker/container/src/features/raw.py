import pandas as pd
def read_data(path,input_col=None,**kwargs):
    """
        
        This funtion takes in a file path - CSV, excel or parquet and reads the data
        based on the input columns specified 
        
        Returns:
            dataset to be used for training
        
    """
        
    if path.endswith('.csv'):
        data = pd.read_csv(path, usecols = input_col)
        print('\nCSV file read sucessfully')
        data = data.reindex(columns = input_col)
        return data
            
    elif path.endswith('.parquet'):
        data = pd.read_parquet(path, engine = 'pyarrow', columns = input_col)
        print('Parquet file read sucessfully')
        data.columns = data.columns.astype(str)
        data = data.reindex(columns = input_col)
        return data
        
    elif path.endswith('.xls'):
        data = pd.read_excel(path, usecols = input_col)
        print('Excel file read success')
        data = data.reindex(columns = input_col)
        return data
        
    else:
        return ('No CSV file or Parquet file or Excel file was passed')
    