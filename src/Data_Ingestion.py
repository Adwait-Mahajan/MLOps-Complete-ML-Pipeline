import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging Configuration
## Made the logger object from logging module
logger = logging.getLogger('Data_Ingestion') #creating logger object with name data ingestion
logger.setLevel('DEBUG') #setting a level called Debug. Basic level information is what it gives us. Least critical.

## Made the console handler using logging module
console_handler = logging.StreamHandler() #Console Handler -> Prints logs in the terminal
console_handler.setLevel('DEBUG')

## Made the file handler using logging module
log_file_path = os.path.join(log_dir, 'Data_Ingestion.log') # File Handler -> Saves logs as files. Thus made a new path for it.
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

## Made the fomatter to display outout in Time - Name - LevelName - Message format for both console and file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') #Time - Name - LevelName - Message
console_handler.setFormatter(formatter) # Set the console handler to formatter
file_handler.setFormatter(formatter) # Set the file handler to formatter

## Now we plan to send these console and file handlers back to the logger object we first created

logger.addHandler(console_handler)
logger.addHandler(file_handler)

#Yaml Setup
def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


# Data loading function now

def load_data(data_url: str) -> pd.DataFrame:
    """Load Data from a CSV File."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data Loaded from %s',data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s',e)
        raise
    except Exception as e:
        logger.error('Unnexpected error occurred while loading the data: %s',e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
        df.rename(columns={"v1":"target","v2":"text"}, inplace=True)
        logger.debug('Data Preprocessing Completed.')
        return df
    except KeyError as e:
        logger.error("Missing column in the dataframe: %s", e)
        raise
    except Exception as e:
        logger.error("Unnexpected error during preprocessing: %s",e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index= False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index= False)
        logger.debug("Train and Test Data Saved to %s", raw_data_path)
    except Exception as e:
        logger.error("Unnexpected error occurred while saving the data: %s", e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        #test_size = 0.2
        data_path = "https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/spam.csv"
        df = load_data(data_url=data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data, test_data, data_path='./data') #Makes a folder at the root of the project named data
    
    except Exception as e:
        logger.error('Failed to complete the Data Ingestion Process: %s',e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()