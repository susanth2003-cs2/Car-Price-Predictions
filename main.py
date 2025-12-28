import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('main')

from sklearn.model_selection import train_test_split
from random_sample import RSITechnique
from var_out import VT_OUT
#from feature_selection import FEATURE_SELECTION
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from imbalanced_data import SCALE_DATA
from models import CAR_PRICE_MODEL

class CAR_PRICE_PREDICTOR:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(path)

            logger.info('Data loaded')
            logger.info(f'{self.df.shape}')
            logger.info(f'{self.df.head()}')
            #Dependent column
            self.y = self.df['Selling_Price']
            # Independent columns (features)
            self.X = self.df.drop(['Selling_Price'], axis=1)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,random_state=42)
            logger.info(f'{self.y_train.sample(5)}')
            logger.info(f'{self.y_test.sample(5)}')
            logger.info(f'{self.X_train['Car_Name'].unique()}')

            logger.info(f'Training data size : {self.X_train.shape}')
            logger.info(f'Testing data size : {self.X_test.shape}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def missing_values(self):
        try:
            logger.info(f'Missing Values')
            logger.info(f"X_train columns: {self.X_train.columns}")
            logger.info(f"X_test columns: {self.X_test.columns}")

            if self.X_train.isnull().sum().any() > 0 or self.X_test.isnull().sum().any() > 0:
                self.X_train, self.X_test = RSITechnique.random_sample_imputation_technique(self.X_train, self.X_test)
            else:
                logger.info(f'No Missing Values')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def VarTrasform_Outliers(self):
        try:
            logger.info(f'Variable Transform Outliers Columns')
            logger.info(f"X_train columns: {self.X_train.columns}")
            logger.info(f"X_test columns: {self.X_test.columns}")

            self.X_train_num = self.X_train.select_dtypes(exclude = 'object')
            self.X_train_cat = self.X_train.select_dtypes(include = 'object')
            self.X_test_num = self.X_test.select_dtypes(exclude = 'object')
            self.X_test_cat = self.X_test.select_dtypes(include = 'object')

            logger.info(f'{self.X_train_num.columns}')
            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_num.columns}')
            logger.info(f'{self.X_test_cat.columns}')

            logger.info(f'{self.X_train_num.shape}')
            logger.info(f'{self.X_train_cat.shape}')
            logger.info(f'{self.X_test_num.shape}')
            logger.info(f'{self.X_test_cat.shape}')

            self.X_train_num, self.X_test_num = VT_OUT.variable_transformation_outliers(self.X_train_num, self.X_test_num)

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def cat_to_num(self):
        try:
            logger.info('Categorical to Numerical Conversion Started')

            # ------------------ Define categorical columns ------------------
            cat_cols = ['Car_Name', 'Fuel_Type', 'Seller_Type', 'Transmission']

            logger.info(f"Categorical Columns : {cat_cols}")
            logger.info(f"X_train_cat columns : {self.X_train_cat.columns}")
            logger.info(f"X_test_cat columns  : {self.X_test_cat.columns}")

            # ------------------ One Hot Encoder ------------------
            one_hot = OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore')

            # Fit only on training data
            one_hot.fit(self.X_train_cat[cat_cols])

            # ------------------ Transform Train ------------------
            train_encoded = one_hot.transform(self.X_train_cat[cat_cols])
            train_encoded_df = pd.DataFrame(train_encoded,columns=one_hot.get_feature_names_out(cat_cols))

            # ------------------ Transform Test ------------------
            test_encoded = one_hot.transform(self.X_test_cat[cat_cols])
            test_encoded_df = pd.DataFrame(test_encoded,columns=one_hot.get_feature_names_out(cat_cols))

            # Reset index before concat
            self.X_train_cat.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)

            train_encoded_df.reset_index(drop=True, inplace=True)
            test_encoded_df.reset_index(drop=True, inplace=True)

            # Drop original categorical columns
            self.X_train_cat = self.X_train_cat.drop(columns=cat_cols)
            self.X_test_cat = self.X_test_cat.drop(columns=cat_cols)

            # Add encoded columns
            self.X_train_cat = pd.concat([self.X_train_cat, train_encoded_df], axis=1)
            self.X_test_cat = pd.concat([self.X_test_cat, test_encoded_df], axis=1)

            # ------------------ Combine numerical + categorical ------------------
            self.X_train_num.reset_index(drop=True, inplace=True)
            self.X_test_num.reset_index(drop=True, inplace=True)

            self.training_data = pd.concat([self.X_train_num, self.X_train_cat], axis=1)
            self.testing_data = pd.concat([self.X_test_num, self.X_test_cat], axis=1)

            # ------------------ Logging ------------------
            logger.info(f"Training Data Shape : {self.training_data.shape}")
            logger.info(f"Testing Data Shape  : {self.testing_data.shape}")

            logger.info(f'Training Data : {self.training_data.head}')
            logger.info(f'Testing Data : {self.testing_data.head}')

            logger.info(f"Training Nulls:\n{self.training_data.isnull().sum()}")
            logger.info(f"Testing Nulls:\n{self.testing_data.isnull().sum()}")

            logger.info("Categorical to Numerical Conversion Completed Successfully")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error in Line no : {error_line.tb_lineno} due to {error_msg}')
    '''
    def fs(self):
        try:
            logger.info(f'Feature Selection')
            logger.info(f" Before : {self.training_data.columns} -> {self.training_data.shape}")
            logger.info(f"Before : {self.testing_data.columns} -> {self.testing_data.shape}")

            self.X_train, self.X_test = FEATURE_SELECTION.complete_feature_selection(self.training_data,self.testing_data,self.y_train)

            logger.info(f" After : {self.training_data.columns} -> {self.training_data.shape}")
            logger.info(f"After : {self.testing_data.columns} -> {self.testing_data.shape}")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
    '''
    def data_balance(self):
        try:
            logger.info('Scaling Data Before Regression')
            self.X_train, self.X_test = SCALE_DATA.scale(self.training_data, self.testing_data)
            CAR_PRICE_MODEL.train_linear_regression(self.X_train, self.y_train, self.X_test, self.y_test)

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

if __name__ == "__main__":
    try:
        obj = CAR_PRICE_PREDICTOR('C:\\Users\\Rajesh\\Downloads\\Mini Projects\\Predict Car Values\\car price.csv')
        obj.missing_values()
        obj.VarTrasform_Outliers()
        obj.cat_to_num()
        #obj.fs()
        obj.data_balance()

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

