import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('scaling')

from sklearn.preprocessing import StandardScaler
import pickle


class SCALE_DATA:
    def scale(X_train, X_test):
        try:
            logger.info('Feature Scaling Started (Car Price Prediction)')
            logger.info(f'Train Data Before Scaling : {X_train.shape}')
            logger.info(f'{X_train.columns}')
            logger.info(f'{X_train.head}')
            logger.info(f'Test Data Before Scaling  : {X_test.shape}')
            logger.info(f'{X_test.head}')

            scaler = StandardScaler()

            # Fit only on training data
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns, index=X_train.index)

            # Transform test data
            X_test_scaled = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns,index=X_test.index)

            logger.info(f'Train Data After Scaling : {X_train_scaled.shape}')
            logger.info(f'{X_train_scaled.head}')
            logger.info(f'Test Data After Scaling  : {X_test_scaled.shape}')
            logger.info(f'{X_test_scaled.head}')

            # Save scaler for inference
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

            logger.info("Feature Scaling Completed Successfully")

            return X_train_scaled, X_test_scaled

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
