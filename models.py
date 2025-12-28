import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('models')

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle


class CAR_PRICE_MODEL:
    def train_linear_regression(X_train, y_train, X_test, y_test):
        try:
            # ================= SAVE FEATURE NAMES =================
            feature_names = X_train.columns.tolist()
            with open("features.pkl", "wb") as f:
                pickle.dump(feature_names, f)

            # ================= LINEAR REGRESSION =================
            logger.info("Training Linear Regression Model (Car Price)")

            model = LinearRegression()
            model.fit(X_train, y_train)

            logger.info(f"Intercept : {model.intercept_}")
            logger.info(f"Coefficients : {model.coef_}")

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # ================= METRICS =================
            logger.info(f"Train R2 Score : {r2_score(y_train, y_train_pred)}")
            logger.info(f"Test R2 Score  : {r2_score(y_test, y_test_pred)}")

            logger.info(f"Train RMSE : {np.sqrt(mean_squared_error(y_train, y_train_pred))}")
            logger.info(f"Test RMSE  : {np.sqrt(mean_squared_error(y_test, y_test_pred))}")

            logger.info(f"Train MAE : {mean_absolute_error(y_train, y_train_pred)}")
            logger.info(f"Test MAE  : {mean_absolute_error(y_test, y_test_pred)}")

            # ================= SAVE MODEL =================
            with open("car_price_model.pkl", "wb") as f:
                pickle.dump(model, f)

            logger.info("Car Price Model Saved Successfully")

            # ================= SAMPLE PREDICTION =================
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            '''
            # Example car input (MUST MATCH feature order)
            # Example:
            # Year, Present_Price, Kms_Driven, Owner,
            # Fuel_Type_Diesel, Seller_Type_Individual, Transmission_Manual
            sample_input = np.array([[2017, 8.5, 45000, 0, 1, 0, 1]])

            sample_scaled = scaler.transform(sample_input)
            predicted_price = model.predict(sample_scaled)

            logger.info(f"Sample Predicted Car Price : ₹{predicted_price[0]:.2f} Lakhs")
            '''
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
