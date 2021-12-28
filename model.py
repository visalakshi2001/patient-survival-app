import os
import joblib
import xgboost

curr_path = os.path.dirname(os.path.realpath(__file__))
xgb_final = joblib.load(curr_path + "/xgb_tuned_final.pkl.compressed" )
print("model loaded")


def predict_survival(x_test):
    y_hat = xgb_final.predict(x_test._get_numeric_data())

    return y_hat
