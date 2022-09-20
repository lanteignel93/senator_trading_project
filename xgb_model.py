import pandas as pd
import numpy as np
import tensorflow as tf
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from tqdm.auto import tqdm, trange
from xgboost import XGBRegressor
from pandas.tseries.offsets import *
import warnings

warnings.filterwarnings("ignore")

test_data = pd.read_pickle('test_data.pkl')
test_data_sectoral = pd.read_pickle('test_data_sectoral.pkl')


class XGBoost:
    def __init__(self, data, period):
        self.period = period
        self.data = data
        self.spy_col = 'spy_' + str(period)
        self.ret_col = 'ret_' + str(period)
        self.data['adjusted_ret'] = self.data[self.spy_col] * self.data['Beta']
        self.data['win/loss'] = np.where(
            ((self.data['order_type'] == 1) & (self.data[self.ret_col] > self.data['adjusted_ret'])) | (
                    (self.data['order_type'] == 0) & (self.data[self.ret_col] < self.data['adjusted_ret'])), 1, 0)
        self.report = pd.DataFrame(index=[0])

    def process_data(self, sector=False):
        if sector == True:
            features_num = [
                "order_type", "Beta"
            ]
            features_cat = [
                "full_name", "ticker", 'sector'
            ]
        else:
            features_num = [
                "order_type", "size", "Beta"
            ]
            features_cat = [
                "full_name", "ticker"
            ]
        transformer_num = make_pipeline(
            SimpleImputer(strategy="constant"),
            StandardScaler(),
        )
        transformer_cat = make_pipeline(
            SimpleImputer(strategy="constant", fill_value="NA"),
            OneHotEncoder(handle_unknown='ignore'),
        )

        preprocessor = make_column_transformer(
            (transformer_num, features_num),
            (transformer_cat, features_cat),
        )

        X = self.data.copy()

        X = X[features_num + features_cat + ['win/loss']].dropna()
        self.r = len(X) / X['win/loss'].sum()
        self.report.loc[0, "r"] = self.r
        y = X.pop('win/loss')
        test_size = int(0.8 * len(X))
        self.X_test = X.iloc[test_size:]
        self.y_test = y.iloc[test_size:]
        self.X = X.iloc[:test_size]
        self.y = y.iloc[:test_size]

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X, self.y, stratify=self.y,
                                                                                  train_size=0.75)

        self.X_train = preprocessor.fit_transform(self.X_train)
        self.X_train = self.X_train.toarray()
        self.X_valid = preprocessor.transform(self.X_valid)
        self.X_valid = self.X_valid.toarray()
        # input_shape = [self.X_train.shape[1]]
        self.X_test = preprocessor.transform(self.X_test)
        self.X_test = self.X_test.toarray()

    def fit_model(self, w=1):

        self.model = XGBRegressor(n_estimators=1000, learning_rate=0.05, scale_pos_weight=w)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):

        self.pred = self.model.predict(self.X_test).reshape(self.y_test.shape[0], 1)
        self.pred = np.where(self.pred > 0.5, 1, 0)
        self.pred = self.pred.reshape(self.y_test.shape[0], )
        self.accuracy = np.where(self.pred == self.y_test, 1, 0)
        self.report.loc[0, "accuracy"] = 100 * sum(self.accuracy / len(self.accuracy))
        self.win_accuracy = (np.where((self.pred == 1) & (self.y_test == 1), 1, 0).sum()) / ((self.pred == 1).sum())
        self.report.loc[0, "win_accuracy"] = self.win_accuracy
        self.false_pos = (np.where((self.pred == 1) & (self.y_test == 0), 1, 0).sum()) / (self.pred == 1).sum()
        self.report.loc[0, "false_pos"] = self.false_pos

    def print_evaluation(self):
        # print("Return period:",self.period)
        print("Optimized w: {:.4f}".format(self.opt_w))
        print("Dataset win/loss ratio: {:.4f}%".format(100 / self.r))
        print("out-of-sample accuracy: {:.4f}%".format(100 * sum(self.accuracy / len(self.accuracy))))
        print("out-of-sample win accuracy: {:.4f}%".format(100 * self.win_accuracy))
        print("out-of-sample false positive accuracy: {:.4f}%".format(100 * self.false_pos))

    def xgboost_tree_true_positives(self, w):
        self.fit_model(w[0])
        self.evaluate_model()
        return -self.win_accuracy

    def optimize_model(self):
        res = sp.optimize.minimize(self.xgboost_tree_true_positives, 1, method='nelder-mead',
                                   options={'maxiter': 10, 'disp': True})
        self.opt_w = res.x[0]
        self.report.loc[0, 'w'] = self.opt_w
        self.fit_model(self.opt_w)


def print_evaluation(sector=False):
    if sector == False:
        try:
            report = pd.read_pickle('xgboost_report.pkl')
            print("First Specification Performance:")
        except:
            model = XGBoost(test_data, 90)
            model.process_data()
            model.optimize_model()
            model.evaluate_model()
            model.report.to_pickle('xgboost_report.pkl')
            report = model.report
    if sector == True:
        try:
            report = pd.read_pickle('xgboost_report_sector.pkl')
            print("Second Specification Performance:")
        except:
            model = XGBoost(test_data_sectoral, 90)
            model.process_data()
            model.optimize_model()
            model.evaluate_model()
            model.report.to_pickle('xgboost_report_sector.pkl')
            report = model.report
    print("Optimized w: {:.4f}".format(report.loc[0, 'w']))
    print("Dataset win/loss ratio: {:.4f}%".format(100 / report.loc[0, 'r']))
    print("out-of-sample accuracy: {:.4f}%".format(report.loc[0, 'accuracy']))
    print("out-of-sample win accuracy: {:.4f}%".format(100 * report.loc[0, 'win_accuracy']))
    print("out-of-sample false positive accuracy: {:.4f}%".format(100 * report.loc[0, 'false_pos']))

