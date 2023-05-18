import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
from sklearn.linear_model import QuantileRegressor
from quantile_forest import RandomForestQuantileRegressor
import copy

class LinearQR:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.qr_low = QuantileRegressor(quantile=alpha/2, alpha=0, solver='highs')
        self.qr_upp = QuantileRegressor(quantile=1-alpha/2, alpha=0, solver='highs')

    def fit(self, X, Y):
        self.qr_low.fit(X,Y)
        self.qr_upp.fit(X,Y)
        return self

    def predict(self, X):
        lower = self.qr_low.predict(X)
        upper = self.qr_upp.predict(X)
        # Correct any possible quantile crossings
        pred = np.concatenate([lower.reshape(len(lower),1), upper.reshape(len(upper),1)],1)
        pred = np.sort(pred,1)
        lower = pred[:,0]
        upper = pred[:,1]
        return lower, upper


class RFQR:
    def __init__(self, alpha=0.1, n_estimators=100, min_samples_split=10):
        self.alpha = alpha
        self.qr = RandomForestQuantileRegressor(min_samples_split=min_samples_split, min_samples_leaf=5, 
                                                n_estimators=n_estimators)
        self.qr_upp = QuantileRegressor(quantile=1-alpha/2, alpha=0, solver='highs')

    def fit(self, X, Y):
        self.qr.fit(X, Y)
        return self

    def predict(self, X):
        pred = self.qr.predict(X, quantiles=[self.alpha/2, 1-self.alpha/2])
        lower = pred[:,0]
        upper = pred[:,1]
        # Correct any possible quantile crossings
        pred = np.concatenate([lower.reshape(len(lower),1), upper.reshape(len(upper),1)],1)
        pred = np.sort(pred,1)
        return lower, upper
