import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb



class LinearModel():
    def __init__(self, features_columns, target_column, hyperparameters: dict = None):
        self.features_columns = features_columns
        self.target_column = target_column
        self.model_params = hyperparameters if hyperparameters else {}
        self.model = LinearRegression()

    def train_model(self, df):
        X = df[self.features_columns]
        y = df[self.target_column]
        
        scale = StandardScaler()
        X_scaled = scale.fit_transform(X)
        
        self.model.fit(X_scaled, y)
        self.scaler = scale 

    def predict_model(self, df):
        X = df[self.features_columns]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate_model(self, df):
        X = df[self.features_columns]
        y = df[self.target_column]
        
        y_pred = self.predict_model(df)
        mse = mean_squared_error(y, y_pred).round()
        return mse

class LassoModel():
    def __init__(self, features_columns, target_column):
        self.features_columns = features_columns
        self.target_column = target_column
        self.model = Lasso()

    def train_model(self, df, param_grid):
        X = df[self.features_columns]
        y = df[self.target_column]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # GridSearchCV and Crossvalidation
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_scaled, y)

        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

    def predict_model(self, df):
        X = df[self.features_columns]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate_model(self, df):
        X = df[self.features_columns]
        y = df[self.target_column]

        y_pred = self.predict_model(df)
        mse = mean_squared_error(y, y_pred)
        return mse

class RidgeModel():
    def __init__(self, features_columns, target_column):
        self.features_columns = features_columns
        self.target_column = target_column
        self.model = Ridge()

    def train_model(self, df, param_grid):
        X = df[self.features_columns]
        y = df[self.target_column]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # GridSearchCV and Crossvalidation
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_scaled, y)

        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

    def predict_model(self, df):
        X = df[self.features_columns]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate_model(self, df):
        X = df[self.features_columns]
        y = df[self.target_column]

        y_pred = self.predict_model(df)
        mse = mean_squared_error(y, y_pred)
        return mse



class NonLinearModel():
    def __init__(self, model_class, features_columns, target_column):
        self.model_class = model_class
        self.features_columns = features_columns
        self.target_column = target_column
        self.model = None
        self.scaler = None
        self.random_seed = 42

    def train_model(self, df, param_grid):
        X = df[self.features_columns]
        y = df[self.target_column]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # GridSearchCV and CrossValidation
        grid_search = GridSearchCV(self.model_class(random_state=self.random_seed), param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_scaled, y)

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

    def predict_model(self, df):
        X = df[self.features_columns]
        X_scaled = self.scaler.transform(X) 
        return self.model.predict(X_scaled)

    def evaluate_model(self, df):
        X = df[self.features_columns]
        y = df[self.target_column]

        y_pred = self.predict_model(df)
        mse = mean_squared_error(y, y_pred)
        return mse
