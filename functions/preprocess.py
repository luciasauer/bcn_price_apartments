import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from abc import ABC, abstractmethod
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder

class DataLoader():
    def __init__(self, path, columns_to_drop):
        self.path = path
        self.columns_to_drop = columns_to_drop
    
    def load_and_split(self):
        df = pd.read_csv(self.path)
        df['floor'] = df['door'].str.extract(r'(\d{1,2})º')
        df = df.drop(columns = self.columns_to_drop)
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=10)
        return df_train, df_test
    
class NumericalFeaturesDataCleaner:
    def __init__(self, df):
        self.df = df

    def impute_square_meters(self):
        #Replace negative values with nan and then impute with knn and randomforest regressor
        self.df['square_meters'] = self.df['square_meters'].apply(lambda x: np.nan if x < 0 else x)
        imputer = KNNImputer(n_neighbors=5)
        #self.df['square_meters_knn'] = imputer.fit_transform(self.df[['square_meters']])
        #imputer_2 = IterativeImputer(estimator=RandomForestRegressor(), random_state=10)
        self.df['square_meters'] = imputer.fit_transform(self.df[['square_meters', 'num_rooms', 'square_meters', 'year_built']])
        #self.df['square_meters'] = self.df['square_meters'].fillna(self.df['square_meters'].mean())
        
    def impute_num_rooms(self):
        # Cambia valores mayores a 10 en 'num_rooms' a NaN
        self.df['num_rooms'] = self.df['num_rooms'].apply(lambda x: np.nan if x > 10 else x)
        # Imputar solo los nulos en 'num_rooms' basados en 'neighborhood' y en rangos de 'square_meters'
        #imputed_rooms = self.df.groupby(['neighborhood', pd.cut(self.df['square_meters'], bins=4)])['num_rooms'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else self.df['num_rooms'].mode().iloc[0])
        # Combinar el resultado de imputación solo en posiciones nulas
        #self.df['num_rooms'] = self.df['num_rooms'].combine_first(imputed_rooms)
        #self.df['num_rooms'] = self.df['num_rooms'].fillna(self.df['num_rooms'].mean().round())
        imputer_2 = IterativeImputer(estimator=RandomForestRegressor(), random_state=10)
        imputer = KNNImputer(n_neighbors=5)
        self.df['num_rooms'] = imputer.fit_transform(self.df[['num_rooms', 'square_meters']]).round()

    def impute_num_baths(self):
        # Usa la moda de 'num_baths' condicionada por 'num_rooms'
        #self.df['num_baths'] = self.df.groupby(['num_rooms', 'square_meters'])['num_baths'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else 1)
        imputer_2 = IterativeImputer(estimator=RandomForestRegressor(), random_state=10)
        imputer = KNNImputer(n_neighbors=5)
        self.df['num_baths'] = imputer.fit_transform(self.df[['num_baths', 'num_rooms', 'square_meters']]).round()
        #self.df['num_baths'] = self.df['num_baths'].fillna(self.df['num_baths'].mean().round())
    def impute_year_built(self):
        # Usa la moda de 'year_built' basada en 'num_rooms', 'square_meters', y 'neighborhood'
        # imputed_years = self.df.groupby(['neighborhood', 'num_rooms', pd.cut(self.df['square_meters'], bins=4)])['year_built'] \
        #                              .transform(lambda x: x.mean() )
        # self.df['year_built'] = self.df['year_built'].combine_first(imputed_years)
        imputer_2 = IterativeImputer(estimator=RandomForestRegressor(), random_state=10)
        imputer = KNNImputer(n_neighbors=5)
        self.df['year_built'] = imputer.fit_transform(self.df[['year_built', 'square_meters']]).round()
        #self.df['year_built'] = self.df['year_built'].fillna(self.df['year_built'].mean())

    def impute_floor(self):
        # Convierte 'floor' a numérico
        self.df['floor'] = pd.to_numeric(self.df['floor'], errors='coerce')
        # imputer_2 = IterativeImputer(estimator=RandomForestRegressor(), random_state=32)
        # self.df['floor'] = imputer_2.fit_transform(self.df[['floor']])
        # self.df['floor'] = self.df[['floor']].round()
        #self.df['floor'] = self.df['floor'].fillna(self.df['floor'].mean().round())
        imputer_2 = IterativeImputer(estimator=RandomForestRegressor(), random_state=10)
        self.df['floor'] = imputer_2.fit_transform(self.df[['floor','year_built']]).round()
    
    def impute_num_crimes(self):
        # self.df['num_crimes'] = pd.to_numeric(self.df['num_crimes'], errors='coerce')
        # imputer_2 = IterativeImputer(estimator=RandomForestRegressor(), random_state=42)
        # self.df['num_crimes'] = imputer_2.fit_transform(self.df[['num_crimes']]).round()
        #self.df['num_crimes'] = self.df['num_crimes'].fillna(self.df['num_crimes'].mode()[0]) 
        imputer_2 = IterativeImputer(estimator=RandomForestRegressor(), random_state=10)
        self.df['num_crimes'] = imputer_2.fit_transform(self.df[['num_crimes']]).round()
    
    def process_all(self):
        self.impute_square_meters()
        self.impute_num_rooms()
        self.impute_num_baths()
        self.impute_year_built()
        self.impute_floor()
        self.impute_num_crimes()
        return self.df
    
    def return_results(self):
        return self.df

class CategoricalFeaturesDataCleaner:
    np.random.seed(43)
    def __init__(self, df, columns):
        self.df = df
        self.columns = columns

    #def impute_categorical_randomly(self):
        #for c in self.columns:
        # self.df[f'{self.column}_random'] = self.df[self.column].fillna(pd.Series(np.random.choice([0, 1], size=self.df[self.column].isnull().sum())))
            #self.df[f'{c}_random'] = self.df[c].fillna(np.random.choice([0, 1]))

    def impute_is_furnished(self):
        self.df['is_furnished'] = self.df['is_furnished'].fillna(self.df['is_furnished'].mode()[0])
    
    def impute_has_ac(self):
        self.df['has_ac'] = self.df['has_ac'].fillna(self.df['has_ac'].mode()[0])
    
    def impute_has_pool(self):
        self.df['has_pool'] = self.df['has_pool'].fillna(self.df['has_pool'].mode()[0])
    
    def impute_accepts_pets(self):
        self.df['accepts_pets'] = self.df['accepts_pets'].fillna(self.df['accepts_pets'].mode()[0])
      

    def impute_neighborhodd(self):
        self.df['neighborhood'] = self.df['neighborhood'].fillna(self.df['neighborhood'].mode()[0])

    def process_all(self):
        #self.impute_categorical_randomly()
        self.impute_is_furnished()
        self.impute_has_ac()
        self.impute_has_pool()
        self.impute_accepts_pets()
        self.impute_neighborhodd()
        return self.df



class CreateDummies:
    def __init__(self, df, ordinal_columns, categorical_columns, encoder=None):
        self.df = df
        self.ordinal_columns = ordinal_columns
        self.categorical_columns = categorical_columns
        self.encoder = encoder if encoder is not None else OrdinalEncoder()

    def transformer_get_dummies(self):
        self.df = pd.get_dummies(self.df, columns=self.categorical_columns, drop_first=True)

    def transformer_ordinalencoder(self, fit=True):
        if fit:
            encoded_variable = self.encoder.fit_transform(self.df[self.ordinal_columns])
        else:
            encoded_variable = self.encoder.transform(self.df[self.ordinal_columns])

        df_encoded = pd.DataFrame(encoded_variable, columns=self.ordinal_columns).reset_index(drop=True)

        self.df = self.df.drop(columns=self.ordinal_columns).reset_index(drop=True)
        self.df = pd.concat([self.df, df_encoded], axis=1)

    def process_all(self, fit=True):
        self.transformer_get_dummies()
        self.transformer_ordinalencoder(fit=fit)
        return self.df

###################################################
####General Functions for Feature Engineering######
###################################################

def year_group(df):
    if 'year' in df.columns:
        df['year_group'] = df['year'].apply(lambda x: 1 if x <= 1970 else (2 if x <= 1990 else 3))
    return df

def safety_level(df):
    if 'num_crimes' in df.columns: 
        df['safety_level'] = df['num_crimes'].apply(lambda x: 1 if x == 0 else 0)
    return df

def neigh_safety(df):
    neighborhood_columns = [col for col in df.columns if col.startswith('neigh')]
    for col in neighborhood_columns:
        df[f'safety_{col}'] = df[col] * df['safety_level']
    return df

def neigh_crime(df):
    neighborhood_columns = [col for col in df.columns if col.startswith('neigh')]
    for col in neighborhood_columns:
        df[f'crime_{col}'] = df[col] * df['num_crimes']
    return df

def has_ac_pool(df):
    if 'has_pool' in df.columns:
        df['has_ac_pool'] = df['has_ac'] * df['has_pool']
    return df

def square_met_2(df):
    if 'square_meters' in df.columns:
        df['square_met_2'] = df['square_meters'] ** 2
    return df
