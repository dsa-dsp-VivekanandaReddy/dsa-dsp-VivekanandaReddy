import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
    
ROOT_DIR = Path('.').resolve().parents[1].absolute()
target_column = 'SalePrice'


def fin_preprocessing(df):
    df['LotFrontage'] = df['LotFrontage'].fillna(np.mean(df['LotFrontage']))
    df['LotArea'] = df['LotArea'].fillna(np.mean(df['LotArea']))
    df['Utilities'] = df['Utilities'].fillna('AllPub')
    df['BsmtQual'] = df['BsmtQual'].fillna('Unf')
    df['BsmtCond'] = df['BsmtCond'].fillna('Unf')
    df['Electrical'] = df['Electrical'].fillna('Unf')
    df['FireplaceQu'] = df['FireplaceQu'].fillna('Unf')
    df['Fence'] = df['Fence'].fillna('Unf')
    df['BsmtExposure'] = df['BsmtExposure'].fillna('Unf')
    df['BsmtFinType1'] = df['BsmtFinType1'].fillna('Unf')
    df['BsmtFinType2'] = df['BsmtFinType2'].fillna('Unf')
    df['GarageQual'] = df['GarageQual'].fillna('Unf')
    df['GarageCond'] = df['GarageCond'].fillna('Unf')
    df['MasVnrType'] = df['MasVnrType'].fillna('Unf')
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    df['Alley'] = df['Alley'].fillna('Na')

    return df


def drop_cols(df, to_drop):
    df = df.drop(columns=to_drop, axis=1)
    df = df.dropna()

    return df


def encode(df, categ_cols, one_hot):
    
    hp_df = one_hot.transform(df[categ_cols]).toarray()

    categ_col_name = one_hot.get_feature_names(categ_cols)
    hp_df = pd.DataFrame(hp_df, index=df.index, columns=categ_col_name)
    df_without_categ = df.drop(columns=categ_cols)
    df = pd.concat([hp_df, df_without_categ], axis=1)

    dump(one_hot, ROOT_DIR / 'models' / 'onehot.pkl')

    return df


def split_data(df):
    X, y = df.drop(target_column, axis=1), df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X, y, X_train, X_test, y_train, y_test


def linear_model(df, X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def predict(model, X_test):
    y_pred = model.predict(X_test)
    # Replace negative predictions with 0
    y_pred = np.where(y_pred < 0, 0, y_pred)

    return y_pred

def preprocess_cont(df_continuous):
    df_continuous.select_dtypes(include='number')
    df_continuous = df_continuous.dropna()
    
    return df_continuous
 
def final_preprocessing(df):
    
    df = fin_preprocessing(df)
    df = df.dropna(axis=1)
    cols = df.select_dtypes(include='object').columns

    label_encoder = LabelEncoder()

    for col in cols:
        df[col] = label_encoder.fit_transform(df[col])
    
    #one_hot = OneHotEncoder(handle_unknown='ignore')
    #one_hot = one_hot.fit(df[cols])
    
    
    #df = encode(df, cols, one_hot)
    

    
    return df
    
def inference(inference_df):
    pass
    
def submission(submission_df):
    pass