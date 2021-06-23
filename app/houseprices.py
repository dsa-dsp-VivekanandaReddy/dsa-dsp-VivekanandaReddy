import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
    
ROOT_DIR = Path('.').resolve().parents[1].absolute()
target_column = 'SalePrice'


def fin_preprocessing(df):
        
    df['LotFrontage'] = df['LotFrontage'].fillna(np.mean(df['LotFrontage']))
    df['LotArea'] = df['LotArea'].fillna(np.mean(df['LotArea']))
    df['Utilities'] = df['Utilities'].fillna('AllPub')
    
    cols = df.select_dtypes(include='object').columns
    
    for c in cols:
        df[c] = df[c].fillna('notavailable')
    
    cols = df.select_dtypes(include='number').columns
    
    for c in cols:
        df[c] = df[c].fillna(0)

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
    df_continuous = df_continuous.select_dtypes(include='number')
    df_continuous = df_continuous.dropna()
    
    return df_continuous
 
def final_preprocessing(df):
    
    df = fin_preprocessing(df)
    df = df.dropna()
    cols = df.select_dtypes(include='object').columns

    #label_encoder = LabelEncoder()

    #for col in cols:
    #    df[col] = label_encoder.fit_transform(df[col])
    
    one_hot = OneHotEncoder(handle_unknown='ignore')
    one_hot = one_hot.fit(df[cols])
    
    
    df = encode(df, cols, one_hot)
    

    
    return df

def first_inference(inference_df):
    continuous_inference_df = inference_df.select_dtypes(include='number')
    continuous_inference_df = continuous_inference_df.dropna()
    
    return continuous_inference_df

def final_inference(final_inference_df):
    final_inference_df = fin_preprocessing(final_inference_df)
    final_inference_df = final_inference_df.dropna()
    
    categ_cols = final_inference_df.select_dtypes(include='object').columns
    
    one_hot = load(ROOT_DIR / 'models' / 'onehot.pkl')
    
    hp_df = one_hot.transform(final_inference_df[categ_cols]).toarray()
    
    categ_col_name = one_hot.get_feature_names(categ_cols)
    hp_df = pd.DataFrame(hp_df, index=final_inference_df.index, columns=categ_col_name)
    df_without_categ = final_inference_df.drop(columns=categ_cols)
    final_inference_df = pd.concat([hp_df, df_without_categ], axis=1)
    
    return final_inference_df

def submission(submit_df, inference_df, predictions):
    submit_df[target_column] = predictions
    submit_df = submit_df[[target_column]].reset_index()
    inference_ids_df = inference_df.reset_index()[['Id']]
    submission_df = inference_ids_df.merge(submit_df, on='Id', how='left')
    submission_df[target_column].isna().sum() == len(inference_ids_df) - len(submit_df)
    submission_df[target_column] = submission_df[target_column].fillna(0)

    return submission_df



def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)

def evaluation(model, X_test, y_test, y_pred):
    y_pred = model.predict(X_test)
    # Replace negative predictions with 0
    y_pred = np.where(y_pred < 0, 0, y_pred)
    print(compute_rmsle(y_test, y_pred))

    return y_pred
    