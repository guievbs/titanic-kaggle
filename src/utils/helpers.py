import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def is_male(df, column):
    return df[column].apply(lambda x: 1 if x == 'male' else 0)


def apply_one_hot_encoding(df, columns) -> pd.DataFrame:
    ohe = OneHotEncoder(handle_unknown='ignore', dtype='int32')
    transformed_data = ohe.fit_transform(df[columns]).toarray()
    ohe_df = pd.DataFrame(transformed_data, columns=ohe.get_feature_names_out())
    df = pd.concat([df, ohe_df], axis=1)
    return df

