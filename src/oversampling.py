import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import smogn

def oversampling_smogn(X, y, target):
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    train_data = pd.concat([pd.DataFrame(X), pd.Series(y, name=target)], axis=1)

    train_data_resampled = smogn.smoter(
        data=train_data,
        y=target,
        samp_method='extreme'
    )
    X = train_data_resampled.drop(columns=[target]).to_numpy()
    y = train_data_resampled[target].to_numpy()
    X = pd.DataFrame(X)
    dataSet_resampled = pd.DataFrame(X, columns=X.columns)
    dataSet_resampled[target] = y
    file_path = "../dataset/Automobile_resampled.csv"
    dataSet_resampled.to_csv(file_path, index=False)
    return X, y
