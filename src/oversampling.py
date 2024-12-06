import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import KBinsDiscretizer

def oversampling(X, y, target):
    # Discretizzo y per poter usare adasyn
    y = np.array(y).reshape(-1, 1)
    discretizer = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
    y_discretized = discretizer.fit_transform(y).flatten()

    # Inizializzo adasyn
    adasyn = ADASYN(sampling_strategy='minority', n_neighbors=1, random_state=42)
    # Applicazione di adasyn al dataset
    X_resampled, y_resampled = adasyn.fit_resample(X, y_discretized)
    # Scrivo il nuovo dataset ottenuto
    X = pd.DataFrame(X)
    dataSet_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    dataSet_resampled[target] = y_resampled
    file_path = "../dataset/Automobile_resampled.csv"
    dataSet_resampled.to_csv(file_path, index=False)

    return X_resampled, y_resampled