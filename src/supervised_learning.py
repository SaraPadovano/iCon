import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold, learning_curve, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor


# Funzione che crea la curva di apprendimento di un modello di regressione
def plot_learning_curves(regression_model, X, y, regression_model_name, file_end, log_file):
    # Calcola la curva dato il modello di regressione
    train_sizes, train_scores, test_scores = learning_curve(regression_model, X, y, cv=10, scoring='neg_mean_squared_error')

    # Scrive i dati della curva in un file apposito del modello
    log_file.write("\n{:<8}{:<25}{:<25}{:<25}{:<25}\n".format(
        "Size",
        "Mean Train Score",
        "Variance Train Score",
        "Std Train Score",
        "Train Scores"
    ))
    for i in range(0, len(train_scores)):
        mean = np.mean(train_scores[i])
        var = np.var(train_scores[i])
        std = np.std(train_scores[i])
        log_file.write(
            "{:<8}{:<25}{:<25}{:<25}{:<25}\n".format(str(i), str(mean), str(var), str(std), str(train_scores[i])))

    log_file.write("\n{:<8}{:<25}{:<25}{:<25}{:<25}\n".format(
        "Size",
        "Mean Test Score",
        "Variance Test Score",
        "Std Test Score",
        "Test Scores"
    ))
    for i in range(0, len(test_scores)):
        mean = np.mean(test_scores[i])
        var = np.var(test_scores[i])
        std = np.std(test_scores[i])
        log_file.write(
            "{:<8}{:<25}{:<25}{:<25}{:<25}\n".format(str(i), str(mean), str(var), str(std), str(test_scores[i])))

    # Converte l'errore negativo in positivo
    mean_train_errors = np.mean(-train_scores, axis=1)
    mean_test_errors = np.mean(-test_scores, axis=1)

    # Calcola la devizione standard dell'errore
    train_errors_std = np.std(-train_scores, axis=1)
    test_errors_std = np.std(-test_scores, axis=1)

    # Disegna la curva di apprendimento
    plt.figure(figsize=(12, 8))  # Dimensione del grafico
    plt.plot(train_sizes, mean_train_errors, 'o-', color='#1f77b4', label='Training Error', linewidth=2, markersize=6)
    plt.plot(train_sizes, mean_test_errors, 'o-', color='#ff7f0e', label='Validation Error', linewidth=2, markersize=6)
    plt.fill_between(train_sizes,
                     mean_train_errors - train_errors_std,
                     mean_train_errors + train_errors_std,
                     color='#1f77b4', alpha=0.2)
    plt.fill_between(train_sizes,
                     mean_test_errors - test_errors_std,
                     mean_test_errors + test_errors_std,
                     color='#ff7f0e', alpha=0.2)
    plt.xlabel('Training examples', fontsize=14)
    plt.ylabel('Mean Error', fontsize=14)
    plt.legend(loc='best')
    plt.title(regression_model_name + ' Learning Curves')
    plt.savefig(file_end.png)
    plt.close()

#Funzione che restituisce i migliori iperparametri per ogni modello
def get_best_hyperparameters(X, y, regression_model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if regression_model_name == 'DecisionTree':
        dtr = DecisionTreeRegressor()
        DecisionTreeHyperparameters = {
            'DecisionTree__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'DecisionTree__max_depth': [None, 5, 10],
            'DecisionTree__splitter': ['best'],
            'DecisionTree__min_samples_split': [2, 5, 10, 20],
            'DecisionTree__min_samples_leaf': [1, 2, 5, 10, 20]}
        gridSearchCV_dtr = GridSearchCV(Pipeline([('DecisionTree', dtr)]), DecisionTreeHyperparameters, cv=5, scoring='neg_mean_squared_error', error_score='raise')
        gridSearchCV_dtr.fit(X_train, y_train)
        best_parameters = {
            'DecisionTree__criterion': gridSearchCV_dtr.best_params_['DecisionTree__criterion'],
            'DecisionTree__max_depth': gridSearchCV_dtr.best_params_['DecisionTree__max_depth'],
            'DecisionTree__min_samples_split': gridSearchCV_dtr.best_params_['DecisionTree__min_samples_split'],
            'DecisionTree__min_samples_leaf': gridSearchCV_dtr.best_params_['DecisionTree__min_samples_leaf']
        }
        return best_parameters

    if regression_model_name == 'RandomForest':
        rfr = RandomForestRegressor()
        RandomForestHyperparameters = {
            'RandomForest__n_estimators': [10, 20, 50, 100],
            'RandomForest__criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
            'RandomForest__max_depth': [None, 5, 10]}
        gridSearchCV_rfr = GridSearchCV(Pipeline([('RandomForest', rfr)]), RandomForestHyperparameters, cv=5, scoring='neg_mean_squared_error', error_score='raise')
        gridSearchCV_rfr.fit(X_train, y_train)
        best_parameters = {
            'RandomForest__n_estimators': gridSearchCV_rfr.best_params_['RandomForest__n_estimators'],
            'RandomForest__max_depth': gridSearchCV_rfr.best_params_['RandomForest__max_depth'],
            'RandomForest__criterion': gridSearchCV_rfr.best_params_['RandomForest__criterion']
        }
        return best_parameters

    if regression_model_name == 'LightGBM':
        lgbmr = LGBMRegressor()
        LGBMHyperparameters = {
            'LGBM__n_estimators': [10, 20, 50, 100],
            'LGBM__learning_rate': [0.01, 0.05, 0.1],
            'LGBM__max_depth': [None, 5, 10],
            'LGBM__class_weight': ['balanced']}
        gridSearchCV_lgbmr = GridSearchCV(Pipeline([('LightGBM', lgbmr)]), LGBMHyperparameters, cv=5, scoring='neg_mean_squared_error', error_score='raise')
        gridSearchCV_lgbmr.fit(X_train, y_train)
        best_parameters = {
            'LGBM__n_estimators': gridSearchCV_lgbmr.best_params_['LGBM__n_estimators'],
            'LGBM__learning_rate': gridSearchCV_lgbmr.best_params_['LGBM__learning_rate'],
            'LGBM__max_depth': gridSearchCV_lgbmr.best_params_['LGBM__max_depth']
        }
        return best_parameters

    if regression_model_name == 'CatBoost':
        cbr = CatBoostRegressor()
        CatBoostHyperparameters = {
            'CatBoost__learning_rate': [0.01, 0.05, 0.1],
            'CatBoost__n_estimators': [10, 20, 50, 100],
            'CatBoost__l2_leaf_reg': [3, 5, 7, 10],
            'CatBoost__border_count': [32, 64, 128],
            'CatBoost__max_depth': [None, 5, 10]
        }
        gridSearchCV_cbr = GridSearchCV(Pipeline([('CatBoost', cbr)]), CatBoostHyperparameters, cv=5, scoring='neg_mean_squared_error', error_score='raise')
        gridSearchCV_cbr.fit(X_train, y_train)
        best_parameters = {
            'CatBoost__learning_rate': gridSearchCV_cbr.best_params_['CatBoost__learning_rate'],
            'CatBoost__n_estimators': gridSearchCV_cbr.best_params_['CatBoost__n_estimators'],
            'CatBoost__l2_leaf_reg': gridSearchCV_cbr.best_params_['CatBoost__l2_leaf_reg'],
            'CatBoost__border_count': gridSearchCV_cbr.best_params_['CatBoost__border_count'],
            'CatBoost__max_depth': gridSearchCV_cbr.best_params_['CatBoost__max_depht']
        }
        return best_parameters




