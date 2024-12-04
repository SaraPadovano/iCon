import matplotlib
matplotlib.use('Agg')  # Imposta il backend a 'Agg' (non interattivo)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold, learning_curve, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score


# Funzione che crea la curva di apprendimento di un modello di regressione
def plot_learning_curves(regression_model, X, y, regression_model_name, file_end):
    print(f"Generazione della curva di apprendimento per il modello: {regression_model_name}")
    # Calcola la curva dato il modello di regressione
    train_sizes, train_scores, test_scores = learning_curve(regression_model, X, y, cv=10, scoring='neg_mean_squared_error')
    print(f"Curva calcolata. Dimensioni dei dati di addestramento: {train_sizes}")

    log_file = open(f"../text/log_{regression_model_name}.txt", "w")
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
    plt.xlabel('Training examples', fontsize=14)
    plt.ylabel('Mean Error', fontsize=14)
    plt.legend(loc='best')
    plt.title(regression_model_name + ' Learning Curves')
    plt.savefig(f"{file_end}.png")
    print("Ottenuto grafico")
    plt.close()

#Funzione che restituisce i migliori iperparametri per ogni modello
def get_best_hyperparameters(X, y, regression_model_name):
    print(f"Inizio la ricerca dei migliori iperparametri per il modello: {regression_model_name}")
    if regression_model_name == 'DecisionTree':
        print("Ricerca iperparametri per DecisionTree...")
        dtr = DecisionTreeRegressor()
        DecisionTreeHyperparameters = {
            'DecisionTree__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'DecisionTree__max_depth': [None, 5, 10],
            'DecisionTree__splitter': ['best'],
            'DecisionTree__min_samples_split': [2, 5, 10, 20],
            'DecisionTree__min_samples_leaf': [1, 2, 5, 10, 20],
            'DecisionTree__random_state': [42]}
        gridSearchCV_dtr = GridSearchCV(Pipeline([('DecisionTree', dtr)]), DecisionTreeHyperparameters, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', error_score='raise')
        gridSearchCV_dtr.fit(X, y)
        best_parameters = {
            'DecisionTree__criterion': gridSearchCV_dtr.best_params_['DecisionTree__criterion'],
            'DecisionTree__max_depth': gridSearchCV_dtr.best_params_['DecisionTree__max_depth'],
            'DecisionTree__min_samples_split': gridSearchCV_dtr.best_params_['DecisionTree__min_samples_split'],
            'DecisionTree__min_samples_leaf': gridSearchCV_dtr.best_params_['DecisionTree__min_samples_leaf']
        }
        print(f"Migliori iperparametri trovati per DecisionTree: {best_parameters}")
        # Scrive i risultati nel file
        with open("../text/hyperparameters.txt", "w") as hyperparameters_file:
            hyperparameters_file.write("Best DecisionTree parameters found:\n")
            for param, value in gridSearchCV_dtr.best_params_.items():
                hyperparameters_file.write("{:<35}{:<10}\n".format(param, str(value)))
            hyperparameters_file.close()
        return best_parameters

    if regression_model_name == 'RandomForest':
        print("Ricerca iperparametri per RandomForest...")
        rfr = RandomForestRegressor()
        RandomForestHyperparameters = {
            'RandomForest__n_estimators': [10, 20, 50, 100],
            'RandomForest__criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
            'RandomForest__max_depth': [None, 5, 10],
            'RandomForest__random_state': [42]}
        gridSearchCV_rfr = GridSearchCV(Pipeline([('RandomForest', rfr)]), RandomForestHyperparameters, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', error_score='raise')
        gridSearchCV_rfr.fit(X, y)
        best_parameters = {
            'RandomForest__n_estimators': gridSearchCV_rfr.best_params_['RandomForest__n_estimators'],
            'RandomForest__max_depth': gridSearchCV_rfr.best_params_['RandomForest__max_depth'],
            'RandomForest__criterion': gridSearchCV_rfr.best_params_['RandomForest__criterion']
        }
        print(f"Migliori iperparametri trovati per RandomForest: {best_parameters}")
        with open("../text/hyperparameters.txt", "a") as hyperparameters_file:
            hyperparameters_file.write("Best RandomForest parameters found:\n")
            for param, value in gridSearchCV_rfr.best_params_.items():
                hyperparameters_file.write("{:<35}{:<10}\n".format(param, str(value)))
            hyperparameters_file.close()
        return best_parameters

    if regression_model_name == 'LightGBM':
        print("Ricerca iperparametri per LightGBM...")
        lgbmr = LGBMRegressor()
        LGBMHyperparameters = {
            'LightGBM__n_estimators': [10, 20, 50, 100],
            'LightGBM__learning_rate': [0.01, 0.05, 0.1],
            'LightGBM__max_depth': [None, 5, 10],
            'LightGBM__class_weight': ['balanced'],
            'LightGBM__verbose': [-1],
            'LightGBM__random_state': [42]}
        gridSearchCV_lgbmr = GridSearchCV(Pipeline([('LightGBM', lgbmr)]), LGBMHyperparameters, cv=5, n_jobs=-1,
                                          scoring='neg_mean_squared_error', error_score='raise')
        gridSearchCV_lgbmr.fit(X, y)
        best_parameters = {
            'LightGBM__n_estimators': gridSearchCV_lgbmr.best_params_['LightGBM__n_estimators'],
            'LightGBM__learning_rate': gridSearchCV_lgbmr.best_params_['LightGBM__learning_rate'],
            'LightGBM__max_depth': gridSearchCV_lgbmr.best_params_['LightGBM__max_depth']
        }
        print(f"Migliori iperparametri trovati per LightGBM: {best_parameters}")
        with open("../text/hyperparameters.txt", "a") as hyperparameters_file:
            hyperparameters_file.write("Best LightGBM parameters found:\n")
            for param, value in gridSearchCV_lgbmr.best_params_.items():
                hyperparameters_file.write("{:<35}{:<10}\n".format(param, str(value)))
            hyperparameters_file.close()
        return best_parameters


# Funzione che addestra il modello
def train_valuate_model(X, y):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    print("Avvio della ricerca degli iperparametri...")
    h_dcr = get_best_hyperparameters(X, y, 'DecisionTree')
    h_rfr = get_best_hyperparameters(X, y, 'RandomForest')
    h_lgbmr = get_best_hyperparameters(X, y, 'LightGBM')
    print("Iperparametri trovati per tutti i modelli.")
    dtr = DecisionTreeRegressor(criterion=h_dcr['DecisionTree__criterion'],
                                 splitter='best',
                                 max_depth=h_dcr['DecisionTree__max_depth'],
                                 min_samples_split=h_dcr['DecisionTree__min_samples_split'],
                                 min_samples_leaf=h_dcr['DecisionTree__min_samples_leaf'],
                                random_state=42)

    rfr = RandomForestRegressor(n_estimators=h_rfr['RandomForest__n_estimators'],
                                 max_depth=h_rfr['RandomForest__max_depth'],
                                criterion=h_rfr['RandomForest__criterion'],
                                random_state=42)

    lgbmr = LGBMRegressor(n_estimators=h_lgbmr['LightGBM__n_estimators'],
                          learning_rate=h_lgbmr['LightGBM__learning_rate'],
                          max_depth=h_lgbmr['LightGBM__max_depth'],
                          class_weight='balanced',
                          verbose=-1,
                          random_state=42)
    print("Modelli configurati con i migliori iperparametri.")

    cv = RepeatedKFold(n_splits=5, n_repeats=3)
    print("Valutazioni dei modelli con le varie metriche")

    # Metriche per valutare il modello
    metrics = {
        'MSE': make_scorer(mean_squared_error),
        'MAE': make_scorer(mean_absolute_error),
        'MSLE': make_scorer(mean_squared_log_error),
        'R2': make_scorer(r2_score)
    }

    metric_file = open("../text/metrics.txt", "w")
    metric_file.write(
        "\n{:<10}{:<10}{:<25}{:<25}{:<25}\n".format("Metric", "Model Name", "Score Mean", "Score Variance", "Score Std"))

    # Valuta ciascuna metrica per ogni modello
    for metric_name, metric_scorer in metrics.items():
        # Computa cross-validated score per ogni modello
        scores_dict = {
            "DecisionTree": cross_val_score(dtr, X, y, scoring=metric_scorer, cv=cv),
            "RandomForest": cross_val_score(rfr, X, y, scoring=metric_scorer, cv=cv),
            "LightGBM": cross_val_score(lgbmr, X, y, scoring=metric_scorer, cv=cv)
        }

        # Scrive i risultati per ogni modello
        for model_name, scores in scores_dict.items():
            mean = np.mean(scores)
            var = np.var(scores)
            std = np.std(scores)
            metric_file.write("{:<10}{:<10}{:<25}{:<25}{:<25}\n".format(metric_name, model_name, str(mean), str(var), str(std)))
    print("Valutazione completata.")
    print("Generazione delle curve di apprendimento...")
    # Disegna la curva per ogni modello
    plot_learning_curves(dtr, X, y, 'DecisionTree', '../png/decisionTree_curve')
    plot_learning_curves(rfr, X, y, 'RandomForest', '../png/randomForest_curve')
    plot_learning_curves(lgbmr, X, y, 'LightGBM', '../png/lightGBM')
    print("Curva di apprendimento generata per tutti i modelli.")




