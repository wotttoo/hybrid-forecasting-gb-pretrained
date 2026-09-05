import lightgbm as lgb
import xgboost as xgb
import optuna
import numpy as np
import pandas as pd

'''
Các hàm huấn luyện mô hình lightgbm, xgboost
'''
def get_lightgbm_params():
    return {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 7,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1
    }

def get_xgboost_params():
    return {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': 7,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'verbosity': 0
    }

def train_lightgbm(X_train, y_train, X_val, y_val, params=None, num_rounds=1000):
    if params is None:
        params = get_lightgbm_params()
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    model = lgb.train(
        params, 
        train_data, 
        num_boost_round=num_rounds,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    return model

def train_xgboost(X_train, y_train, X_val, y_val, params=None, num_rounds=1000):
    if params is None:
        params = get_xgboost_params()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round = num_rounds,
        evals = [(dval, 'eval')],
        callbacks=[xgb.callback.EarlyStopping(rounds=50)],
        verbose_eval=100 #in ra giá trị của metrics sau mỗi 100 rounds
    )
    return model

'''
Các hàm dự đoán của mô hình lightgbm, xgboost
'''
def predict_lightgbm(model, X):
    return model.predict(X)
def predict_xgboost(model, X):
    dtest = xgb.DMatrix(X)
    return model.predict(dtest)
 
'''
Thiết lập các hàm mục tiêu lightgbm, xgboost cho optuna để học các bộ tham số tốt nhất
'''
def objective_lightgbm(trial, X_train, y_train, X_val, y_val):
    from sklearn.metrics import mean_squared_error
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        'verbose': -1
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    model = lgb.train(
        params, 
        train_data, 
        num_boost_round = 500,
        valid_sets = [val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0)]
    )
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

def objective_xgboost(trial, X_train, y_train, X_val, y_val):
    from sklearn.metrics import mean_squared_error
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'alpha': trial.suggest_float('alpha', 0.0, 10.0),
        'lambda': trial.suggest_float('lambda', 0.0, 10.0),
        'tree_method': 'hist',
        'verbosity': 0
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round = 500,
        evals = [(dval, 'eval')],
        callbacks=[xgb.callback.EarlyStopping(rounds=50)],
        verbose_eval=False
    )
    y_pred = model.predict(dval)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

'''
Các hàm tối ưu hóa bộ tham só tốt nhất cho các mô hình lightgbm, xgboost 
'''
def optimize_xgboost_params(X_train, y_train, X_val, y_val, n_trials=100, timeout=3600, seed=42):
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    #bộ lấy mẫu Tree-structured Parzen Estimator
    sampler = TPESampler(seed=seed)
    #Bộ tỉa nhánh pruner
    pruner = MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner
    )
    study.optimize(
        lambda trial: objective_xgboost(trial, X_train, y_train, X_val, y_val), 
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    return study

def optimize_lightgbm_params(X_train, y_train, X_val, y_val, n_trials=100, timeout=3600, seed=42):
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    #bộ lấy mẫu Tree-structured Parzen Estimator
    sampler = TPESampler(seed=seed)
    #Bộ tỉa nhánh pruner
    pruner = MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner
    )
    study.optimize(
        lambda trial: objective_lightgbm(trial, X_train, y_train, X_val, y_val), 
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    return study

def train_with_best_params(study_type, study, X_train, y_train, X_val, y_val):
    best_params = study.best_trial.params
    if study_type == 'lightgbm':
        best_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1
        })
        return train_lightgbm(X_train, y_train, X_val, y_val, 
                params=best_params, num_rounds=1000)
    if study_type == 'xgboost':
        best_params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'verbosity': 0
        })
        best_params['verbosity'] = 0
        return train_xgboost(X_train, y_train, X_val, y_val, 
                params=best_params, num_rounds=1000)
    
def plot_optuna_results(study):
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_parallel_coordinate(study).show()
    optuna.visualization.plot_slice(study).show()