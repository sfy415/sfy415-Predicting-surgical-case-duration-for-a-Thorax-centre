# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.model_selection import train_test_split,KFold, cross_val_score,GridSearchCV
from sklearn.metrics import mean_squared_error as mse
import xgboost as xgb
import warnings
warnings.filterwarnings(action='ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('max_columns',50)
pd.set_option('max_colwidth',150)


data_dir='../data/'
tmp_dir='../tmp/'


def linear_model(X_train,y_train,X_test,y_test,y_planed):
    model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True,n_jobs=4)
    cv = KFold(n_splits=5, shuffle=True, random_state=2019)

    mean_mse = np.mean((cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error", n_jobs=4)))
    print('linear model 5 fold mean_mse:',mean_mse)

    model.fit(X_train, y_train)
    # model predict
    y_pred = model.predict(X_test)
    mse1=mse(y_test,y_pred)
    mse2=mse(y_test,y_planed)
    print('Linear model predict mse score on test_set:',mse1)
    print('planed mse score on test_set:',mse2)

    plt.plot(range(X_test.shape[0]),y_test,color='b',label='Real Surgery_duration')
    plt.plot(range(X_test.shape[0]),y_planed,color='g',label='Planned Surgery_duration(mse:%.2f)'%mse2)
    plt.plot(range(X_test.shape[0]),y_pred,color='r',label='Pred Surgery_duration(mse:%.2f)'%mse1)
    plt.legend(loc='best')
    plt.title('linear model predicted Vs planned  on test_set')
    plt.xlabel('index')
    plt.ylabel('Surgery_duration')
    plt.savefig(tmp_dir+'linear_result.png')
    plt.close()
    # plt.show()


# gbdt GridSearch  find the best params
def gbdt_model(X,y):
    cv = KFold(n_splits=5, shuffle=True, random_state=2019)
    param={'max_depth':range(2,7),
           'n_estimators':range(200,601,100),
           }

    model=GradientBoostingRegressor(loss='ls', learning_rate=0.05,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 min_impurity_decrease=0.,
                 min_impurity_split=None, init=None, random_state=25,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, presort='auto')

    gsearch=GridSearchCV(model, param, scoring='neg_mean_squared_error',cv=cv, n_jobs=4, verbose=5)
    gsearch.fit(X,y)
    best_param=gsearch.best_params_
    best_score=gsearch.best_score_

    print('gbdt best param:\n',best_param)
    print('gbdt best score:',best_score)
    print('best_estimator_:\n',gsearch.best_estimator_)
    return gsearch.best_estimator_


# gbdt model predict
def gbdt_predict(X_train,y_train,X_test,y_test,y_planed):
    model=GradientBoostingRegressor(loss='ls', max_depth=4,learning_rate=0.05,n_estimators=400,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 min_impurity_decrease=0.,
                 min_impurity_split=None, init=None, random_state=None,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                presort='auto')

    cv = KFold(n_splits=5, shuffle=True, random_state=2019)
    mean_mse = np.mean((cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error", n_jobs=4)))
    print('model 5 fold mean_mse:',mean_mse)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse1=mse(y_test,y_pred)
    mse2=mse(y_test,y_planed)
    print('gbdt model predict mse score on test_set:',mse1)
    print('planed mse score on test_set:',mse2)

    plt.plot(range(X_test.shape[0]),y_test,color='b',label='Real Surgery_duration')
    plt.plot(range(X_test.shape[0]),y_planed,color='g',label='Planned Surgery_duration(mse:%.2f)'%mse2)
    plt.plot(range(X_test.shape[0]),y_pred,color='r',label='Pred Surgery_duration(mse:%.2f)'%mse1)
    plt.legend(loc='best')
    plt.title('gbdt model predicted Vs planned  on test_set')
    plt.xlabel('index')
    plt.ylabel('Surgery_duration')
    plt.savefig(tmp_dir+'gbdt_result.png')
    # plt.show()
    plt.close()


# xgboost GridSearch find the best params
def xgb_model(X,y):
    cv = KFold(n_splits=5, shuffle=True, random_state=2019)
    param={'max_depth':range(2,6),
           'n_estimators':range(200,601,100),
           'booster':['gbtree','gblinear','dart']
           }

    model = xgb.XGBRegressor(
                learning_rate=0.05,objective="reg:linear",
                n_jobs=4,random_state=25,silent=True,
                gamma=0.05, min_child_weight=1, max_delta_step=0,
                subsample=0.8, colsample_bytree=1, colsample_bylevel=1,
                reg_alpha=0, reg_lambda=1,
    )

    gsearch=GridSearchCV(model, param, scoring='neg_mean_squared_error',cv=cv, n_jobs=4, verbose=5)
    gsearch.fit(X,y)
    best_param=gsearch.best_params_
    best_score=gsearch.best_score_
    print('xgboost best param:\n',best_param)
    print('xgboost best score:',best_score)
    print('best_estimator_:\n',gsearch.best_estimator_)
    return gsearch.best_estimator_
    #  {'booster': 'dart', 'max_depth': 4, 'n_estimators': 200}
    # best score: -4103.268041496694


# xgboost model predict
def xgb_predict(X_train,y_train,X_test,y_test,y_planed):
    model = xgb.XGBRegressor(
                max_depth=3, learning_rate=0.05, n_estimators=400,
                silent=True, objective="reg:linear", booster='dart',
                n_jobs=4, gamma=0.05, min_child_weight=1, max_delta_step=0,
                subsample=1, colsample_bytree=1, colsample_bylevel=1,
                reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                random_state=25,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=2019)

    mean_mse = np.mean((cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error", n_jobs=4)))
    print('model 5 fold mean_mse:',mean_mse)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse1=mse(y_test,y_pred)
    mse2=mse(y_test,y_planed)
    print('model predict mse score on test_set:',mse1)
    print('planed mse score on test_set:',mse2)
    res_df=pd.DataFrame()
    res_df['real']=y_test
    res_df['planed']=y_planed
    res_df['pred']=y_pred
    res_df.to_csv(tmp_dir+'model_pred.csv',index=False)

    plt.plot(range(X_test.shape[0]),y_test,color='b',label='Real Surgery_duration')
    plt.plot(range(X_test.shape[0]),y_planed,color='g',label='Planned Surgery_duration(mse:%.2f)'%mse2)
    plt.plot(range(X_test.shape[0]),y_pred,color='r',label='Pred Surgery_duration(mse:%.2f)'%mse1)
    plt.legend(loc='best')
    plt.title('xgboost model predicted Vs planned  on test_set')
    plt.xlabel('index')
    plt.ylabel('Surgery_duration')
    plt.savefig(tmp_dir+'xgb_result.png')
    # plt.show()
    plt.close()


if __name__=="__main__":
    data=pd.read_excel(data_dir+'new_Data4.xlsx')
    targets=['Planned_surgery_duration','Surgery_duration','Hospital_days','Intensive_care_days']
    feats=[col for col in data.columns if col not in targets]

    num_feats=['Age','Euroscore1','BMI']
    cate_feats=[col for col in feats if col not in num_feats]

    scaler=StandardScaler()
    data[num_feats]=scaler.fit_transform(data[num_feats])
    data=pd.get_dummies(data,columns=cate_feats,prefix=cate_feats)

    new_feats=[col for col in data.columns if col not in targets]

    X=data[new_feats].values
    y=data[targets[0:2]].values

    gbdt_model(X,y[:,1])
    xgb_model(X,y[:,1])

    X_train,X_test,y_train_com,y_test_com=train_test_split(X,y,test_size=0.3,random_state=25)
    print('X_train shape:',X_train.shape)
    print('X_test shape:',X_test.shape)
    y_train=y_train_com[:,1]
    y_train_plan=y_train_com[:,0]
    y_test=y_test_com[:,1]
    y_test_plan=y_test_com[:,0]

    linear_model(X_train,y_train,X_test,y_test,y_test_plan)

    gbdt_predict(X_train,y_train,X_test,y_test,y_test_plan)

    xgb_predict(X_train,y_train,X_test,y_test,y_test_plan)

