import numpy as np
import pandas as pd
import xgboost as xgb
import time

# 记录程序运行时间
start_time = time.time()

#  划分数据集
train_frame = pd.read_csv('data/new_train.csv')
test_frame = pd.read_csv('data/new_test.csv')
X_train = train_frame.iloc[:, 2:-1]
Y_train = train_frame.iloc[:, -1:].values
X_test = test_frame.iloc[:, 2:]

# 模型参数设置
param = {'booster': 'gbtree',
         'max_depth': 6, 'learning_rate': 0.1,
         'objective': 'binary:logistic', 'silent': True,
         'sample_type': 'uniform',
         'normalize_type': 'tree',
         'rate_drop': 0.1,
         'skip_drop': 0.5}
num_round = 50

#  进行预测
xg_train = xgb.DMatrix(X_train[:], label=Y_train[:])
bst = xgb.train(param, xg_train, num_round)
xg_test = xgb.DMatrix(X_test)
Y_test = bst.predict(xg_test)


def convert_re(survived):
    if survived > 0.5:
        return 1
    else:
        return 0

sur = pd.Series(Y_test)

re_frame = pd.read_csv('data/gender_submission.csv')

re_frame['Survived'] = sur.map(convert_re)

re_frame.to_csv('data/re.csv', index=False)
