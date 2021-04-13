# %%
from DataProcess import *

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor, ExtraTreesRegressor)
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor
from sklearn.model_selection import KFold

from sklearn.metrics import r2_score

from sklearn.neural_network import MLPRegressor
# %%
# 获取数据
U, V, Alpha = [], [], []
folder_path = "SV/"
files = os.listdir(folder_path)
for file in files:
    U = U+get_data(folder_path+file, 15)[0]
    V = V+get_data(folder_path+file, 15)[1]
    Alpha = Alpha+get_data(folder_path+file, 15)[2]

# 备份一份
U_backup = copy.deepcopy(U)
V_backup = copy.deepcopy(V)
Alpha_backup = copy.deepcopy(Alpha)
# %%
'''
（一）建立数据集，并预处理
1. 将 U、V 裁剪使之与 Alpha 长度相等
2. 构造特征，并将 U、V、Alpha的所有特征拼接
'''
# 从备份中复制数据
U = copy.deepcopy(U_backup)
V = copy.deepcopy(V_backup)
Alpha = copy.deepcopy(Alpha_backup)

# 裁剪 U、V
for u, v in zip(U, V):
    key = list(u.keys())[0]
    u_value = list(u.values())[0][2:, :]
    u[key] = u_value
    v_value = list(v.values())[0][1:, :]
    v[key] = v_value

# 构造特征，并拼接，最终特征(样本数，时间，点号，特征)，分出测试集
U_feature, U_targets = ConstructFeatrue(U)
V_feature, V_targets = ConstructFeatrue(V)
Alpha_feature, Alpha_targets = ConstructFeatrue(Alpha)

wave = np.concatenate((U_feature, V_feature, Alpha_feature), axis=3)
X_train, X_test, Y_train, Y_test = train_test_split(wave, U_targets,
                                                    test_size=0.235,
                                                    shuffle=True)

# （二）构建Stacking模型

def get_stacking(clf, x_train, y_train, x_test, y_test, n_folds=5):
    """
    stacking的核心，使用交叉验证的方法得到次级训练集
    """
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds, shuffle=True)

    scores = []  # 提取各个模型的效果
    for i, (train_index, valid_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_vld, y_vld = x_train[valid_index], y_train[valid_index]

        clf.fit(x_tra, y_tra)
        scores.append(clf.score(x_test, y_test))
        second_level_train_set[valid_index] = clf.predict(x_vld)
        test_nfolds_sets[:, i] = clf.predict(x_test)

        second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set, np.array(scores).mean()


def creat_meta_data(x_train, y_train, x_test, y_test):
    models = [RandomForestRegressor(),
              RandomForestRegressor(),
              # AdaBoostRegressor(),
              GradientBoostingRegressor(),
              ExtraTreesRegressor(),
              # SGDRegressor()
              SVR()
              ]

    train_sets = []
    test_sets = []
    scores = []
    for m in models:
        train_set, test_set, score = get_stacking(
            m, x_train, y_train, x_test, y_test)
        train_sets.append(train_set)
        test_sets.append(test_set)
        scores.append(score)

    m_train = np.concatenate([result_set.reshape(-1, 1)
                              for result_set in train_sets], axis=1)
    m_test = np.concatenate([y_test_set.reshape(-1, 1)
                             for y_test_set in test_sets], axis=1)

    return m_train, m_test, np.array(scores)


# Meta：(样本数，集成分类器数，点数，构造特征数)
Meta_train = np.zeros(
    shape=(X_train.shape[0], 5, X_train.shape[2], X_train.shape[3]))
Meta_test = np.zeros(
    shape=(X_test.shape[0], 5, X_test.shape[2], X_test.shape[3]))
# 元学习机统计（点号，构造特征数，元学习机数）
Scores = np.zeros(shape=(X_test.shape[2], X_test.shape[3], 5))
# i：点号，j：特征号
for i in range(X_train.shape[2]):
    for j in range(X_train.shape[3]):
        meta_train, meta_test, scores = creat_meta_data(X_train[:, :, i, j],
                                                        Y_train,
                                                        X_test[:, :, i, j],
                                                        Y_test)
        Meta_train[:, :, i, j] = meta_train
        Meta_test[:, :, i, j] = meta_test
        Scores[i, j, :] = scores

        print("点号: ", i,
              " 特征号: ", j, 
              " 平均分: ", round(scores.mean(), 4),
              " 最高分: ", round(scores.max(),4))

# 方案1，把四维的样本整理成二维，然后直接学习
#       长度 5点x9特征x5元学习机
Train = Meta_train.reshape(Meta_train.shape[0], -1)
Test = Meta_test.reshape(Meta_test.shape[0], -1)
# 次级分类器
dt_model = ExtraTreesRegressor()
dt_model.fit(Train, Y_train)
df_predict = dt_model.predict(Test)
print(dt_model.score(Test, Y_test))

# %%
# 使用单个点测试
point_no = 1
feature_no = 5
# 单个点的 Stacking
meta_train, meta_test, scores = creat_meta_data(
    X_train[:, :, point_no, feature_no],
    Y_train,
    X_test[:, :,  point_no, feature_no],
    Y_test)
# 次级分类器
dt_model = ExtraTreesRegressor()
dt_model.fit(meta_train, Y_train)
df_predict = dt_model.predict(meta_test)


print(dt_model.score(meta_test, Y_test))

# 单个点的 传统算法
rf2 = ExtraTreesRegressor()
rf2 = rf2.fit(X_train[:, :, point_no, feature_no], Y_train)
p = rf2.predict(X_test[:, :, point_no, feature_no])
rf2.score(X_test[:, :, point_no, feature_no], Y_test)
# %%
