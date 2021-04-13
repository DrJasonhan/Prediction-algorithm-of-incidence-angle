# %%
import os
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor


import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
# %matplotlib qt5

# %%
'''
数据处理方法：
1. 时间序列求导方法
2. 筛选重复数据方法
3. 从Excel中取方法
4. 去除时间列、缩放至 500 维，并去除时间列
5. 将数据按照ABCDE点拆分，并归一化

'''


def diff_data(data):
    """数据求导，如果有零，用前一时刻的值替换

    Args:
        data ([type]): [description]

    Returns:
        [type]: [description]
    """
    dD = []
    for i in range(1, data.shape[0]):
        dt = data[i, 0]-data[i-1, 0]
        # 注意不要将时间列求导
        new_X = np.zeros(shape=(len(data[i]),))
        new_X[0] = data[i, 0]
        new_X[1:] = (data[i, 1:]-data[i-1, 1:])/dt
        dD.append(new_X)
    dD = np.array(dD)

    for i in range(dD.shape[1]):
        for j in range(1, dD.shape[0]):
            if dD[j, i] == 0:
                dD[j, i] = dD[j-1, i]

    return dD


def filter_data(data):
    '''消除时间列相同的行'''
    index = []
    for i in range(1, data.shape[0]):
        if data[i, 0] != data[i-1, 0]:
            index.append(i)
    data = data[index]
    return data


def get_data(path, angle_space):
    """获取数据

    Args:
        path ([str]): 数据excel表的路径
        angle_space([float64]):隔多少度做一次实验
    Returns:
        [list]: 带label的数据集 u：位移，v：速度，alpha：加速度
    """
    u,v,alpha = [],[],[]
    for name in pd.ExcelFile(path).sheet_names:
        sheet = pd.read_excel(path, name)
        for c in range(7):
            X = sheet.iloc[2:, 7*c:7*c+6]
            X = X.dropna(axis=0).values         
            X = filter_data(X)
            u.append({c*angle_space: X})
            # 转成速度
            X = diff_data(X)
            v.append({c*angle_space: X})
            # 转成加速度
            X = diff_data(X)
            y = angle_space
            alpha.append({c*angle_space: X})
    return u,v,alpha


def zoom(X):
    """
    1. 将所有的数据缩放至 450 维
    2. 去除时间列
    Args:
        X ([narray]): 待处理的数据

    Returns:
        [narray]: 处理好的数据
    """
    # 去除时间列
    X = X[:, 1:]
    # 缩放
    step = (X.shape[0]+1)/451
    X_new = []
    # X_new.append(X[0])
    for i in range(450):
        idx = i*step
        before_idx = int(np.floor(i*step))
        after_idx = int(np.ceil(i*step))
        X_new.append(
            X[before_idx] +
            (X[after_idx]-X[before_idx])*(idx-before_idx))
    return np.array(X_new)


def split_point_data(data_list, shuffle=False):
    """
    将数据按照ABCDE点拆分，并归一化
    Args:
        data_list ([type]): [description]
        shuffle (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """

    if shuffle == True:
        random.shuffle(data_list)
    A, B, C, D, E = [], [], [], [], []
    for i in range(len(data_list)):
        key = str(list(data_list[i].keys())[0])
        value = list(data_list[i].values())[0]
        A.append(np.append(value[:, 0], float(key)))
        B.append(np.append(value[:, 1], float(key)))
        C.append(np.append(value[:, 2], float(key)))
        D.append(np.append(value[:, 3], float(key)))
        E.append(np.append(value[:, 4], float(key)))
    # 转成数组格式，并归一化
    A = StandardScaler().fit_transform(np.array(A))
    B = StandardScaler().fit_transform(np.array(B))
    C = StandardScaler().fit_transform(np.array(C))
    D = StandardScaler().fit_transform(np.array(D))
    E = StandardScaler().fit_transform(np.array(E))

    return A, B, C, D, E


def ConstructFeatrue(data):
    M = 126  # 样本数
    F1 = data[:M]
    F2 = data[M:]
    # 数据集 X 形状 (样本数，时间、点号、构造特征)
    X = np.zeros(shape=(M, 450, 5, 3))
    Y = np.zeros(shape=(M,))

    G = np.ones(shape=(M,))
    G[:42] = 0.1
    G[42:84] = 0.2
    G[84:] = 0.3

    for i in range(M):
        # 提取标签
        key = list(F1[i].keys())[0]
        # 提取、构造特征
        f1 = np.array(list(F1[i].values())[0], dtype='float32')
        f2 = np.array(list(F2[i].values())[0], dtype='float32')
        f3 = np.divide(f1, f2)
        f4 = np.divide(f1, np.sqrt(f1**2+f2**2))
        f5 = np.divide(f2, np.sqrt(f1**2+f2**2))
        # 特征维度缩放
        f1 = zoom(f1)
        f2 = zoom(f2)
        f3 = zoom(f3)
        f4 = zoom(f4)
        f5 = zoom(f5)
        # 特征标准化 ！！！
        # f1 = MinMaxScaler().fit_transform(np.array(f1))
        # f2 = MinMaxScaler().fit_transform(np.array(f2))
        # f3 = MinMaxScaler().fit_transform(np.array(f3))
        # f4 = MinMaxScaler().fit_transform(np.array(f4))
        # 特征拼接
        f1 = np.expand_dims(f1, axis=2)
        f2 = np.expand_dims(f2, axis=2)
        f3 = np.expand_dims(f3, axis=2)
        f4 = np.expand_dims(f4, axis=2)
        f5 = np.expand_dims(f5, axis=2)
        F = np.concatenate((f3, f4,f5), axis=2)
        X[i] = F
        Y[i] = key

    return X, Y


# %%
if __name__ == "__main__":

    # 获取数据
    wave_data = []
    files = os.listdir(r"Pbo/")
    for file in files:
        wave_data = wave_data+get_data("Pbo/"+file, 15)[2]
    wave_data_backup = copy.deepcopy(wave_data)

    '''
    综合使用 XY 两个特征，并构造新特征
    构造特征包括：
        F1，x分量
        F2，y分量
        F3，x/y
        F4，(sqrt(x**2+y**2))/0.1
    '''
    # 0. 建立数据集
    wave_data = copy.deepcopy(wave_data_backup)
    X, Y = ConstructFeatrue(wave_data)

    # 打乱数据
    index = np.arange(X.shape[0])
    random.shuffle(index)
    X = X[index]
    Y = Y[index]

    # 拆分成单个点预测

    def model_of_sigle_point(dataset, targets, algorithm):
        X_train, X_test, Y_train, Y_test = train_test_split(dataset, targets,
                                                            test_size=0.2, shuffle=False)
        model = make_pipeline(algorithm)
        model = model.fit(X_train, Y_train)
        prediction = model.predict(X_test)
        score = model.score(X_test, Y_test)
        return model, [X_train, X_test, Y_train, Y_test], prediction, score

    feature = 3
    model_x_y_A, dataset_x_y_A, prediction_x_y_A, score_x_y_A = model_of_sigle_point(
        X[:, :, 0, feature], Y, RandomForestRegressor())
    model_x_y_B, dataset_x_y_B, prediction_x_y_B, score_x_y_B = model_of_sigle_point(
        X[:, :, 1, feature], Y, RandomForestRegressor())
    model_x_y_C, dataset_x_y_C, prediction_x_y_C, score_x_y_C = model_of_sigle_point(
        X[:, :, 2, feature], Y, RandomForestRegressor())
    model_x_y_D, dataset_x_y_D, prediction_x_y_D, score_x_y_D = model_of_sigle_point(
        X[:, :, 3, feature], Y, RandomForestRegressor())
    model_x_y_E, dataset_x_y_E, prediction_x_y_E, score_x_y_E = model_of_sigle_point(
        X[:, :, 4, feature], Y, RandomForestRegressor())

    print("A:", score_x_y_A, '\n',
          "B:", score_x_y_B, '\n',
          "C:", score_x_y_C, '\n',
          "D:", score_x_y_D, '\n',
          "E:", score_x_y_E)

# %%
# %%
# '''
# 方案一：（混用、不区分XY）
# 1. 按ABCDE划分成5个数据集
# 2. 分别建立五个预测模型
# 3. 最后综合五个模型的建立最终预测模型
# '''
# # 步骤0：整体缩放数据
# P = copy.deepcopy(P_backup)
# for i in range(len(P)):
#     key = list(P[i].keys())[0]
#     value3 = list(P[i].values())[0]
#     P[i][key] = zoom(value3)

# # 步骤1：分成五个数据集，并分别加入标签
# A, B, C, D, E = split_point_data(P, shuffle=True)

# # 步骤2：分别训练5个模型


# def model_of_sigle_point(dataset, algorithm):
#     X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, :-1], dataset[:, -1],
#                                                         test_size=0.2, shuffle=False)
#     model = make_pipeline(algorithm)
#     model = model.fit(X_train, Y_train)
#     prediction = model.predict(X_test)
#     score = model.score(X_test, Y_test)
#     return model, [X_train, X_test, Y_train, Y_test], prediction, score


# model_A, dataset_A, prediction_A, score_A = model_of_sigle_point(
#     A, RandomForestRegressor())
# model_B, dataset_B, prediction_B, score_B = model_of_sigle_point(
#     B, RandomForestRegressor())
# model_C, dataset_C, prediction_C, score_C = model_of_sigle_point(
#     C, RandomForestRegressor())
# model_D, dataset_D, prediction_D, score_D = model_of_sigle_point(
#     D, RandomForestRegressor())
# model_E, dataset_E, prediction_E, score_E = model_of_sigle_point(
#     E, RandomForestRegressor())

# # 步骤3：综合五个模型，建立综合模型
# X_train = np.concatenate((model_A.predict(dataset_A[0]).reshape(-1, 1),
#                           model_B.predict(dataset_B[0]).reshape(-1, 1),
#                           model_C.predict(dataset_C[0]).reshape(-1, 1),
#                           model_D.predict(dataset_D[0]).reshape(-1, 1),
#                           model_E.predict(dataset_E[0]).reshape(-1, 1)), axis=1)
# X_test = np.concatenate((model_A.predict(dataset_A[1]).reshape(-1, 1),
#                          model_B.predict(dataset_B[1]).reshape(-1, 1),
#                          model_C.predict(dataset_C[1]).reshape(-1, 1),
#                          model_D.predict(dataset_D[1]).reshape(-1, 1),
#                          model_E.predict(dataset_E[1]).reshape(-1, 1)), axis=1)

# Y_train, Y_test = dataset_A[2], dataset_A[3]
# model_sum = make_pipeline(StandardScaler(), linear_model.LinearRegression())
# model_sum = model_sum.fit(X_train, Y_train)
# print("A:", score_A, '\n',
#       "B:", score_B, '\n',
#       "C:", score_C, '\n',
#       "D:", score_D, '\n',
#       "E:", score_E)

# print("综合:", model_sum.score(X_test, Y_test))

# plt.scatter(dataset_A[-1], model_A.predict(dataset_A[1]))
# plt.scatter(dataset_B[-1], model_B.predict(dataset_B[1]))
# plt.scatter(dataset_C[-1], model_C.predict(dataset_C[1]))
# plt.scatter(dataset_D[-1], model_D.predict(dataset_D[1]))
# plt.scatter(dataset_E[-1], model_E.predict(dataset_E[1]))
