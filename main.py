import pandas as pd
import numpy as np
from sklearn.utils import shuffle

### ввод
df = pd.read_csv('./dataset.csv')
df.drop(df.columns[0], axis=1, inplace=True)
df.drop('Post Promotion Status', axis=1, inplace=True)


### нормируем
def is_binary(x):
    return x.unique().shape[0] == 2


def normalize(x):
    return (x - x.mean()) / x.std()


bin_free = df.columns[~df.apply(is_binary)]
df[bin_free] = df[bin_free].apply(normalize, axis=0)
df['w0_reg_constant'] = 1
df = df.drop(df[df.Target > df.Target.quantile(.95)].index)

### перемешиваем
df = shuffle(df)


# функция градиентного спуска
# где new_w задает измение веса с помощью градиентного спуска
def gradient_descent(X, y, theta=1e-3, epsilon=1e-5):
    dw = 1000
    w = np.random.normal(size=X.shape[1])
    N = X.shape[0]

    while dw >= epsilon:
        # new_w = 2*theta/N*(X.T@X@w - X.T@y)
        # dw = np.linalg.norm(w - new_w)
        # print(X.T@X )

        new_w = 2 / N * (X.T @ X @ w - X.T @ y)
        w = w - theta * new_w
        dw = np.linalg.norm(new_w)

    return w


# остальные функции
def R2(x, y):
    return 1 - np.sum(np.power(y - x, 2)) / np.sum(np.power(y - y.mean(), 2))


def reg_prediction(X, w):
    return X @ w


def RMSE(x, y):
    return np.sqrt(np.sum(np.power(y - x, 2)) / y.shape[0])


# делим на фолды и мутим свое грязное дело
# вторые копии для наглядости
folds_index = 5
fold_size = round(df.shape[0] / folds_index)

features = pd.DataFrame()
RMSE_test = []
RMSE_train = []
R2_test = []
R2_train = []

features2 = pd.DataFrame()
RMSE_test2 = []
RMSE_train2 = []
R2_test2 = []
R2_train2 = []

for i in range(folds_index):
    test = df[i * fold_size:(i + 1) * fold_size]
    if i == 0:
        train = df[(i + 1) * fold_size:]
    else:
        train = df[:i * fold_size]
        if i != 4:
            train = train.append(df[(i + 1) * fold_size:], ignore_index=False)

    Features = train.drop('Target', axis=1)
    Target = train['Target']
    w = gradient_descent(Features, Target, 1e-3)
    print(w)
    features = features.append(w, ignore_index=True)
    w2 = gradient_descent(Features, Target, 1e-5)
    features2 = features2.append(w2, ignore_index=True)
    print(w2)

    train_pred = reg_prediction(train.drop('Target', axis=1), w)
    R2_train.append(R2(train_pred, train['Target']))
    RMSE_train.append(RMSE(train_pred, train['Target']))

    test_pred = reg_prediction(test.drop('Target', axis=1), w)
    R2_test.append(R2(test_pred, test['Target']))
    RMSE_test.append(RMSE(test_pred, test['Target']))

    train_pred2 = reg_prediction(train.drop('Target', axis=1), w2)
    R2_train2.append(R2(train_pred2, train['Target']))
    RMSE_train2.append(RMSE(train_pred2, train['Target']))

    test_pred2 = reg_prediction(test.drop('Target', axis=1), w2)
    R2_test2.append(R2(test_pred2, test['Target']))
    RMSE_test2.append(RMSE(test_pred2, test['Target']))

# вывод результатов с шагом 0.001
res_df = pd.DataFrame(np.vstack([R2_test, R2_train, RMSE_test, RMSE_train]),
                      index=['R2_test', 'R2_train', 'RMSE_test', 'RMSE_train'])
res_df = res_df.append(features.T)
res_df.columns = ['T1', 'T2', 'T3', 'T4', 'T5']
res_df = pd.concat([res_df, res_df.mean(axis=1).rename('E(mean)'), res_df.std(axis=1).rename('STD')], axis=1)

print(res_df)

# вывод результатов с шагом 0.00001
res_df = pd.DataFrame(np.vstack([R2_test2, R2_train2, RMSE_test2, RMSE_train2]),
                      index=['R2_test', 'R2_train', 'RMSE_test', 'RMSE_train'])
res_df = res_df.append(features2.T)
res_df.columns = ['T1', 'T2', 'T3', 'T4', 'T5']
res_df = pd.concat([res_df, res_df.mean(axis=1).rename('E(mean)'), res_df.std(axis=1).rename('STD')], axis=1)

print(res_df)
# res_df.to_csv('out.csv', sep='\t', encoding='utf-8')
