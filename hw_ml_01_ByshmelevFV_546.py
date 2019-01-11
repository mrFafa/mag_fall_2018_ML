import pandas as pd
import numpy as np
import copy as cp


def get_weights(X, y):
    A = np.dot(X.T, X)
    b = np.dot(X.T, y)
    return np.linalg.solve(A, b)


def gradient_descent(X, y, theta=1e-2, epsilon=1e-5):
    dw = np.inf
    w = get_weights(X, y)
    np.random.seed(42)

    while dw >= epsilon:
        rand_ind = np.random.randint(X.shape[0])
        new_w = gradient_step(X, y, w, rand_ind, theta)
        dw = np.linalg.norm(w - new_w)
        w = new_w
    return w


def gradient_step(X, y, w, train_ind, theta=0.01):
    N = X.shape[0]
    x = X.iloc[train_ind, :]
    y_pred = reg_prediction(x, w)
    rs = (y_pred - y.iloc[train_ind])
    # print(N, '\n', x, '\n', y_pred, '\n', rs, '\n')
    return w - 2 * theta / N * x * rs


def is_binary(x):
    return x.unique().shape[0] == 2


def normalize(x):
    return (x - x.mean()) / x.std()


def R2(x, y):
    return 1 - np.sum(np.power(y - x, 2)) / np.sum(np.power(y - y.mean(), 2))


def reg_prediction(X, w):
    return np.dot(X, w)


def RMSE(x, y):
    return np.sqrt(np.sum(np.power(y - x, 2)) / y.shape[0])


### ввод
df = pd.read_csv('./dataset.csv')
df.drop(df.columns[0], axis=1, inplace=True)
df.drop('Post Promotion Status', axis=1, inplace=True)

### нормализуем
bin_free = df.columns[~df.apply(is_binary)]
df[bin_free] = df[bin_free].apply(normalize, axis=0)
df['w0_reg_constant'] = 1

### перемешиваем
df = df.sample(frac=1).reset_index(drop=True)
folds_index = 5
fold_size = round(df.shape[0] / folds_index)

features = pd.DataFrame()
RMSE_test = []
RMSE_train = []
R2_test = []
R2_train = []

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
    w = gradient_descent(Features, Target)
    features = features.append(w, ignore_index=True)

    train_pred = reg_prediction(train.drop('Target', axis=1), w)
    R2_train.append(R2(train_pred, train['Target']))
    RMSE_train.append(RMSE(train_pred, train['Target']))

    test_pred = reg_prediction(test.drop('Target', axis=1), w)
    R2_test.append(R2(test_pred, test['Target']))
    RMSE_test.append(RMSE(test_pred, test['Target']))

res_df = pd.DataFrame(np.vstack([R2_test, R2_train, RMSE_test, RMSE_train]),
                      index=['R2_test', 'R2_train', 'RMSE_test', 'RMSE_train'])
res_df = res_df.append(features.T)
res_df.columns = ['T1', 'T2', 'T3', 'T4', 'T5']
res_df = pd.concat([res_df, res_df.mean(axis=1).rename('E(mean)'), res_df.std(axis=1).rename('STD')], axis=1)

print(res_df)
# res_df.to_csv('out.csv', sep='\t', encoding='utf-8')
