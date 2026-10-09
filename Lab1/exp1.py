import numpy as np

raw = np.loadtxt('data/semeion.data') 

X = raw[:, :256]                             
y = np.argmax(raw[:, 256:], axis=1)        

print('样本数:', X.shape[0], '| 特征数(像素):', X.shape[1], '| 前10个标签:', y[:10])


# 欧氏距离
def euclidean_distances(X_train, x_test):
    squares = (X_train-x_test) ** 2
    dists = np.sqrt(squares.sum(axis=1))
    return dists


# kNN
def knn_predict(X_train, y_train, x_test, k):
    dists = euclidean_distances(X_train, x_test)
    k_idx = np.argsort(dists)[:k]
    k_labels = y_train[k_idx]
    pred = np.bincount(k_labels).argmax()
    return pred


# LOO
def loo_eval(X, y, k):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        x_test = X[i]
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        pred = knn_predict(X_train, y_train, x_test, k)
        if pred == y[i]:
            correct += 1
    acc = correct / n
    return acc

if __name__ == '__main__':
    for k in [1, 3, 5]:
        acc = loo_eval(X, y, k)
        print(f'k={k}  LOO 准确率 = {acc:.4f}')
