"""手写回归算法。只依赖 NumPy；最后一列约定为截距项。"""
from itertools import combinations_with_replacement
import numpy as np


def split_indices(n, test_ratio=0.2, seed=42):
    idx = np.random.RandomState(seed).permutation(n)
    k = int(n * test_ratio)
    return idx[k:], idx[:k]


def standardize(train, other):
    mean, std = train.mean(axis=0), train.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (train - mean) / std, (other - mean) / std, mean, std


def design(X):
    return np.column_stack([X, np.ones(len(X))])


def mse(y, pred):
    return float(np.mean((y - pred) ** 2))


def normal_fit(X, y):
    """正规方程，solve 避免显式求逆；奇异或病态时回退 SVD 伪逆。"""
    gram = X.T @ X
    if np.linalg.cond(gram) < 1e12:
        return np.linalg.solve(gram, X.T @ y)
    return np.linalg.pinv(X, rcond=1e-10) @ y


def bgd(X, y, lr=0.05, epochs=3000):
    """目标 J=MSE，梯度为 2 X^T(X theta-y)/n，每次迭代使用全部样本。"""
    theta = np.zeros(X.shape[1])
    hist = []
    for _ in range(epochs):
        with np.errstate(over='ignore', invalid='ignore'):
            theta -= lr * (2.0 / len(X)) * (X.T @ (X @ theta - y))
            loss = mse(y, X @ theta)
        hist.append(loss)
        if not np.isfinite(loss) or loss > 1e20:
            break  # 显式记录发散，不把失败曲线当成收敛结果
    return theta, np.asarray(hist)


def ridge_fit(X, y, lam=1.0):
    """最小化 SSE + lambda*||w||²，截距不惩罚。"""
    if lam < 0:
        raise ValueError('lambda must be nonnegative')
    if lam == 0:
        return normal_fit(X, y)
    penalty = np.eye(X.shape[1])
    penalty[-1, -1] = 0
    return np.linalg.solve(X.T @ X + lam * penalty, X.T @ y)


def polynomial_features(X, degree):
    """生成总次数 <= degree 的所有单项式，含交互项；不含常数列。"""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    if degree < 1:
        raise ValueError('degree must be positive')
    powers = [c for d in range(1, degree + 1)
              for c in combinations_with_replacement(range(X.shape[1]), d)]
    return np.column_stack([np.prod(X[:, c], axis=1) for c in powers])


def poly_fit(X, y, degree):
    # 先标准化原始特征，再标准化展开后的特征，统计量均仅来自训练集。
    Z, _, mean, std = standardize(X, X)
    P = polynomial_features(Z, degree)
    P, _, pm, ps = standardize(P, P)
    # SVD 最小二乘与正规方程目标相同，避免四阶时形成 X^T X 放大条件数。
    theta, _, rank, singular = np.linalg.lstsq(design(P), y, rcond=1e-10)
    return dict(theta=theta, mean=mean, std=std, pm=pm, ps=ps,
                degree=degree, rank=int(rank), features=P.shape[1],
                condition=float(singular[0] / singular[-1]))


def poly_predict(model, X):
    P = polynomial_features((X - model['mean']) / model['std'], model['degree'])
    return design((P - model['pm']) / model['ps']) @ model['theta']
