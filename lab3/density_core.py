"""手写高斯分类、核密度与kNN密度；sklearn只用于数据和交叉验证。"""
import math
import numpy as np


def samples(X, dimension=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not len(X) or not X.shape[1] or not np.isfinite(X).all():
        raise ValueError('X must be a nonempty finite 2D array')
    if dimension is not None and X.shape[1] != dimension:
        raise ValueError('Feature dimension mismatch')
    return X


class GaussianBayes:
    def __init__(self, covariance='full', epsilon=1e-9):
        if covariance not in ('full', 'diag') or epsilon <= 0:
            raise ValueError('Invalid covariance or epsilon')
        self.covariance = covariance
        self._epsilon = epsilon

    def fit(self, X, y):
        X = samples(X); y = np.asarray(y)
        if y.shape != (len(X),): raise ValueError('Label shape mismatch')
        self.classes_, counts = np.unique(y, return_counts=True)
        if len(counts) != 2: raise ValueError('Two classes are required')
        self.priors_ = counts / len(X)
        self.means_ = np.array([X[y == c].mean(0) for c in self.classes_])
        self.vars_ = np.array([X[y == c].var(0) for c in self.classes_])
        self.covs_ = []
        self.cholesky_ = []
        self.log_dets_ = []
        for i, c in enumerate(self.classes_):
            centered = X[y == c] - self.means_[i]
            cov = centered.T @ centered / counts[i]
            if self.covariance == 'diag': cov = np.diag(np.diag(cov))
            cov = cov + self._epsilon * np.eye(X.shape[1])
            L = np.linalg.cholesky(cov)
            self.covs_.append(cov); self.cholesky_.append(L)
            self.log_dets_.append(2 * np.log(np.diag(L)).sum())
        self.covs_ = np.array(self.covs_)
        return self

    def _calculate_log_likelihood(self, X):
        X = samples(X, self.means_.shape[1])
        values = []
        for mean, L, logdet in zip(self.means_, self.cholesky_, self.log_dets_):
            z = np.linalg.solve(L, (X - mean).T)
            values.append(-0.5 * (X.shape[1] * np.log(2*np.pi) + logdet + (z*z).sum(0)))
        return np.column_stack(values)

    def decision_function(self, X):
        joint = self._calculate_log_likelihood(X) + np.log(self.priors_)
        return joint[:, 1] - joint[:, 0]

    def predict(self, X):
        joint = self._calculate_log_likelihood(X) + np.log(self.priors_)
        return self.classes_[np.argmax(joint, axis=1)]

    def predict_proba(self, X):
        joint = self._calculate_log_likelihood(X) + np.log(self.priors_)
        exp = np.exp(joint - joint.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


class MyGaussianNB(GaussianBayes):
    def __init__(self): super().__init__('diag')


def apply_LRT_rule(model, X):
    log_like = model._calculate_log_likelihood(X)
    log_ratio = log_like[:, 1] - log_like[:, 0]
    threshold = np.log(model.priors_[0]) - np.log(model.priors_[1])
    return model.classes_[(log_ratio > threshold).astype(int)]


class MyGaussianKernelDensity:
    def fit(self, X):
        self.X_train_ = samples(X).copy()
        self.N_, self.d_ = self.X_train_.shape
        return self

    def predict_log_density(self, X_new, h):
        X_new = samples(X_new, self.d_)
        if not np.isfinite(h) or h <= 0: raise ValueError('h must be positive')
        out = []
        normalizer = np.log(self.N_) + self.d_*np.log(h) + self.d_/2*np.log(2*np.pi)
        for begin in range(0, len(X_new), 512):
            diff = (X_new[begin:begin+512, None, :] - self.X_train_[None, :, :]) / h
            logk = -0.5 * np.sum(diff*diff, axis=2)
            m = logk.max(axis=1)
            out.append(m + np.log(np.exp(logk-m[:, None]).sum(axis=1)) - normalizer)
        return np.concatenate(out)


class MyKernelClassifier:
    def fit(self, X, y):
        X = samples(X); y = np.asarray(y)
        if y.shape != (len(X),): raise ValueError('Label shape mismatch')
        self.classes_, counts = np.unique(y, return_counts=True)
        if len(counts) != 2: raise ValueError('Two classes are required')
        self.priors_ = counts / len(X)
        self.models_ = [MyGaussianKernelDensity().fit(X[y == c]) for c in self.classes_]
        self.kde_c0_, self.kde_c1_ = self.models_
        self.log_prior_c0_, self.log_prior_c1_ = np.log(self.priors_)
        return self

    def decision_function(self, X, h):
        a, b = [m.predict_log_density(X, h) for m in self.models_]
        return b - a - (self.log_prior_c0_ - self.log_prior_c1_)

    def predict(self, X, h):
        return self.classes_[(self.decision_function(X, h) > 0).astype(int)]


class MyKNNDensityEstimator:
    def fit(self, X):
        self.X_train_ = samples(X).copy()
        self.N_, self.d_ = self.X_train_.shape
        return self

    def radii(self, X_new, k, exclude_indices=None):
        X_new = samples(X_new, self.d_)
        limit = self.N_ - (exclude_indices is not None)
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= limit:
            raise ValueError('k is outside the valid range')
        if exclude_indices is not None:
            exclude_indices = np.asarray(exclude_indices)
            if exclude_indices.shape != (len(X_new),) or not np.issubdtype(exclude_indices.dtype, np.integer) or np.any((exclude_indices < 0) | (exclude_indices >= self.N_)):
                raise ValueError('Invalid excluded row indices')
        out=[]
        for begin in range(0,len(X_new),512):
            q = X_new[begin:begin+512]
            distances = np.sum((q[:,None,:]-self.X_train_[None,:,:])**2,axis=2)
            if exclude_indices is not None:
                distances[np.arange(len(q)),exclude_indices[begin:begin+len(q)]]=np.inf
            out.append(np.sqrt(np.partition(distances,k-1,axis=1)[:,k-1]))
        return np.concatenate(out)

    def predict(self, X_new, k, exclude_indices=None):
        r = self.radii(X_new, k, exclude_indices)
        unit_volume = np.pi**(self.d_/2) / math.gamma(self.d_/2+1)
        # 零半径的理论密度为无穷；保留数学含义，不用有限大数冒充有效密度。
        volume = unit_volume * r**self.d_
        effective_n = self.N_ - (exclude_indices is not None)
        return np.divide(float(k), effective_n*volume, out=np.full_like(r,np.inf), where=volume>0)
