"""算法数值核验与数据检查：在 ml 环境中运行，无 sklearn。"""
import json
import numpy as np
from regression_core import (normal_fit, ridge_fit, design, polynomial_features,
                             poly_fit, poly_predict, split_indices)
from run_all import DATA, OUT, wine
import pandas as pd


def main():
    rng = np.random.RandomState(7)
    X = design(rng.normal(size=(60, 3)))
    y = X @ np.array([2., -1., .5, 3.]) + rng.normal(0,.05,60)
    ols = normal_fit(X,y)
    np.testing.assert_allclose(ols,np.linalg.lstsq(X,y,rcond=None)[0],atol=1e-10)
    duplicate = np.column_stack([X[:,0],X[:,0],np.ones(len(X))])
    pred = duplicate @ normal_fit(duplicate,y)
    np.testing.assert_allclose(pred,duplicate@np.linalg.lstsq(duplicate,y,rcond=None)[0],atol=1e-9)
    ridge = ridge_fit(X,y,10)
    np.testing.assert_allclose(X.T@(X@ridge-y)+10*np.r_[ridge[:-1],0],0,atol=1e-10)
    shifted = ridge_fit(X,y+100,10)
    np.testing.assert_allclose(shifted-ridge,[0,0,0,100],atol=1e-10)
    # 有限差分独立核对 MSE 梯度。
    theta=rng.normal(size=4); eps=1e-6
    analytic=2*X.T@(X@theta-y)/len(y)
    numeric=np.array([(np.mean((X@(theta+eps*e)-y)**2)-np.mean((X@(theta-eps*e)-y)**2))/(2*eps) for e in np.eye(4)])
    np.testing.assert_allclose(analytic,numeric,rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(polynomial_features(np.array([[2.,3.]]),2),[[2,3,4,6,9]])
    pm=poly_fit(X[:,:-1],y,1)
    np.testing.assert_allclose(poly_predict(pm,X[:,:-1]),X@ols,atol=1e-9)
    df, features, target, tr, te=wine()
    assert len(set(tr)&set(te))==0 and len(tr)+len(te)==len(df)
    assert np.isfinite(features).all() and np.isfinite(target).all()
    train_rows=set(map(tuple,df.iloc[tr].to_numpy()))
    cross=sum(tuple(row) in train_rows for row in df.iloc[te].to_numpy())
    summary={'numerical_checks':'passed','wine_rows':len(df),'wine_features':features.shape[1],
             'wine_missing_values':int(df.isna().sum().sum()),'wine_duplicate_rows':int(df.duplicated().sum()),
             'test_rows_identical_to_a_training_row':cross,
             'split':'random row split as required; duplicate rows retained',
             'regression_rows':len(pd.read_csv(DATA/'dataset_regression.csv'))}
    (OUT/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
