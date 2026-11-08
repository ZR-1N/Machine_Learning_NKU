"""独立数值路径复核；只将参考实现用于验证，不用于主实验训练。"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import json
import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import simpson
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from density_core import MyGaussianNB, GaussianBayes, apply_LRT_rule, MyGaussianKernelDensity, MyKNNDensityEstimator, MyKernelClassifier
from run_all import ROOT, OUT, DATA, datasets, H_VALUES

def main():
    checks=[]
    def done(name,**kw):checks.append({'name':name,'passed':True,**kw});print('PASS',name,flush=True)
    rng=np.random.RandomState(17)
    X=rng.normal(size=(37,2));Q=rng.normal(size=(19,2));y=np.r_[np.zeros(19,int),np.ones(18,int)]
    g=GaussianBayes().fit(X,y)
    ref=np.column_stack([multivariate_normal.logpdf(Q,mean=g.means_[i],cov=g.covs_[i]) for i in [0,1]])
    np.testing.assert_allclose(g._calculate_log_likelihood(Q),ref,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(g.predict_proba(Q).sum(1),1,atol=1e-15)
    done('Full Gaussian log density agrees with scipy; posterior sums to one')
    kde=MyGaussianKernelDensity().fit(X)
    for h in H_VALUES:
        ref=KernelDensity(kernel='gaussian',bandwidth=h).fit(X).score_samples(Q)
        np.testing.assert_allclose(kde.predict_log_density(Q,h),ref,rtol=1e-12,atol=1e-12)
    done('Normalized Gaussian KDE agrees with sklearn at all five bandwidths')
    # One-dimensional normalized kernel must integrate to one over a wide numerical range.
    singleton=MyGaussianKernelDensity().fit(np.array([[0.]]))
    grid=np.linspace(-8,8,10001);area=simpson(np.exp(singleton.predict_log_density(grid[:,None],1)),x=grid)
    np.testing.assert_allclose(area,1,atol=1e-12)
    done('KDE normalization integral',integral=float(area))
    far=np.array([[100.,-100.]])
    assert np.isfinite(kde.predict_log_density(far,.1)).all()
    model=MyGaussianNB().fit(X,y)
    assert np.isfinite(model._calculate_log_likelihood(far)).all()
    done('Log-space prediction remains finite for distant queries')
    kd=MyKNNDensityEstimator().fit(X)
    for k in [1,3,5]:
        distances,_=NearestNeighbors(n_neighbors=k).fit(X).kneighbors(Q)
        expected=k/(len(X)*np.pi*distances[:,-1]**2)
        np.testing.assert_allclose(kd.predict(Q,k),expected,rtol=1e-12)
    assert np.isinf(kd.predict(X[:1],1)[0])
    loo_dist2 = np.sum((X[:,None,:]-X[None,:,:])**2,axis=2)
    np.fill_diagonal(loo_dist2,np.inf)
    expected_loo = 1/((len(X)-1)*np.pi*loo_dist2.min(axis=1))
    np.testing.assert_allclose(kd.predict(X,1,exclude_indices=np.arange(len(X))),expected_loo,rtol=1e-12)
    done('kNN density matches independent neighbor search; zero-radius and exclusion semantics')
    # Ties choose the first class for both MAP and strict LRT.
    tie=MyGaussianNB().fit(np.array([[-1.],[1.],[-1.],[1.]]),np.array([0,0,1,1]))
    np.testing.assert_array_equal(tie.predict(np.array([[0.]])),[0])
    np.testing.assert_array_equal(apply_LRT_rule(tie,np.array([[0.]])),[0])
    g=MyGaussianNB().fit(X,np.where(y==0,3,9))
    np.testing.assert_array_equal(g.predict(Q),apply_LRT_rule(g,Q))
    done('MAP and LRT handle ties and nonconsecutive labels consistently')
    degenerate=GaussianBayes().fit(np.array([[0.,0.],[0.,0.],[1.,1.],[1.,1.]]),[0,0,1,1])
    assert np.isfinite(degenerate._calculate_log_likelihood(Q)).all()
    done('Regularized covariance supports rank-deficient samples')
    for action in [lambda:kde.predict_log_density(Q,0),lambda:kd.predict(Q,0),lambda:kd.predict(Q,38),lambda:kd.predict(Q,1.5)]:
        try:action()
        except ValueError:pass
        else:raise AssertionError('Invalid input should raise ValueError')
    done('Invalid bandwidth and k rejected')
    saved=json.loads((OUT/'results_all.json').read_text(encoding='utf-8'))
    import csv
    with (OUT/'task2_cv.csv').open(encoding='utf-8-sig') as f:cv=list(csv.DictReader(f))
    for name,sep,title,A,b in datasets():
        disk=np.loadtxt(DATA/f'{name}.csv',delimiter=',',skiprows=1,encoding='utf-8-sig')
        np.testing.assert_allclose(disk[:,1:3],A,atol=0,rtol=0)
        np.testing.assert_array_equal(disk[:,3],b)
        nb=MyGaussianNB().fit(A,b)
        oracle=GaussianNB(var_smoothing=0).fit(A,b)
        np.testing.assert_array_equal(nb.predict(A),oracle.predict(A))
        np.testing.assert_array_equal(nb.predict(A),apply_LRT_rule(nb,A))
        assert float(np.mean(nb.predict(A)!=b))==saved['datasets'][name]['task1']['map_error']
        # Cross-validation rows independently recomputed with sklearn's density estimator.
        folds=list(KFold(5,shuffle=True,random_state=42).split(A))
        seen=[]
        for h in H_VALUES:
            for i,(tr,va) in enumerate(folds,1):
                assert not set(tr)&set(va)
                if h==H_VALUES[0]:seen.extend(va.tolist())
                joint=[]
                for c in [0,1]:
                    Z=A[tr][b[tr]==c]
                    density=KernelDensity(bandwidth=h).fit(Z).score_samples(A[va])
                    joint.append(density+np.log(len(Z)/len(tr)))
                error=float(np.mean(np.argmax(np.column_stack(joint),axis=1)!=b[va]))
                row=next(r for r in cv if r['dataset']==name and r['scope']=='full' and float(r['h'])==h and int(r['fold'])==i)
                np.testing.assert_allclose(float(row['error_rate']),error,atol=1e-15)
        assert sorted(seen)==list(range(len(A)))
        done(f'{name}: saved data, Gaussian predictions and all 25 CV errors independently verified')
    report={'status':'passed','checks':checks,'count':len(checks)}
    (OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'All {len(checks)} checks passed.')

if __name__=='__main__':main()
