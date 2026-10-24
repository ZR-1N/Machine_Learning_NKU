"""运行全部实验：conda run -n ml python Lab2/run_all.py。无 sklearn。"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from regression_core import (split_indices, standardize, design, mse, normal_fit,
                             bgd, ridge_fit, poly_fit, poly_predict)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / '\u4ee3\u7801\u6a21\u7248' / 'data'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'figure.dpi': 120})
RESULTS = {}


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / (name + '.png'), dpi=180)
    plt.close()


def table(name, rows):
    pd.DataFrame(rows).to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')
    RESULTS[name] = rows
    print(name, json.dumps(rows, ensure_ascii=False), flush=True)


def wine():
    df = pd.read_csv(DATA / 'winequality-white.csv', sep=';')
    X, y = df.iloc[:, :-1].to_numpy(float), df.iloc[:, -1].to_numpy(float)
    tr, te = split_indices(len(X))
    return df, X, y, tr, te


def wine_design():
    df, X, y, tr, te = wine()
    A, B, mean, std = standardize(X[tr], X[te])
    return df, design(A), design(B), y[tr], y[te], mean, std


def task1():
    df = pd.read_csv(DATA / 'dataset_regression.csv')
    x, y = df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)
    tr, te = split_indices(len(x))
    theta = normal_fit(design(x[tr]), y[tr])
    table('task1_metrics', [dict(n_train=len(tr), n_test=len(te), w=theta[0], b=theta[1],
          train_mse=mse(y[tr], design(x[tr]) @ theta), test_mse=mse(y[te], design(x[te]) @ theta))])
    new = np.array([-12, -3, 0, 4.5, 11])
    table('task1_predictions', [dict(x=a, prediction=b) for a, b in zip(new, design(new) @ theta)])
    plt.figure(figsize=(7, 4))
    plt.scatter(x[tr], y[tr], s=14, label='Train')
    plt.scatter(x[te], y[te], s=18, marker='x', label='Test')
    xs = np.linspace(min(x.min(), -12), max(x.max(), 11), 200)
    plt.plot(xs, design(xs) @ theta, color='crimson', label=f'y={theta[0]:.4f}x+{theta[1]:.4f}')
    plt.xlabel('x'); plt.ylabel('y'); plt.legend(); savefig('task1_fit')


def task2():
    df, A, B, yt, ye, mean, std = wine_design()
    theta, hist = bgd(A, yt)
    oracle = normal_fit(A, yt)
    table('task2_metrics', [dict(lr=0.05, epochs=len(hist), n_train=len(A), n_test=len(B),
          train_mse=mse(yt, A @ theta), test_mse=mse(ye, B @ theta),
          closed_form_train_mse=mse(yt, A @ oracle), theta_distance=float(np.linalg.norm(theta-oracle)))])
    table('task2_weights', [dict(feature=f, standardized_weight=w, raw_weight=w/s)
          for f,w,s in zip(df.columns[:-1], theta[:-1], std)] +
          [dict(feature='intercept', standardized_weight=theta[-1], raw_weight=theta[-1]-mean @ (theta[:-1]/std))])
    pd.DataFrame({'epoch':np.arange(1,len(hist)+1), 'train_mse':hist}).to_csv(OUT/'task2_history.csv',index=False)
    plt.figure(figsize=(7,4)); plt.plot(np.arange(1,len(hist)+1), hist)
    plt.yscale('log'); plt.xlabel('Epoch'); plt.ylabel('Train MSE (log scale)'); savefig('task2_convergence')


def task3():
    _, A, B, yt, ye, _, _ = wine_design()
    rows, histories = [], {}
    ref = mse(yt, A @ normal_fit(A, yt))
    bound = float(1 / np.linalg.eigvalsh(A.T @ A / len(A)).max())
    plt.figure(figsize=(8,4.6))
    for lr in [0.001, 0.01, 0.05, 0.1, 0.3, 1.0]:
        theta, hist = bgd(A, yt, lr=lr, epochs=3000)
        stable = bool(np.isfinite(hist[-1]) and hist[-1] < 1e20)
        hit = np.flatnonzero(hist <= ref + 1e-4)
        rows.append(dict(lr=lr, epochs=len(hist), status='stable' if stable else 'diverged',
                    train_mse=float(hist[-1]), test_mse=mse(ye,B@theta) if stable else None,
                    epoch_to_1e4=int(hit[0]+1) if len(hit) else None, stable_lr_upper_bound=bound))
        histories[str(lr)] = pd.Series(hist)
        plt.plot(np.arange(1,len(hist)+1),hist,label=f'lr={lr}')
    plt.yscale('log'); plt.ylim(ref*.8,1e4); plt.xlabel('Epoch'); plt.ylabel('Train MSE (log scale)')
    plt.legend(ncol=2); savefig('task3_learning_rates')
    table('task3_metrics',rows)
    pd.DataFrame(histories).rename_axis('zero_based_iteration').to_csv(OUT/'task3_histories.csv')


def task4():
    _, X, y, tr, te = wine()
    fit, val = split_indices(len(tr), seed=43)
    Z, V, _, _ = standardize(X[tr[fit]], X[tr[val]])
    A, B, mean, std = standardize(X[tr], X[te])
    rows = []
    for lam in [0, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
        inner = ridge_fit(design(Z), y[tr[fit]], lam)
        theta = ridge_fit(design(A), y[tr], lam)
        rows.append(dict(lam=lam, train_mse=mse(y[tr], design(A)@theta),
                         validation_mse=mse(y[tr[val]],design(V)@inner),
                         test_mse=mse(y[te],design(B)@theta), weight_norm=float(np.linalg.norm(theta[:-1]))))
    best = min(rows,key=lambda r:r['validation_mse'])['lam']
    for r in rows: r['selected_by_validation'] = r['lam']==best
    table('task4_metrics', rows)
    theta = ridge_fit(design(A),y[tr],1.0)
    table('task4_weights',[dict(feature=f,weight=w) for f,w in zip(list(wine()[0].columns[:-1])+['intercept'],theta)])
    fig, axes = plt.subplots(1,2,figsize=(9,3.7))
    for col in ['train_mse','validation_mse','test_mse']:
        axes[0].plot([r['lam'] for r in rows],[r[col] for r in rows],'o-',label=col)
    axes[0].set_xscale('symlog',linthresh=.01); axes[0].set_xlabel('Lambda'); axes[0].set_ylabel('MSE'); axes[0].legend(fontsize=8)
    axes[1].plot([r['lam'] for r in rows],[r['weight_norm'] for r in rows],'o-')
    axes[1].set_xscale('symlog',linthresh=.01); axes[1].set_xlabel('Lambda'); axes[1].set_ylabel('L2 norm of weights')
    savefig('task4_ridge')


def task_ext():
    df, X, y, tr, te = wine()
    fit, val = split_indices(len(tr), seed=43)
    rows, predictions = [], {'true_quality':y[te]}
    for degree in [1,2,4]:
        print(f'Polynomial degree {degree}: fitting all 11 features and interactions',flush=True)
        inner = poly_fit(X[tr[fit]], y[tr[fit]], degree)
        vm = mse(y[tr[val]],poly_predict(inner,X[tr[val]]))
        model = poly_fit(X[tr], y[tr], degree)
        test_pred=poly_predict(model,X[te])
        predictions[f'degree_{degree}']=test_pred
        rows.append(dict(degree=degree,features=model['features'],rank=model['rank'],
                         train_mse=mse(y[tr],poly_predict(model,X[tr])),validation_mse=vm,
                         test_mse=mse(y[te],test_pred),condition=model['condition']))
    best=min(rows,key=lambda r:r['validation_mse'])['degree']
    for r in rows: r['selected_by_validation']=r['degree']==best
    table('task_ext_metrics',rows)
    pd.DataFrame(predictions).to_csv(OUT/'task_ext_test_predictions.csv',index=False)
    # 11 维模型没有唯一的一维拟合曲线；增加同一 wine 数据的 alcohol 单特征实验，
    # 让散点与曲线的自变量一致，而不误把高维条件切片当成所有样本的拟合值。
    alcohol=X[:,list(df.columns[:-1]).index('alcohol')][:,None]
    xs=np.linspace(alcohol.min(),alcohol.max(),400)[:,None]
    plt.figure(figsize=(8,4.5)); plt.scatter(alcohol[tr],y[tr],s=9,alpha=.15,label='Wine train data')
    curves=[]
    for degree in [1,2,4]:
        model=poly_fit(alcohol[tr],y[tr],degree)
        plt.plot(xs[:,0],poly_predict(model,xs),label=f'degree={degree}',linewidth=2)
        curves.append(dict(degree=degree,train_mse=mse(y[tr],poly_predict(model,alcohol[tr])),test_mse=mse(y[te],poly_predict(model,alcohol[te]))))
    plt.xlabel('Alcohol (% vol)'); plt.ylabel('Quality'); plt.legend(); savefig('task_ext_polynomial_curves')
    table('task_ext_alcohol_metrics',curves)
    plt.figure(figsize=(7,4))
    for col in ['train_mse','validation_mse','test_mse']:
        plt.plot([r['degree'] for r in rows],[r[col] for r in rows],'o-',label=col)
    plt.yscale('log'); plt.xticks([1,2,4]); plt.xlabel('Degree'); plt.ylabel('MSE (log scale)'); plt.legend(); savefig('task_ext_model_selection')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--task',choices=['all','1','2','3','4','ext'],default='all')
    args=parser.parse_args()
    functions={'1':task1,'2':task2,'3':task3,'4':task4,'ext':task_ext}
    for name,func in functions.items():
        if args.task in ('all',name): func()
    RESULTS['environment']={'python':sys.version,'executable':sys.executable,'numpy':np.__version__,
       'pandas':pd.__version__,'matplotlib':matplotlib.__version__,'platform':platform.platform(),
       'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*.csv')}}
    (OUT/f'results_{args.task}.json').write_text(json.dumps(RESULTS,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__': main()
