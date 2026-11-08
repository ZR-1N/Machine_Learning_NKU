"""在当前文件所在目录保存数据、指标、预测和图像。"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
from pathlib import Path
import argparse, csv, hashlib, json, platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, train_test_split
import sklearn
from density_core import MyGaussianNB, apply_LRT_rule, GaussianBayes, MyGaussianKernelDensity, MyKernelClassifier, MyKNNDensityEstimator
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'; DATA=ROOT/'data'
H_VALUES=[0.1,0.5,1,1.5,2]; K_VALUES=[1,3,5]
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'figure.dpi':120})
COLORS=['#b54549','#2670a8'];CMAP=ListedColormap(['#f4d9da','#d9e7f4'])

def save_csv(path, rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def datasets():
    for name,sep,title in [('high',2.0,'高分离度'),('low',0.5,'低分离度')]:
        X,y=make_classification(n_samples=500,n_features=2,n_redundant=0,n_informative=2,n_classes=2,n_clusters_per_class=1,class_sep=sep,random_state=42)
        yield name,sep,title,X,y

def cv_bandwidth(X,y, h_values=H_VALUES, n_splits=5):
    folds=list(KFold(n_splits=n_splits,shuffle=True,random_state=42).split(X))
    rows=[]
    for h in h_values:
        errors=[]
        for i,(tr,va) in enumerate(folds,1):
            model=MyKernelClassifier().fit(X[tr],y[tr])
            e=float(np.mean(model.predict(X[va],h)!=y[va]));errors.append(e)
            rows.append({'h':h,'fold':i,'train_n':len(tr),'validation_n':len(va),'error_rate':e,'errors':int(round(e*len(va)))})
    means=[np.mean([r['error_rate'] for r in rows if r['h']==h]) for h in h_values]
    return h_values[int(np.argmin(means))],rows

def find_best_h(X,y,h_values=H_VALUES,n_splits=5):return cv_bandwidth(X,y,h_values,n_splits)[0]

def mesh(X,n=170):
    xx,yy=np.meshgrid(np.linspace(X[:,0].min()-1,X[:,0].max()+1,n),np.linspace(X[:,1].min()-1,X[:,1].max()+1,n))
    return xx,yy,np.column_stack([xx.ravel(),yy.ravel()])

def boundary(ax,X,y,model,title,h=None,heldout=None):
    xx,yy,Q=mesh(X)
    score=model.decision_function(Q) if h is None else model.decision_function(Q,h)
    ax.contourf(xx,yy,(score>0).reshape(xx.shape),levels=[-.5,.5,1.5],cmap=CMAP)
    if score.min()<0<score.max():ax.contour(xx,yy,score.reshape(xx.shape),levels=[0],colors='#3d3d3d',linewidths=1)
    for c in [0,1]:ax.scatter(X[y==c,0],X[y==c,1],s=10,c=COLORS[c],alpha=.65,label=f'类别{c}')
    q=X if heldout is None else X[heldout]
    truth=y if heldout is None else y[heldout]
    pred=model.predict(q) if h is None else model.predict(q,h)
    bad=pred!=truth
    if bad.any():ax.scatter(q[bad,0],q[bad,1],s=40,c='#dc9c12',marker='x',linewidths=1.2,label='误分类' if heldout is None else '测试误分类')
    if heldout is not None:ax.scatter(q[:,0],q[:,1],s=19,facecolors='none',edgecolors='#252525',linewidths=.4,label='测试样本')
    ax.set(title=title,xlabel='特征1',ylabel='特征2');ax.legend(fontsize=8,loc='best')

def confusion(y,pred):return [[int(np.sum((y==i)&(pred==j))) for j in [0,1]] for i in [0,1]]

def run(task='all'):
    OUT.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
    summary={'protocol':{'n_samples':500,'seed':42,'class_sep':[2.0,0.5],'flip_y':0.01,'bandwidths':H_VALUES,'folds':5,'test_size':.3,'covariance_regularization':1e-9,'scaled_features':False},'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__},'datasets':{}}
    metrics=[];cv_all=[];knn_all=[];parameters=[]
    for name,sep,title,X,y in datasets():
        print(f'{name}: running task {task}',flush=True)
        save_csv(DATA/f'{name}.csv',[{'row':i,'x1':x[0],'x2':x[1],'class':int(c)} for i,(x,c) in enumerate(zip(X,y))])
        indices=np.arange(len(X));tr,te=train_test_split(indices,test_size=.3,random_state=42)
        save_csv(DATA/f'{name}_split.csv',[{'row':i,'subset':'test' if i in te else 'train'} for i in indices])
        result={'class_sep':sep,'class_counts':np.bincount(y).tolist(),'train_counts':np.bincount(y[tr]).tolist(),'test_counts':np.bincount(y[te]).tolist(),'feature_correlation_by_class':[float(np.corrcoef(X[y==c].T)[0,1]) for c in [0,1]],'data_sha256':hashlib.sha256((DATA/f'{name}.csv').read_bytes()).hexdigest()}
        if task in ('all','1'):
            model=MyGaussianNB().fit(X,y);p=model.predict(X);lrt=apply_LRT_rule(model,X)
            _,_,Q=mesh(X);grid_diff=int(np.sum(model.predict(Q)!=apply_LRT_rule(model,Q)))
            mhold=MyGaussianNB().fit(X[tr],y[tr]);pred=mhold.predict(X[te])
            result['task1']={'map_error':float(np.mean(p!=y)),'lrt_error':float(np.mean(lrt!=y)),'disagreements':int(np.sum(p!=lrt)),'grid_disagreements':grid_diff,'confusion':confusion(y,p),'holdout_error':float(np.mean(pred!=y[te])),'priors':model.priors_.tolist(),'means':model.means_.tolist(),'variances':model.vars_.tolist(),'log_lrt_threshold':float(np.log(model.priors_[0]/model.priors_[1]))}
            for method,pp in [('MAP',p),('LRT',lrt)]:metrics.append({'dataset':name,'task':1,'method':method,'evaluation':'full_fit','n':len(y),'errors':int(np.sum(pp!=y)),'error_rate':float(np.mean(pp!=y))})
            fig,axes=plt.subplots(1,2,figsize=(11,4.3))
            for ax,method in zip(axes,['MAP','LRT']):boundary(ax,X,y,model,f'{title}：{method}，错误率={np.mean(p!=y):.3f}')
            fig.tight_layout();fig.savefig(OUT/f'task1_{name}.png',dpi=200);plt.close(fig)
            save_csv(OUT/f'task1_{name}_predictions.csv',[{'row':i,'actual':int(y[i]),'MAP':int(p[i]),'LRT':int(lrt[i])} for i in indices])
        if task in ('all','2'):
            best,cv=cv_bandwidth(X,y)
            for row in cv:cv_all.append({'dataset':name,'scope':'full',**row})
            model=MyKernelClassifier().fit(X,y);p=model.predict(X,best)
            best_hold,cvt=cv_bandwidth(X[tr],y[tr])
            for row in cvt:cv_all.append({'dataset':name,'scope':'holdout_train',**row})
            hold=MyKernelClassifier().fit(X[tr],y[tr]);hp=hold.predict(X[te],best_hold)
            result['task2']={'best_h':best,'full_fit_error':float(np.mean(p!=y)),'cv_errors':{str(h):float(np.mean([r['error_rate'] for r in cv if r['h']==h])) for h in H_VALUES},'cv_fold_std':{str(h):float(np.std([r['error_rate'] for r in cv if r['h']==h],ddof=1)) for h in H_VALUES},'confusion':confusion(y,p),'holdout_best_h':best_hold,'holdout_error':float(np.mean(hp!=y[te]))}
            metrics.append({'dataset':name,'task':2,'method':'KDE-LRT','evaluation':'full_fit','n':len(y),'errors':int(np.sum(p!=y)),'error_rate':float(np.mean(p!=y))})
            metrics.append({'dataset':name,'task':2,'method':'KDE-LRT','evaluation':'holdout','n':len(te),'errors':int(np.sum(hp!=y[te])),'error_rate':float(np.mean(hp!=y[te]))})
            fig,axes=plt.subplots(1,2,figsize=(11,4.3));boundary(axes[0],X,y,model,f'{title}：全数据拟合，h={best}',best);boundary(axes[1],X,y,hold,f'{title}：留出测试，h={best_hold}',best_hold,te)
            fig.tight_layout();fig.savefig(OUT/f'task2_{name}.png',dpi=200);plt.close(fig)
            save_csv(OUT/f'task2_{name}_predictions.csv',[{'row':i,'actual':int(y[i]),'full_fit_prediction':int(p[i])} for i in indices])
        if task in ('all','3'):
            model=MyKNNDensityEstimator().fit(X);xx,yy,Q=mesh(X,160)
            densities=[model.predict(Q,k) for k in K_VALUES]
            pooled=np.concatenate(densities);vmin=float(np.quantile(pooled,.01));vmax=float(np.quantile(pooled,.995))
            fig,axes=plt.subplots(1,3,figsize=(13,4.1),layout='constrained')
            for ax,k,d in zip(axes,K_VALUES,densities):
                im=ax.pcolormesh(xx,yy,d.reshape(xx.shape),cmap='viridis',norm=LogNorm(vmin,vmax),shading='auto')
                ax.scatter(X[:,0],X[:,1],s=3,c='white',alpha=.45)
                ax.set(title=f'{title}，k={k}',xlabel='特征1',ylabel='特征2')
                loo=model.predict(X,k,exclude_indices=indices)
                stats={'dataset':name,'k':k,'grid_min':float(d.min()),'grid_median':float(np.median(d)),'grid_p95':float(np.quantile(d,.95)),'grid_p99':float(np.quantile(d,.99)),'grid_max':float(d.max()),'loo_median':float(np.median(loo)),'loo_p95':float(np.quantile(loo,.95)),'loo_max':float(loo.max())};knn_all.append(stats)
                save_csv(OUT/f'task3_{name}_k{k}_density.csv',[{'x1':q[0],'x2':q[1],'density':v} for q,v in zip(Q,d)])
            fig.colorbar(im,ax=axes,label='估计密度（对数色标；上端按99.5%分位显示）',extend='max',shrink=.85)
            fig.savefig(OUT/f'task3_{name}.png',dpi=200);plt.close(fig)
            # 固定位置查询用来比较k的作用，不将有标签的分类任务混入密度估计。
            query=np.vstack([np.zeros(2),X.mean(axis=0)])
            save_csv(OUT/f'task3_{name}_queries.csv',[{'query':i,'x1':q[0],'x2':q[1],**{f'k{k}':float(model.predict(q[None,:],k)[0]) for k in K_VALUES}} for i,q in enumerate(query)])
            result['task3']={'grid_shape':list(xx.shape),'grid_points':len(Q),'stats':[r for r in knn_all if r['dataset']==name],'color_limits':[vmin,vmax],'labels_used':False}
        if task in ('all','4'):
            full=GaussianBayes('full').fit(X[tr],y[tr]);diag=MyGaussianNB().fit(X[tr],y[tr])
            result['task4']={'train_n':len(tr),'test_n':len(te),'models':{}}
            for kind,m in [('full',full),('diag',diag)]:
                pp=m.predict(X[te]);tp=m.predict(X[tr]);lrt=apply_LRT_rule(m,X[te])
                result['task4']['models'][kind]={'train_error':float(np.mean(tp!=y[tr])),'test_error':float(np.mean(pp!=y[te])),'confusion':confusion(y[te],pp),'priors':m.priors_.tolist(),'means':m.means_.tolist(),'covariances':m.covs_.tolist(),'map_lrt_disagreements':int(np.sum(pp!=lrt))}
                for scope,yy,pr in [('train',y[tr],tp),('test',y[te],pp)]:metrics.append({'dataset':name,'task':4,'method':f'Gaussian-{kind}','evaluation':scope,'n':len(yy),'errors':int(np.sum(pr!=yy)),'error_rate':float(np.mean(pr!=yy))})
                for ci,c in enumerate(m.classes_):parameters.append({'dataset':name,'model':kind,'class':int(c),'prior':float(m.priors_[ci]),'mean1':m.means_[ci,0],'mean2':m.means_[ci,1],'cov11':m.covs_[ci,0,0],'cov12':m.covs_[ci,0,1],'cov22':m.covs_[ci,1,1]})
            fig,axes=plt.subplots(1,2,figsize=(11,4.3))
            for ax,m,kind in zip(axes,[full,diag],['完整协方差','对角协方差']):boundary(ax,X,y,m,f'{title}：{kind}，测试错误率={np.mean(m.predict(X[te])!=y[te]):.3f}',heldout=te)
            fig.tight_layout();fig.savefig(OUT/f'task4_{name}.png',dpi=200);plt.close(fig)
            fp,dp=full.predict(X[te]),diag.predict(X[te]);proba=full.predict_proba(X[te])
            save_csv(OUT/f'task4_{name}_test_predictions.csv',[{'row':int(j),'actual':int(y[j]),'full_prediction':int(fp[i]),'diag_prediction':int(dp[i]),'prob_c0':proba[i,0],'prob_c1':proba[i,1]} for i,j in enumerate(te)])
        summary['datasets'][name]=result
    save_csv(OUT/('metrics.csv' if task=='all' else f'metrics_{task}.csv'),metrics)
    if cv_all:save_csv(OUT/'task2_cv.csv',cv_all)
    if knn_all:save_csv(OUT/'task3_statistics.csv',knn_all)
    if parameters:save_csv(OUT/'task4_parameters.csv',parameters)
    if task in ('all','2'):
        fig,ax=plt.subplots(figsize=(7.5,4.2))
        for name,color in zip(['high','low'],COLORS):
            stats=summary['datasets'][name]['task2'];means=[stats['cv_errors'][str(h)] for h in H_VALUES]
            ax.plot(H_VALUES,means,'o-',color=color,label='高分离度' if name=='high' else '低分离度')
        ax.set(xlabel='带宽 h',ylabel='五折验证平均错误率',xticks=H_VALUES);ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'task2_cv_curve.png',dpi=200);plt.close(fig)
    (OUT/('results_all.json' if task=='all' else f'results_{task}.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'metrics':metrics,'best_h':{n:r.get('task2',{}).get('best_h') for n,r in summary['datasets'].items()}},ensure_ascii=False,indent=2),flush=True)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['all','1','2','3','4'],default='all');run(parser.parse_args().task)
