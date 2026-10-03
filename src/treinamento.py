from pathlib import Path
import json,pickle,platform,sys,time
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,r2_score
from .features import CausalFeatures,NUMERIC,CATEGORICAL,CONTEXT,temporal_folds

def preprocess():
    return ColumnTransformer([('numericas',StandardScaler(),NUMERIC),
        ('categoricas',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATEGORICAL)],
        verbose_feature_names_out=False)
def pipe(model):
    return Pipeline([('features',CausalFeatures()),('encoding',preprocess()),('modelo',model)])
def metrics(y,p): return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(root_mean_squared_error(y,p)),'R2':float(r2_score(y,p))}

def train(df,root):
    root=Path(root);out=root/'resultados'
    dates=sorted(df.data.unique());cutoff=dates[int(len(dates)*.8)-1]
    train_df=df[df.data<=cutoff].copy();test_df=df[df.data>cutoff].copy()
    X=train_df[CONTEXT].reset_index(drop=True);y=train_df.quantidade.reset_index(drop=True)
    Xt=test_df[CONTEXT].reset_index(drop=True);yt=test_df.quantidade.reset_index(drop=True)
    assert X.data.max()<Xt.data.min() and set(X.venda_id).isdisjoint(Xt.venda_id)
    X.to_csv(out/'X_train_contexto.csv',index=False);Xt.to_csv(out/'X_test_contexto.csv',index=False)
    y.to_frame('quantidade').to_csv(out/'y_train.csv',index=False);yt.to_frame('quantidade').to_csv(out/'y_test.csv',index=False)
    models={'Ridge':Ridge(alpha=10.0),'RandomForest':RandomForestRegressor(n_estimators=200,max_depth=6,min_samples_leaf=8,random_state=42,n_jobs=2),
            'HistGradientBoosting':HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=10,early_stopping=False,random_state=42),
            'BaselineMedia':DummyRegressor(strategy='mean')}
    folds=list(temporal_folds(X,3)); cvrows=[]; summaries=[]
    for name,est in models.items():
        print('VALIDACAO',name,flush=True); scores=[]
        for i,(tr,va) in enumerate(folds,1):
            p=pipe(clone(est));p.fit(X.iloc[tr],y.iloc[tr]); pred=p.predict(X.iloc[va])
            m=metrics(y.iloc[va],pred);scores.append(m['MAE'])
            cvrows.append({'modelo':name,'fold':i,'treino_n':len(tr),'validacao_n':len(va),
                'treino_fim':X.iloc[tr].data.max(),'validacao_inicio':X.iloc[va].data.min(),
                'validacao_fim':X.iloc[va].data.max(),**m})
        summaries.append({'modelo':name,'CV_MAE_media':float(np.mean(scores)),'CV_MAE_desvio':float(np.std(scores))})
    selected=min([s for s in summaries if s['modelo']!='BaselineMedia'],key=lambda r:r['CV_MAE_media'])['modelo']
    testrows=[];best=None;bestpred=None
    for name,est in models.items():
        p=pipe(clone(est));p.fit(X,y);pred=p.predict(Xt)
        m=metrics(yt,pred);testrows.append({'modelo':name,**next(s for s in summaries if s['modelo']==name),**m,'selecionado_por_CV':name==selected})
        if name==selected: best=p;bestpred=pred
    pd.DataFrame(cvrows).to_csv(out/'validacao_temporal.csv',index=False)
    pd.DataFrame(testrows).to_csv(out/'comparacao_modelos.csv',index=False)
    # Matriz exata do treino, usando fit_transform causal, e transform congelado do teste.
    features=clone(best.named_steps['features'])
    F=features.fit_transform(X,y);Ft=features.transform(Xt)
    enc=clone(best.named_steps['encoding']);A=enc.fit_transform(F);At=enc.transform(Ft)
    names=list(enc.get_feature_names_out())
    pd.DataFrame(A,columns=names).to_csv(out/'X_train.csv',index=False)
    pd.DataFrame(At,columns=names).to_csv(out/'X_test.csv',index=False)
    F.to_csv(out/'features_train_antes_encoding.csv',index=False)
    Ft.to_csv(out/'features_test_antes_encoding.csv',index=False)
    # Correlacoes apenas no treino. Numericas sem escala e dummies 0/1.
    corr_frame=pd.DataFrame(A,columns=names);corr_frame['target_quantidade']=y.to_numpy()
    pearson=corr_frame.corr(method='pearson')['target_quantidade'].drop('target_quantidade')
    spearman=corr_frame.corr(method='spearman')['target_quantidade'].drop('target_quantidade')
    pd.DataFrame({'feature':names,'pearson':pearson.values,'spearman':spearman.values,'n':len(X)}).to_csv(out/'correlacoes_target_treino.csv',index=False)
    preds=pd.DataFrame({'venda_id':Xt.venda_id,'data':Xt.data,'real':yt,'previsto':bestpred,'residuo':yt-bestpred})
    preds.to_csv(out/'previsoes_teste.csv',index=False)
    artifact={'pipeline':best,'target':'quantidade','modelo':selected,'cutoff_treino':cutoff,
              'contexto_colunas':CONTEXT,'sklearn':sklearn.__version__,'python':platform.python_version(),
              'protocolo':'Teste em lote; historicos congelados no fim do treino; sem targets do teste.'}
    with (root/'modelos/modelo_final.pkl').open('wb') as f:pickle.dump(artifact,f,protocol=pickle.HIGHEST_PROTOCOL)
    with (root/'modelos/modelo_final.pkl').open('rb') as f:loaded=pickle.load(f)
    assert np.allclose(loaded['pipeline'].predict(Xt),bestpred,rtol=0,atol=1e-12)
    result={'target':'quantidade','selecao':'Menor MAE medio dos 3 folds temporais, entre os 3 modelos ML; baseline separado.',
       'modelo_selecionado':selected,'seed':42,'treino_n':len(X),'teste_n':len(Xt),
       'treino_inicio':X.data.min(),'treino_fim':X.data.max(),'teste_inicio':Xt.data.min(),'teste_fim':Xt.data.max(),
       'features_numericas':len(NUMERIC),'campos_categoricos':len(CATEGORICAL),'features_apos_encoding':len(names),
       'comparacao':testrows,'modelo_recarregado_previsoes_iguais':True,
       'versoes':{'python':platform.python_version(),'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__},
       'limitacao':'Dados ficticios M3; correlacoes fracas e R2 negativo sao resultados validos. Nao comprova utilidade em producao.',
       'protocolo_temporal':'Dias inteiros em folds; nenhuma linha do mesmo dia atravessa fronteira. Historico previo estrito por dia.',
       'target_excluido_X':True,'valor_total_excluido_X':True}
    (out/'metricas_ml.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('MODELOS:',json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result
