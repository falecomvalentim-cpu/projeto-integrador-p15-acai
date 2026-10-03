"""Historicos causais no treino; historico congelado no lote de teste."""
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

NUMERIC=['mes','dia_semana','dia_mes','dia_ano','trimestre','fim_semana',
         'mes_sin','mes_cos','semana_sin','semana_cos','preco_unitario',
         'cliente_vendas_previas','cliente_media_qtd_previa','cliente_dias_desde_ultima',
         'produto_vendas_previas','produto_media_qtd_previa',
         'municipio_vendas_previas','global_media_qtd_previa']
CATEGORICAL=['produto_id','categoria','segmento','municipio','uf','regiao_imediata']
CONTEXT=['venda_id','cliente_id','produto_id','data','municipio','uf','preco_unitario','segmento','categoria','regiao_imediata']

class CausalFeatures(TransformerMixin,BaseEstimator):
    def __init__(self,prior=5.0,smoothing=5.0): self.prior=prior;self.smoothing=smoothing
    def _state(self): return {'cliente':{},'produto':{},'municipio':{},'global':[0,0.0]}
    def _row(self,r,state):
        d=pd.Timestamp(r['data']); out={}
        out.update(mes=d.month,dia_semana=d.dayofweek,dia_mes=d.day,dia_ano=d.dayofyear,
                   trimestre=d.quarter,fim_semana=int(d.dayofweek>=5),
                   mes_sin=np.sin(2*np.pi*d.month/12),mes_cos=np.cos(2*np.pi*d.month/12),
                   semana_sin=np.sin(2*np.pi*d.dayofweek/7),semana_cos=np.cos(2*np.pi*d.dayofweek/7),
                   preco_unitario=float(r['preco_unitario']))
        n,s=state['global']; global_mean=(s+self.smoothing*self.prior)/(n+self.smoothing)
        out['global_media_qtd_previa']=global_mean
        for kind,key in [('cliente','cliente_id'),('produto','produto_id'),('municipio','municipio')]:
            count,total,last=state[kind].get(str(r[key]),(0,0.0,None))
            out[kind+'_vendas_previas']=count
            if kind!='municipio':
                out[kind+'_media_qtd_previa']=(total+self.smoothing*global_mean)/(count+self.smoothing)
            if kind=='cliente':
                out['cliente_dias_desde_ultima']=(d-last).days if last is not None else 365
        for c in CATEGORICAL: out[c]=str(r[c])
        return out
    def _update(self,r,q,state):
        d=pd.Timestamp(r['data'])
        for kind,key in [('cliente','cliente_id'),('produto','produto_id'),('municipio','municipio')]:
            count,total,last=state[kind].get(str(r[key]),(0,0.0,None))
            state[kind][str(r[key])]=(count+1,total+float(q),d)
        state['global'][0]+=1;state['global'][1]+=float(q)
    def fit_transform(self,X,y=None,**fit_params):
        if y is None: raise ValueError('Treino requer y para historico causal')
        X=X.copy().reset_index(drop=True); y=np.asarray(y,dtype=float)
        if len(X)!=len(y): raise ValueError('X/y desalinhados')
        dates=pd.to_datetime(X['data'])
        if not dates.is_monotonic_increasing: raise ValueError('Ordenar treino por data')
        state=self._state(); result=[None]*len(X)
        # Vendas no mesmo dia nunca veem os targets umas das outras.
        for day,indices in X.groupby('data',sort=True).groups.items():
            for i in indices: result[i]=self._row(X.iloc[i],state)
            for i in indices: self._update(X.iloc[i],y[i],state)
        self.state_=state; self.cutoff_=dates.max()
        self.n_features_in_=X.shape[1]
        return pd.DataFrame(result,columns=NUMERIC+CATEGORICAL)
    def fit(self,X,y=None):
        self.fit_transform(X,y);return self
    def transform(self,X):
        check_is_fitted(self,['state_','cutoff_'])
        dates=pd.to_datetime(X['data'])
        if len(X) and (dates<=self.cutoff_).any():
            raise ValueError('Inferencia deve usar datas posteriores ao ultimo dia do treino')
        # Nenhum target do teste participa das features, mesmo em lotes.
        return pd.DataFrame([self._row(r,self.state_) for r in X.to_dict('records')],
                            columns=NUMERIC+CATEGORICAL,index=X.index)
    def get_feature_names_out(self,input_features=None): return np.array(NUMERIC+CATEGORICAL,dtype=object)

def temporal_folds(X,n_splits=3):
    dates=np.array(sorted(X['data'].unique()))
    blocks=np.array_split(dates,n_splits+1)
    for k in range(1,len(blocks)):
        train_dates=np.concatenate(blocks[:k]);valid_dates=blocks[k]
        train=np.flatnonzero(X['data'].isin(train_dates).to_numpy())
        valid=np.flatnonzero(X['data'].isin(valid_dates).to_numpy())
        assert X.iloc[train]['data'].max()<X.iloc[valid]['data'].min()
        yield train,valid
