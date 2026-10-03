from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
COLOR='#087F8C'
def save(fig,root,name):
    fig.tight_layout();fig.savefig(Path(root)/'graficos'/name,dpi=160,bbox_inches='tight');plt.close(fig)
def eda(df,metrics,root):
    g=Path(root)/'graficos';g.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,4)); counts=df.quantidade.value_counts().sort_index()
    ax.bar(counts.index,counts.values,color=COLOR);ax.set(xlabel='Quantidade (target)',ylabel='Vendas',title='Distribuição da quantidade nas 592 vendas válidas')
    save(fig,root,'01_target.png')
    monthly=df.assign(mes=pd.to_datetime(df.data).dt.month).groupby('mes').agg(vendas=('venda_id','size'),receita=('valor_total','sum'))
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].bar(monthly.index,monthly.vendas,color=COLOR);axes[0].set(title='Vendas por mês',xlabel='Mês',ylabel='Vendas')
    axes[1].plot(monthly.index,monthly.receita,marker='o',color='#D58127');axes[1].set(title='Receita mensal',xlabel='Mês',ylabel='R$')
    save(fig,root,'02_temporal.png')
    city=df.groupby('municipio').valor_total.sum().sort_values()
    fig,ax=plt.subplots(figsize=(8,4));ax.barh(city.index,city.values,color=COLOR);ax.set(title='Receita por município',xlabel='R$')
    save(fig,root,'03_municipios.png')
    fig,ax=plt.subplots(figsize=(8,4));data=[x.quantidade.to_numpy() for _,x in df.groupby('produto_id')];labels=list(df.groupby('produto_id').groups)
    ax.boxplot(data,tick_labels=labels);ax.set(title='Quantidade por produto',xlabel='Produto',ylabel='Quantidade')
    save(fig,root,'04_produtos.png')
    reasons=pd.Series(metrics['motivos_exclusivos']).sort_values()
    fig,ax=plt.subplots(figsize=(9,4));ax.barh(reasons.index,reasons.values,color='#D58127');ax.set(title='148 rejeições - motivo exclusivo por linha',xlabel='Registros')
    save(fig,root,'05_limpeza.png')
    numeric=df[['quantidade','preco_unitario','valor_total']].corr()
    numeric.to_csv(Path(root)/'resultados/correlacao_eda.csv')
    (Path(root)/'resultados/resumo_eda.json').write_text(df[['quantidade','preco_unitario','valor_total']].describe().to_json(indent=2),encoding='utf8')
    monthly.to_csv(Path(root)/'resultados/eda_mensal.csv')
def model_charts(root):
    root=Path(root)
    c=pd.read_csv(root/'resultados/correlacoes_target_treino.csv').dropna(subset=['pearson'])
    c=c.loc[c.pearson.abs().nlargest(15).index].sort_values('pearson')
    fig,ax=plt.subplots(figsize=(10,6));ax.barh(c.feature,c.pearson,color=COLOR);ax.axvline(0,color='#999',lw=.7);ax.set(title='15 maiores correlações absolutas com o target - apenas treino',xlabel='Pearson')
    save(fig,root,'06_correlacoes.png')
    m=pd.read_csv(root/'resultados/comparacao_modelos.csv')
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(m));ax.bar(x-.18,m.CV_MAE_media,.36,label='Validação temporal',color=COLOR);ax.bar(x+.18,m.MAE,.36,label='Teste final',color='#D58127');ax.set_xticks(x,m.modelo,rotation=10);ax.set(ylabel='MAE (unidades)',title='Comparação dos modelos e baseline');ax.legend()
    save(fig,root,'07_modelos.png')
    p=pd.read_csv(root/'resultados/previsoes_teste.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].scatter(p.real,p.previsto,s=22,alpha=.6,color=COLOR);a=min(p.real.min(),p.previsto.min());b=max(p.real.max(),p.previsto.max());axes[0].plot([a,b],[a,b],'--',color='#999');axes[0].set(xlabel='Quantidade real',ylabel='Quantidade prevista',title='Teste: real vs. previsto')
    axes[1].hist(p.residuo,bins=15,color=COLOR);axes[1].set(xlabel='Real - previsto',ylabel='Vendas',title='Resíduos do teste')
    save(fig,root,'08_residuos.png')
