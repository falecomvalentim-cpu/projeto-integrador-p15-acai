"""Limpeza deterministica, sem estatisticas aprendidas do teste."""
from pathlib import Path
from decimal import Decimal, InvalidOperation
from datetime import datetime
import csv, json, re, hashlib
import pandas as pd

COLS=['venda_id','cliente_id','produto_id','data','municipio','uf','quantidade','preco_unitario']
CITIES={'Macapá':'AP','Santana':'AP','Belém':'PA','Santarém':'PA','Manaus':'AM'}

def norm(v): return re.sub(r'\s+',' ',str(v or '').strip())
def date(v):
    for fmt in ('%Y-%m-%d','%d/%m/%Y'):
        try: return datetime.strptime(norm(v),fmt).date()
        except ValueError: pass
    return None
def price(v):
    v=re.sub(r'R\$|\s','',norm(v))
    if ',' in v: v=v.replace('.','').replace(',','.')
    if not re.fullmatch(r'\d+(\.\d{1,2})?',v): return None
    try:
        d=Decimal(v)
        return d if d.is_finite() and d>0 else None
    except InvalidOperation: return None

def clean(root):
    root=Path(root)
    raw=pd.read_csv(root/'dados/vendas.csv',sep=';',dtype=str,keep_default_na=False)
    customers=json.loads((root/'dados/clientes.json').read_text(encoding='utf-8-sig'))
    clients={norm(c['cliente_id']).upper():{'segmento':norm(c['segmento']).lower()} for c in customers}
    products=pd.read_csv(root/'dados/produtos.csv',sep=';',dtype=str,keep_default_na=False)
    prods={norm(r['produto_id']).upper():{'categoria':norm(r['categoria']).lower()} for r in products.to_dict('records')}
    assert len(clients)==len(customers) and len(prods)==len(products)
    rows=[]; rejected=[]; seen=set(); changed={k:0 for k in COLS}; defects={}
    normalized_ids=raw['venda_id'].map(lambda x:norm(x).upper())
    before_null={c:int(raw[c].map(norm).eq('').sum()) for c in COLS}
    for order,r in enumerate(raw.to_dict('records')):
        c={k:norm(v) for k,v in r.items()}
        for k in ('venda_id','cliente_id','produto_id','uf'): c[k]=c[k].upper()
        c['municipio']=c['municipio'].title()
        d=date(c['data']); p=price(c['preco_unitario'])
        q=int(c['quantidade']) if re.fullmatch(r'[+-]?\d+',c['quantidade']) else None
        checks=[
            ('id_ausente',not bool(c['venda_id'])),
            ('cliente_desconhecido',c['cliente_id'] not in clients),
            ('produto_desconhecido',c['produto_id'] not in prods),
            ('data_invalida',d is None),
            ('data_fora_periodo',d is not None and d.year!=2025),
            ('quantidade_invalida',q is None or not 1<=q<=1000),
            ('preco_invalido',p is None),
            ('municipio_uf_invalido',CITIES.get(c['municipio'])!=c['uf'])]
        for key,bad in checks:
            if bad: defects[key]=defects.get(key,0)+1
        reason=next((key for key,bad in checks if bad),None)
        if reason is None and c['venda_id'] in seen: reason='duplicada'
        if reason:
            rejected.append({'ordem':order,'venda_id':c['venda_id'],'motivo':reason})
            continue
        seen.add(c['venda_id'])
        c.update(data=d.isoformat(),quantidade=q,preco_unitario=float(p),valor_total=float(p*q),
                 **clients[c['cliente_id']],**prods[c['produto_id']])
        for k in COLS:
            canonical=f'{p:.2f}' if k=='preco_unitario' else str(c[k])
            if canonical!=str(r[k]): changed[k]+=1
        rows.append(c)
    df=pd.DataFrame(rows).sort_values(['data','venda_id']).reset_index(drop=True)
    reasons=pd.Series([r['motivo'] for r in rejected]).value_counts().to_dict()
    metrics={'antes':{'registros':len(raw),'ausentes_por_coluna':before_null,
       'duplicatas_linha_completa':int(raw.duplicated().sum()),
       'duplicatas_id_normalizado':int(normalized_ids.duplicated().sum()),
       'inconsistencias_independentes':defects},
       'depois':{'registros':len(df),'ausentes_por_coluna':df.isna().sum().astype(int).to_dict(),
       'duplicatas_id':int(df.venda_id.duplicated().sum()),'receita':round(float(df.valor_total.sum()),2)},
       'rejeitadas':len(rejected),'motivos_exclusivos':{k:int(v) for k,v in reasons.items()},
       'campos_normalizados_em_linhas_aceitas':changed,
       'politica':'Rejeitar registros invalidos; primeira ocorrencia valida por venda_id. Sem imputar target/preco.',
       'sha256_originais':{f:hashlib.sha256((root/'dados'/f).read_bytes()).hexdigest()
                          for f in ['vendas.csv','clientes.json','produtos.csv']}}
    assert len(raw)==len(df)+len(rejected)
    assert len(df)>=500 and df.venda_id.is_unique
    assert len(df)==592 and round(df.valor_total.sum(),2)==130405.70
    return df,pd.DataFrame(rejected),metrics
