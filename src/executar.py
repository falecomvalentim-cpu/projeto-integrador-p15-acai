"""P15: coleta -> raw PostgreSQL -> ETL -> modelo de negócio -> ML."""
from pathlib import Path
import argparse, csv, hashlib, json, os, tempfile, urllib.request
from datetime import datetime, timezone
import pandas as pd
import psycopg
from psycopg.types.json import Jsonb
from .limpeza import clean, norm
from .treinamento import train
from .eda import eda, model_charts

ROOT = Path(__file__).resolve().parents[1]
IDS = [1600303,1600600,1501402,1506807,1302603]
URL = 'https://servicodados.ibge.gov.br/api/v1/localidades/municipios/' + '|'.join(map(str, IDS))

def collect(root, refresh=False):
    snapshot=root/'dados/ibge_municipios.json'
    manifest=root/'dados/fontes.json'
    if refresh or not snapshot.exists():
        req=urllib.request.Request(URL, headers={'User-Agent':'ProjetoAcademicoP15/1.0'})
        with urllib.request.urlopen(req, timeout=60) as response:
            payload=response.read()
        records=json.loads(payload)
        if {r['id'] for r in records} != set(IDS):
            raise ValueError('Resposta IBGE não contém os cinco municípios solicitados')
        snapshot.write_bytes(payload)
        metadata={'ibge':{'url':URL,'documentacao':'https://servicodados.ibge.gov.br/api/docs/localidades',
            'coletado_em_utc':datetime.now(timezone.utc).isoformat(),'metodo':'GET HTTPS; resposta JSON original preservada',
            'sha256':hashlib.sha256(payload).hexdigest(),'registros':len(records)},
            'm3':{'origem':'Base fictícia da atividade M3, reutilizada da M5',
                'metodo':'Cópia local dos arquivos originais, sem alterar bytes',
                'natureza':'Vendas, clientes e produtos simulados; uma única origem',
                'arquivos':{name:hashlib.sha256((root/'dados'/name).read_bytes()).hexdigest()
                    for name in ['vendas.csv','clientes.json','produtos.csv']}}}
        manifest.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
    metadata=json.loads(manifest.read_text(encoding='utf8'))
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==metadata['ibge']['sha256']
    for name, digest in metadata['m3']['arquivos'].items():
        assert hashlib.sha256((root/'dados'/name).read_bytes()).hexdigest()==digest
    return metadata

def main(root=ROOT, refresh=False):
    root=Path(root)
    for folder in ['resultados','graficos','modelos','evidencias']:
        (root/folder).mkdir(exist_ok=True)
    dsn=os.environ.get('P15_DATABASE_URL')
    if not dsn: raise RuntimeError('Defina P15_DATABASE_URL para um PostgreSQL dedicado')
    metadata=collect(root, refresh)
    source={
        'vendas':pd.read_csv(root/'dados/vendas.csv',sep=';',dtype=str,keep_default_na=False).to_dict('records'),
        'clientes':json.loads((root/'dados/clientes.json').read_text(encoding='utf-8-sig')),
        'produtos':pd.read_csv(root/'dados/produtos.csv',sep=';',dtype=str,keep_default_na=False).to_dict('records'),
        'ibge':json.loads((root/'dados/ibge_municipios.json').read_text(encoding='utf8'))}
    with psycopg.connect(dsn) as con:
        with con.cursor() as cur:
            cur.execute((root/'sql/estrutura.sql').read_text(encoding='utf8'))
            cur.execute('TRUNCATE p15.vendas,p15.clientes,p15.produtos,p15.municipios,p15.rejeitadas,p15.raw,p15.fontes')
            cur.executemany('INSERT INTO p15.fontes VALUES (%s,%s)',[(key,Jsonb(value)) for key,value in metadata.items()])
            cur.executemany('INSERT INTO p15.raw VALUES (%s,%s,%s)',
                [(kind,i,Jsonb(row)) for kind,rows in source.items() for i,row in enumerate(rows)])
            # O ETL recebe a releitura raw do banco, e não os arquivos de entrada.
            staged={}
            for kind in source:
                cur.execute('SELECT payload FROM p15.raw WHERE fonte=%s ORDER BY ordem',(kind,))
                staged[kind]=[r[0] for r in cur.fetchall()]
                assert staged[kind]==source[kind]
            cities=[]
            for r in staged['ibge']:
                immediate=r['regiao-imediata'];uf=immediate['regiao-intermediaria']['UF']
                cities.append((r['id'],r['nome'],uf['sigla'],immediate['nome']))
            cur.executemany('INSERT INTO p15.municipios VALUES (%s,%s,%s,%s)',cities)
            with tempfile.TemporaryDirectory() as directory:
                tmp=Path(directory);(tmp/'dados').mkdir()
                for key in ['vendas','produtos']:
                    pd.DataFrame(staged[key]).to_csv(tmp/'dados'/f'{key}.csv',sep=';',index=False)
                (tmp/'dados/clientes.json').write_text(json.dumps(staged['clientes'],ensure_ascii=False),encoding='utf8')
                df,rejected,clean_metrics=clean(tmp)
            # Substitui os hashes dos arquivos temporários pelos hashes da origem preservada.
            clean_metrics['sha256_originais']=metadata['m3']['arquivos']
            cur.executemany('INSERT INTO p15.clientes VALUES (%s,%s,%s)',
                [(norm(r['cliente_id']).upper(),norm(r['nome']),norm(r['segmento']).lower()) for r in staged['clientes']])
            cur.executemany('INSERT INTO p15.produtos VALUES (%s,%s,%s)',
                [(norm(r['produto_id']).upper(),norm(r['nome']),norm(r['categoria']).lower()) for r in staged['produtos']])
            geography=pd.DataFrame(cities,columns=['municipio_id','municipio','uf','regiao_imediata'])
            df=df.merge(geography,on=['municipio','uf'],how='left',validate='many_to_one')
            if df.municipio_id.isna().any(): raise ValueError('Venda sem município no IBGE')
            columns=['venda_id','cliente_id','produto_id','data','municipio_id','quantidade','preco_unitario','valor_total']
            cur.executemany('INSERT INTO p15.vendas VALUES (%s,%s,%s,%s,%s,%s,%s,%s)',
                [tuple(row) for row in df[columns].itertuples(index=False,name=None)])
            cur.executemany('INSERT INTO p15.rejeitadas VALUES (%s,%s,%s)',
                [tuple(row) for row in rejected.itertuples(index=False,name=None)])
            cur.execute('SELECT version()'); version=cur.fetchone()[0]
            cur.execute('SELECT * FROM p15.dataset_ml ORDER BY data,venda_id')
            names=[c.name for c in cur.description]
            loaded=pd.DataFrame(cur.fetchall(),columns=names)
            loaded['data']=loaded.data.map(str)
            for col in ['preco_unitario','valor_total']: loaded[col]=loaded[col].astype(float)
            expected=df[names].sort_values(['data','venda_id']).reset_index(drop=True)
            pd.testing.assert_frame_equal(loaded,expected,check_dtype=False)
            cur.execute('SELECT count(*),sum(valor_total) FROM p15.vendas')
            count,revenue=cur.fetchone()
            assert count==592 and float(revenue)==130405.70
        con.commit()
    loaded.to_csv(root/'dados/vendas_integradas.csv',index=False)
    rejected.to_csv(root/'resultados/rejeitadas.csv',index=False)
    def save(name,value): (root/'resultados'/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    save('metricas_limpeza.json',clean_metrics)
    save('postgresql.json',{'executado':True,'versao':version,'schema':'p15','registros':count,
        'receita':str(revenue),'raw_releitura_igual':True,'dataset_releitura_igual':True,
        'municipios_ibge':len(cities),'vendas_integradas_ibge':len(loaded),'commit_confirmado':True})
    eda(loaded,clean_metrics,root)
    metrics=train(loaded,root);model_charts(root)
    save('execucao.json',{'concluido':True,'duas_origens':['M3 fictícia','IBGE Localidades'],
        'fluxo':'Fontes -> raw PostgreSQL -> ETL Python -> tabelas PostgreSQL -> features -> ML -> métricas',
        'modelo':metrics['modelo_selecionado']})
    print('P15 concluída; PostgreSQL e duas fontes integradas.',flush=True)
    return metrics

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--atualizar-ibge',action='store_true')
    main(refresh=parser.parse_args().atualizar_ibge)
