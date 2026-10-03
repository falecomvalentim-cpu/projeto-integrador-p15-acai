CREATE SCHEMA IF NOT EXISTS p15;
CREATE TABLE IF NOT EXISTS p15.fontes (fonte text PRIMARY KEY, metadados jsonb NOT NULL);
CREATE TABLE IF NOT EXISTS p15.raw (fonte text NOT NULL, ordem integer NOT NULL, payload jsonb NOT NULL, PRIMARY KEY(fonte,ordem));
CREATE TABLE IF NOT EXISTS p15.municipios (municipio_id integer PRIMARY KEY, nome text NOT NULL, uf char(2) NOT NULL, regiao_imediata text NOT NULL, UNIQUE(nome,uf));
CREATE TABLE IF NOT EXISTS p15.clientes (cliente_id text PRIMARY KEY, nome text NOT NULL, segmento text NOT NULL);
CREATE TABLE IF NOT EXISTS p15.produtos (produto_id text PRIMARY KEY, nome text NOT NULL, categoria text NOT NULL);
CREATE TABLE IF NOT EXISTS p15.vendas (
 venda_id text PRIMARY KEY, cliente_id text NOT NULL REFERENCES p15.clientes,
 produto_id text NOT NULL REFERENCES p15.produtos, data date NOT NULL,
 municipio_id integer NOT NULL REFERENCES p15.municipios,
 quantidade integer NOT NULL CHECK(quantidade BETWEEN 1 AND 1000),
 preco_unitario numeric(12,2) NOT NULL CHECK(preco_unitario>0),
 valor_total numeric(16,2) NOT NULL CHECK(valor_total=quantidade*preco_unitario));
CREATE TABLE IF NOT EXISTS p15.rejeitadas (ordem integer PRIMARY KEY,venda_id text NOT NULL,motivo text NOT NULL);
CREATE OR REPLACE VIEW p15.dataset_ml AS
 SELECT v.venda_id,v.cliente_id,v.produto_id,v.data,m.nome AS municipio,m.uf,
 v.quantidade,v.preco_unitario,v.valor_total,c.segmento,p.categoria,
 v.municipio_id,m.regiao_imediata
 FROM p15.vendas v JOIN p15.municipios m USING(municipio_id)
 JOIN p15.clientes c USING(cliente_id) JOIN p15.produtos p USING(produto_id);
