# P15 — Pipeline de Engenharia de Dados e ML para vendas de açaí

Projeto Integrador individual. Objetivo: estimar a quantidade vendida por transação de açaí e produtos relacionados, usando informações conhecidas antes da venda.

**Justificativa da realização individual:** Optei pela realização individual do Projeto Integrador em razão dos meus horários de trabalho como docente em duas faculdades, que dificultam a conciliação de agendas para o desenvolvimento em equipe.

## Fontes e procedência

- **Fonte 1 — M3 fictícia:** vendas.csv, clientes.json e produtos.csv reutilizados da M5, sem alteração dos bytes. Esses arquivos são uma única origem, não três fontes distintas. As vendas são simuladas, não registros comerciais reais.
- **Fonte 2 — IBGE Localidades:** cadastro oficial de Macapá, Santana, Belém, Santarém e Manaus. Coleta por GET HTTPS na API https://servicodados.ibge.gov.br/api/v1/localidades/municipios/1600303|1600600|1501402|1506807|1302603 . Documentação: https://servicodados.ibge.gov.br/api/docs/localidades . O JSON original, URL, data UTC e SHA256 estão em dados/ibge_municipios.json e dados/fontes.json.

O snapshot permite reprodução sem rede. --atualizar-ibge realiza nova coleta, valida os cinco códigos e atualiza o manifesto. A integração usa nome do município e UF, produz código IBGE e região imediata. O cadastro foi consultado retrospectivamente: não é uma fotografia territorial histórica de 2025. Não contém targets nem informações de vendas futuras.

## Organização e fluxo

- src/executar.py: coleta e verificação, carga raw, ETL e orquestração.
- src/limpeza.py: normalização, validação e rejeições determinísticas.
- src/features.py: atributos de calendário e históricos estritamente anteriores.
- src/treinamento.py: validação temporal, modelos, métricas e serialização.
- src/eda.py: análise exploratória e gráficos.
- sql/estrutura.sql: modelagem PostgreSQL com chaves e restrições.
- dados/: originais, snapshot, manifesto e dataset integrado.
- resultados/: qualidade, validação temporal, features, métricas e previsões.
- modelos/modelo_final.pkl: pipeline ML escolhido por validação.
- graficos/: oito figuras exploratórias e de avaliação.
- notebooks/final.ipynb: notebook final; executar de cima para baixo.
- evidencias/: verificação da execução do notebook.

Fontes → tabelas fontes/raw no PostgreSQL → releitura raw → ETL Python → dimensões clientes/produtos/municípios e tabela vendas → view dataset_ml → releitura conferida → features → ML → métricas.

As entradas são preservadas como JSONB, com fonte e ordem. O ETL normaliza datas, textos, IDs e preços; valida cadastros, quantidade, preço, município/UF e duplicatas. Não imputa target/preço. Registra 148 rejeições, aceita 592 vendas e integra todas com o IBGE. Vendas possui chaves estrangeiras e CHECK de valor_total=quantidade*preco_unitario. A recarga transacional afeta apenas as tabelas do schema p15; execute em banco dedicado.

## Reprodução

Requisitos: Python 3.12, PostgreSQL 17, terminal na raiz deste projeto.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edite .env: defina usuário, banco e uma senha própria.
docker compose up -d
# Aguarde o healthcheck indicar banco saudável.
$env:P15_DATABASE_URL='postgresql://p15user:SUA_SENHA@127.0.0.1:55436/p15'
python -m src.executar
```

Docker é uma opção de reprodução. Também é possível usar PostgreSQL nativo e apontar P15_DATABASE_URL para um banco dedicado existente. Use a mesma senha configurada no .env; caracteres especiais devem ser codificados para uma URL. O arquivo .env não é carregado automaticamente pelo Python.

```powershell
# Opcional: atualizar a fonte externa; requer internet.
python -m src.executar --atualizar-ibge
# Abrir o notebook com o ambiente e a conexão acima:
python -m pip install jupyterlab
python -m jupyter lab notebooks/final.ipynb
```

O notebook executa o pipeline completo, sem depender de resultados antigos. A primeira célula identifica a raiz do projeto. Não registre senha em células ou arquivos versionados. O .gitignore exclui .env, ambientes, caches e arquivos temporários. O modelo exige estas versões e a pasta src importável; carregue somente o artefato confiável deste projeto.

## Features, seleção e métricas

18 atributos numéricos: calendário, seno/cosseno de mês e semana, preço e históricos de cliente, produto, município e média global. Seis campos categóricos: produto, categoria, segmento, município, UF e região imediata. Código IBGE é chave cadastral, não variável numérica. Quantidade e valor_total nunca entram em X.

Treino com os primeiros 80% dos dias e teste com os dias finais; dias inteiros não atravessam fronteiras. Históricos do treino usam apenas dias anteriores. Teste em lote usa histórico congelado, sem y_test. Média global prévia suavizada com prior 5 e peso 5; médias locais suavizadas com a média global prévia. Categoria nova é ignorada no one-hot. Escala e encoding são ajustados somente no treino, inclusive dentro dos folds.

Ridge, Random Forest e HistGradientBoosting usam os parâmetros reutilizados da M5, semente 42 e três folds expansivos. Seleção do modelo ML pelo menor MAE médio de validação. DummyRegressor da média é comparado separadamente, com a mesma divisão. MAE e RMSE são erros em unidades; R² pode ser negativo. O resultado efetivamente obtido está em resultados/comparacao_modelos.csv e no notebook. Não se deve presumir melhoria por acrescentar a região do IBGE.

## Limitações e interpretação

Dados fictícios e amostra pequena não comprovam previsão de demanda real. Município, UF e região imediata são parcialmente redundantes. O IBGE acrescenta integridade cadastral e contexto geográfico; não cria relações no target nem torna as vendas reais. Na M5, o baseline superou os modelos; esta P15 deve apresentar sua própria comparação e reconhecer eventual desempenho inferior. O modelo ML é salvo para demonstrar o processo, não como recomendação de implantação.

O teste da M5 já era conhecido: esta extensão não é um novo experimento independente com teste nunca observado. Não há ajuste de parâmetros pelo teste. Correlacões de features são calculadas apenas no treino; a EDA global é descritiva. A verificação inclui releitura raw e dataset campo a campo, comparação do modelo recarregado, separação temporal e alteração de target para conferir causalidade.

## Entrega no Moodle

Informar um **link acessível do repositório Git** no campo de texto e anexar **notebooks/final.ipynb**. Este diretório é a raiz a publicar, sem runtimes locais, credenciais ou entregas M3/M4/M5. A publicação remota depende da definição da conta e do repositório de destino. Nenhum envio ao Moodle é realizado automaticamente.

## Execução verificada em 03/10/2026

PostgreSQL 17.11 nativo, banco p15: 592 vendas, 148 rejeições, cinco municípios IBGE e receita descritiva R$ 130.405,70. Releituras raw e dataset verificadas campo a campo. Notebook final executado: 18 células, nove de código, zero erros. Treino: 476 registros; teste: 116.

| Modelo | MAE teste | RMSE teste | R² teste |
|---|---:|---:|---:|
| Ridge | 2.788392 | 3.445562 | -0.331468 |
| RandomForest | 2.618156 | 3.222327 | -0.164528 |
| HistGradientBoosting | 2.903426 | 3.625553 | -0.474210 |
| BaselineMedia | 2.522964 | 3.038288 | -0.035306 |

Random Forest foi escolhido entre os modelos ML por validação. O baseline da média supera esse modelo tanto no MAE médio de validação quanto no MAE do teste. O resultado não demonstra ganho preditivo com a região do IBGE.
