
# RESUMO EXECUTIVO
## Modelo Preditivo de NPS
## Objetivo, base de dados, metodologia e como reproduzir os resultados
## Tech Challenge · Fase 1 · Ciência de Dados

### 1. Objetivo do projeto
Antecipar clientes com alta probabilidade de virarem detratores (NPS baixo), usando dados do pedido — sem esperar a resposta da pesquisa de satisfação.
Em uma frase transformar dados operacionais (entrega, atendimento, reclamações) em um modelo que sinaliza risco de insatisfação a tempo de agir.

### 2. Descrição da base de dados
2.500 pedidos / linhas	
19 variáveis
0% nulos/duplicados

**Arquivo desafio_nps_fase_1.csv — 1 linha por pedido (customer_id pode se repetir). Principais grupos de variáveis:**
    • Perfil do cliente — idade, região, tempo de casa
    • Pedido — valor, itens, desconto, parcelas
    • Logística — tempo de entrega, atraso, frete, tentativas
    • Atendimento — contatos, tempo de resolução, reclamações
    • Satisfação — CSAT interno e nps_score (0-10, variável-base)
    • Pós-venda — repeat_purchase_30d (recompra em até 30 dias)

### 3. Metodologia utilizada
1.	Ingestão e diagnóstico — carga do CSV, inspeção inicial, checagem de nulos/duplicidades e regras de negócio (NPS 0-10, valores não negativos etc.).
2.	Tratamento — cópia da base, remoção de duplicidades, conversão forçada de tipos numéricos.
3.	Feature engineering — classificação de NPS (Detrator/Neutro/Promotor), flags de atraso/reclamação/atendimento, faixas de idade e entrega.
4.	Análise exploratória — NPS geral, distribuição das notas, NPS por atraso, atendimento, região, correlação de Spearman e recompra por classe.
5.	Modelagem — alvo is_detractor (NPS ≤ 6), 15 features, split 80/20 estratificado, pipeline com imputação + one-hot, dois modelos comparados (Logistic Regression e Random Forest), avaliação por classification report, ROC-AUC, matriz de confusão e importância de variáveis.
6.	Síntese e exportação — resumo de insights, tabelas em CSV e modelo treinado salvo em .pkl.

Modelo escolhido: Random Forest
Maior recall na classe detrator (identifica mais clientes insatisfeitos de fato), com class_weight="balanced" para compensar o desbalanceamento (~74% detratores na base).

### 4. Como reproduzir os resultados

 • **Pré-requisitos** — Python 3 com pandas, numpy, matplotlib, seaborn, scipy, scikit-learn e joblib.
   
 • **Dados** — posicionar ***desafio_nps_fase_1.csv***
   
 • **Execução** — rodar as células em ordem sequencial (1 a 14); cada seção depende das anteriores.
    
 • **Reprodutibilidade** — garantida por random_state=42 no split e no Random Forest.
    
 • **Saída gerada** — tabelas em reports/tables/ e modelo treinado em models/nps_detractor_model.pkl.
