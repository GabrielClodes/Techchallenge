# -*- coding: utf-8 -*-
"""Data Scientist-TechChallenge.ipynb"""

# ============================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mannwhitneyu, spearmanr

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import joblib

# Configurações visuais
sns.set_theme(
    style="whitegrid",
    palette="Set2"
)

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


# ============================================================
# 2. CARREGANDO OS DADOS
# ============================================================

df = pd.read_csv("/content/desafio_nps_fase_1.csv")

print(f"Linhas: {df.shape[0]:,}")
print(f"Colunas: {df.shape[1]}")

# ============================================================
# 3. VISÃO GERAL
# ============================================================

display(df.head())

# A linha df["nps_class"].value_counts() foi removida daqui, 
# pois a coluna só é criada na Seção 7.

df.info()

display(df.describe(include="all").T)

# ============================================================
# 4. QUALIDADE DOS DADOS
# ============================================================

missing = (
    df.isnull()
      .sum()
      .to_frame("missing_count")
)

missing["missing_pct"] = (
    missing["missing_count"] / len(df) * 100
)

missing = missing.sort_values(
    "missing_pct",
    ascending=False
)

display(missing)

duplicated_rows = df.duplicated().sum()

print(
    f"Quantidade de linhas duplicadas: {duplicated_rows:,}"
)

print(
    f"Clientes únicos: {df['customer_id'].nunique():,}"
)

print(
    f"Pedidos únicos: {df['order_id'].nunique():,}"
)

print(
    f"Linhas da base: {len(df):,}"
)

orders_per_customer = (
    df.groupby("customer_id")["order_id"]
      .nunique()
)

print(
    f"Média de pedidos por cliente: "
    f"{orders_per_customer.mean():.2f}"
)

# ============================================================
# 5. VALIDAÇÕES
# ============================================================

print(
    "NPS fora do intervalo 0-10:",
    (~df["nps_score"].between(0, 10)).sum()
)

print(
    "Idades inválidas:",
    (df["customer_age"] <= 0).sum()
)

print(
    "Pedidos com valor negativo:",
    (df["order_value"] < 0).sum()
)

print(
    "Tempo de entrega negativo:",
    (df["delivery_time_days"] < 0).sum()
)

print(
    "Atraso negativo:",
    (df["delivery_delay_days"] < 0).sum()
)

print(
    "Quantidade de itens <= 0:",
    (df["items_quantity"] <= 0).sum()
)

# ============================================================
# 6. TRATAMENTO DOS DADOS
# ============================================================

df_clean = df.copy()

# Remover duplicidades exatas
df_clean = df_clean.drop_duplicates()

# Garantir tipos numéricos
numeric_columns = [
    "customer_age",
    "customer_tenure_months",
    "order_value",
    "items_quantity",
    "discount_value",
    "payment_installments",
    "delivery_time_days",
    "delivery_delay_days",
    "freight_value",
    "delivery_attempts",
    "customer_service_contacts",
    "resolution_time_days",
    "complaints_count",
    "repeat_purchase_30d",
    "csat_internal_score",
    "nps_score"
]

for col in numeric_columns:
    df_clean[col] = pd.to_numeric(
        df_clean[col],
        errors="coerce"
    )

# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

# Classificação tradicional do NPS
def classify_nps(score):
    if pd.isna(score):
        return np.nan
    elif score <= 6:
        return "Detrator"
    elif score <= 8:
        return "Neutro"
    else:
        return "Promotor"


df_clean["nps_class"] = (
    df_clean["nps_score"]
    .apply(classify_nps)
)

df_clean["has_delivery_delay"] = (
    df_clean["delivery_delay_days"] > 0
).astype(int)

df_clean["contacted_customer_service"] = (
    df_clean["customer_service_contacts"] > 0
).astype(int)

df_clean["has_complaint"] = (
    df_clean["complaints_count"] > 0
).astype(int)

df_clean["multiple_delivery_attempts"] = (
    df_clean["delivery_attempts"] > 1
).astype(int)

df_clean["delivery_time_group"] = pd.cut(
    df_clean["delivery_time_days"],
    bins=[-np.inf, 2, 5, 10, np.inf],
    labels=[
        "Até 2 dias",
        "3-5 dias",
        "6-10 dias",
        "Mais de 10 dias"
    ]
)

df_clean["age_group"] = pd.cut(
    df_clean["customer_age"],
    bins=[0, 25, 35, 45, 55, np.inf],
    labels=[
        "Até 25",
        "26-35",
        "36-45",
        "46-55",
        "56+"
    ]
)

# ============================================================
# 8. NPS GERAL
# ============================================================

nps_valid = df_clean.dropna(
    subset=["nps_score"]
)

promoters_pct = (
    (nps_valid["nps_score"] >= 9).mean()
)

detractors_pct = (
    (nps_valid["nps_score"] <= 6).mean()
)

nps = (
    promoters_pct - detractors_pct
) * 100

print(f"NPS geral: {nps:.2f}")
print(f"Promotores: {promoters_pct * 100:.2f}%")
print(f"Detratores: {detractors_pct * 100:.2f}%")
print(
    f"Neutros: "
    f"{((nps_valid['nps_score'].between(7, 8)).mean() * 100):.2f}%"
)

# ============================================================
# 9. DISTRIBUIÇÃO DO NPS
# ============================================================

plt.figure(figsize=(10, 5))

sns.countplot(
    data=nps_valid,
    x="nps_score",
    color="#4472C4"
)

plt.title("Distribuição das notas de NPS")
plt.xlabel("Nota NPS")
plt.ylabel("Quantidade de clientes")

plt.show()

nps_distribution = (
    df_clean["nps_class"]
    .value_counts(normalize=True)
    .mul(100)
    .reindex([
        "Detrator",
        "Neutro",
        "Promotor"
    ])
)

display(
    nps_distribution.to_frame(
        "percentual"
    )
)

plt.figure(figsize=(8, 5))

sns.barplot(
    x=nps_distribution.index,
    y=nps_distribution.values
)

plt.title("Perfil dos clientes por classificação de NPS")
plt.xlabel("")
plt.ylabel("% dos clientes")

plt.show()

# ============================================================
# 10. NPS x ATRASO
# ============================================================

nps_delay = (
    df_clean
    .groupby("has_delivery_delay")
    .agg(
        nps_medio=("nps_score", "mean"),
        quantidade=("nps_score", "count")
    )
)

nps_delay.index = [
    "Sem atraso",
    "Com atraso"
]

display(nps_delay)

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    data=nps_delay.reset_index(),
    x="index",
    y="nps_medio",
    hue="index",
    palette={
        "Sem atraso": "#2ECC71",
        "Com atraso": "#E74C3C"
    },
    legend=False
)

# Adiciona o valor do NPS em cima das barras
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f",
        padding=3,
        fontsize=11,
        fontweight="bold"
    )

plt.title(
    "NPS médio por ocorrência de atraso",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("")
plt.ylabel("NPS médio")

plt.ylim(0, 8)

plt.tight_layout()
plt.show()

delay_group = df_clean.loc[
    df_clean["has_delivery_delay"] == 1,
    "nps_score"
].dropna()

no_delay_group = df_clean.loc[
    df_clean["has_delivery_delay"] == 0,
    "nps_score"
].dropna()

stat, p_value = mannwhitneyu(
    delay_group,
    no_delay_group,
    alternative="two-sided"
)

print(f"Estatística: {stat:.2f}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.05:
    print(
        "INSIGHT: existe evidência estatística de diferença "
        "no NPS entre clientes com e sem atraso."
    )
else:
    print(
        "INSIGHT: não foi encontrada evidência estatística "
        "suficiente de diferença no NPS entre os grupos."
    )

delivery_analysis = (
    df_clean
    .groupby("delivery_time_group", observed=True)
    .agg(
        nps_medio=("nps_score", "mean"),
        clientes=("nps_score", "count")
    )
    .reset_index()
)

display(delivery_analysis)

plt.figure(figsize=(10, 5))

sns.barplot(
    data=delivery_analysis,
    x="delivery_time_group",
    y="nps_medio"
)

plt.title("NPS médio por faixa de tempo de entrega")
plt.xlabel("Tempo de entrega")
plt.ylabel("NPS médio")

plt.xticks(rotation=20)

plt.show()

service_analysis = (
    df_clean
    .groupby("contacted_customer_service")
    .agg(
        nps_medio=("nps_score", "mean"),
        clientes=("nps_score", "count")
    )
)

service_analysis.index = [
    "Não entrou em contato",
    "Entrou em contato"
]

display(service_analysis)

complaint_analysis = (
    df_clean
    .groupby("has_complaint")
    .agg(
        nps_medio=("nps_score", "mean"),
        clientes=("nps_score", "count")
    )
)

complaint_analysis.index = [
    "Sem reclamação",
    "Com reclamação"
]

display(complaint_analysis)

resolution_corr = df_clean[
    [
        "resolution_time_days",
        "nps_score"
    ]
].corr(
    method="spearman"
)

print(
    "Correlação de Spearman:",
    resolution_corr.loc[
        "resolution_time_days",
        "nps_score"
    ]
)

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df_clean,
    x="resolution_time_days",
    y="nps_score",
    alpha=0.4
)

plt.title(
    "Relação entre tempo de resolução e NPS"
)

plt.xlabel("Tempo de resolução (dias)")
plt.ylabel("NPS")

plt.show()

csat_corr = df_clean[
    [
        "csat_internal_score",
        "nps_score"
    ]
].corr(
    method="spearman"
)

print(
    "Correlação CSAT x NPS:",
    csat_corr.loc[
        "csat_internal_score",
        "nps_score"
    ]
)

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df_clean,
    x="csat_internal_score",
    y="nps_score",
    alpha=0.4
)

plt.title("Relação entre CSAT e NPS")
plt.xlabel("CSAT")
plt.ylabel("NPS")

plt.show()

region_analysis = (
    df_clean
    .groupby("customer_region")
    .agg(
        nps_medio=("nps_score", "mean"),
        clientes=("nps_score", "count")
    )
    .sort_values(
        "nps_medio",
        ascending=False
    )
)

display(region_analysis)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=region_analysis.reset_index(),
    x="nps_medio",
    y="customer_region"
)

plt.title("NPS médio por região")
plt.xlabel("NPS médio")
plt.ylabel("Região")

plt.show()

repurchase_analysis = (
    df_clean
    .groupby("nps_class")
    .agg(
        recompra_30d=("repeat_purchase_30d", "mean"),
        clientes=("customer_id", "count")
    )
    .reindex([
        "Detrator",
        "Neutro",
        "Promotor"
    ])
)

repurchase_analysis["recompra_30d_pct"] = (
    repurchase_analysis["recompra_30d"] * 100
)

display(repurchase_analysis)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=repurchase_analysis.reset_index(),
    x="nps_class",
    y="recompra_30d_pct"
)

plt.title(
    "Taxa de recompra em 30 dias por classificação de NPS"
)

plt.xlabel("")
plt.ylabel("Recompra em 30 dias (%)")

plt.show()

# ============================================================
# 11. COMPARAÇÃO DOS DRIVERS
# ============================================================

driver_columns = [
    "delivery_time_days",
    "delivery_delay_days",
    "delivery_attempts",
    "customer_service_contacts",
    "resolution_time_days",
    "complaints_count",
    "order_value",
    "customer_tenure_months",
    "csat_internal_score"
]

driver_comparison = (
    df_clean
    .groupby("nps_class")[driver_columns]
    .mean()
    .T
)

driver_comparison = driver_comparison[
    [
        "Detrator",
        "Neutro",
        "Promotor"
    ]
]

display(driver_comparison)

correlation_columns = [
    "customer_age",
    "customer_tenure_months",
    "order_value",
    "items_quantity",
    "discount_value",
    "payment_installments",
    "delivery_time_days",
    "delivery_delay_days",
    "freight_value",
    "delivery_attempts",
    "customer_service_contacts",
    "resolution_time_days",
    "complaints_count",
    "csat_internal_score",
    "nps_score"
]

correlation_matrix = (
    df_clean[correlation_columns]
    .corr(method="spearman")
)

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0
)

plt.title("Correlação entre variáveis")

plt.show()

# ============================================================
# 12. MODELO PREDITIVO
# ============================================================

model_df = df_clean.dropna(
    subset=["nps_score"]
).copy()

model_df["is_detractor"] = (
    model_df["nps_score"] <= 6
).astype(int)

features = [
    "customer_age",
    "customer_region",
    "customer_tenure_months",
    "order_value",
    "items_quantity",
    "discount_value",
    "payment_installments",
    "delivery_time_days",
    "delivery_delay_days",
    "freight_value",
    "delivery_attempts",
    "customer_service_contacts",
    "resolution_time_days",
    "complaints_count",
    "csat_internal_score"
]

target = "is_detractor"

X = model_df[features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Treino:", X_train.shape)
print("Teste:", X_test.shape)

categorical_features = [
    "customer_region"
]

numeric_features = [
    col
    for col in features
    if col not in categorical_features
]

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)

logistic_model.fit(
    X_train,
    y_train
)

y_pred = logistic_model.predict(X_test)

y_prob = logistic_model.predict_proba(
    X_test
)[:, 1]

print(
    classification_report(
        y_test,
        y_pred
    )
)

print(
    f"ROC-AUC: "
    f"{roc_auc_score(y_test, y_prob):.3f}"
)

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

random_forest_model.fit(
    X_train,
    y_train
)

y_pred_rf = random_forest_model.predict(
    X_test
)

y_prob_rf = random_forest_model.predict_proba(
    X_test
)[:, 1]

print(
    classification_report(
        y_test,
        y_pred_rf
    )
)

print(
    f"ROC-AUC: "
    f"{roc_auc_score(y_test, y_prob_rf):.3f}"
)

model_comparison = pd.DataFrame({
    "Modelo": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y_pred_rf)
    ],
    "Precision": [
        precision_score(y_test, y_pred),
        precision_score(y_test, y_pred_rf)
    ],
    "Recall": [
        recall_score(y_test, y_pred),
        recall_score(y_test, y_pred_rf)
    ],
    "F1": [
        f1_score(y_test, y_pred),
        f1_score(y_test, y_pred_rf)
    ],
    "ROC_AUC": [
        roc_auc_score(y_test, y_prob),
        roc_auc_score(y_test, y_prob_rf)
    ]
})

display(
    model_comparison.sort_values(
        "ROC_AUC",
        ascending=False
    )
)

best_model = random_forest_model

best_predictions = best_model.predict(
    X_test
)

cm = confusion_matrix(
    y_test,
    best_predictions
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Matriz de confusão")
plt.xlabel("Predito")
plt.ylabel("Real")

plt.show()

rf_pipeline = random_forest_model

preprocessor_fitted = (
    rf_pipeline
    .named_steps["preprocessor"]
)

rf_fitted = (
    rf_pipeline
    .named_steps["model"]
)

feature_names = (
    preprocessor_fitted
    .get_feature_names_out()
)

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_fitted.feature_importances_
})

feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
)

display(
    feature_importance.head(15)
)

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="importance",
    y="feature"
)

plt.title(
    "Principais fatores utilizados pelo modelo"
)

plt.xlabel("Importância")
plt.ylabel("Variável")

plt.show()

# ============================================================
# 13. RESUMO DOS INSIGHTS
# ============================================================

print("=" * 70)
print("RESUMO DOS PRINCIPAIS INSIGHTS")
print("=" * 70)

print(
    f"\n1. NPS geral: {nps:.1f}"
)

delay_nps_without = nps_delay.loc[
    "Sem atraso",
    "nps_medio"
]

delay_nps_with = nps_delay.loc[
    "Com atraso",
    "nps_medio"
]

difference = (
    delay_nps_without -
    delay_nps_with
)

print(
    f"\n2. Impacto do atraso:"
)

print(
    f"   NPS sem atraso: {delay_nps_without:.2f}"
)

print(
    f"   NPS com atraso: {delay_nps_with:.2f}"
)

print(
    f"   Diferença: {difference:.2f} pontos"
)

promoter_repurchase = (
    repurchase_analysis
    .loc["Promotor", "recompra_30d_pct"]
)

detractor_repurchase = (
    repurchase_analysis
    .loc["Detrator", "recompra_30d_pct"]
)

print(
    f"\n3. Recompra:"
)

print(
    f"   Promotores: {promoter_repurchase:.2f}%"
)

print(
    f"   Detratores: {detractor_repurchase:.2f}%"
)

print(
    f"   Diferença: "
    f"{promoter_repurchase - detractor_repurchase:.2f} p.p."
)

print("\n4. RECOMENDAÇÕES INICIAIS")
print("-" * 50)

if delay_nps_with < delay_nps_without:
    print(
        "• Priorizar redução de atrasos logísticos, "
        "pois clientes com atraso apresentam menor NPS."
    )

if promoter_repurchase > detractor_repurchase:
    print(
        "• Tratar satisfação como indicador estratégico, "
        "pois clientes promotores apresentam maior taxa "
        "de recompra."
    )

if (
    df_clean["has_complaint"].mean() > 0
):
    print(
        "• Monitorar clientes com reclamações e priorizar "
        "resolução rápida dos problemas."
    )

# ============================================================
# 14. EXPORTAÇÃO DOS RESULTADOS
# ============================================================

import os

os.makedirs(
    "reports/tables",
    exist_ok=True
)

os.makedirs(
    "reports/figures",
    exist_ok=True
)

driver_comparison.to_csv(
    "reports/tables/nps_drivers.csv"
)

region_analysis.to_csv(
    "reports/tables/nps_by_region.csv"
)

repurchase_analysis.to_csv(
    "reports/tables/nps_repurchase.csv"
)

model_comparison.to_csv(
    "reports/tables/model_comparison.csv",
    index=False
)

feature_importance.to_csv(
    "reports/tables/feature_importance.csv",
    index=False
)

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    random_forest_model,
    "models/nps_detractor_model.pkl"
)
