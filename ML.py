import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------
# Configuração da página
# -------------------------
st.set_page_config(page_title="Previsão de Câncer de Mama", layout="wide")
st.title("🏥 Previsão de Câncer de Mama")
st.markdown("Compare diferentes estratégias de decisão usando Machine Learning.")

# -------------------------
# Carregar dados e traduzir colunas
# -------------------------
@st.cache_data
def carregar_dados():
    dados = load_breast_cancer()
    colunas_pt = [
        "Raio_Médio", "Textura_Média", "Perímetro_Médio", "Área_Média", "Suavidade_Média",
        "Compacidade_Média", "Concavidade_Média", "Pontos_Concavidade_Média",
        "Simetria_Média", "Dimensão_Fractal_Média", "Raio_SE", "Textura_SE",
        "Perímetro_SE", "Área_SE", "Suavidade_SE", "Compacidade_SE", "Concavidade_SE",
        "Pontos_Concavidade_SE", "Simetria_SE", "Dimensão_Fractal_SE", "Raio_Maior",
        "Textura_Maior", "Perímetro_Maior", "Área_Maior", "Suavidade_Maior",
        "Compacidade_Maior", "Concavidade_Maior", "Pontos_Concavidade_Maior",
        "Simetria_Maior", "Dimensão_Fractal_Maior"
    ]
    df = pd.DataFrame(dados.data, columns=colunas_pt)
    df["Diagnóstico"] = dados.target  # 0 = maligno, 1 = benigno
    return df

df = carregar_dados()

# -------------------------
# Descrição das features
# -------------------------
descricao_features = {
    "Raio_Médio": "Média da distância do centro até a borda do tumor",
    "Textura_Média": "Desvio padrão dos tons de cinza da imagem",
    "Perímetro_Médio": "Média do perímetro do tumor",
    "Área_Média": "Média da área do tumor",
    "Suavidade_Média": "Média da suavidade da borda do tumor",
    "Compacidade_Média": "Média da compacidade (perímetro² / área - 1)",
    "Concavidade_Média": "Média da concavidade da borda do tumor",
    "Pontos_Concavidade_Média": "Número médio de pontos côncavos na borda",
    "Simetria_Média": "Média da simetria do tumor",
    "Dimensão_Fractal_Média": "Média da dimensão fractal da borda",
    "Raio_SE": "Erro padrão do raio",
    "Textura_SE": "Erro padrão da textura",
    "Perímetro_SE": "Erro padrão do perímetro",
    "Área_SE": "Erro padrão da área",
    "Suavidade_SE": "Erro padrão da suavidade",
    "Compacidade_SE": "Erro padrão da compacidade",
    "Concavidade_SE": "Erro padrão da concavidade",
    "Pontos_Concavidade_SE": "Erro padrão do número de pontos côncavos",
    "Simetria_SE": "Erro padrão da simetria",
    "Dimensão_Fractal_SE": "Erro padrão da dimensão fractal",
    "Raio_Maior": "Valor máximo do raio",
    "Textura_Maior": "Valor máximo da textura",
    "Perímetro_Maior": "Valor máximo do perímetro",
    "Área_Maior": "Valor máximo da área",
    "Suavidade_Maior": "Valor máximo da suavidade",
    "Compacidade_Maior": "Valor máximo da compacidade",
    "Concavidade_Maior": "Valor máximo da concavidade",
    "Pontos_Concavidade_Maior": "Valor máximo do número de pontos côncavos",
    "Simetria_Maior": "Valor máximo da simetria",
    "Dimensão_Fractal_Maior": "Valor máximo da dimensão fractal"
}

# -------------------------
# Visualização dinâmica dos dados
# -------------------------
st.subheader("📊 Visualização dos Dados")
st.dataframe(df, height=400)

st.subheader("📘 Descrição das Features")
descricao_df = pd.DataFrame(list(descricao_features.items()), columns=["Feature", "Descrição"])
st.dataframe(descricao_df, height=400)

# -------------------------
# Preparar dados
# -------------------------
variaveis = df.columns[:-1]
X = df[variaveis]
y = df["Diagnóstico"]
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# Random Forest robusto
# -------------------------
st.subheader("⚡ Treinamento do Modelo de Machine Learning")
st.markdown("Treinando um modelo Random Forest robusto como referência...")

pipeline_rf = Pipeline([
    ("normalizador", StandardScaler()),
    ("modelo", RandomForestClassifier(random_state=42))
])
parametros = {"modelo__n_estimators":[500], "modelo__max_depth":[None]}

with st.spinner("Treinando modelo robusto..."):
    grid = GridSearchCV(pipeline_rf, parametros, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_treino, y_treino)

melhor_rf = grid.best_estimator_
pred_rf = melhor_rf.predict(X_teste)
acuracia_rf = accuracy_score(y_teste, pred_rf)
importancias = melhor_rf.named_steps["modelo"].feature_importances_

st.success(f"Modelo treinado com acurácia: {acuracia_rf*100:.2f}%")

# -------------------------
# Estratégias do usuário
# -------------------------
st.sidebar.header("⚙️ Estratégia do Usuário")
estrategia = st.sidebar.selectbox(
    "Escolha a estratégia:",
    ["Pesos Iguais", "Regras Clínicas Simples", "Modelo Simplificado (3 grupos)"]
)

# Explicações de cada estratégia
explicacoes = {
    "Pesos Iguais": "Usa todas as variáveis igualmente, sem priorizar nenhuma. Regressão logística simples sobre todas as features.",
    "Regras Clínicas Simples": "Escolhe duas variáveis mais importantes e aplica um limiar de risco ajustável pelo usuário. Limiar controla quando considerar maligno ou benigno.",
    "Modelo Simplificado (3 grupos)": "Divide todas as features em 3 grupos (Morfologia, Textura, Outras). O usuário ajusta o peso de cada grupo, que influencia a decisão final."
}
st.sidebar.info(explicacoes[estrategia])

# Estratégia 1: Pesos Iguais
if estrategia == "Pesos Iguais":
    modelo_usuario = LogisticRegression(max_iter=1000)
    modelo_usuario.fit(X_treino, y_treino)
    pred_usuario = modelo_usuario.predict(X_teste)

# Estratégia 2: Regras Clínicas Simples
elif estrategia == "Regras Clínicas Simples":
    limiar = st.sidebar.slider("Limiar de risco", 0.0, 1.0, 0.5)
    idx_top = np.argsort(importancias)[-2:]
    X_sub = X_teste.iloc[:, idx_top]
    # Normaliza para que limiar realmente influencie
    X_norm = (X_sub - X_sub.min()) / (X_sub.max() - X_sub.min())
    score_usuario = X_norm.mean(axis=1)
    pred_usuario = (score_usuario > limiar).astype(int)

# Estratégia 3: Modelo Simplificado (3 grupos)
elif estrategia == "Modelo Simplificado (3 grupos)":
    peso_grupo1 = st.sidebar.slider("Grupo 1 (Morfologia)", 0.0, 1.0, 0.3)
    peso_grupo2 = st.sidebar.slider("Grupo 2 (Textura)", 0.0, 1.0, 0.3)
    peso_grupo3 = st.sidebar.slider("Grupo 3 (Outras)", 0.0, 1.0, 0.4)

    grupo1 = X_teste.iloc[:, :10]
    grupo2 = X_teste.iloc[:, 10:20]
    grupo3 = X_teste.iloc[:, 20:]
    grupo1_norm = (grupo1 - grupo1.min()) / (grupo1.max() - grupo1.min())
    grupo2_norm = (grupo2 - grupo2.min()) / (grupo2.max() - grupo2.min())
    grupo3_norm = (grupo3 - grupo3.min()) / (grupo3.max() - grupo3.min())
    score_usuario = (peso_grupo1*grupo1_norm.mean(axis=1) +
                     peso_grupo2*grupo2_norm.mean(axis=1) +
                     peso_grupo3*grupo3_norm.mean(axis=1))
    pred_usuario = (score_usuario > 0.5).astype(int)

# -------------------------
# Avaliar acurácia
# -------------------------
acuracia_usuario = accuracy_score(y_teste, pred_usuario)

# -------------------------
# Resultados
# -------------------------
st.subheader("📊 Comparação de Desempenho")
col1, col2 = st.columns(2)
col1.metric("🧠 Estratégia do Usuário", f"{acuracia_usuario*100:.2f}%")
col2.metric("🌳 Modelo de Machine Learning", f"{acuracia_rf*100:.2f}%")

# -------------------------
# Importância das variáveis
# -------------------------
st.subheader("📊 Variáveis Mais Importantes")
df_importancia = pd.DataFrame({
    "Variável": variaveis,
    "Importância": importancias
}).sort_values("Importância", ascending=False)
st.dataframe(df_importancia, height=400)

# -------------------------
# Mensagem final
# -------------------------
st.markdown("---")
st.info("O modelo de Machine Learning robusto garante o melhor desempenho.")
st.markdown(
    """
💡 **Insight:**  
- Estratégias simples ajudam a entender padrões  
- Machine Learning captura padrões complexos automaticamente  
- A escolha das variáveis é mais importante do que ajustar pesos arbitrários  
"""
)