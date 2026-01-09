import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification, make_moons, make_circles

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ML Algorithm Playground", layout="wide")
st.title("🤖 Interactive ML Algorithm Playground")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📊 Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

# ---------------- DATA LOADING ----------------
def load_data():
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.subheader("Dataset Preview")
        st.sidebar.write(df.head())

        target_col = st.sidebar.selectbox("Select Target Column", df.columns)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    else:
        dataset_type = st.sidebar.selectbox(
            "Synthetic Dataset",
            ["Classification", "Moons", "Circles"]
        )

        if dataset_type == "Classification":
            X, y = make_classification(
                n_samples=800,
                n_features=2,
                n_informative=2,
                n_redundant=0,
                random_state=42
            )
        elif dataset_type == "Moons":
            X, y = make_moons(n_samples=800, noise=0.3, random_state=42)
        else:
            X, y = make_circles(n_samples=800, noise=0.2, factor=0.5)

        X = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
        y = pd.Series(y, name="Target")

        categorical_cols = []
        numerical_cols = X.columns.tolist()

    return X, y, categorical_cols, numerical_cols

X, y, categorical_cols, numerical_cols = load_data()

# ---------------- MODEL SELECTION ----------------
classifier_name = st.sidebar.selectbox(
    "Select Algorithm",
    [
        "Logistic Regression",
        "KNN",
        "SVM",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "Naive Bayes"
    ]
)

# ---------------- HYPERPARAMETERS ----------------
def get_model(name):
    if name == "Logistic Regression":
        C = st.sidebar.slider("C", 0.01, 10.0, 1.0)
        solver = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear"])
        return LogisticRegression(C=C, solver=solver, max_iter=1000)

    elif name == "KNN":
        k = st.sidebar.slider("Neighbors", 1, 30, 5)
        weights = st.sidebar.selectbox("Weights", ["uniform", "distance"])
        metric = st.sidebar.selectbox("Metric", ["euclidean", "manhattan"])
        return KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)

    elif name == "SVM":
        C = st.sidebar.slider("C", 0.1, 20.0, 1.0)
        kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])
        gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])
        degree = st.sidebar.slider("Degree (poly)", 2, 5, 3)
        return SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, probability=True)

    elif name == "Decision Tree":
        depth = st.sidebar.slider("Max Depth", 1, 30, 5)
        split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
        leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, 1)
        return DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=split,
            min_samples_leaf=leaf
        )

    elif name == "Random Forest":
        trees = st.sidebar.slider("Trees", 50, 500, 100)
        depth = st.sidebar.slider("Max Depth", 2, 30, 10)
        max_feat = st.sidebar.selectbox("Max Features", ["sqrt", "log2", None])
        return RandomForestClassifier(
            n_estimators=trees,
            max_depth=depth,
            max_features=max_feat
        )

    elif name == "Gradient Boosting":
        lr = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
        trees = st.sidebar.slider("Estimators", 50, 500, 100)
        subsample = st.sidebar.slider("Subsample", 0.5, 1.0, 1.0)
        return GradientBoostingClassifier(
            learning_rate=lr,
            n_estimators=trees,
            subsample=subsample
        )

    else:
        smooth = st.sidebar.slider("Var Smoothing", 1e-12, 1e-7, 1e-9, format="%.1e")
        return GaussianNB(var_smoothing=smooth)

base_model = get_model(classifier_name)

# ---------------- PREPROCESSING ----------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", base_model)
])

# ---------------- TRAIN / TEST ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# ---------------- RESULTS ----------------
st.subheader("📈 Model Performance")
st.metric("Accuracy", f"{accuracy:.4f}")

col1, col2 = st.columns(2)

# Confusion Matrix
with col1:
    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

# Prediction Probability
with col2:
    if hasattr(model.named_steps["classifier"], "predict_proba"):
        st.markdown("### Prediction Probability Distribution")
        probs = model.predict_proba(X_test)
        figp, axp = plt.subplots(figsize=(4, 3))
        axp.hist(probs[:, 1], bins=20)
        axp.set_xlabel("Probability")
        axp.set_ylabel("Frequency")
        st.pyplot(figp)

# ---------------- PCA VISUALIZATION ----------------
st.subheader("📉 PCA Projection (2D)")

X_transformed = model.named_steps["preprocessing"].transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_transformed)

fig_pca, ax_pca = plt.subplots(figsize=(5, 4))
scatter = ax_pca.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="viridis",
    s=15
)
ax_pca.set_xlabel("PCA Component 1")
ax_pca.set_ylabel("PCA Component 2")
st.pyplot(fig_pca)

# ---------------- FEATURE IMPORTANCE ----------------
if hasattr(model.named_steps["classifier"], "feature_importances_"):
    st.subheader("🌲 Feature Importance")
    importances = model.named_steps["classifier"].feature_importances_
    fig_imp, ax_imp = plt.subplots(figsize=(6, 3))
    ax_imp.bar(range(len(importances)), importances)
    ax_imp.set_ylabel("Importance")
    st.pyplot(fig_imp)

# ---------------- CORRELATION HEATMAP ----------------
if len(numerical_cols) > 1:
    st.subheader("🔗 Feature Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
    sns.heatmap(X[numerical_cols].corr(), cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)
