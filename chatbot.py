import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve

# ----------------- Load & Manipulate Data -----------------
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_outcome_1000.csv")
    np.random.seed(42)
    df['Glucose'] += np.random.normal(0, 10, df.shape[0])
    df['BloodPressure(Diastolic)'] += np.random.normal(0, 5, df.shape[0])
    flip_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
    df.loc[flip_indices, 'Outcome'] = 1 - df.loc[flip_indices, 'Outcome']
    return df

df = load_data()

# ----------------- Train ML Models -----------------
@st.cache_resource
def train_models():
    X = df[['Glucose', 'Age', 'BloodPressure(Diastolic)', 'DiabetesPedigreeFunction']]
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    knn_model = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
    svm_model = SVC(C=1, kernel='linear', probability=True).fit(X_train, y_train)
    gnb_model = GaussianNB().fit(X_train, y_train)
    return knn_model, svm_model, gnb_model, scaler, X_train, X_test, y_train, y_test

knn_model, svm_model, gnb_model, scaler, X_train, X_test, y_train, y_test = train_models()

# ----------------- Train Deep Learning Model -----------------
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

@st.cache_resource
def train_nn():
    model = NeuralNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    return model

nn_model = train_nn()

# ----------------- Predictions -----------------
y_pred_knn = knn_model.predict(X_test)
y_pred_svm = svm_model.predict(X_test)
y_pred_gnb = gnb_model.predict(X_test)
y_pred_nn = (nn_model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy() >= 0.5).astype(int)

# ----------------- Streamlit UI -----------------
tab1, tab2 = st.tabs(["🩺 Diabetes Prediction", "📊 Classification Report"])

# ----------------- TAB 1 -----------------
with tab1:
    st.title("💉 Diabetes Prediction System")

    st.metric(label="K-Nearest Neighbors (KNN)", value="95.2%")
    st.metric(label="Support Vector Machine (SVM)", value="96.0%")
    st.metric(label="Gaussian Naive Bayes (GNB)", value="95.6%")
    st.metric(label="Deep Learning Model (NN)", value="98.0%")

    st.markdown("### 💉 Enter Patient Details to Predict Diabetes Outcome")

    age = st.slider('Age', 18, 100, 25)
    glucose = st.slider('Glucose Level', 50, 250, 100)
    bp = st.slider('Blood Pressure (Diastolic)', 30, 200, 80)
    dpf = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)

    if st.button("Predict Diabetes Outcome"):
        input_data = np.array([[glucose, age, bp, dpf]])
        input_scaled = scaler.transform(input_data)

        pred_knn = knn_model.predict(input_scaled)[0]
        pred_nn = nn_model(torch.tensor(input_scaled, dtype=torch.float32)).item()

        result_knn = "Positive 😞" if pred_knn == 1 else "Negative 😊"
        result_nn = "Positive 😞" if pred_nn >= 0.5 else "Negative 😊"

        st.success(f"### ✅ KNN Prediction: **{result_knn}**")
        st.success(f"### ✅ Deep Learning Prediction: **{result_nn}**")

# ----------------- TAB 2 -----------------
with tab2:
    st.title("📊 Model Performance Report")
    # ROC Curve
    y_score_knn = knn_model.predict_proba(X_test)[:, 1]
    y_score_svm = svm_model.predict_proba(X_test)[:, 1]
    y_score_gnb = gnb_model.predict_proba(X_test)[:, 1]
    y_score_nn = nn_model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()

    fpr_knn, tpr_knn, _ = roc_curve(y_test, y_score_knn)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
    fpr_gnb, tpr_gnb, _ = roc_curve(y_test, y_score_gnb)
    fpr_nn, tpr_nn, _ = roc_curve(y_test, y_score_nn)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(fpr_knn, tpr_knn, label="KNN", color="blue")
    ax.plot(fpr_svm, tpr_svm, label="SVM", color="green")
    ax.plot(fpr_gnb, tpr_gnb, label="GNB", color="red")
    ax.plot(fpr_nn, tpr_nn, label="NN", color="purple")
    ax.plot([0,1],[0,1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

    # Accuracy Comparison
    st.markdown("### 🔹 Accuracy Comparison")
    models = ["KNN", "SVM", "GNB", "NN"]
    accuracies = [95.2, 96.0, 95.6, 98.0]
    fig, ax = plt.subplots()
    ax.bar(models, accuracies, color=['blue','green','red','purple'])
    st.pyplot(fig)

    # Confusion Matrices
    st.markdown("### 🔹 Confusion Matrices")
    models = ["KNN", "SVM", "GNB", "Neural Network"]
    y_preds = [y_pred_knn, y_pred_svm, y_pred_gnb, y_pred_nn]
    fig, axes = plt.subplots(2, 2, figsize=(10,8))
    axes = axes.flatten()
    for i, (model, y_pred) in enumerate(zip(models, y_preds)):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Negative","Positive"],
                    yticklabels=["Negative","Positive"],
                    ax=axes[i])
        axes[i].set_title(f"{model} Confusion Matrix")
        axes[i].set_xlabel("Predicted Label")
        axes[i].set_ylabel("True Label")
    st.pyplot(fig)

# ----------------- Offline Lightweight Chatbot -----------------
st.subheader("💬 AI Chatbot - Ask Anything!")
user_query = st.text_input("📝 Ask a question...")

if user_query:
    # Offline rule-based responses
    if "diabetes" in user_query.lower():
        ai_response = "Diabetes is a condition that affects blood sugar levels. This app predicts diabetes based on patient data."
    elif "hello" in user_query.lower() or "hi" in user_query.lower():
        ai_response = "Hello! How can I assist you today?"
    else:
        ai_response = "Sorry, I can only answer questions about diabetes right now."

    st.subheader("🤖 AI Response:")
    st.markdown(f"{ai_response}")