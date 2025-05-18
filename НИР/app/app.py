import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Прогноз сердечных заболеваний")
st.markdown("Веб-приложение для классификации риска сердечно-сосудистых заболеваний.")

@st.cache_data
def preprocess_data(df):
    df = df.copy()
    
    # Кодируем категориальные признаки
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    df['ChestPainType'] = df['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
    df['RestingECG'] = df['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1})
    df['ST_Slope'] = df['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})
    
    return df

@st.cache_data
def load_data():
    raw = pd.read_csv("../data/heart.csv")
    processed = preprocess_data(raw)
    return processed


@st.cache_data
def load_scaler():
    return joblib.load("../models/scaler.pkl")

df = load_data()
scaler = load_scaler()

st.sidebar.header("Ввод параметров пациента")

def user_input_features():
    age = st.sidebar.slider("Возраст", 20, 80, 50)
    sex = st.sidebar.selectbox("Пол", ["Мужчина", "Женщина"])
    cp = st.sidebar.selectbox("Тип боли в груди", ["Тип 0", "Тип 1", "Тип 2", "Тип 3"])
    bp = st.sidebar.slider("Артериальное давление", 80, 200, 120)
    cholesterol = st.sidebar.slider("Холестерин", 100, 600, 200)
    fbs = st.sidebar.selectbox("Сахар > 120 мг/дл", ["Нет", "Да"])
    restecg = st.sidebar.selectbox("ЭКГ в покое", ["Норма", "Отклонение", "Гипертрофия"])
    maxhr = st.sidebar.slider("Макс. пульс", 60, 220, 150)
    exang = st.sidebar.selectbox("Стенокардия при нагрузке", ["Нет", "Да"])
    oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0, step=0.1)
    slope = st.sidebar.selectbox("Наклон ST", ["Плавный", "Вверх", "Вниз"])

    data = {
        "Age": age,
        "Sex": 1 if sex == "Мужчина" else 0,
        "ChestPainType": int(cp.split()[-1]),
        "RestingBP": bp,
        "Cholesterol": cholesterol,
        "FastingBS": 1 if fbs == "Да" else 0,
        "RestingECG": restecg_map(restecg),
        "MaxHR": maxhr,
        "ExerciseAngina": 1 if exang == "Да" else 0,
        "Oldpeak": oldpeak,
        "ST_Slope": slope_map(slope),
    }
    return pd.DataFrame([data])

def restecg_map(value):
    return {"Норма": 0, "Отклонение": 1, "Гипертрофия": 2}[value]

def slope_map(value):
    return {"Плавный": 0, "Вверх": 1, "Вниз": 2}[value]

input_df = user_input_features()

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_scaled = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

n_estimators = st.slider("Количество деревьев в Random Forest", 10, 200, 100, step=10)

model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)

y_pred = model.predict(X_test)

st.subheader("📈 Метрики модели")
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.write(f"**Accuracy:** {acc:.2f}")
st.write(f"**Precision:** {prec:.2f}")
st.write(f"**Recall:** {rec:.2f}")
st.write(f"**F1-мера:** {f1:.2f}")

result_text = "🔴 Высокий риск" if prediction[0] == 1 else "🟢 Низкий риск"
st.subheader("🧠 Прогноз:")
st.write(result_text)

st.subheader("📌 Важность признаков")
importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({"Признак": features, "Важность": importances}).sort_values("Важность", ascending=False)

fig, ax = plt.subplots()
sns.barplot(x="Важность", y="Признак", data=importance_df, ax=ax)
st.pyplot(fig)

st.markdown("---")
st.markdown("**Вывод:**")
st.markdown("""
- Модель обучается заново при изменении параметра `n_estimators`
- Самые важные признаки: `Oldpeak`, `ST_Slope`, `ExerciseAngina`
- Можно использовать для оценки риска у новых пациентов
""")
