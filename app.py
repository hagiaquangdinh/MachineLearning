import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="App Phân Loại Khách Hàng", page_icon="🚗", layout="wide")

if not os.path.exists('models'):
    os.makedirs('models')

# --- HÀM 1: LOAD VÀ TIỀN XỬ LÝ DỮ LIỆU ---
@st.cache_data
def load_data():
    df = pd.read_csv('Data/Train.csv') 
    
    cols_all = ['Age', 'Spending_Score', 'Family_Size', 'Ever_Married', 'Profession', 'Work_Experience', 'Graduated']
    data = df[cols_all].dropna().copy()
    
    # Tách dữ liệu để huấn luyện (BỎ Spending_Score)
    data_train = data.drop(columns=['Spending_Score'])
    
    data_train['Ever_Married_Num'] = data_train['Ever_Married'].map({'No': 0, 'Yes': 1})
    data_train['Graduated_Num'] = data_train['Graduated'].map({'No': 0, 'Yes': 1})
    
    data_numeric = data_train.drop(columns=['Ever_Married', 'Graduated'])
    data_processed = pd.get_dummies(data_numeric, columns=['Profession'])
    
    return data, data_processed 

# --- HÀM 2: HUẤN LUYỆN VÀ LƯU MODEL ---
def train_and_save_model(data_processed):
    st.info("🔄 Đang huấn luyện mô hình dựa trên Nhân khẩu học...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_processed)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    model_artifacts = {
        'scaler': scaler,
        'kmeans': kmeans,
        'features': data_processed.columns.tolist() 
    }
    joblib.dump(model_artifacts, 'models/model.pkl')
    st.success("✅ Đã lưu mô hình mới vào models/model.pkl!")

# --- HÀM 3: LOAD MODEL ---
@st.cache_resource
def load_models():
    model_path = 'models/model.pkl'
    
    if not os.path.exists(model_path):
        _, data_processed = load_data()
        train_and_save_model(data_processed)
        
    artifacts = joblib.load(model_path)
    return artifacts['scaler'], artifacts['kmeans'], artifacts['features']

# ==========================================
# THỰC THI CHÍNH
# ==========================================
df_raw, df_processed = load_data()
scaler, kmeans, expected_features = load_models()

page = st.sidebar.radio("Cấu trúc ứng dụng", 
                        ["Giới thiệu & EDA", 
                         "Triển khai mô hình", 
                         "Đánh giá & Hiệu năng"])

# ==========================================
# TRANG 1: EDA
# ==========================================
if page == "Giới thiệu & EDA":
    st.title("📊 Khám phá dữ liệu (EDA)")
    
    if st.button("🔄 Cập nhật và Huấn luyện lại Model"):
        train_and_save_model(df_processed)
        st.cache_resource.clear()
        st.rerun()

    st.markdown("**Mô hình K-Means (4 Nhóm):** Phân nhóm khách hàng chỉ dựa trên đặc điểm nhân khẩu học.")
    st.divider()
    
    st.subheader("1. Một phần dữ liệu thô")
    st.dataframe(df_raw.head(10))

# ==========================================
# TRANG 2: DỰ ĐOÁN
# ==========================================
elif page == "Triển khai mô hình":
    st.title("⚙️ Dự đoán Phân khúc Khách hàng Mới")
    
    st.subheader("Nhập thông tin khách hàng")
    col1, col2 = st.columns(2)
    
    with col1:
        age_input = st.number_input("Độ tuổi:", min_value=18, max_value=100, value=30)
        family_input = st.number_input("Quy mô gia đình:", min_value=1, max_value=20, value=2)
        work_input = st.number_input("Số năm kinh nghiệm:", min_value=0, max_value=50, value=5)
        
    with col2:
        profession_list = df_raw['Profession'].dropna().unique().tolist()
        profession_input = st.selectbox("Nghề nghiệp:", profession_list)
        married_input = st.selectbox("Đã kết hôn?", ["Yes", "No"])
        graduated_input = st.selectbox("Đã tốt nghiệp ĐH?", ["Yes", "No"])

    if st.button("Phân loại Khách hàng", type="primary"):
        input_dict = {
            'Age': age_input,
            'Family_Size': family_input,
            'Work_Experience': work_input,
            'Ever_Married_Num': 1 if married_input == "Yes" else 0,
            'Graduated_Num': 1 if graduated_input == "Yes" else 0
        }
        
        input_df = pd.DataFrame([input_dict])
        
        for feature in expected_features:
            if feature.startswith('Profession_'):
                prof_name = feature.replace('Profession_', '')
                input_df[feature] = 1 if profession_input == prof_name else 0
                
        input_df = input_df[expected_features]
        
        input_scaled = scaler.transform(input_df)
        cluster_id = kmeans.predict(input_scaled)[0]
        
        # --- ÁNH XẠ ID NHÓM THÀNH TÊN NHÓM ---
        # Bạn có thể đổi tên các nhóm này cho phù hợp với dữ liệu thực tế của mình
        cluster_names = {
            0: "🧑‍🎓 Nhóm Trẻ tuổi / Độc thân (Tiềm năng phân khúc xe cỡ nhỏ, giá rẻ)",
            1: "👨‍👩‍👧‍👦 Nhóm Gia đình (Ưu tiên xe SUV/MPV 7 chỗ, rộng rãi, an toàn)",
            2: "💼 Nhóm Doanh nhân / Thu nhập cao (Tiềm năng dòng xe Sedan hạng sang)",
            3: "👴 Nhóm Trung niên / Hưu trí (Ưu tiên xe thực dụng, bền bỉ, tiết kiệm)"
        }
        
        # Lấy tên nhóm dựa trên ID dự đoán được
        persona = cluster_names.get(cluster_id, "Nhóm Khách hàng chung")
        
        st.success(f"### 🎯 Khách hàng này thuộc: {persona}")
        st.caption(f"*(Thông tin kỹ thuật: Hệ thống phân loại vào Cụm số {cluster_id})*")

# ==========================================
# TRANG 3: ĐÁNH GIÁ
# ==========================================
elif page == "Đánh giá & Hiệu năng":
    st.title("📈 Đánh giá Hiệu năng Mô hình")
    
    X_scaled = scaler.transform(df_processed)
    labels = kmeans.labels_
    
    silhouette_avg = silhouette_score(X_scaled, labels)
    st.metric(label="Chỉ số Silhouette Score", value=f"{silhouette_avg:.4f}")
    
    st.subheader("Mối liên hệ giữa Nhóm dự đoán và Mức chi tiêu")
    
    df_plot = df_raw.copy()
    
    # Thay thế ID nhóm trong biểu đồ bằng tên nhóm luôn cho đồng bộ
    cluster_short_names = {
        0: "Nhóm Trẻ tuổi",
        1: "Nhóm Gia đình",
        2: "Nhóm Doanh nhân",
        3: "Nhóm Trung niên"
    }
    df_plot['Cluster_Name'] = [cluster_short_names[label] for label in labels]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df_plot, y='Cluster_Name', hue='Spending_Score', palette='viridis', ax=ax)
    plt.title("Phân phối Mức chi tiêu theo từng Chân dung Khách hàng")
    plt.ylabel("Chân dung Khách hàng")
    plt.xlabel("Số lượng Khách hàng")
    st.pyplot(fig)