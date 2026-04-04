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

# Tạo thư mục models nếu chưa tồn tại
if not os.path.exists('models'):
    os.makedirs('models')

# --- HÀM 1: LOAD VÀ TIỀN XỬ LÝ DỮ LIỆU ---
@st.cache_data
def load_data():
    # Đọc dữ liệu
    df = pd.read_csv('Data/Train.csv') 
    
    # Lấy 7 cột quan trọng nhất
    cols = ['Age', 'Spending_Score', 'Family_Size', 'Ever_Married', 'Profession', 'Work_Experience', 'Graduated']
    data = df[cols].dropna().copy()
    
    # 1. Mã hóa các biến Ordinal và Binary
    data['Spending_Score_Num'] = data['Spending_Score'].map({'Low': 1, 'Average': 2, 'High': 3})
    data['Ever_Married_Num'] = data['Ever_Married'].map({'No': 0, 'Yes': 1})
    data['Graduated_Num'] = data['Graduated'].map({'No': 0, 'Yes': 1})
    
    # Bỏ các cột chữ gốc sau khi đã chuyển thành số
    data_numeric = data.drop(columns=['Spending_Score', 'Ever_Married', 'Graduated'])
    
    # 2. Mã hóa One-Hot cho cột Profession
    data_processed = pd.get_dummies(data_numeric, columns=['Profession'])
    
    return df, data_processed

# --- HÀM 2: HUẤN LUYỆN VÀ LƯU MODEL ---
def train_and_save_model(data_processed):
    st.info("🔄 Đang huấn luyện mô hình đa chiều (7 tiêu chí)...")
    
    # 1. Chuẩn hóa
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_processed)
    
    # 2. Huấn luyện K-Means (4 cụm)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # 3. Đóng gói và lưu model
    model_artifacts = {
        'scaler': scaler,
        'kmeans': kmeans,
        'features': data_processed.columns.tolist() 
    }
    joblib.dump(model_artifacts, 'models/model.pkl')
    st.success("✅ Đã lưu mô hình vào models/model.pkl!")

# --- HÀM 3: LOAD MODEL ---
@st.cache_resource
def load_models():
    model_path = 'models/model.pkl'
    
    # Nếu chưa có file thì gọi hàm huấn luyện ngay
    if not os.path.exists(model_path):
        _, data_processed = load_data()
        train_and_save_model(data_processed)
        
    # Đọc file .pkl
    artifacts = joblib.load(model_path)
    return artifacts['scaler'], artifacts['kmeans'], artifacts['features']

# ==========================================
# THỰC THI CHÍNH
# ==========================================
df_raw, df_processed = load_data()
scaler, kmeans, expected_features = load_models()

# Điều hướng các trang
page = st.sidebar.radio("Cấu trúc ứng dụng", 
                        ["Giới thiệu & EDA", 
                         "Triển khai mô hình", 
                         "Đánh giá & Hiệu năng"])

# ==========================================
# TRANG 1: EDA
# ==========================================
if page == "Giới thiệu & EDA":
    st.title("📊 Khám phá dữ liệu (EDA)")
    st.markdown("""
    **Tên đề tài:** Phân loại khách hàng Ô tô dựa trên Đa đặc trưng  
    **Giá trị thực tiễn:** Tối ưu ngân sách quảng cáo và cá nhân hóa thông điệp bán xe.
    """)
    st.markdown("**Mô hình hiện tại đã được nâng cấp đánh giá đa chiều với 7 đặc trưng.**")
    st.divider()
    
    st.subheader("1. Một phần dữ liệu thô")
    st.dataframe(df_raw[['Age', 'Spending_Score', 'Family_Size', 'Ever_Married', 'Profession', 'Work_Experience', 'Graduated']].head(10))

    st.subheader("2. Tỷ lệ nhóm Nghề nghiệp trong dữ liệu")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(y='Profession', data=df_raw, order=df_raw['Profession'].value_counts().index, palette='Set2', ax=ax)
    st.pyplot(fig)

# ==========================================
# TRANG 2: DỰ ĐOÁN
# ==========================================
elif page == "Triển khai mô hình":
    st.title("⚙️ Triển khai mô hình K-Means")
    
    st.subheader("Nhập thông tin khách hàng mới")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age_input = st.number_input("Độ tuổi:", min_value=18, max_value=100, value=30)
        family_input = st.number_input("Quy mô gia đình:", min_value=1, max_value=20, value=2)
        work_input = st.number_input("Số năm kinh nghiệm:", min_value=0, max_value=50, value=5)
        
    with col2:
        spending_input = st.selectbox("Mức chi tiêu:", ["Low", "Average", "High"])
        married_input = st.selectbox("Đã kết hôn?", ["Yes", "No"])
        graduated_input = st.selectbox("Đã tốt nghiệp ĐH?", ["Yes", "No"])
        
    with col3:
        profession_list = df_raw['Profession'].dropna().unique().tolist()
        profession_input = st.selectbox("Nghề nghiệp:", profession_list)

    if st.button("Dự đoán Phân khúc", type="primary"):
        # 1. Khởi tạo Dictionary map với dữ liệu nhập
        input_dict = {
            'Age': age_input,
            'Family_Size': family_input,
            'Work_Experience': work_input,
            'Spending_Score_Num': {'Low': 1, 'Average': 2, 'High': 3}[spending_input],
            'Ever_Married_Num': 1 if married_input == "Yes" else 0,
            'Graduated_Num': 1 if graduated_input == "Yes" else 0
        }
        
        # 2. Chuyển thành DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # 3. Tạo One-Hot Encoding cho Profession khớp với lúc Train
        for feature in expected_features:
            if feature.startswith('Profession_'):
                prof_name = feature.replace('Profession_', '')
                input_df[feature] = 1 if profession_input == prof_name else 0
                
        # 4. Ép thứ tự cột chuẩn xác 100%
        input_df = input_df[expected_features]
        
        # 5. Transform và Predict
        input_scaled = scaler.transform(input_df)
        cluster_id = kmeans.predict(input_scaled)[0]
        
        # 6. In kết quả
        st.success(f"### Khách hàng này thuộc: Nhóm {cluster_id}")
        st.caption("Do dữ liệu phân nhóm đa chiều, mỗi nhóm sẽ đại diện cho một cụm tổng hòa các đặc trưng về gia đình, thu nhập và nghề nghiệp.")

# ==========================================
# TRANG 3: ĐÁNH GIÁ
# ==========================================
elif page == "Đánh giá & Hiệu năng":
    st.title("📈 Đánh giá Hiệu năng Mô hình")
    
    # Đánh giá trên toàn bộ tập dữ liệu đã qua xử lý
    X_scaled = scaler.transform(df_processed)
    labels = kmeans.labels_
    
    silhouette_avg = silhouette_score(X_scaled, labels)
    st.metric(label="Chỉ số Silhouette Score (Trên không gian 7 chiều)", value=f"{silhouette_avg:.4f}")
    
    st.subheader("Phân tán 2D (Age vs Spending) trên mô hình Đa chiều")
    st.caption("*Lưu ý: Biểu đồ này chỉ hiển thị 2 trong số 7 chiều dữ liệu thực tế để dễ trực quan hóa.*")
    
    # Lấy lại DataFrame raw để vẽ kèm theo nhãn Cluster
    df_plot = df_raw.dropna(subset=['Age', 'Spending_Score', 'Family_Size', 'Ever_Married', 'Profession', 'Work_Experience', 'Graduated']).copy()
    df_plot['Cluster'] = labels
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x='Age', y='Spending_Score', hue='Cluster', palette='viridis', data=df_plot, ax=ax)
    plt.title("Kết quả phân nhóm K-Means")
    st.pyplot(fig)