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

# Cấu hình trang
st.set_page_config(page_title="App Phân Loại Khách Hàng", page_icon="🚗", layout="wide")

# Tạo thư mục models nếu chưa tồn tại
if not os.path.exists('models'):
    os.makedirs('models')

# --- HÀM 1: LOAD DỮ LIỆU ---
@st.cache_data
def load_data():
    df = pd.read_csv('Data/Train.csv') 
    
    # Lấy thêm các cột mới
    data = df[['Age', 'Spending_Score', 'Income', 'Gender', 'Profession']].dropna().copy()
    
    # 1. Xử lý Spending_Score (như cũ)
    score_mapping = {'Low': 1, 'Average': 2, 'High': 3}
    data['Spending_Score_Num'] = data['Spending_Score'].map(score_mapping)
    
    # 2. Xử lý Gender (Giới tính) thành 0 và 1
    data['Gender_Num'] = data['Gender'].map({'Male': 0, 'Female': 1})
    
    # 3. Xử lý Profession (Nghề nghiệp) bằng One-Hot Encoding
    # pd.get_dummies sẽ biến mỗi nghề nghiệp thành 1 cột riêng biệt (0 hoặc 1)
    data = pd.get_dummies(data, columns=['Profession'], drop_first=True)
    
    return df, data

# --- HÀM 2: HUẤN LUYỆN VÀ LƯU MODEL (CHỈ CHẠY KHI THIẾU FILE) ---
def train_and_save_model(data_processed):
    st.info("🔄 Đang huấn luyện mô hình với nhiều tiêu chí...")
    
    # Chọn TẤT CẢ các cột số để đưa vào K-Means
    # Lưu ý: Bỏ qua các cột chữ gốc (như Spending_Score, Gender)
    feature_columns = [col for col in data_processed.columns if col not in ['Age', 'Spending_Score', 'Gender']]
    # Ví dụ feature_columns sẽ gồm: Age, Spending_Score_Num, Income, Gender_Num, Profession_Doctor...
    
    X = data_processed[feature_columns] # Lưu lại tên các cột để sau này dùng
    
    # 1. Chuẩn hóa toàn bộ dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Huấn luyện (Giả sử tăng lên 4 nhóm cho chi tiết hơn)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # 3. ĐÓNG GÓI THÊM TÊN CỘT (Rất quan trọng)
    model_artifacts = {
        'scaler': scaler,
        'kmeans': kmeans,
        'features': X.columns.tolist() # Lưu lại danh sách các cột để lúc dự đoán nhập cho đúng thứ tự
    }
    joblib.dump(model_artifacts, 'models/model.pkl')
    st.success("✅ Đã lưu mô hình mới!")

# --- HÀM 3: LOAD MODEL TỪ FILE .PKL ---
@st.cache_resource
def load_models():
    model_path = 'models/model.pkl'
    
    # Nếu chưa có file thì gọi hàm huấn luyện ngay
    if not os.path.exists(model_path):
        _, data_processed = load_data()
        train_and_save_model(data_processed)
        
    # Đọc file .pkl
    artifacts = joblib.load(model_path)
    return artifacts['scaler'], artifacts['kmeans']

# ==========================================
# THỰC THI CHÍNH
# ==========================================
df_raw, df_processed = load_data()
# Lấy scaler và kmeans từ file .pkl (hoặc huấn luyện mới nếu chưa có)
scaler, kmeans = load_models()

# Điều hướng các trang
page = st.sidebar.radio("Cấu trúc ứng dụng", 
                        ["Giới thiệu & EDA", 
                         "Triển khai mô hình", 
                         "Đánh giá & Hiệu năng"])

if page == "Giới thiệu & EDA":
    st.title("📊 Khám phá dữ liệu (EDA)")
    st.markdown("""
    **Tên đề tài:** Phân loại khách hàng Ô tô dựa trên Độ tuổi và Mức chi tiêu  
    **Giá trị thực tiễn:** Tối ưu ngân sách quảng cáo và cá nhân hóa thông điệp bán xe.
    """)
    st.divider()
    st.subheader("1. Một phần dữ liệu thô")
    st.dataframe(df_raw.head(10))

    st.subheader("2. Biểu đồ phân tích đặc trưng")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        sns.histplot(df_raw['Age'], bins=20, kde=True, color='skyblue', ax=ax1)
        ax1.set_title("Phân phối Độ tuổi (Age)")
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots()
        sns.countplot(x='Spending_Score', data=df_raw, order=['Low', 'Average', 'High'], palette='viridis', ax=ax2)
        ax2.set_title("Phân phối Mức độ chi tiêu")
        st.pyplot(fig2)

elif page == "Triển khai mô hình":
    st.title("⚙️ Triển khai mô hình nâng cao")
    
    artifacts = joblib.load('models/model.pkl')
    scaler = artifacts['scaler']
    kmeans = artifacts['kmeans']
    expected_features = artifacts['features'] # Danh sách cột mô hình cần
    
    st.subheader("Nhập thông tin khách hàng mới")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age_input = st.number_input("Độ tuổi:", 18, 100, 30)
        gender_input = st.selectbox("Giới tính:", ["Male", "Female"])
    with col2:
        income_input = st.number_input("Thu nhập/năm ($):", 0, 200000, 50000)
    with col3:
        spending_input = st.selectbox("Mức chi tiêu:", ["Low", "Average", "High"])
        profession_input = st.selectbox("Nghề nghiệp:", ["Engineer", "Doctor", "Artist", "Lawyer"]) # Thay bằng các nghề trong data của em

    if st.button("Dự đoán Phân khúc", type="primary"):
        # 1. Tạo một DataFrame giả lập chứa đúng 1 dòng dữ liệu người dùng nhập
        input_dict = {
            'Age': age_input,
            'Income': income_input,
            'Spending_Score_Num': {'Low': 1, 'Average': 2, 'High': 3}[spending_input],
            'Gender_Num': 0 if gender_input == 'Male' else 1
        }
        
        # Tạo DataFrame ban đầu
        input_df = pd.DataFrame([input_dict])
        
        # 2. Khớp các cột Profession (One-Hot Encoding thủ công)
        # Vì lúc train dùng get_dummies, giờ mình phải tự tạo ra các cột 0/1 tương ứng
        for feature in expected_features:
            if feature.startswith('Profession_'):
                prof_name = feature.replace('Profession_', '')
                input_df[feature] = 1 if profession_input == prof_name else 0
                
        # 3. Sắp xếp lại thứ tự cột cho đúng với lúc Train
        input_df = input_df[expected_features]
        
        # 4. Dự đoán
        input_scaled = scaler.transform(input_df)
        cluster_id = kmeans.predict(input_scaled)[0]
        
        # st.success(f"Khách hàng thuộc Nhóm: {cluster_id}")
        # Cập nhật lại phần Text miêu tả cho phù hợp với 4 nhóm mới
        
        # Lấy ID cụm (0, 1 hoặc 2)
        cluster_id = kmeans.predict(input_scaled)[0]
        
        # Ánh xạ ID sang câu văn miêu tả chi tiết
        if cluster_id == 0:
            description = "Khách hàng trẻ, chi tiêu thấp. Xe giá rẻ, trả góp."
        elif cluster_id == 1:
            description = "Khách hàng VIP, chi tiêu cao. Xe sang, sự kiện độc quyền."
        else:
            description = "Khách hàng trung niên, thực dụng. Xe gia đình bền bỉ."
            
        # In thẳng câu miêu tả ra khung màu xanh
        st.success(f"### Khách hàng này thuộc nhóm: {description}")

elif page == "Đánh giá & Hiệu năng":
    st.title("📈 Đánh giá Hiệu năng Mô hình")
    
    # Dùng scaler và kmeans đã load để tính toán
    X = df_processed[['Age', 'Spending_Score_Num']]
    X_scaled = scaler.transform(X)
    labels = kmeans.labels_
    
    silhouette_avg = silhouette_score(X_scaled, labels)
    st.metric(label="Chỉ số Silhouette Score", value=f"{silhouette_avg:.4f}")
    
    st.subheader("Biểu đồ phân tán K-Means (Scatter Plot)")
    df_processed['Cluster'] = labels
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x='Age', y='Spending_Score_Num', hue='Cluster', palette='viridis', data=df_processed, ax=ax)
    plt.title("Kết quả phân nhóm từ file model.pkl")
    st.pyplot(fig)