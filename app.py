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
            1: "👨‍👩‍👧‍👦 Nhóm Gia đình (Ưu tiên xe rộng rãi, an toàn)",
            2: "💼 Nhóm Doanh nhân / Thu nhập cao (Tiềm năng dòng xe hạng sang)",
            3: "👴 Nhóm Trung niên / Hưu trí (Ưu tiên xe thực dụng, bền bỉ, tiết kiệm)"
        }
        
        # Lấy tên nhóm dựa trên ID dự đoán được
        persona = cluster_names.get(cluster_id, "Nhóm Khách hàng chung")
        
        st.success(f"### 🎯 Khách hàng này thuộc: {persona}")
        st.caption(f"*(Thông tin kỹ thuật: Hệ thống phân loại vào Cụm số {cluster_id})*")

# ==========================================
# TRANG 3: ĐÁNH GIÁ
# ==========================================
# ==========================================
# TRANG 3: ĐÁNH GIÁ & HIỆU NĂNG
# ==========================================
elif page == "Đánh giá & Hiệu năng":
    st.title("📈 Đánh giá & Hiệu năng (Evaluation)")
    st.markdown("Chứng minh mô hình hoạt động tốt và đáng tin cậy dựa trên các chỉ số phân cụm.")

    X_scaled = scaler.transform(df_processed)
    labels = kmeans.labels_

    # --- YÊU CẦU 1: CÁC CHỈ SỐ ĐO LƯỜNG ---
    st.subheader("1. Các chỉ số đo lường (Metrics)")
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    
    # Tính toán 2 chỉ số quan trọng nhất cho Clustering
    silhouette_avg = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)

    col1, col2 = st.columns(2)
    col1.metric(label="Chỉ số Silhouette (Càng gần 1 càng tốt)", value=f"{silhouette_avg:.4f}")
    col2.metric(label="Chỉ số Davies-Bouldin (Càng nhỏ càng tốt)", value=f"{db_score:.4f}")
    st.caption("*Lưu ý: Vì đây là bài toán Phân cụm (Học không giám sát), ta sử dụng Silhouette và Davies-Bouldin thay cho Accuracy/F1-score.*")

    # --- YÊU CẦU 2: BIỂU ĐỒ KỸ THUẬT ---
    st.subheader("2. Biểu đồ kỹ thuật")
    st.markdown("Trực quan hóa mức độ phân tách của các cụm trong không gian (Hiển thị 2 chiều đặc trưng: Tuổi và Chi tiêu).")
    
    df_plot = df_raw.dropna(subset=['Age', 'Spending_Score', 'Family_Size', 'Ever_Married', 'Profession', 'Work_Experience', 'Graduated']).copy()
    df_plot['Cluster'] = labels

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x='Age', y='Spending_Score', hue='Cluster', palette='viridis', data=df_plot, ax=ax)
    st.pyplot(fig)

    # --- YÊU CẦU 3: PHÂN TÍCH SAI SỐ ---
    st.subheader("3. Phân tích sai số & Hướng cải thiện")
    st.info("""
    **Nhận định về vùng dữ liệu khó phân tách (Overlap):**
    * **Đặc điểm thuật toán:** Thuật toán K-Means không có khái niệm "đoán sai nhãn", nhưng điểm yếu của nó là thường ép dữ liệu thành các cụm hình cầu.
    * **Trường hợp dễ phân nhóm chưa tối ưu:** Dựa vào biểu đồ, ta thấy các cụm có sự chồng lấn nhất định. Khách hàng nằm ở vùng ranh giới (ví dụ: tuổi trung niên nhưng mức chi tiêu đan xen giữa Low và Average) rất dễ bị gán vào cụm lân cận.
    * **Hướng cải thiện (Future Works):**
      1. **Thay đổi thuật toán:** Thử nghiệm DBSCAN hoặc Hierarchical Clustering để gom cụm theo mật độ dữ liệu thay vì khoảng cách, giúp bắt được các hình dạng dữ liệu phức tạp.
      2. **Feature Engineering (Trích xuất đặc trưng):** Tạo thêm các biến số mới (Ví dụ: Tỷ lệ Số năm kinh nghiệm / Tuổi) để tăng khoảng cách giữa các điểm ảnh hưởng.
      3. **Bổ sung dữ liệu:** Thu thập thêm dữ liệu về Thu nhập thực tế (Income) sẽ giúp phân khúc tách bạch hơn rất nhiều so với việc chỉ dùng điểm Spending Score.
    """)