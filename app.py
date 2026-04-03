# Tên file: app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import silhouette_score
import os

# Cấu hình trang
st.set_page_config(page_title="App Phân Loại Khách Hàng", page_icon="🚗", layout="wide")

# --- YÊU CẦU KỸ THUẬT: SỬ DỤNG CACHE ---
@st.cache_data
def load_data():
    df = pd.read_csv('Data/Train.csv')
    data = df[['Age', 'Spending_Score']].dropna().copy()
    score_mapping = {'Low': 1, 'Average': 2, 'High': 3}
    data['Spending_Score_Num'] = data['Spending_Score'].map(score_mapping)
    return df, data

@st.cache_resource
def load_models():
    # 1. Đọc chiếc hộp duy nhất
    artifacts = joblib.load('models/model.pkl')
    
    # 2. Lấy từng món ra từ trong hộp
    scaler = artifacts['scaler']
    kmeans = artifacts['kmeans']
    
    return scaler, kmeans

# Load dữ liệu
df_raw, df_processed = load_data()

# Điều hướng các trang
page = st.sidebar.radio("Cấu trúc ứng dụng", 
                        ["Giới thiệu & EDA", 
                         "Triển khai mô hình", 
                         "Đánh giá & Hiệu năng"])

# ==========================================
# TRANG 1: GIỚI THIỆU & KHÁM PHÁ DỮ LIỆU
# ==========================================
if page == "Giới thiệu & EDA":
    st.title("📊 Khám phá dữ liệu (EDA)")
    
    # Thông tin bắt buộc
    st.markdown("""
    **Tên đề tài:** Phân loại khách hàng Ô tô dựa trên Độ tuổi và Mức chi tiêu (K-Means)  
    **Họ tên SV:** [Điền tên em vào đây] | **MSSV:** [Điền MSSV vào đây]  
    **Giá trị thực tiễn:** Tự động hóa việc phân khúc khách hàng, giúp phòng Marketing tối ưu ngân sách quảng cáo và cá nhân hóa thông điệp bán xe (xe cỡ nhỏ giá rẻ vs. xe SUV hạng sang).
    """)
    st.divider()

    # Hiển thị dữ liệu thô
    st.subheader("1. Một phần dữ liệu thô")
    st.dataframe(df_raw.head(10))

    # Biểu đồ phân tích
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

    # Giải thích
    st.info("**Nhận xét dữ liệu:** Dữ liệu độ tuổi tập trung đông nhất ở nhóm 20-35 tuổi (có sự lệch nhẹ về bên phải). Mức chi tiêu 'Low' chiếm tỷ trọng lớn nhất, cho thấy đối tượng mua xe lần đầu hoặc xe giá rẻ là tệp khách hàng tiềm năng cốt lõi.")

# ==========================================
# TRANG 2: TRIỂN KHAI MÔ HÌNH
# ==========================================
elif page == "Triển khai mô hình":
    st.title("⚙️ Triển khai mô hình K-Means")
    
    scaler, kmeans = load_models()
    
    st.subheader("Nhập thông tin khách hàng mới")
    col1, col2 = st.columns(2)
    
    with col1:
        age_input = st.number_input("Nhập Độ tuổi:", min_value=18, max_value=100, value=30, step=1)
    with col2:
        spending_input = st.selectbox("Chọn Mức độ chi tiêu:", ["Low", "Average", "High"])

    if st.button("Dự đoán Phân khúc", type="primary"):
        # Xử lý logic giống hệt lúc huấn luyện
        score_mapping = {'Low': 1, 'Average': 2, 'High': 3}
        spending_num = score_mapping[spending_input]
        
        input_data = np.array([[age_input, spending_num]])
        input_scaled = scaler.transform(input_data)
        
        # Dự đoán
        cluster_id = kmeans.predict(input_scaled)[0]
        
        # Tính khoảng cách tới tâm cụm (Đóng vai trò như độ tự tin trong Học không giám sát)
        distances = kmeans.transform(input_scaled)[0]
        min_distance = np.min(distances)
        
        st.success(f"### Khách hàng này thuộc: Nhóm {cluster_id}")
        st.caption(f"Khoảng cách (Distance) tới tâm cụm: {min_distance:.4f} (Càng nhỏ càng chính xác)")
        
        # Diễn giải kết quả kinh doanh
        if cluster_id == 0:
            st.write("🎯 **Khuyến nghị Marketing:** Khách hàng trẻ tuổi, chi tiêu thấp. Phù hợp chạy quảng cáo các dòng xe cỡ nhỏ, hỗ trợ trả góp trên kênh TikTok.")
        elif cluster_id == 1:
            st.write("🎯 **Khuyến nghị Marketing:** Khách hàng VIP, chi tiêu cao. Phù hợp tư vấn xe SUV hạng sang, phiên bản Full Option, chăm sóc qua Email hoặc tổ chức sự kiện.")
        else:
            st.write("🎯 **Khuyến nghị Marketing:** Khách hàng trung niên, chi tiêu trung bình. Đề cao sự thực dụng, nên quảng cáo các dòng xe sedan gia đình bền bỉ, tiết kiệm nhiên liệu.")

# ==========================================
# TRANG 3: ĐÁNH GIÁ & HIỆU NĂNG
# ==========================================
elif page == "Đánh giá & Hiệu năng":
    st.title("📈 Đánh giá Hiệu năng Mô hình")
    
    scaler, kmeans = load_models()
    X = df_processed[['Age', 'Spending_Score_Num']]
    X_scaled = scaler.transform(X)
    labels = kmeans.labels_
    
    # 1. Chỉ số đo lường
    silhouette_avg = silhouette_score(X_scaled, labels)
    st.metric(label="Chỉ số Silhouette Score (Từ -1 đến 1)", value=f"{silhouette_avg:.4f}", 
              delta="Tốt" if silhouette_avg > 0.5 else "Bình thường")
    st.caption("*Silhouette Score càng gần 1, chứng tỏ các cụm phân tách càng rõ ràng và chất lượng phân cụm càng cao.*")
    
    # 2. Biểu đồ kỹ thuật
    st.subheader("Biểu đồ phân tán K-Means (Scatter Plot)")
    df_processed['Cluster'] = labels
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x='Age', y='Spending_Score_Num', hue='Cluster', palette='viridis', data=df_processed, ax=ax, s=60)
    plt.yticks([1, 2, 3], ['Low (1)', 'Average (2)', 'High (3)'])
    plt.title("Kết quả phân nhóm trên tập dữ liệu huấn luyện")
    st.pyplot(fig)
    
    # 3. Phân tích sai số
    st.subheader("Phân tích hạn chế & Hướng cải thiện")
    st.warning("""
    - **Nhận định:** Mô hình K-Means sử dụng khoảng cách Euclidean, nên đôi khi nó quá nhạy cảm với những khách hàng có thông tin bất thường (Outliers). Việc chỉ dùng 2 biến (Age và Spending) có thể chưa khắc họa đủ sâu chân dung khách hàng.
    - **Trường hợp dễ nhầm lẫn:** Những người nằm ở "ranh giới" độ tuổi (ví dụ 35 tuổi) có thể bị xếp nhảy qua lại giữa 2 nhóm nếu mức độ chi tiêu của họ không quá rõ ràng.
    - **Hướng cải thiện:** Thu thập thêm dữ liệu định lượng (như `Income` - Thu nhập hàng tháng) hoặc ứng dụng thuật toán **Fuzzy C-Means** để tính xác suất mềm thay vì gán nhãn cứng nhắc.
    """)