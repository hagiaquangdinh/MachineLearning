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
    st.title("📊 Giới thiệu & Khám phá dữ liệu (EDA)")

    # --- YÊU CẦU 1: THÔNG TIN BẮT BUỘC ---
    st.subheader("📌 Thông tin đồ án")
    st.markdown("""
    * **Tên đề tài:** Phân loại khách hàng sử dụng thuật toán K-Means Clustering dựa vào mức độ chi tiêu theo độ tuổi nhằm cải thiện kế hoạch marketing kinh doanh xe ô tô.
    * **Họ tên SV:** HÀ GIA QUANG ĐỊNH
    * **MSSV:** 21T1020303
    * **Giá trị thực tiễn:** Giải pháp này sẽ giúp bộ phận Marketing chấm dứt việc quảng cáo đại trà lãng phí, từ đó thiết kế các kịch bản tiếp thị cá nhân hóa sâu sắc cho từng nhóm khách hàng. Thông qua đó để gia tăng mạnh mẽ tỷ lệ chuyển đổi và doanh thu cho công ty.
    """)
    st.divider()

    if st.button("🔄 Cập nhật và Huấn luyện lại Model"):
        train_and_save_model(df_processed)
        st.cache_resource.clear()
        st.rerun()

    # --- YÊU CẦU 2: NỘI DUNG KỸ THUẬT ---
    st.subheader("1. Dữ liệu thô (Raw Data)")
    st.markdown("Hiển thị 10 dòng đầu tiên để có cái nhìn tổng quan về cấu trúc các đặc trưng của khách hàng.")
    st.dataframe(df_raw.head(10))

    st.subheader("2. Biểu đồ phân tích (Data Visualization)")
    st.markdown("Phân tích phân phối của một số đặc trưng cốt lõi trong bộ dữ liệu.")
    
    # Tạo 2 cột để chứa 2 biểu đồ
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Biểu đồ phân phối Độ tuổi (Age)**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(df_raw['Age'], bins=20, kde=True, color='skyblue', ax=ax1)
        ax1.set_xlabel('Tuổi')
        ax1.set_ylabel('Số lượng khách hàng')
        st.pyplot(fig1)

    with col2:
        st.markdown("**Thống kê Nghề nghiệp (Profession)**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        # Sắp xếp từ cao xuống thấp để biểu đồ đẹp hơn
        sns.countplot(y='Profession', data=df_raw, order=df_raw['Profession'].value_counts().index, palette='pastel', ax=ax2)
        ax2.set_xlabel('Số lượng khách hàng')
        ax2.set_ylabel('Nghề nghiệp')
        st.pyplot(fig2)

    # --- YÊU CẦU 3: GIẢI THÍCH VÀ NHẬN XÉT ---
    st.subheader("3. Nhận xét và Giải thích dữ liệu")
    st.info("""
    **🔍 Đánh giá nhanh về đặc trưng tập dữ liệu:**
    * **Sự phân bố độ tuổi (Độ lệch dữ liệu):** Nhìn vào biểu đồ Histogram, ta thấy phân phối độ tuổi hơi lệch phải (Right-skewed). Khách hàng chủ yếu tập trung đông đúc ở nhóm thanh niên và trung niên (từ 20 đến 45 tuổi). Số lượng khách hàng cao tuổi (trên 60) giảm dần và chiếm tỷ trọng nhỏ. 
    * **Sự mất cân bằng biến phân loại:** Ở biểu đồ Nghề nghiệp, nhóm *Artist (Nghệ sĩ)* và *Healthcare (Y tế)* chiếm số lượng áp đảo so với các nhóm như *Homemaker (Nội trợ)*. Sự chênh lệch này là phản ánh đúng thực tế, nhưng nó sẽ khiến thuật toán K-Means nhạy cảm hơn với các nhóm đông người.
    * **Đặc trưng quan trọng:** Trong số 7 biến được sử dụng, các biến dạng số liên tục có dải giá trị rộng như *Độ tuổi (Age)* và *Số năm kinh nghiệm (Work_Experience)* có khả năng cao sẽ đóng góp trọng số lớn vào việc phân chia ranh giới giữa các cụm khách hàng.
    """)
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
        # st.caption(f"*(Thông tin kỹ thuật: Hệ thống phân loại vào Cụm số {cluster_id})*")

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
    # st.caption("*Lưu ý: Vì đây là bài toán Phân cụm (Học không giám sát), ta sử dụng Silhouette và Davies-Bouldin thay cho Accuracy/F1-score.*")

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