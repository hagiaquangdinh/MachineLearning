@@ -19,33 +19,51 @@ if not os.path.exists('models'):
# --- HÀM 1: LOAD DỮ LIỆU ---
@st.cache_data
def load_data():
    # Lưu ý: Kiểm tra chính xác tên thư mục là 'Data' hay 'data' trên GitHub của em
    df = pd.read_csv('Data/Train.csv') 
    data = df[['Age', 'Spending_Score']].dropna().copy()



    score_mapping = {'Low': 1, 'Average': 2, 'High': 3}
    data['Spending_Score_Num'] = data['Spending_Score'].map(score_mapping)








    return df, data

# --- HÀM 2: HUẤN LUYỆN VÀ LƯU MODEL (CHỈ CHẠY KHI THIẾU FILE) ---
def train_and_save_model(data_processed):
    st.info("🔄 Đang huấn luyện mô hình lần đầu...")
    X = data_processed[['Age', 'Spending_Score_Num']]






    
    # 1. Chuẩn hóa
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Huấn luyện K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # 3. Đóng gói và lưu vào 1 file .pkl duy nhất
    model_artifacts = {
        'scaler': scaler,
        'kmeans': kmeans

    }
    joblib.dump(model_artifacts, 'models/model.pkl')
    st.success("✅ Đã lưu mô hình vào models/model.pkl!")

# --- HÀM 3: LOAD MODEL TỪ FILE .PKL ---
@st.cache_resource


@@ -98,51 +116,82 @@
        st.pyplot(fig2)

elif page == "Triển khai mô hình":
    st.title("⚙️ Triển khai mô hình K-Means")





    
    st.subheader("Nhập thông tin khách hàng mới")
    col1, col2 = st.columns(2)

    with col1:
        age_input = st.number_input("Nhập Độ tuổi:", min_value=18, max_value=100, value=30)

    with col2:
        spending_input = st.selectbox("Chọn Mức độ chi tiêu:", ["Low", "Average", "High"])




    if st.button("Dự đoán Phân khúc", type="primary"):
        score_mapping = {'Low': 1, 'Average': 2, 'High': 3}
        spending_num = score_mapping[spending_input]






















        
        # CHÚ Ý: Sử dụng 'scaler' đã load từ file, KHÔNG tạo scaler mới
        input_data = np.array([[age_input, spending_num]])
        input_scaled = scaler.transform(input_data)
        
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