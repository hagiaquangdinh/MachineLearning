# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
# (Giả sử em đã tải file Train.csv về chung thư mục với file code)
df = pd.read_csv('Train.csv')

# Lấy 2 cột mục tiêu và loại bỏ các dòng bị khuyết dữ liệu (NaN) để tránh lỗi
data = df[['Age', 'Spending_Score']].dropna().copy()

# Chuyển đổi mức chi tiêu (Chữ) thành số thứ bậc (Số)
# Bước này giúp K-Means hiểu được Low thấp hơn Average và Average thấp hơn High
score_mapping = {'Low': 1, 'Average': 2, 'High': 3}
data['Spending_Score_Num'] = data['Spending_Score'].map(score_mapping)

# Tách riêng các đặc trưng để đưa vào mô hình
X = data[['Age', 'Spending_Score_Num']]

# CHUẨN HÓA DỮ LIỆU (Cực kỳ quan trọng)
# Vì Age (18-89) lớn hơn rất nhiều so với Điểm chi tiêu (1-3).
# Nếu không chuẩn hóa, AI sẽ chỉ tập trung phân nhóm theo Tuổi mà bỏ qua Điểm chi tiêu.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. XÂY DỰNG MÔ HÌNH K-MEANS
# Thầy sẽ thiết lập thử chia thành 3 nhóm chiến lược (K=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# 3. TRỰC QUAN HÓA KẾT QUẢ
plt.figure(figsize=(10, 6))
# Vẽ biểu đồ phân tán (Scatter Plot)
sns.scatterplot(x='Age', y='Spending_Score_Num', hue='Cluster', 
                data=data, palette='viridis', s=100, alpha=0.7)

# Trang trí biểu đồ cho chuyên nghiệp
plt.yticks([1, 2, 3], ['Low (1)', 'Average (2)', 'High (3)'])
plt.title('Phân cụm Khách hàng Ô tô: Độ tuổi vs Mức chi tiêu', fontsize=14, fontweight='bold')
plt.xlabel('Độ tuổi (Age)', fontsize=12)
plt.ylabel('Mức độ chi tiêu (Spending Score)', fontsize=12)
plt.legend(title='Nhóm Khách hàng')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 4. TỔNG HỢP ĐẶC ĐIỂM TRUNG BÌNH ĐỂ LẬP KẾ HOẠCH MARKETING
cluster_summary = data.groupby('Cluster')[['Age', 'Spending_Score_Num']].mean().round(1)
print("\n--- ĐẶC ĐIỂM TRUNG BÌNH CỦA CÁC NHÓM ---")
print(cluster_summary)