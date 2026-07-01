import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def resize_nearest(img, new_h, new_w):
    old_h, old_w = img.shape[:2]
    row_idx = (np.linspace(0, old_h - 1, new_h)).astype(int)
    col_idx = (np.linspace(0, old_w - 1, new_w)).astype(int)
    grid_y, grid_x = np.meshgrid(row_idx, col_idx, indexing='ij')
    return img[grid_y, grid_x]

def compute_Roundness(mask):
    area = np.sum(mask)
    padded = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
    perimeter_mask = (mask & (
        (~padded[:-2, 1:-1]) | (~padded[2:, 1:-1]) |
        (~padded[1:-1, :-2]) | (~padded[1:-1, 2:])
    ))
    perimeter = np.sum(perimeter_mask)
    if perimeter > 0:
        roundness = 4 * np.pi * area / (perimeter ** 2)
    else:
        roundness = 0
    return roundness

def compute_Elongation(mask):
    ys, xs = np.where(mask)
    coords = np.column_stack((xs, ys))
    cov = np.cov(coords, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals[0] > 0:
        elongation = np.sqrt(eigvals[1]) / np.sqrt(eigvals[0])
    else:
        elongation = np.inf
    return elongation

def FeatureExtraction(img_raw):
    if img_raw.dtype != np.uint8:
        img_raw = (img_raw * 255).astype(np.uint8)
    if len(img_raw.shape) == 3 and img_raw.shape[2] == 4:
        img_raw = img_raw[:, :, :3]
    img = resize_nearest(img_raw, 200, 200)
    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = gray.size
    current_max, threshold = 0, 0
    sum_total = np.dot(np.arange(256), hist)
    sum_background, weight_background = 0.0, 0.0

    for i in range(256):
        weight_background += hist[i]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += i * hist[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        between_var = (weight_background * weight_foreground * (mean_background - mean_foreground) ** 2)
        if between_var > current_max:
            current_max = between_var
            threshold = i

    mask = gray < threshold
    roundness = compute_Roundness(mask)
    elongation = compute_Elongation(mask)
    object_pixels = img[mask]
    if object_pixels.size > 0:
        avg_r, avg_g, avg_b = object_pixels.mean(axis=0)
    else:
        avg_r, avg_g, avg_b = -1, -1, -1
    return np.array([roundness, elongation, avg_r, avg_g, avg_b])

def normal_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

class NaiveBayesClassifier:
    def __init__(self, _DataLoc, _ClassName):
        self.DataLoc = _DataLoc
        self.ClassName = _ClassName

    def compute_posterior_probability(self, queried_x):
        df = pd.read_csv(self.DataLoc)
        X = df.iloc[:, :-1].to_numpy(dtype=float)
        y = df.iloc[:, -1].to_numpy(str)
        N = y.size
        class_labels = np.unique(y)
        posterior_probs = []
        for c in class_labels:
            P_c = np.sum(y == c) / N
            indices = np.where(y == c)[0]
            likelihood = 1.0
            for i in range(len(queried_x)):
                mu = np.mean(X[indices, i])
                sigma = np.std(X[indices, i])
                sigma = sigma if sigma > 0 else 1e-6
                prob = normal_pdf(queried_x[i], mu, sigma)
                likelihood *= prob
            posterior_probs.append(P_c * likelihood)
        posterior_probs = np.array(posterior_probs)
        total = np.sum(posterior_probs)
        if total == 0:
            posterior_probs = np.zeros_like(posterior_probs)
        else:
            posterior_probs = posterior_probs / total
        return posterior_probs

# BƯỚC 1: TẠO DATASET
print('=' * 60)
print('BƯỚC 1: TRÍCH XUẤT ĐẶC TRƯNG TỪ ẢNH HUẤN LUYỆN')
print('=' * 60)

class_folders = ['apple', 'banana', 'orange']
dataset = []
for class_name in class_folders:
    print(f'\n📁 Xử lý loại: {class_name.upper()}')
    for i in range(5):
        img_path = f'images/{class_name}/image{i}.bmp'
        try:
            img_raw = plt.imread(img_path)
            feature_vec = FeatureExtraction(img_raw)
            row = feature_vec.tolist() + [class_name]
            dataset.append(row)
            print(f'  ✓ image{i}.bmp -> Roundness={feature_vec[0]:.4f}, Elongation={feature_vec[1]:.4f}')
        except Exception as e:
            print(f'  ✗ image{i}.bmp: Lỗi - {e}')

df = pd.DataFrame(dataset, columns=['Roundness', 'Elongation', 'Avg_R', 'Avg_G', 'Avg_B', 'Label'])
df.to_csv('fruit_feature_dataset.csv', index=False)
print(f'\n✓ Đã tạo dataset: fruit_feature_dataset.csv')
print('\nDataset (15 mẫu):\n')
print(df.to_string(index=False))

# BƯỚC 2: PHÂN LOẠI ẢNH TEST
print('\n' + '=' * 60)
print('BƯỚC 2: PHÂN LOẠI ẢNH TEST BẰNG NAIVE BAYES')
print('=' * 60)

test_folder = 'images/Test_Images_FruitClassification/'
test_images = sorted([f for f in os.listdir(test_folder) if f.endswith(('.bmp', '.jpg', '.png'))])

classifier = NaiveBayesClassifier('fruit_feature_dataset.csv', class_folders)

results = []
for img_file in test_images:
    img_path = os.path.join(test_folder, img_file)
    try:
        img_raw = plt.imread(img_path)
        query = FeatureExtraction(img_raw)
        probs = classifier.compute_posterior_probability(query)
        pred_idx = np.argmax(probs)
        pred_label = class_folders[pred_idx]
        max_prob = probs[pred_idx]
        
        result = {
            'Ảnh': img_file,
            'Dự đoán': pred_label,
            'Xác suất': f'{max_prob:.4f}',
            'Apple': f'{probs[0]:.4f}',
            'Banana': f'{probs[1]:.4f}',
            'Orange': f'{probs[2]:.4f}'
        }
        results.append(result)
        print(f'\n📷 {img_file}')
        print(f'   → Dự đoán: {pred_label.upper()} (xác suất: {max_prob:.4f})')
        print(f'   → Apple: {probs[0]:.4f}, Banana: {probs[1]:.4f}, Orange: {probs[2]:.4f}')
    except Exception as e:
        print(f'\n✗ Lỗi xử lý {img_file}: {e}')

# Lưu kết quả
results_df = pd.DataFrame(results)
results_df.to_csv('ket_qua_task1.csv', index=False)
print(f'\n✓ Đã lưu kết quả vào: ket_qua_task1.csv')
print('\nKết quả phân loại:\n')
print(results_df.to_string(index=False))

print('\n' + '=' * 60)
print('✓ HOÀN THÀNH!')
print('=' * 60)
