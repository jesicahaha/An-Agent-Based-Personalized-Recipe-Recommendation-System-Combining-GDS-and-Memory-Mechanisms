"""
二層隱藏層結構實驗：5 -> 20 -> H2 -> 1

說明：
- 使用同一組「熱杯子五指溫度」資料（cup_nn_train_data.csv）
- 先切成 train/valid/test = 6:2:2
- 第一層隱藏層固定 20 個神經元（ReLU）
- 第二層隱藏層大小 H2 為候選超參數，例如 [3,5,8,10]
- 對每個 H2 做多次實驗（不同 seed），在 valid 上計算平均精準度
- 看「加第二層」對表現的影響，再跟之前的單層 5->H->1 比較

輸出：
- hidden2_runs.csv：每個 H2、每次 run 的 train/valid accuracy
- hidden2_summary.csv：每個 H2 的 validation 平均與標準差
- hidden2_validation_curve.png：第二層神經元數量 vs validation 精準度
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import rcParams

# =============================
# 中文字型設定
# =============================
rcParams["font.sans-serif"] = ["SimHei", "Microsoft JhengHei", "Taipei Sans TC Beta"]
rcParams["axes.unicode_minus"] = False

# -----------------------------
# 激活函數
# -----------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# -----------------------------
# 從 CSV 載入資料
# -----------------------------

def load_data_from_csv(filename):
    X_list = []
    y_list = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        if header is not None:
            header_lower = [h.lower() for h in header]
            try:
                y_idx = header_lower.index("y")
            except ValueError:
                y_idx = len(header_lower) - 1
        else:
            y_idx = -1

        for row in reader:
            if not row:
                continue
            if len(row) <= max(4, y_idx):
                continue
            try:
                thumb  = float(row[0])
                index  = float(row[1])
                middle = float(row[2])
                ring   = float(row[3])
                pinky  = float(row[4])
                y_val  = int(float(row[y_idx]))
            except ValueError:
                continue

            X_list.append([thumb, index, middle, ring, pinky])
            y_list.append([y_val])

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int).reshape(-1, 1)
    return X, y

# -----------------------------
# 切分資料成 train / valid / test
# -----------------------------

def split_train_valid_test(X, y, train_ratio=0.6, valid_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    n_test  = n - n_train - n_valid

    X_train = X_shuffled[:n_train]
    y_train = y_shuffled[:n_train]

    X_valid = X_shuffled[n_train:n_train + n_valid]
    y_valid = y_shuffled[n_train:n_train + n_valid]

    X_test  = X_shuffled[n_train + n_valid:]
    y_test  = y_shuffled[n_train + n_valid:]

    return X_train, y_train, X_valid, y_valid, X_test, y_test

# -----------------------------
# 二層隱藏層 NN：5 -> 20 -> H2 -> 1
# -----------------------------

class TwoLayerNN:
    def __init__(self,
                 input_dim=5,
                 hidden1_dim=20,
                 hidden2_dim=5,
                 output_dim=1,
                 learning_rate=0.01,
                 seed=1):
        rng = np.random.default_rng(seed)

        # 第一層：5 -> 20
        self.W1 = rng.normal(loc=0.0, scale=0.1,
                             size=(input_dim, hidden1_dim))
        self.b1 = np.zeros((1, hidden1_dim))

        # 第二層：20 -> H2
        self.W2 = rng.normal(loc=0.0, scale=0.1,
                             size=(hidden1_dim, hidden2_dim))
        self.b2 = np.zeros((1, hidden2_dim))

        # 輸出層：H2 -> 1
        self.W3 = rng.normal(loc=0.0, scale=0.1,
                             size=(hidden2_dim, output_dim))
        self.b3 = np.zeros((1, output_dim))

        self.lr = learning_rate

    def forward(self, X):
        # 隱藏層 1
        self.z1 = X @ self.W1 + self.b1
        self.h1 = relu(self.z1)

        # 隱藏層 2
        self.z2 = self.h1 @ self.W2 + self.b2
        self.h2 = relu(self.z2)

        # 輸出層
        self.z3 = self.h2 @ self.W3 + self.b3
        self.y_hat = sigmoid(self.z3)

        return self.y_hat

    def compute_loss(self, y_true, y_hat):
        eps = 1e-8
        y_hat_clipped = np.clip(y_hat, eps, 1 - eps)
        loss = -np.mean(
            y_true * np.log(y_hat_clipped) +
            (1 - y_true) * np.log(1 - y_hat_clipped)
        )
        return loss

    def backward(self, X, y_true):
        n_samples = X.shape[0]

        # 輸出層
        dL_dz3 = (self.y_hat - y_true) / n_samples       # (n,1)
        dL_dW3 = self.h2.T @ dL_dz3                     # (H2,1)
        dL_db3 = np.sum(dL_dz3, axis=0, keepdims=True)  # (1,1)

        # 傳回隱藏層2
        dL_dh2 = dL_dz3 @ self.W3.T                     # (n,H2)
        d_relu2 = relu_derivative(self.z2)              # (n,H2)
        dL_dz2 = dL_dh2 * d_relu2                       # (n,H2)

        dL_dW2 = self.h1.T @ dL_dz2                     # (20,H2)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)  # (1,H2)

        # 傳回隱藏層1
        dL_dh1 = dL_dz2 @ self.W2.T                     # (n,20)
        d_relu1 = relu_derivative(self.z1)              # (n,20)
        dL_dz1 = dL_dh1 * d_relu1                       # (n,20)

        dL_dW1 = X.T @ dL_dz1                           # (5,20)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)  # (1,20)

        # 更新權重
        self.W3 -= self.lr * dL_dW3
        self.b3 -= self.lr * dL_db3
        self.W2 -= self.lr * dL_dW2
        self.b2 -= self.lr * dL_db2
        self.W1 -= self.lr * dL_dW1
        self.b1 -= self.lr * dL_db1

    def train(self, X, y, epochs=300):
        for epoch in range(1, epochs + 1):
            y_hat = self.forward(X)
            loss = self.compute_loss(y, y_hat)
            self.backward(X, y)
        return loss

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

# -----------------------------
# 主程式：掃第二層 hidden2_dim
# -----------------------------

if __name__ == "__main__":
    # 1. 載入資料並切成 train/valid/test
    X_all, y_all = load_data_from_csv("cup_nn_train_data.csv")
    print("全部資料筆數:", X_all.shape[0])
    print("y=1 比例:", float(np.mean(y_all)))
    print()

    X_train, y_train, X_valid, y_valid, X_test, y_test = split_train_valid_test(
        X_all, y_all, train_ratio=0.6, valid_ratio=0.2, seed=0
    )
    print("Train 筆數:", X_train.shape[0])
    print("Valid 筆數:", X_valid.shape[0])
    print("Test  筆數:", X_test.shape[0])
    print()

    hidden2_candidates = list(range(3, 21))  # 3,4,5,...,20
    num_runs_per_h2 = 3

    results = []  # 每列: [hidden2_dim, run_id, train_acc, valid_acc]

    for h2 in hidden2_candidates:
        print(f"=== hidden1_dim = 20, hidden2_dim = {h2} 的多次實驗 ===")
        for run in range(num_runs_per_h2):
            seed = 200 + run
            model = TwoLayerNN(input_dim=5,
                               hidden1_dim=20,
                               hidden2_dim=h2,
                               output_dim=1,
                               learning_rate=0.05,
                               seed=seed)

            final_loss = model.train(X_train, y_train, epochs=300)

            # train / valid accuracy
            y_proba_train = model.predict_proba(X_train)
            y_pred_train = (y_proba_train >= 0.5).astype(int)
            train_acc = float(np.mean(y_pred_train == y_train))

            y_proba_valid = model.predict_proba(X_valid)
            y_pred_valid = (y_proba_valid >= 0.5).astype(int)
            valid_acc = float(np.mean(y_pred_valid == y_valid))

            results.append([h2, run, train_acc, valid_acc])

            print(f"  run = {run} | train_acc = {train_acc*100:.2f}% | "
                  f"valid_acc = {valid_acc*100:.2f}% | loss = {final_loss:.4f}")
        print()

    # 3. 計算 summary（每個 H2 的 validation 平均與標準差）
    summary = []
    for h2 in hidden2_candidates:
        vals = [r[3] for r in results if r[0] == h2]
        mean_acc = float(np.mean(vals))
        std_acc  = float(np.std(vals))
        summary.append([h2, mean_acc, std_acc])

    # 4. 存成 CSV
    with open("hidden2_runs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hidden2_dim", "run_id", "train_acc", "valid_acc"])
        for h2, run, train_acc, valid_acc in results:
            writer.writerow([h2, run, train_acc, valid_acc])

    with open("hidden2_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hidden2_dim", "valid_acc_mean", "valid_acc_std"])
        for h2, mean_acc, std_acc in summary:
            writer.writerow([h2, mean_acc, std_acc])

    # 5. 畫出第二層大小 vs validation 精準度
    hidden2_dims = [row[0] for row in summary]
    valid_means = [row[1] * 100 for row in summary]
    valid_stds  = [row[2] * 100 for row in summary]

    plt.figure(figsize=(8, 5))
    plt.errorbar(hidden2_dims, valid_means, yerr=valid_stds,
                 fmt="o-", capsize=5)
    plt.xlabel("第二層隱藏層神經元數量 (hidden2_dim)")
    plt.ylabel("Validation 精準度平均值 (%) ± 標準差")
    plt.title("在第一層固定 20 個神經元下，第二層大小對表現的影響")
    plt.grid(True)
    plt.xticks(hidden2_dims)
    plt.tight_layout()
    plt.savefig("hidden2_validation_curve.png", dpi=150)
    plt.show()

    print("已輸出：")
    print("- hidden2_runs.csv：每個 hidden2_dim、每次 run 的 train/valid accuracy")
    print("- hidden2_summary.csv：每個 hidden2_dim 的 validation 平均與標準差")
    print("- hidden2_validation_curve.png：第二層神經元數量 vs validation 精準度圖")
    print()

    # 6. 選出 validation 平均最佳的 hidden2_dim
    best_row = max(summary, key=lambda r: r[1])
    best_h2, best_mean, best_std = best_row
    print(f"在此次實驗中，hidden1_dim=20 下，validation 平均表現最佳的第二層大小約為 {best_h2}，")
    print(f"平均精準度 ≈ {best_mean*100:.2f}%，標準差 ≈ {best_std*100:.2f}%。")
    print()

    # 7. 最後用最佳 (20, best_h2) 在 train+valid 上重訓一次，並在 test 上評估
    X_train_full = np.vstack([X_train, X_valid])
    y_train_full = np.vstack([y_train, y_valid])

    final_model = TwoLayerNN(input_dim=5,
                             hidden1_dim=20,
                             hidden2_dim=best_h2,
                             output_dim=1,
                             learning_rate=0.05,
                             seed=999)
    final_model.train(X_train_full, y_train_full, epochs=300)

    y_proba_test = final_model.predict_proba(X_test)
    y_pred_test = (y_proba_test >= 0.5).astype(int)
    test_acc = float(np.mean(y_pred_test == y_test))

    print(f"使用結構 5 -> 20 -> {best_h2} -> 1，在 train+valid 上重新訓練後，")
    print(f"在 test 資料上的最終精準度 ≈ {test_acc*100:.2f}%。")