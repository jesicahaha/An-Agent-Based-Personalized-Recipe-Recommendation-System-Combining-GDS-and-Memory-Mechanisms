# -*- coding: utf-8 -*-
"""
二層隱藏層結構實驗：5 -> 43 -> H2 -> 1

說明：
- 使用同一組「熱杯子五指溫度」資料（cup_nn_train_data.csv）
- 先切成 train/valid/test = 6:2:2
- 第一層隱藏層固定 43 個神經元（ReLU）（由單層實驗得出最佳值）
- 第二層隱藏層大小 H2 為候選超參數，範圍 3~50
- 對每個 H2 做 3 次實驗（不同 seed），在 valid 上計算平均精準度
- 找出最佳 H2，用 train+valid 重訓，最後在 test 上評估

輸出：
- h2_runs.csv：每個 H2、每次 run 的 train/valid accuracy
- h2_summary.csv：每個 H2 的 validation 平均與標準差
- h2_validation_curve.png：第二層神經元數量 vs validation 精準度
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import rcParams

# =============================
# 中文字型設定
# =============================
rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Taipei Sans TC Beta",
                                "Noto Sans CJK TC", "PingFang TC"]
rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────────────────
# 激活函數
# ─────────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# ─────────────────────────────────────────────
# 1. 讀取 CSV 資料
# ─────────────────────────────────────────────

def load_data_from_csv(filename):
    X_list, y_list = [], []
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
            if not row or len(row) <= max(4, y_idx):
                continue
            try:
                X_list.append([float(row[i]) for i in range(5)])
                y_list.append([int(float(row[y_idx]))])
            except ValueError:
                continue

    return np.array(X_list, dtype=float), np.array(y_list, dtype=int).reshape(-1, 1)

# ─────────────────────────────────────────────
# 2. 切 train / valid / test = 6:2:2
# ─────────────────────────────────────────────

def split_train_valid_test(X, y, train_ratio=0.6, valid_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    X, y = X[idx], y[idx]

    n = X.shape[0]
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    return (X[:n_train], y[:n_train],
            X[n_train:n_train+n_valid], y[n_train:n_train+n_valid],
            X[n_train+n_valid:], y[n_train+n_valid:])

# ─────────────────────────────────────────────
# 3. 二層隱藏層神經網路：5 -> 43 -> H2 -> 1
# ─────────────────────────────────────────────

class TwoLayerNN:
    def __init__(self,
                 input_dim=5,
                 #hidden1_dim=43,
                 hidden1_dim=9,
                 hidden2_dim=10,
                 output_dim=1,
                 learning_rate=0.05,
                 seed=1):
        rng = np.random.default_rng(seed)

        # 第一層：5 -> 43
        self.W1 = rng.normal(0.0, 0.1, size=(input_dim, hidden1_dim))
        self.b1 = np.zeros((1, hidden1_dim))

        # 第二層：43 -> H2
        self.W2 = rng.normal(0.0, 0.1, size=(hidden1_dim, hidden2_dim))
        self.b2 = np.zeros((1, hidden2_dim))

        # 輸出層：H2 -> 1
        self.W3 = rng.normal(0.0, 0.1, size=(hidden2_dim, output_dim))
        self.b3 = np.zeros((1, output_dim))

        self.lr = learning_rate

    def forward(self, X):
        # 隱藏層 1（ReLU）
        self.z1 = X @ self.W1 + self.b1
        self.h1 = relu(self.z1)

        # 隱藏層 2（ReLU）
        self.z2 = self.h1 @ self.W2 + self.b2
        self.h2 = relu(self.z2)

        # 輸出層（Sigmoid）
        self.z3 = self.h2 @ self.W3 + self.b3
        self.y_hat = sigmoid(self.z3)

        return self.y_hat

    def compute_loss(self, y_true, y_hat):
        eps = 1e-8
        yh = np.clip(y_hat, eps, 1 - eps)
        return -np.mean(y_true * np.log(yh) + (1 - y_true) * np.log(1 - yh))

    def backward(self, X, y_true):
        n = X.shape[0]

        # 輸出層梯度
        dz3 = (self.y_hat - y_true) / n          # (n,1)
        dW3 = self.h2.T @ dz3                    # (H2,1)
        db3 = np.sum(dz3, axis=0, keepdims=True) # (1,1)

        # 傳回隱藏層2
        dh2 = dz3 @ self.W3.T                    # (n,H2)
        dz2 = dh2 * relu_derivative(self.z2)     # (n,H2)
        dW2 = self.h1.T @ dz2                    # (43,H2)
        db2 = np.sum(dz2, axis=0, keepdims=True) # (1,H2)

        # 傳回隱藏層1
        dh1 = dz2 @ self.W2.T                    # (n,43)
        dz1 = dh1 * relu_derivative(self.z1)     # (n,43)
        dW1 = X.T @ dz1                          # (5,43)
        db1 = np.sum(dz1, axis=0, keepdims=True) # (1,43)

        # 更新權重
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=300):
        for _ in range(epochs):
            self.forward(X)
            self.backward(X, y)
        return self.compute_loss(y, self.y_hat)

    def accuracy(self, X, y):
        pred = (self.forward(X) >= 0.5).astype(int)
        return float(np.mean(pred == y))

# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. 載入資料 ──────────────────────────
    print("=" * 55)
    print("1. 讀取 CSV 資料")
    print("=" * 55)
    X_all, y_all = load_data_from_csv("cup_nn_train_data.csv")
    print(f"全部資料筆數: {X_all.shape[0]}")
    print(f"y=1 比例: {float(np.mean(y_all)):.3f}")
    print()

    # ── 2. 切分資料 ──────────────────────────
    print("=" * 55)
    print("2. 切 train / valid / test = 6:2:2")
    print("=" * 55)
    X_train, y_train, X_valid, y_valid, X_test, y_test = \
        split_train_valid_test(X_all, y_all, train_ratio=0.6, valid_ratio=0.2, seed=0)
    print(f"Train 筆數: {X_train.shape[0]}")
    print(f"Valid 筆數: {X_valid.shape[0]}")
    print(f"Test  筆數: {X_test.shape[0]}")
    print()

    # ── 4. 掃描 H2 = 3~50（3 seeds）────────
    print("=" * 55)
    print("4. 掃描 H2 = 3~50（hidden1_dim 固定 43，3 個 seed）")
    print("=" * 55)

    #hidden2_candidates = list(range(3, 51))  # 3,4,...,50
    hidden2_candidates = list(range(3, 21))
    num_runs_per_h2 = 3
    epochs = 300
    lr = 0.05

    results = []  # [h2, run, train_acc, valid_acc, loss]

    for h2 in hidden2_candidates:
        #print(f"=== hidden1_dim = 43, hidden2_dim = {h2} 的多次實驗 ===")
        print(f"=== hidden1_dim = 9, hidden2_dim = {h2} 的多次實驗 ===")
        for run in range(num_runs_per_h2):
            seed = 200 + run
            model = TwoLayerNN(input_dim=5,
                               #hidden1_dim=43,
                               hidden1_dim=9,
                               hidden2_dim=h2,
                               output_dim=1,
                               learning_rate=0.05,
                               seed=seed)

            final_loss = model.train(X_train, y_train, epochs=300)

            # train / valid accuracy
            y_proba_train = model.forward(X_train)
            y_pred_train = (y_proba_train >= 0.5).astype(int)
            train_acc = float(np.mean(y_pred_train == y_train))

            y_proba_valid = model.forward(X_valid)
            y_pred_valid = (y_proba_valid >= 0.5).astype(int)
            valid_acc = float(np.mean(y_pred_valid == y_valid))

            results.append([h2, run, train_acc, valid_acc, final_loss])

            print(f"  run = {run} | train_acc = {train_acc*100:.2f}% | "
                  f"valid_acc = {valid_acc*100:.2f}% | loss = {final_loss:.4f}")
        print()

    # ── 5. 計算 summary ───────────────────────
    print("=" * 55)
    print("5. 各 H2 的 Validation 精準度摘要")
    print("=" * 55)

    summary = []
    for h2 in hidden2_candidates:
        rows = [r for r in results if r[0] == h2]
        va = [r[3] for r in rows]
        tr = [r[2] for r in rows]
        summary.append({
            "h2": h2,
            "train_mean": float(np.mean(tr)),
            "train_std":  float(np.std(tr)),
            "valid_mean": float(np.mean(va)),
            "valid_std":  float(np.std(va)),
        })
        print(f"  H2={h2:2d} | 平均精準度 ≈ {np.mean(va)*100:.2f}%，標準差 ≈ {np.std(va)*100:.2f}%。")
    print()

    # ── 6. 找最佳 H2 ────────────────────────
    best = max(summary, key=lambda x: x["valid_mean"])
    best_h2 = best["h2"]
    #print(f"在此次實驗中，hidden1_dim=43 下，validation 平均表現最佳的第二層大小約為 {best_h2}，")
    print(f"在此次實驗中，hidden1_dim=9 下，validation 平均表現最佳的第二層大小約為 {best_h2}，")
    print(f"平均精準度 ≈ {best['valid_mean']*100:.2f}%，標準差 ≈ {best['valid_std']*100:.2f}%。")
    print()

    # ── 存 CSV ───────────────────────────────
    with open("h2_runs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hidden2_dim", "run_id", "train_acc", "valid_acc", "loss"])
        for h2, run, tr, va, lo in results:
            writer.writerow([h2, run, tr, va, lo])

    with open("h2_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hidden2_dim", "train_mean", "train_std", "valid_mean", "valid_std"])
        for s in summary:
            writer.writerow([s["h2"], s["train_mean"], s["train_std"],
                             s["valid_mean"], s["valid_std"]])

    # ── 7. 用最佳 H2 在 train+valid 上重訓 ──
    print("=" * 55)
    print("7. 用最佳 H2 重新訓練（train + valid）")
    print("=" * 55)

    X_tv = np.vstack([X_train, X_valid])
    y_tv = np.vstack([y_train, y_valid])

    final_model = TwoLayerNN(input_dim=5,
                             #hidden1_dim=43,
                             hidden1_dim=9,
                             hidden2_dim=best_h2,
                             output_dim=1,
                             learning_rate=lr,
                             seed=999)
    final_model.train(X_tv, y_tv, epochs)

    # ── 8. Test 最終評估 ─────────────────────
    test_acc = final_model.accuracy(X_test, y_test)
    #print(f"使用結構 5 -> 43 -> {best_h2} -> 1，在 train+valid 上重新訓練後，")
    print(f"使用結構 5 -> 9 -> {best_h2} -> 1，在 train+valid 上重新訓練後，")
    print(f"在 test 資料上的最終精準度 ≈ {test_acc*100:.2f}%。")
    print()

    # ── 9. 畫圖 ─────────────────────────────
    h2_list    = [s["h2"]         for s in summary]
    valid_mean = [s["valid_mean"] * 100 for s in summary]
    valid_std  = [s["valid_std"]  * 100 for s in summary]
    train_mean = [s["train_mean"] * 100 for s in summary]
    train_std  = [s["train_std"]  * 100 for s in summary]

    plt.figure(figsize=(12, 5))

    # ── Validation 精準度曲線 ────────────────
    plt.subplot(1, 2, 1)
    plt.errorbar(h2_list, valid_mean, yerr=valid_std,
                 fmt="o-", capsize=4, color="steelblue", label="Valid")
    plt.axvline(best_h2, color="red", linestyle="--", label=f"最佳 H2={best_h2}")
    plt.xlabel("第二層隱藏層神經元數量 (H2)")
    plt.ylabel("Validation 精準度 (%) ± 標準差")
    #plt.title("5 → 43 → H2 → 1\nValidation 精準度 vs H2")
    plt.title("5 → 9 → H2 → 1\nValidation 精準度 vs H2")
    plt.legend()
    plt.grid(True)
    #plt.xticks(range(3, 51, 3))
    plt.xticks(range(3, 21, 3))

    # ── Train 精準度曲線 ─────────────────────
    plt.subplot(1, 2, 2)
    plt.errorbar(h2_list, train_mean, yerr=train_std,
                 fmt="s-", capsize=4, color="darkorange", label="Train")
    plt.axvline(best_h2, color="red", linestyle="--", label=f"最佳 H2={best_h2}")
    plt.xlabel("第二層隱藏層神經元數量 (H2)")
    plt.ylabel("Train 精準度 (%) ± 標準差")
    #plt.title("5 → 43 → H2 → 1\nTrain 精準度 vs H2")
    plt.title("5 →9 → H2 → 1\nTrain 精準度 vs H2")
    plt.legend()
    plt.grid(True)
    #plt.xticks(range(3, 51, 3))
    plt.xticks(range(3, 21, 3))

    plt.tight_layout()
    plt.savefig("h2_validation_curve.png", dpi=150)
    plt.show()

    print("已輸出：")
    print("- h2_runs.csv：每個 H2、每次 run 的 train/valid accuracy")
    print("- h2_summary.csv：每個 H2 的 validation 平均與標準差")
    print("- h2_validation_curve.png：Train / Validation 精準度 vs H2 圖")