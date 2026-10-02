import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ============================================================
# Mean-Field Glauber Dynamics
#
# dm/dt = -m + tanh(beta * W @ m)
# ============================================================

def mean_field_glauber(t, m, W, beta):
    return -m + np.tanh(beta * (W @ m))


# ============================================================
# パラメータ
# ============================================================

N = 20

# エッジが存在する確率
EDGE_PROBABILITY = 0.20

# 再現性
np.random.seed(42)


# ============================================================
# スパースな対称結合行列 W を生成
# ============================================================

# まずランダムな実数値結合を作る
W_random = np.random.randn(N, N) / np.sqrt(N)

# 対称化
W = (W_random + W_random.T) / 2

# ------------------------------------------------------------
# エッジの有無を決める
# ------------------------------------------------------------

# 上三角部分だけでエッジの有無を決める
edge_mask = np.random.rand(N, N) < EDGE_PROBABILITY

# 対称化
edge_mask = np.triu(edge_mask, k=1)
edge_mask = edge_mask + edge_mask.T

# エッジがない場所を0にする
W[edge_mask == 0] = 0

# 自己結合なし
np.fill_diagonal(W, 0)


# ============================================================
# ネットワークの確認
# ============================================================

num_possible_edges = N * (N - 1) // 2
num_edges = np.sum(np.triu(W != 0, k=1))

print("ノード数 =", N)
print("エッジ存在確率 =", EDGE_PROBABILITY)
print("実際のエッジ数 =", num_edges)
print("最大可能エッジ数 =", num_possible_edges)
print("エッジ密度 =", num_edges / num_possible_edges)


# ============================================================
# 最大固有値から臨界点を計算
# ============================================================

eigenvalues = np.linalg.eigvalsh(W)
lambda_max = eigenvalues[-1]

beta_c = 1.0 / lambda_max
T_c = lambda_max

print("最大固有値 =", lambda_max)
print("臨界逆温度 beta_c =", beta_c)
print("臨界温度 T_c =", T_c)


# ============================================================
# 初期状態
# ============================================================

m0 = np.random.uniform(-0.5, 0.5, N)


# ============================================================
# 時系列を生成
# ============================================================

beta = 1.2 * beta_c

t_start = 0
t_end = 50
num_points = 500

t_eval = np.linspace(t_start, t_end, num_points)

sol = solve_ivp(
    mean_field_glauber,
    [t_start, t_end],
    m0,
    args=(W, beta),
    method="RK45",
    t_eval=t_eval
)


# ============================================================
# 結果
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(sol.t, sol.y.T)

plt.xlabel("Time t")
plt.ylabel("Magnetization m_i(t)")
plt.title("Mean-Field Glauber Dynamics")


# ============================================================
# 時系列データを保存
# ============================================================

data = np.column_stack((sol.t, sol.y.T))

np.savetxt(
    "time_series2.csv",
    data,
    delimiter=",",
    header="time," + ",".join([f"node_{i+1}" for i in range(N)]),
    comments=""
)


# ============================================================
# 結合行列 W を保存
# ============================================================

np.savetxt(
    "W2.csv",
    W,
    delimiter=","
)

print("時系列データを time_series2.csv に保存しました")
print("結合行列 W を W2.csv に保存しました")


plt.show()