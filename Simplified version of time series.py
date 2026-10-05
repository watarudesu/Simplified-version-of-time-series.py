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

# エッジ密度
# 例：0.20 → 最大可能エッジ数の20%を必ず使用
EDGE_DENSITY = 0.20

# 再現性
np.random.seed(42)


# ============================================================
# 0/1の対称結合行列 W を生成
# ============================================================

# 最大可能エッジ数
num_possible_edges = N * (N - 1) // 2

# 指定したエッジ密度から実際のエッジ数を決定
num_edges = round(EDGE_DENSITY * num_possible_edges)

# 上三角部分の全候補エッジ
upper_i, upper_j = np.triu_indices(N, k=1)

# エッジをランダムに選択
selected = np.random.choice(
    len(upper_i),
    size=num_edges,
    replace=False
)

# 0行列からスタート
W = np.zeros((N, N), dtype=float)

# 選ばれたエッジを1にする
i = upper_i[selected]
j = upper_j[selected]

W[i, j] = 1
W[j, i] = 1

# 自己結合なし
np.fill_diagonal(W, 0)


# ============================================================
# ネットワークの確認
# ============================================================

print("ノード数 =", N)
print("指定エッジ密度 =", EDGE_DENSITY)
print("実際のエッジ数 =", num_edges)
print("最大可能エッジ数 =", num_possible_edges)
print("実際のエッジ密度 =", num_edges / num_possible_edges)


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

plt.tight_layout()


# ============================================================
# 時系列データを保存
# ============================================================

data = np.column_stack((sol.t, sol.y.T))

np.savetxt(
    "time_series1.csv",
    data,
    delimiter=",",
    header="time," + ",".join([f"node_{i+1}" for i in range(N)]),
    comments=""
)


# ============================================================
# 結合行列 W を保存
# ============================================================

np.savetxt(
    "W1.csv",
    W,
    delimiter=","
)

print("時系列データを time_series1.csv に保存しました")
print("結合行列 W を W1.csv に保存しました")


plt.show()