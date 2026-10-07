import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams["font.size"] = 18


# ============================================================
# 1. 元のネットワーク W を作成
# ============================================================

N = 20
EDGE_DENSITY = 0.20

np.random.seed(42)

num_possible_edges = N * (N - 1) // 2
num_edges = round(EDGE_DENSITY * num_possible_edges)

upper_i, upper_j = np.triu_indices(N, k=1)

selected = np.random.choice(
    len(upper_i),
    size=num_edges,
    replace=False
)

W = np.zeros((N, N), dtype=float)

i = upper_i[selected]
j = upper_j[selected]

W[i, j] = 1
W[j, i] = 1

np.fill_diagonal(W, 0)


# ============================================================
# 2. 元の Tc を計算
# ============================================================

print("========================================")
print("元のネットワーク")
print("========================================")

print("ノード数 =", N)
print("エッジ数 =", num_edges)
print("エッジ密度 =", num_edges / num_possible_edges)


# W は対称行列なので eigh を使用
eigenvalues = np.linalg.eigvalsh(W)

lambda_max = eigenvalues[-1]

Tc = lambda_max

print("最大固有値 =", lambda_max)
print("元の臨界温度 Tc =", Tc)


# ============================================================
# 3. 各ノードの次数と次数中心性を計算
# ============================================================

# 各ノードの次数
degrees = np.sum(W, axis=1)

# 次数中心性
degree_centrality = degrees / (N - 1)


print()
print("========================================")
print("各ノードの次数中心性")
print("========================================")

for i in range(N):

    print(
        f"Node {i+1}: "
        f"degree = {int(degrees[i])}, "
        f"degree centrality = {degree_centrality[i]:.6f}"
    )


# ============================================================
# 4. 各エッジを1本ずつ削除
# ============================================================

results = []

for i in range(N):

    for j in range(i + 1, N):

        # 元々エッジが存在する場合だけ処理
        if W[i, j] == 1:

            # ------------------------------------------------
            # エッジを1本削除
            # ------------------------------------------------

            W_error = W.copy()

            W_error[i, j] = 0
            W_error[j, i] = 0


            # ------------------------------------------------
            # エッジ削除後の Tc
            # ------------------------------------------------

            eigenvalues_error = np.linalg.eigvalsh(
                W_error
            )

            lambda_max_error = eigenvalues_error[-1]

            Tc_error = lambda_max_error


            # ------------------------------------------------
            # Tc の変化量
            # ------------------------------------------------

            delta_Tc = Tc_error - Tc

            # 今回見たいのは影響の大きさ
            abs_delta_Tc = abs(delta_Tc)


            # ------------------------------------------------
            # エッジ両端の次数中心性
            # ------------------------------------------------

            degree_i = degree_centrality[i]
            degree_j = degree_centrality[j]


            # ------------------------------------------------
            # エッジの次数中心性
            #
            # 両端の次数中心性を掛け算
            # ------------------------------------------------

            edge_degree_centrality = (
                degree_i * degree_j
            )


            # ------------------------------------------------
            # 結果を保存
            # ------------------------------------------------

            results.append(
                (
                    i,
                    j,
                    degree_i,
                    degree_j,
                    edge_degree_centrality,
                    Tc_error,
                    delta_Tc,
                    abs_delta_Tc
                )
            )


# ============================================================
# 5. 結果を表示
# ============================================================

print()
print("========================================")
print("エッジごとの次数中心性と Tc の変化")
print("========================================")

print(
    f"{'Edge':<10}"
    f"{'C_i':<12}"
    f"{'C_j':<12}"
    f"{'C_i*C_j':<15}"
    f"{'Delta Tc':<15}"
    f"{'|Delta Tc|':<15}"
)

for result in results:

    (
        i,
        j,
        degree_i,
        degree_j,
        edge_degree_centrality,
        Tc_error,
        delta_Tc,
        abs_delta_Tc
    ) = result

    print(
        f"({i+1},{j+1})"
        f"{degree_i:<12.6f}"
        f"{degree_j:<12.6f}"
        f"{edge_degree_centrality:<15.6f}"
        f"{delta_Tc:<15.6f}"
        f"{abs_delta_Tc:<15.6f}"
    )


# ============================================================
# 6. グラフ用データ
# ============================================================

x_values = np.array([
    result[4]
    for result in results
])

y_values = np.array([
    result[7]
    for result in results
])


# ============================================================
# 7. 次数中心性の積 vs |Delta Tc|
# ============================================================

plt.figure(figsize=(9, 7))

plt.scatter(
    x_values,
    y_values,
    s=60,
    label=r"Edges"
)

plt.xlabel(
    r"$C_i^{\mathrm{degree}} C_j^{\mathrm{degree}}$"
)

plt.ylabel(
    r"$|\Delta T_c|$"
)

plt.title(
    r"Edge Degree Centrality vs. $|\Delta T_c|$"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 8. 画像を保存
# ============================================================

plt.savefig(
    "04_degree_centrality_product_vs_deltaTc.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print(
    "保存完了："
    "04_degree_centrality_product_vs_deltaTc.png"
)


# ============================================================
# 9. 完了
# ============================================================

print()
print("========================================")
print("すべての処理が完了しました")
print("========================================")