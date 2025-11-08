import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vmdpy import VMD
from scipy.stats import entropy
from scipy.optimize import differential_evolution
import antropy as ant


def calculate_sample_entropy(signal, m=2, r=0.2):
    """
    计算样本熵
    m: 模板长度
    r: 容限阈值（通常为0.1-0.25倍的标准差）
    """
    # 标准化信号
    signal = (signal - np.mean(signal)) / np.std(signal)
    return ant.sample_entropy(signal, order=m, metric='chebyshev')


def calculate_permutation_entropy(signal, order=3, delay=1):
    """
    计算排列熵
    order: 排列顺序
    delay: 延迟时间
    """
    return ant.permutation_entropy(signal, order=order, delay=delay, normalize=True)


def vmd_objective_function(params, signal, Fs):
    """
    VMD分解的目标函数
    params: [alpha, K] 参数
    """
    alpha, K = params
    K = int(K)

    # 参数边界检查
    if K < 2 or K > 15:
        return 1e6

    if alpha < 100 or alpha > 5000:
        return 1e6

    try:
        # 执行VMD分解
        u, u_hat, omega = VMD(signal, alpha=alpha, tau=0, K=K, DC=0, init=1, tol=1e-7)

        # 计算每个IMF的熵值
        sample_entropies = []
        permutation_entropies = []

        for i in range(K):
            imf = u[i, :]
            # 样本熵
            sampen = calculate_sample_entropy(imf)
            sample_entropies.append(sampen)

            # 排列熵
            pe = calculate_permutation_entropy(imf)
            permutation_entropies.append(pe)

        # 计算目标函数值
        # 我们希望：高频分量有较高的样本熵，低频分量有较低的样本熵
        # 同时排列熵应该适中，避免过度随机或过度规律

        # 1. 样本熵的梯度惩罚（高频应该高，低频应该低）
        sampen_gradient = np.abs(np.diff(sample_entropies)).mean()

        # 2. 排列熵的稳定性（避免极端值）
        pe_std = np.std(permutation_entropies)

        # 3. 重构误差
        reconstructed = np.sum(u, axis=0)
        reconstruction_error = np.mean((signal - reconstructed) ** 2)

        # 综合目标函数
        objective_value = (
                0.4 * sampen_gradient +  # 样本熵梯度项
                0.3 * pe_std +  # 排列熵稳定性项
                0.3 * reconstruction_error  # 重构误差项
        )

        return objective_value

    except Exception as e:
        print(f"VMD分解失败: {e}")
        return 1e6


def optimize_vmd_parameters(signal, Fs):
    """
    使用差分进化算法优化VMD参数
    """
    print("开始优化VMD参数...")

    # 参数边界: [alpha, K]
    bounds = [
        (500, 3000),  # alpha范围
        (3, 10)  # K范围（整数）
    ]

    # 使用差分进化算法
    result = differential_evolution(
        vmd_objective_function,
        bounds,
        args=(signal, Fs),
        strategy='best1bin',
        maxiter=20,
        popsize=10,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42
    )

    optimal_alpha = result.x[0]
    optimal_K = int(round(result.x[1]))

    print(f"优化完成!")
    print(f"最优参数: alpha={optimal_alpha:.2f}, K={optimal_K}")
    print(f"目标函数值: {result.fun:.6f}")

    return optimal_alpha, optimal_K


def analyze_imf_entropy(u):
    """
    分析各IMF分量的熵特性
    """
    K = u.shape[0]
    entropy_results = []

    for i in range(K):
        imf = u[i, :]

        # 计算各种熵
        sampen = calculate_sample_entropy(imf)
        pe = calculate_permutation_entropy(imf)

        # 判断分量类型
        if sampen > 1.5 and pe > 0.7:
            comp_type = "噪声/高频"
        elif sampen < 0.8 and pe < 0.4:
            comp_type = "趋势/低频"
        else:
            comp_type = "有用信号"

        entropy_results.append({
            'IMF': i + 1,
            'Sample_Entropy': sampen,
            'Permutation_Entropy': pe,
            'Type': comp_type
        })

    return pd.DataFrame(entropy_results)


# 主程序
if __name__ == "__main__":
    # 读取数据
    dataset = pd.read_excel(r'D:\Users\三木\Desktop\beijing.xlsx')
    dataX = dataset['total'].values

    Fs = len(dataX)  # 采样频率
    N = len(dataX)  # 采样点数
    t = np.arange(1, N + 1) / N
    fre_axis = np.linspace(0, Fs / 2, int(N / 2))
    f = dataX

    print("数据基本信息:")
    print(f"数据长度: {N}, 采样频率: {Fs}")

    # 优化VMD参数
    optimal_alpha, optimal_K = optimize_vmd_parameters(f, Fs)

    # 使用最优参数进行VMD分解
    print(f"\n使用最优参数进行VMD分解...")
    u, u_hat, omega = VMD(f, optimal_alpha, 0, optimal_K, 0, 1, 1e-7)

    # 分析IMF的熵特性
    print(f"\n分析IMF分量的熵特性...")
    entropy_df = analyze_imf_entropy(u)
    print(entropy_df)

    # 可视化结果
    plt.figure(figsize=(15, 12))

    # 1. 原始信号和重构信号
    plt.subplot(3, 1, 1)
    plt.plot(t, f, 'b-', label='原始信号', linewidth=1)
    reconstructed = np.sum(u, axis=0)
    plt.plot(t, reconstructed, 'r--', label='重构信号', linewidth=1, alpha=0.8)
    plt.legend()
    plt.title(f'原始信号与重构信号 (alpha={optimal_alpha:.1f}, K={optimal_K})')
    plt.xlabel('时间')
    plt.ylabel('幅值')

    # 2. IMF分量
    plt.subplot(3, 1, 2)
    for i in range(optimal_K):
        plt.plot(t, u[i, :] + i * 0.5, label=f'IMF {i + 1}')
    plt.title('VMD分解结果 - IMF分量')
    plt.xlabel('时间')
    plt.ylabel('幅值')
    plt.legend()

    # 3. 熵分析结果
    plt.subplot(3, 1, 3)
    x_pos = np.arange(optimal_K)
    width = 0.35

    plt.bar(x_pos - width / 2, entropy_df['Sample_Entropy'], width, label='样本熵', alpha=0.7)
    plt.bar(x_pos + width / 2, entropy_df['Permutation_Entropy'], width, label='排列熵', alpha=0.7)

    plt.xlabel('IMF分量')
    plt.ylabel('熵值')
    plt.title('各IMF分量的熵分析')
    plt.xticks(x_pos, [f'IMF{i + 1}' for i in range(optimal_K)])
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 保存结果
    print(f"\n保存分解结果...")
    # 创建包含IMF分量和熵分析结果的DataFrame
    imf_data = {f'IMF_{i + 1}': u[i, :] for i in range(optimal_K)}
    imf_data['Original'] = f
    imf_data['Reconstructed'] = reconstructed

    result_df = pd.DataFrame(imf_data)
    result_df = pd.concat([result_df, dataset], axis=1)

    # 添加参数信息
    result_df.attrs['VMD_alpha'] = optimal_alpha
    result_df.attrs['VMD_K'] = optimal_K

    # 保存到文件
    # result_df.to_excel(r'D:\Users\三木\Desktop\VMD\optimized_decomposition.xlsx', index=False)

    print("处理完成!")

    # 显示最优分解的详细信息
    print(f"\n=== 最优分解方案总结 ===")
    print(f"惩罚系数 alpha: {optimal_alpha:.2f}")
    print(f"模态个数 K: {optimal_K}")
    print(f"\n各分量类型分析:")
    for _, row in entropy_df.iterrows():
        print(f"IMF {row['IMF']}: 样本熵={row['Sample_Entropy']:.3f}, "
              f"排列熵={row['Permutation_Entropy']:.3f}, 类型={row['Type']}")