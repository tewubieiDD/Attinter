# Kobler 等（2022）SPDDSMBN 论文研读笔记

> 论文：Reinmar J. Kobler et al., *SPD Domain-Specific Batch Normalization to Crack Interpretable Unsupervised Domain Adaptation in EEG*, NeurIPS 2022  
> 主题：SPD 流形、切空间映射、域特定批归一化、EEG 无监督域适应  
> 本文档根据论文正文、补充材料及作者官方代码 `rkobler/TSMNet` 整理。

## 1. 论文解决什么问题

EEG 数据在不同受试者和不同采集时段之间存在显著域偏移。即使执行相同任务，电极位置、阻抗、精神状态、脑结构和脑功能差异也会改变 EEG 分布。因此，在旧 session 或旧 subject 上训练的模型，通常不能直接泛化到新域。

论文希望实现：

- 使用多个带标签源域训练模型；
- 面对未见过的目标域时，不使用目标域标签；
- 仅根据目标域 EEG 特征完成无监督适应；
- 保持切空间模型的准确性和神经生理可解释性。

作者提出 SPD domain-specific momentum batch normalization（SPDDSMBN），并将其用于 TSMNet。其核心思想是：

> 为每个域分别估计 SPD 特征分布的 Fréchet 均值和方差，将不同域变换到共享的几何中心和尺度，再用共享切空间分类器分类。

完整数据流为：

\[
X\rightarrow f_\theta(X)\in\mathcal S_D^+
\rightarrow\operatorname{SPDDSMBN}
\rightarrow\operatorname{LogEig}
\rightarrow\text{Linear classifier}
\rightarrow\hat y.
\]

---

## 2. 多源、多目标无监督域适应

### 2.1 什么是域

在本文中，一个域通常对应某个受试者的一次 session。例如：

| 域 | 含义 |
|---|---|
| 域 1 | Subject 1, Session 1 |
| 域 2 | Subject 1, Session 2 |
| 域 3 | Subject 2, Session 1 |

第 \(i\) 个域记为：

\[
\mathcal T_i=\{(X_{ij},y_{ij})\}_{j=1}^{M}\sim P^i_{XY}.
\]

其中 \(i\) 是域编号，\(j\) 是域内样本编号，\(P^i_{XY}\) 是该域的联合分布。

### 2.2 多源和多目标

带标签源域集合为：

\[
\mathcal T^{\mathrm{source}}
=\{\mathcal T_i\mid i\in\mathcal I_d^{\mathrm{source}}\}.
\]

未见目标域集合为：

\[
\mathcal T^{\mathrm{target}}
=\{\mathcal T_l\mid l\in\mathcal I_d^{\mathrm{target}}\},
\qquad
\mathcal I_d^{\mathrm{target}}\cap
\mathcal I_d^{\mathrm{source}}=\varnothing.
\]

“多源”表示从多个带标签域学习；“多目标”表示模型可能依次适应多个新域，每个目标域分别维护自己的统计量。

### 2.3 为什么叫无监督适应

目标域适应只能使用：

\[
\{X_{lj}\}_{j=1}^{M},
\]

不能使用 \(\{y_{lj}\}\)。源域训练仍是有监督的。因此“无监督”只修饰目标域适应阶段，不表示整个学习过程没有标签。

最终预测函数写为：

\[
h:\mathcal X\times\mathcal I_d\rightarrow\mathcal Y,
\qquad \hat y=h(X,i).
\]

输入域编号 \(i\) 是为了选择该域对应的 SPDDSMBN 统计量。

### 2.4 类别先验假设

论文允许不同域的联合分布不同：

\[
P^i_{XY}\ne P^k_{XY},
\]

但假设类别先验相同：

\[
P_Y^i=P_Y.
\]

如果目标域类别比例严重变化，无标签总体均值的差异可能来自标签比例，而非纯粹域偏移；此时总体分布对齐可能损害分类。

---

## 3. SPD 流形预备知识

### 3.1 SPD 矩阵空间

实对称正定矩阵集合为：

\[
\mathcal S_D^+=\{Z\in\mathbb R^{D\times D}:Z^\top=Z,\ Z\succ0\}.
\]

协方差矩阵天然属于该空间。它不是普通欧氏向量空间，而是具有曲率的黎曼流形。

### 3.2 仿射不变黎曼距离

论文使用 affine-invariant Riemannian metric（AIRM）：

\[
\delta(Z_1,Z_2)
=\left\|\log\left(Z_1^{-1/2}Z_2Z_1^{-1/2}\right)\right\|_F.
\]

其重要性质是：对于任意可逆矩阵 \(A\)，

\[
\delta(AZ_1A^\top,AZ_2A^\top)=\delta(Z_1,Z_2).
\]

这对 EEG 很有意义，因为不同域之间的一部分变化可表示成潜在源的不同线性混合。

### 3.3 对数映射和指数映射

\[
\operatorname{Log}_Z(Z_1)
=Z^{1/2}\log(Z^{-1/2}Z_1Z^{-1/2})Z^{1/2},
\]

\[
\operatorname{Exp}_Z(S)
=Z^{1/2}\exp(Z^{-1/2}SZ^{-1/2})Z^{1/2}.
\]

对数映射将流形上的点投影到参考点 \(Z\) 的切空间；指数映射执行逆过程。

### 3.4 Fréchet 均值和方差

一组 SPD 矩阵 \(\{Z_j\}_{j=1}^M\) 的 Fréchet 均值定义为：

\[
G_Z=\arg\min_{G\in\mathcal S_D^+}
\frac1M\sum_{j=1}^M\delta^2(G,Z_j).
\]

它是流形上“到所有样本的平方距离之和最小”的中心。Fréchet 方差为：

\[
\nu_Z^2=\frac1M\sum_{j=1}^M\delta^2(G_Z,Z_j).
\]

两个 SPD 点的加权测地线均值为：

\[
Z_1\#_\gamma Z_2
=Z_1^{1/2}
\left(Z_1^{-1/2}Z_2Z_1^{-1/2}\right)^\gamma
Z_1^{1/2}.
\]

它相当于从 \(Z_1\) 沿最短测地线向 \(Z_2\) 移动比例 \(\gamma\)。

---

## 4. 普通批归一化中的统计量

### 4.1 批统计量

第 \(k\) 个 minibatch 记为 \(\mathcal B_k=\{x_{k,j}\}_{j=1}^M\)。当前批均值和方差为：

\[
b_k=\frac1M\sum_jx_{k,j},
\qquad
s_k^2=\frac1M\sum_j(x_{k,j}-b_k)^2.
\]

它们只描述当前小批次，因此具有采样噪声。batch 越小，通常波动越大。

### 4.2 运行统计量（running statistics）

“Running estimate”应理解为运行估计值或滑动估计值，不是“运行评估”。它通过多个历史 batch 累积：

\[
g_k=(1-\gamma)g_{k-1}+\gamma b_k,
\]

\[
\sigma_k^2=(1-\gamma)\sigma_{k-1}^2+\gamma s_k^2.
\]

三类统计量的区别如下：

| 概念 | 数据来源 | 特点 |
|---|---|---|
| 批统计量 \(b_k,s_k^2\) | 当前 minibatch | 有噪声、变化快 |
| 运行统计量 \(g_k,\sigma_k^2\) | 历史多个 batch | 平滑、较稳定 |
| 数据集真实统计量 | 完整总体 | 希望估计的目标 |

### 4.3 可学习参数不是运行统计量

普通 BN 的输出为：

\[
\widetilde x
=\sigma_\phi\frac{x-b_k}{\sqrt{s_k^2+\varepsilon}}+g_\phi.
\]

- \(g_\phi,\sigma_\phi\)：通过反向传播学习的模型参数；
- \(g_k,\sigma_k^2\)：通过统计更新的状态，不由优化器学习。

### 4.4 普通 BN 与 MBN

普通 BN 通常在训练时使用当前批统计量归一化，在测试时使用运行统计量。Momentum BN（MBN）在训练和测试时都使用运行统计量，并维护两套状态：

- 训练运行统计量 \((\bar g_k,\bar\sigma_k^2)\)，使用衰减的 \(\gamma_{\rm train}(k)\)；
- 测试运行统计量 \((\widetilde g_k,\widetilde\sigma_k^2)\)，使用固定动量 \(\gamma\)。

早期网络特征变化快，训练动量应较大以快速跟踪；后期网络稳定，动量减小以降低统计噪声。

---

## 5. RBN、SPDBN、SPDMBN 与 SPDDSMBN

### 5.1 Riemannian Batch Normalization（RBN）

RBN 用一次 Karcher flow 近似当前批 Fréchet 均值 \(B_k\)，然后把当前 batch 从 \(B_k\) 附近搬运到可学习均值 \(G_\phi\) 附近：

\[
\operatorname{RBN}(Z_j)
=\Gamma_{B_k\rightarrow G_\phi}(Z_j).
\]

它直接使用当前批均值，因此小 batch 时噪声较大。

### 5.2 SPDBN

SPDBN 同时控制 Fréchet 均值和方差：

\[
\operatorname{SPDBN}(Z_j)
=\Gamma_{I\rightarrow G_\phi}
\left(\Gamma_{G_k\rightarrow I}(Z_j)\right)^{
\frac{\nu_\phi}{\nu_k+\varepsilon}}.
\]

含义是：

1. 从运行均值 \(G_k\) 搬到单位矩阵 \(I\) 附近；
2. 用矩阵幂调整离散尺度；
3. 再搬到目标均值 \(G_\phi\) 附近。

### 5.3 SPDMBN

SPDMBN 将 MBN 推广到 SPD 流形。运行均值使用测地线更新：

\[
G_k=G_{k-1}\#_\gamma B_k.
\]

运行方差使用类似指数平滑更新。它同样保留训练和测试两套统计量，并让训练动量随训练过程衰减。

### 5.4 SPDDSMBN

Domain-specific BN 的思想是每个域有自己的归一化统计量。SPDDSMBN 为每个域 \(i\) 保留独立 SPDMBN：

\[
\operatorname{SPDDSMBN}(Z_j,i)
=\operatorname{SPDMBN}_i(Z_j).
\]

每个域具有独立输入统计量：

\[
G_i,\quad \nu_i^2,
\]

但 TSMNet 让所有域共享输出目标：

\[
G_{\phi i}=I,
\qquad
\nu_{\phi i}=\nu_\phi.
\]

所以不同域被对齐到相同中心和尺度。

---

## 6. 为什么 SPDMBN 的运行均值能够工作

### 6.1 三个容易混淆的均值

第 \(k\) 步的特征提取器参数为 \(\theta_k\)，完整数据集的 SPD 表示为：

\[
\mathcal Z_{\theta_k}=\{f_{\theta_k}(x):x\in\mathcal T\}.
\]

需要区分：

| 记号 | 含义 |
|---|---|
| \(G_{\theta_k}\) | 当前网络下完整数据集的真实 Fréchet 均值 |
| \(B_k\) | 当前随机 minibatch 的近似 Fréchet 均值 |
| \(G_k\) | SPDMBN 融合历史 batch 后的运行均值 |

目标是让 \(G_k\) 跟踪 \(G_{\theta_k}\)。困难在于训练时 \(\theta_k\) 不断变化，因此真实目标本身也在移动。

### 6.2 误差的定义

论文写作中的“方差”在这里可理解为均值估计的期望平方误差：

\[
\operatorname{Var}_{\theta_k}(B_k)
=\mathbb E\left[\delta^2(B_k,G_{\theta_k})\right],
\]

\[
\operatorname{Var}_{\theta_k}(G_k)
=\mathbb E\left[\delta^2(G_k,G_{\theta_k})\right].
\]

### 6.3 假设（12）

论文假设：

\[
\operatorname{Var}_{\theta_k}(B_{k-1})
\le
\left(1+\|\theta_k-\theta_{k-1}\|\right)
\operatorname{Var}_{\theta_k}(B_k).
\]

直观含义是：网络参数和特征分布逐渐变化，上一个 batch 均值不会在一步之后突然完全过时。作者认为光滑特征提取器和较小学习率有助于满足该假设，但没有证明它必然成立。

### 6.4 Proposition 1

在相应假设和参数更新约束下：

\[
\operatorname{Var}_{\theta_k}(G_k)
\le
\operatorname{Var}_{\theta_k}(B_k).
\]

即运行均值相对真实 Fréchet 均值的平均误差，不大于单个 batch 均值的误差。所需参数更新条件为：

\[
\|\theta_k-\theta_{k-1}\|
\le
\frac{1-\gamma^2}{(1-\gamma)^2}-1
=\frac{2\gamma}{1-\gamma}.
\]

所以：

- \(\gamma\) 大时，运行均值跟随快，可容许网络快速变化；
- \(\gamma\) 小时，运行均值变化慢，网络参数也必须趋于稳定。

### 6.5 Proposition 与真正收敛的区别

Proposition 1 证明的是误差上界和跟踪能力，并没有直接证明 \(G_k\) 已经收敛。

Remark 1 进一步假设训练后期：

\[
\|\theta_k-\theta^*\|\le\rho,
\]

且特征提取器关于参数足够光滑：

\[
\delta(f_\theta(x),f_{\widetilde\theta}(x))
\le L\|\theta-\widetilde\theta\|.
\]

当 \(\rho L\) 相比数据方差可忽略时，可以认为后期特征分布基本固定。若动量适当衰减，SPD 流形上的大数定律给出：

\[
G_k\xrightarrow{P}G_{\theta^*}.
\]

准确表述应是：SPDMBN 在合理条件下稳定跟踪移动的 Fréchet 均值，并在网络稳定、动量适当衰减时依概率收敛。

---

## 7. SPDMBN 如何学习 Fréchet 均值处的切空间映射

### 7.1 传统 TSM

传统切空间映射为：

\[
P_{G_\mathcal T}(Z)
=\operatorname{upper}\circ
\Gamma_{G_\mathcal T\rightarrow I}\circ
\operatorname{Log}_{G_\mathcal T}(Z),
\]

可化简为：

\[
P_{G_\mathcal T}(Z)
=\operatorname{upper}
\left[
\log(G_\mathcal T^{-1/2}ZG_\mathcal T^{-1/2})
\right].
\]

它做了三件事：

1. 以完整数据集的 Fréchet 均值为参考点；
2. 把均值映射到欧氏原点；
3. 提取对称矩阵的上三角元素，得到 \(D(D+1)/2\) 维向量。

非对角元素乘 \(\sqrt2\)，从而保持 Frobenius 范数。

### 7.2 端到端训练的困难

若 \(Z=f_\theta(X)\)，网络每次更新都会改变 SPD 特征及其 Fréchet 均值。传统方法无法在训练前固定 \(G_\mathcal T\)，而每一步用完整数据集重算又很昂贵。

SPDMBN 使用运行均值 \(G_k\) 跟踪当前 Fréchet 均值，于是可将 TSM 嵌入网络。

### 7.3 SPDMBN + LogEig

作者令 SPDMBN 的目标均值固定为 \(G_\phi=I\)，并定义：

\[
m_\phi=\operatorname{LogEig}\circ\operatorname{SPDMBN}.
\]

得到：

\[
m_\phi(Z)
=\operatorname{upper}
\left[
\frac{\nu_\phi}{\nu_k+\varepsilon}
\log(G_k^{-1/2}ZG_k^{-1/2})
\right].
\]

当 \(G_k\to G_\mathcal T\) 时，它收敛到传统 TSM 的一个缩放版本：

\[
m_\phi(Z)
\rightarrow
\frac{\nu_\phi}{\nu_\mathcal T+\varepsilon}
P_{G_\mathcal T}(Z).
\]

其中 \(\nu_k\) 是数据统计得到的运行 Fréchet 标准差，\(\nu_\phi\) 是反向传播学习的输出尺度。

### 7.4 为什么不直接 LogEig

直接 \(\operatorname{LogEig}(Z)=\operatorname{upper}(\log Z)\) 等于固定在单位矩阵处映射，对应 log-Euclidean 几何，一般不具有 AIRM 的完整仿射不变性。

先在数据 Fréchet 均值处中心化，再进行 LogEig，可让切空间距离局部近似 AIRM 距离，更适合含线性混合变化的 EEG 数据。

---

## 8. 第四章：SPDDSMBN 如何用于 EEG

### 8.1 EEG 生成模型

论文使用：

\[
X_{ij}=A_iS_{ij}+N_{ij}.
\]

- \(X_{ij}\in\mathbb R^{P\times T}\)：电极观测；
- \(S_{ij}\in\mathbb R^{Q\times T}\)：潜在脑源活动；
- \(A_i\in\mathbb R^{P\times Q}\)：域特定混合矩阵；
- \(N_{ij}\)：噪声。

跨 subject 时，脑结构、头部形状和功能网络不同；跨 session 时，电极位置、阻抗和精神状态变化。因此 \(A_i\) 和信号统计量随域变化。

忽略噪声时：

\[
\operatorname{Cov}(X_{ij})
\approx A_i\operatorname{Cov}(S_{ij})A_i^\top.
\]

若源近似不相关，源协方差对角线就是各脑源功率，所以电极协方差编码了源功率、相关性和空间投影。

### 8.2 标签与源功率

论文假设标签与若干判别源的对数功率线性相关：

\[
y_{ij}
=\sum_{k=1}^{K}b_k
\log\left(\operatorname{Var}\{s_{ij}^{(k)}(t)\}\right)
+\varepsilon_{ij}.
\]

这解释了为什么“协方差 + 矩阵对数 + 线性分类器”适合 EEG：在理想源空间中，矩阵对数可把源功率变成对数功率，线性分类器对应上述生成关系。

### 8.3 内在可解释性

模型结构本身被限制为线性滤波、协方差、切空间映射和线性分类器。因此可从模型参数反推出：

- 判别脑源的空间头皮模式；
- 判别频段；
- 每个源对类别的贡献。

源贡献定义为：

\[
c_k=\max(\lambda_k,\lambda_k^{-1}),
\qquad
\lambda_k=\exp\left(\frac{b_k}{\|b\|_2^2}\right).
\]

无论源功率与类别正相关还是负相关，\(c_k\) 都反映其判别强度。

### 8.4 TSMNet 结构

\[
X
\rightarrow\text{TempConv}
\rightarrow\text{SpatConv}
\rightarrow\text{CovPool}
\rightarrow\text{BiMap}
\rightarrow\text{ReEig}
\rightarrow\text{SPDDSMBN}
\rightarrow\text{LogEig}
\rightarrow\text{Linear}.
\]

| 模块 | 作用 |
|---|---|
| TempConv | 学习时间/频率滤波器 |
| SpatConv | 学习空间-频谱成分 |
| CovPool | 沿时间计算协方差，得到 SPD 表示 |
| BiMap | \(W^\top ZW\)，将 \(40\times40\) 降至 \(20\times20\) |
| ReEig | 截断过小特征值，保证数值正定 |
| SPDDSMBN | 按域消除 SPD 中心和尺度差异 |
| LogEig | 矩阵对数并向量化，输出 210 维特征 |
| Linear | 共享线性分类器 |

模型可写为：

\[
h=g_\psi\circ m_\phi\circ f_\theta.
\]

- \(f_\theta\)：所有域共享的线性特征提取器和协方差表示；
- \(m_\phi\)：使用域统计量的切空间映射；
- \(g_\psi\)：所有域共享的线性分类器。

域特定统计差异由 \(m_\phi\) 消除，跨域判别规律由共享的 \(f_\theta,g_\psi\) 学习。

---

## 9. Inter-session UDA 的正确数据划分

假设有 3 个 subject，每个 subject 有 2 个 session。论文代码对每个 subject 独立做 inter-session 评估。

以 Subject 1 为例，某一 fold 中：

\[
S_1=\text{source},
\qquad
S_2=\text{target}.
\]

正确划分为：

\[
S_1
\rightarrow
\begin{cases}
80\% & \text{source training},\\
20\% & \text{source validation},
\end{cases}
\qquad
S_2=\text{完整 target adaptation/test domain}.
\]

| 数据 | 标签用途 | 用途 |
|---|---|---|
| Source 80% | 可见 | 梯度训练 |
| Source 20% | 可见 | 验证、选最佳 epoch |
| Target 全部特征 | 不使用标签 | 估计目标域统计量并预测 |
| Target 标签 | 最后才访问 | 计算 balanced accuracy |

不能把 source 和 target 合并后再按 3:1 划分。那会混淆外层域留出、内层模型选择和最终测试，并可能引入目标标签泄漏。

官方代码使用以 session 为 group 的 `GroupKFold`。当只有两个 session 时会形成两个方向的 fold：

\[
S_1\rightarrow S_2,
\qquad
S_2\rightarrow S_1.
\]

如果实际部署只允许按时间从过去预测未来，也可以只做早期 session 到后期 session，但这与作者的双向留一 session 交叉验证应明确区分。

---

## 10. Target 无监督适应、预测与评估

### 10.1 无监督适应

训练完成并选定最佳模型后，固定 \(\theta,\phi,\psi\)。对完整目标域特征：

\[
Z_{tj}=f_\theta(X_{tj}),
\]

计算：

\[
G_t=\operatorname{Fr\acute echetMean}\{Z_{tj}\},
\]

\[
\nu_t^2=\frac1M\sum_j\delta^2(G_t,Z_{tj}).
\]

该阶段只使用 \(X_t\)，不需要 \(y_t\)，也不更新网络权重。

### 10.2 预测

使用目标域统计量归一化：

\[
\widetilde Z_{tj}
=\left(G_t^{-1/2}Z_{tj}G_t^{-1/2}\right)^{
\frac{\nu_\phi}{\nu_t+\varepsilon}},
\]

随后：

\[
v_{tj}=\operatorname{upper}(\log\widetilde Z_{tj}),
\qquad
\hat y_{tj}=\arg\max g_\psi(v_{tj}).
\]

### 10.3 最终评估

第四步已经产生预测。评估不是再次训练或预测，而是解封真实标签并计算指标：

\[
\operatorname{BalancedAccuracy}
=\frac1C\sum_{c=1}^{C}\operatorname{Recall}_c.
\]

完整关系为：

\[
\underbrace{X_t\rightarrow(G_t,\nu_t)}_{\text{适应}}
\rightarrow
\underbrace{(X_t,G_t,\nu_t)\rightarrow\hat y_t}_{\text{预测}}
\rightarrow
\underbrace{(\hat y_t,y_t)\rightarrow\mathrm{BA}}_{\text{评估}}.
\]

这是离线、传导式 UDA：同一批目标域无标签特征既用于估计域统计量，也用于最终预测。它不使用目标标签，但它不同于严格在线或归纳式测试。

---

## 11. 官方代码中的 REFIT 与 BUFFER

这两个词是代码的测试统计量模式，不是论文正式定义的两种算法。

| 模式 | 是否重算统计量 | 用途 |
|---|---:|---|
| `REFIT` | 是 | 用当前完整域重算 Fréchet 均值和方差 |
| `BUFFER` | 否 | 使用模型中已保存的统计量正式预测 |

### 11.1 Buffer 是什么

PyTorch buffer 是保存在模型状态中、但不由梯度优化的张量。代码保存：

- `running_mean`, `running_var`；
- `running_mean_test`, `running_var_test`。

它们会随模型保存和加载，但不是卷积权重或分类器参数。

### 11.2 REFIT

当测试模式设为 `REFIT`，下一次 forward 调用 `initrunningstats(X)`，对传入整域 SPD 特征运行 Karcher flow，并保存：

\[
G_i,\quad\nu_i^2.
\]

它不是重新训练网络，也没有目标标签损失或反向传播。

### 11.3 BUFFER

`BUFFER` 不重算统计量，直接读取已经保存的 `running_mean_test` 和 `running_var_test` 完成归一化及预测。

作者代码的目标域流程是：

\[
X_t
\xrightarrow{\texttt{REFIT}}
\text{计算并保存 }G_t,\nu_t^2
\xrightarrow{\texttt{BUFFER}}
\text{固定统计量正式预测}.
\]

论文中“完整域可用时直接求 Fréchet 均值和方差”对应 `REFIT`；适应后固定这些统计量推理对应 `BUFFER`。论文还提到在线场景可用 Algorithm 1 的动量更新，但官方 SPD 代码中的 `ADAPT` 尚未实现。

---

## 12. 官方代码对训练流程的验证

作者官方代码位于本地：[`TSMNet`](./TSMNet/README.md)。关键事实如下。

### 12.1 外层按 session 留域

[`experiments/main.py`](./TSMNet/experiments/main.py) 在 inter-session 模式中逐 subject 处理，并使用 `GroupKFold` 按整个 session 划分 source 与 target。因此同一 session 不会同时出现在两侧。

### 12.2 TSMNet 的 target 不进入梯度训练

只有 `DomainAdaptJointTrainableModel`（例如某些 DANN 模型）会把无标签 target 加入联合训练。TSMNet 是 `DomainAdaptFineTuneableModel`，因此其训练索引仍为 source-only。

### 12.3 验证集只从 source 划分

配置 `validation_size: 0.2`。代码在 source `train` 索引内部使用 `StratifiedShuffleSplit`，按“域 + 类别”组合分层。对于单一 source session，这相当于按类别分层。

最佳模型由 `valid_loss_best` 选择，不使用 target 标签或 target 分数。

### 12.4 Target 适应不使用标签

`TSMNet.domainadapt_finetune` 的签名虽然接收 `y`，但函数体没有读取它。函数只在 `torch.no_grad()` 下按域 forward，并用 `REFIT` 重算统计量。

这是无标签适应，但接口设计不够严格；更稳妥的接口应只接收 `x,d`，从结构上杜绝标签误用。

### 12.5 预测后才计算测试分数

适应完成后，代码切回 `BUFFER`，对每个域 forward 得到 `argmax` 预测，最后解除标签屏蔽并计算 target balanced accuracy。

---

## 13. 实验结果和消融结论

论文在多个运动想象和工作负荷 EEG 数据集上进行 inter-session 与 inter-subject UDA。主要结论是：

- TSMNet 在多数设置中达到或接近最优结果；
- inter-session 场景可达到或超过有监督域特定参考方法；
- inter-subject 场景随训练 subject 数量增大而改善；
- 学到的判别源在频谱和头皮分布上具有神经生理合理性。

消融实验表明，成功主要来自：

1. 使用 SPD 表示和流形几何；
2. 在 SPD 流形上做 domain-specific BN；
3. SPDMBN 的自适应动量进一步带来较小但显著的收益。

因此不能把论文贡献简化成“普通 BN 换成流形 BN”。关键组合是：SPD 表示、按域独立统计、运行统计跟踪、Fréchet 均值处 TSM，以及共享线性判别结构。

---

## 14. 方法的边界与批判性理解

### 14.1 理论并非无条件收敛保证

误差界依赖假设（12）与参数更新约束；真正收敛还依赖训练后期特征分布基本固定及动量条件。它说明方法在合理条件下为何稳定，不是对任意网络、学习率和数据分布的无条件保证。

### 14.2 主要对齐总体中心和尺度

SPDDSMBN 对齐每个域的总体 Fréchet 均值和方差，并不显式对齐类条件分布。若标签先验不同、类别结构变化或条件偏移很强，强制总体对齐可能失败。

### 14.3 离线 UDA 使用完整目标域

论文实验在预测前使用完整 target session 计算统计量。这适用于离线、传导式 UDA，但不能直接等同于逐样本在线部署。在线场景需要因果更新策略、冷启动方案和防止早期统计不稳定的机制。

### 14.4 计算复杂度

特征值分解、矩阵幂、矩阵对数和 Karcher flow 的复杂度随 SPD 维度快速增加，因此高维连接矩阵上的应用成本较高。

### 14.5 数值稳定性

实现时必须关注：

- 协方差正则化；
- 小特征值截断（ReEig）；
- 特征值重复时的梯度稳定性；
- 双精度与设备转换；
- batch 内每个域的有效样本数；
- Karcher flow 的收敛阈值和迭代次数。

---

## 15. 核心概念速查

| 概念 | 一句话解释 |
|---|---|
| SPD 矩阵 | 对称正定矩阵，协方差的自然表示 |
| Fréchet 均值 | 流形上最小化总体平方距离的中心 |
| Fréchet 方差 | 数据到 Fréchet 均值的平均平方距离 |
| 批统计量 | 当前 minibatch 计算的均值和方差 |
| 运行统计量 | 跨历史 batch 平滑累积的统计量 |
| RBN | 使用当前 SPD 批均值进行流形归一化 |
| SPDMBN | 使用自适应动量运行统计量的 SPD BN |
| DSBN | 每个域维护独立 BN 统计量 |
| SPDDSMBN | 每个域维护独立 SPDMBN 统计量 |
| TSM | 在 Fréchet 均值处把 SPD 点映射到欧氏切空间 |
| LogEig | 矩阵对数后保范数向量化 |
| REFIT | 用当前完整域重算并保存测试统计量 |
| BUFFER | 使用已保存测试统计量，不再重算 |
| UDA | 源域有标签、目标域适应不使用标签 |

---

## 16. 最终总结

SPDDSMBN 的本质不是简单地“给每个域做标准化”，而是在 SPD 流形上建立域特定坐标系：

\[
Z_{ij}
\rightarrow
(G_i)^{-1/2}Z_{ij}(G_i)^{-1/2}
\rightarrow
\text{统一 Fréchet 尺度}
\rightarrow
\log
\rightarrow
\text{共享欧氏分类器}.
\]

其中：

- \(G_i,\nu_i\) 描述第 \(i\) 个域的位置和尺度；
- SPDMBN 使这些统计量可在端到端训练中跟踪变化的特征分布；
- domain-specific 机制使新目标域只需无标签统计量即可适应；
- LogEig 将对齐后的 SPD 特征变成近似对数源功率特征；
- 线性结构使模型能够反推出判别脑源、频谱与空间模式。

从训练到测试的完整流程是：

\[
\boxed{
\text{按域留出 target}
\rightarrow
\text{source 内部训练/验证}
\rightarrow
\text{固定最佳模型}
\rightarrow
\text{target REFIT 统计量}
\rightarrow
\text{BUFFER 预测}
\rightarrow
\text{解封标签计算 BA}
}
\]

这条链路同时解释了论文的数学方法、代码实现和实验协议。
