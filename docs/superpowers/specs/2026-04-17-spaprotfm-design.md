# SpaProtFM 项目设计文档

**日期**：2026-04-17
**状态**：草稿 v1（待用户审阅）
**目标产出**：发表 *Briefings in Bioinformatics* 级别方法学论文 + 开源代码
**预计周期**：6 个月（v1 投稿）

> **临时代号**：`SpaProtFM` (Spatial Proteomics with Foundation Models)
> 正式定名前必须做 GitHub / PyPI / Scholar 查重。

---

## 1. 项目概述

构建一个**多模态生成式深度学习框架**，用于**空间蛋白质组（CODEX/IMC/t-CyCIF）的 panel 扩展**：给定一张组织切片只测了 ~20 个蛋白通道，结合同切片 H&E 病理图像和同器官的 scRNA-seq 参考，**生成全 panel（~50+ 蛋白）的像素级图像**。

**为什么做**：
- 空间蛋白质组被 *Nature Methods* 评为 **2024 Method of the Year**，方向高热度
- 现有 imputation 方法只有 3 个（7-UP / Murphy U-Net / Kirchgaessner），全部用老架构
- 三大空白：①无人用 transformer/diffusion；②无人做 H&E + scRNA-seq + marker 三模态条件；③跨数据集泛化都很差

---

## 2. 问题定义

### 2.1 形式化

**输入**：
- $X_{obs} \in \mathbb{R}^{H \times W \times C_{obs}}$：观测到的 $C_{obs}$ 个 marker 通道（典型值 $C_{obs}=20$）
- $I_{HE} \in \mathbb{R}^{H \times W \times 3}$：同切片的 H&E 病理图像
- $S_{ref} \in \mathbb{R}^{N \times G}$：同器官的 scRNA-seq 参考（$N$ 个细胞 × $G$ 个基因）
- 元数据：marker 名字、organ、染色平台

**输出**：
- $X_{pred} \in \mathbb{R}^{H \times W \times C_{full}}$：全 panel 预测（$C_{full} \approx 50$）
- 不确定性图 $\sigma \in \mathbb{R}^{H \times W \times C_{full}}$（可选但加分）

### 2.2 任务变体

| 任务 | 协议 |
|---|---|
| **A. Within-sample**：同一张图随机 mask 部分 marker → 重建 | 自监督训练 + 评估 |
| **B. Cross-sample**：训练集和测试集是不同样本，但同器官 | 评估泛化 |
| **C. Cross-dataset**：训练于 HuBMAP（健康），测试于 HTAN（肿瘤） | 攻击 7-UP 短板 |
| **D. Joint panel design**（可选 stretch goal）：同时输出"建议测哪 20 个" | 实用价值高 |

主打 **A + B + C**，D 作 stretch。

---

## 3. 相关工作 & 差异化

### 3.1 三个直接竞品

| 方法 | 架构 | 输入 | 跨数据集 | 关键弱点 |
|---|---|---|---|---|
| 7-UP (PNAS Nexus 2023) | ResNet50 + XGBoost | 7 markers + cell morph | PCC 0.534→0.394 | 老架构、无 H&E、无 scRNA |
| Murphy U-Net (Bioinformatics 2024) | U-Net + 图论选 panel | markers only | 部分 | 单模态，全监督 |
| Kirchgaessner (Nat Commun 2025) | LR/GBT/AE 三选一 | markers + 空间特征 | 仅乳腺癌 | 数据小、无生成式 |

### 3.2 我们的差异化（"为什么 reviewer 不会拒"）

1. **首个用 diffusion + foundation model 做空间蛋白质组 imputation** —— 架构维度直接领先
2. **首个三模态条件**（H&E + scRNA-seq + observed markers）—— 任务定义维度新颖
3. **明确针对跨数据集泛化** —— 解决 7-UP 最大短板
4. **天然支持任意 panel 配置**（masked-modeling 范式） —— 实用性强

---

## 4. 数据集

### 4.1 训练 / 验证（2026-04-17 更新：HuBMAP raw CODEX 因数据规模问题降级为 stretch）

| 数据集 | 平台 | 组织 | Markers | 样本量 | 大小 | 来源 | 用途 |
|---|---|---|---|---|---|---|---|
| **Damond 2019** | IMC | 胰腺 | 38 | 100 张 / 252k 细胞 | ~2 GB | OSN 镜像 (HTTPS) | 训练主力 |
| **Jackson 2020** | IMC | 乳腺癌 | 42 | 100 张 / 286k 细胞 | ~2 GB | OSN 镜像 (HTTPS) | 训练 |
| **HochSchulz 2022** | IMC | 黑色素瘤 | 41 | 50 张 / 326k 细胞 | ~1 GB | OSN 镜像 (HTTPS) | 训练 |
| **Murphy CODEX 脾** | CODEX | 脾 | 29 | 8 张 OME-TIFF | ~42 GB | HuBMAP asset (wget) | 跨平台验证 |
| **Murphy CODEX 淋巴结** | CODEX | 淋巴结 | 29-35 | 9 张 OME-TIFF | ~42 GB | HuBMAP asset (wget) | 跨平台验证 |
| ~~HuBMAP raw CODEX 大/小肠~~ | ~~CODEX~~ | ~~肠道~~ | ~~46~~ | ~~16+16~~ | ~~500-750 GB/数据集~~ | ~~需 cytokit 拼接~~ | **降级为 stretch** |

**论文定位升级**：从"CODEX panel 扩展"改为"**跨平台空间蛋白质组（CODEX + IMC）panel 扩展**"——主流 baseline 全是单平台，多平台是新差异化点。

### 4.2 测试（zero-shot 跨数据集）

| 数据集 | 平台 | 组织 | 用途 |
|---|---|---|---|
| HTAN 胰腺 | IMC | 胰腺癌 | 跨平台 + 跨疾病泛化测试 |
| OPTIMAL 扁桃体 | IMC | 扁桃体（27 markers, 12 batch）| 跨批次稳定性测试 |

### 4.3 辅助参考数据

- **scRNA-seq 参考**：Tabula Sapiens / Human Cell Atlas（按器官选）
- **H&E 配对**：HuBMAP 的 CODEX 数据多数自带配准 H&E

### 4.4 数据获取风险

- **配对 H&E 不一定每张都有** → fallback：用 DAPI 通道或 nuclear morphology 替代
- **marker 命名不统一** → 需要建立 marker 标准化字典（pre-processing 工作量约 1 周）

---

## 5. 方法设计

### 5.1 架构总览

```
                  H&E image                 scRNA-seq reference
                      │                              │
              ┌───────▼───────┐              ┌───────▼───────┐
              │ Histology FM  │              │ scRNA Encoder │
              │ (UNI/Phikon,  │              │ (frozen, e.g. │
              │   frozen)     │              │  scGPT or MLP)│
              └───────┬───────┘              └───────┬───────┘
                      │ patch embeddings              │ organ profile
                      └──────────┬───────────────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │    Cross-attention block    │
                  └──────────────┬──────────────┘
                                 │ context c
              observed markers   │
              + mask token  ─────┤
                                 ▼
                  ┌──────────────────────────────┐
                  │  Conditional Diffusion U-Net │
                  │  (predicts ε for masked      │
                  │   marker channels)           │
                  └──────────────┬───────────────┘
                                 ▼
                       full panel prediction
                       + uncertainty (DDIM variance)
```

### 5.2 核心组件

**(a) Histology Foundation Model 编码器**
- 候选：**UNI** (MIT, Mahmood lab)、**Phikon** (Owkin)、**Virchow** (Paige)
- 全部冻结，只做 inference → 无需训练，省算力
- 输出：每个 patch 的 embedding（256-1024 维）

**(b) scRNA-seq 参考编码器**
- 简单 MLP 聚合该器官 scRNA-seq → 一个 organ-level profile vector
- 进阶（v2）：用 scGPT / Geneformer 提细胞级 embedding

**(c) Marker masking 训练范式**（关键创新）
- 每个 batch 随机选 $k$ 个 marker 作 observed，其余作 target
- $k$ 从 $\{5, 10, 15, 20, 25\}$ 中均匀采样 → 模型学会任意 panel 配置
- 类似 MAE / BERT 的 self-supervision

**(d) 条件扩散主干**
- DDPM 或 DDIM，T=200 步（成像类数据 200 步够用）
- U-Net backbone（参考 Stable Diffusion 的 UNet2DConditionModel）
- 三个条件源（H&E、scRNA、observed markers）通过 cross-attention 注入

**(e) Loss**
- 主：$L_{simple} = \mathbb{E}_{t,\epsilon} \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2$
- 辅：cell-level Pearson correlation loss（在分割 mask 内计算，强化下游可用性）

### 5.3 训练策略

| 阶段 | 数据 | 时长（1×4090） |
|---|---|---|
| Pre-train | HuBMAP 肠道 + 脾淋巴 | 3-5 天 |
| Fine-tune | + Bodenmiller IMC | 2 天 |
| 推理 | HTAN + OPTIMAL（zero-shot） | 几小时 |

### 5.4 推理

- DDIM 50 步采样
- 多次采样取均值 → 不确定性图

---

## 6. 评估方案

### 6.1 像素级指标

| 指标 | 定义 |
|---|---|
| MSE | 全图 mean squared error |
| PCC | per-marker Pearson correlation（图级） |
| SSIM | 结构相似性 |

### 6.2 细胞级指标

- 用 **Mesmer / DeepCell** 分割细胞 → 比较细胞内蛋白表达均值
- Frobenius 范数（参考 Murphy 2024 协议）
- Cell-type classification F1（用预测 marker 训分类器，看准确率）

### 6.3 下游任务

- **Cell typing 准确率**：用预测的全 panel 做细胞分型，对比真实全 panel
- **Tumor vs Normal 区域分类**（HTAN 肿瘤数据）

### 6.4 消融

- 去掉 H&E
- 去掉 scRNA-seq
- 用 U-Net 替换 diffusion
- 用 ResNet 替换 foundation model
- 不做 random masking（固定 panel 训练）

### 6.5 Baseline 对比

必比：7-UP、Murphy U-Net、Kirchgaessner-AE、简单 KNN 插值（sanity check）

---

## 7. 时间表（6 个月）

| 月份 | 任务 |
|---|---|
| **Month 1** | 数据下载、清洗、配准；marker 标准化字典；跑通 Murphy U-Net baseline |
| **Month 2** | 实现 SpaProtFM v0（无 H&E/scRNA，只有 marker masking + diffusion）；跑通端到端 |
| **Month 3** | 加入 H&E foundation model 条件；做 within-sample 实验（任务 A） |
| **Month 4** | 加入 scRNA-seq 条件；跨样本 + 跨数据集实验（任务 B+C） |
| **Month 5** | 完整消融实验、下游任务评估、生成所有图表 |
| **Month 6** | 论文写作、代码整理、投稿 BIB |

---

## 8. 风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| 扩散训练不收敛 | 中 | fallback 到 flow matching 或 condition U-Net |
| H&E 配准不准 | 中 | 用 DAPI 替代，或采用更简单的 image-level condition |
| 跨数据集泛化失败 | 中高 | 至少能在 within-sample 任务上打赢 baseline，仍可投稿 |
| 1 张 GPU 不够 | 中 | 减小图像 patch 大小到 128×128，或用 latent diffusion |
| 名字撞车 | 低 | 投稿前再改名 + 查重 |
| 6 个月内被人 scoop | 中 | bioRxiv 早挂 v0 占坑 |

---

## 9. 锁定假设（保守默认，2026-04-17 锁定）

| # | 假设 | 锁定值 | 防御性设计 |
|---|---|---|---|
| 1 | GPU | **6× RTX 3090 (24GB each, 152GB total)** — 实测 2026-04-17 | 训练用 patch 256×256（无需缩小）；可启用 DDP 多卡并行；可同时跑多个 ablation；不需要 latent diffusion 兜底 |
| 2 | PyTorch 水平 | 中级（能跑能改，不一定能从零写） | 所有模块给完整可运行模板 + 详细注释；优先用 `diffusers` / `lightning` 现成框架 |
| 3 | 每周投入 | 20 小时 | 时间表按此节奏；buffer 已含在 Month 5 |
| 4 | 数据 | 完全公开数据 | 已确认，无私有数据 |
| 5 | 总周期 | 6 个月（v1 投稿） | 锁定；超时则砍掉 H&E 模态保 v1 |
| 6 | 投稿 | BIB 主投，Bioinformatics + Cell Reports Methods 备投 | 锁定 |

**用户应在 Month 1 第 1 周内反馈以上假设是否需要调整**。如有偏差，调整方案：
- GPU 更弱（无 GPU / 8GB）→ 改用 Colab Pro + 小模型，时间表延长 50%
- 时间更少（<10h/周）→ 砍 stretch goal，主投改 Bioinformatics

---

## 10. 下一步

1. 编写实施计划（implementation plan）：周级任务清单 + 仓库结构 + 依赖文件
2. Month 1 W1: 环境搭建（uv + PyTorch + diffusers）+ HuBMAP CODEX 数据试下载
3. Month 1 W2-4: 跑通 Murphy U-Net baseline，建立评估脚本

---

**审阅请关注：**
- 主打题目和差异化是否足以打动 BIB 审稿人
- 时间表是否过于乐观
- 第 9 节假设是否符合你的实际情况
- 是否要砍掉 / 增加任何创新点
