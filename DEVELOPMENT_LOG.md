# SuperMimic 论文从零复现开发与实验演进日志

- **目标论文**：*SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model* (`arXiv:2610.28480`)
- **基础仿真框架**：[`mjlab`](file:///home/sunteng/carbon-test/mjlab)（MuJoCo Warp + Unitree G1 29-DoF 全身控制）
- **数据集**：[`lafan1_g1`](file:///home/sunteng/carbon-test/lafan1_g1)（40 条动捕长序列，重采样至 50 Hz 后共 `441,120` 帧 / `2.451` 小时）
- **严格约束**：全程零外部代码参考（未访问论文官方仓库或主页、未搜索任何网上开源实现、未读取本地或远程任何其他目录/分支代码），完全依据论文正文与附录数学公式独立推导与实现。

---

## 一、 整体架构设计思考（Phase 1: `00:31 – 01:44`, Commit `d79016d`）

在动笔写代码前，通过对论文正文 §3 与 Appendix A–F 的逐项梳理，确立了四个决定复现成败的底层设计原则：

### 1. $\text{SE}(2)$ 航向坐标系规范化（Heading-Frame Canonicalization）
- **思考**：同一套全身动作（如跑步、空翻、起身）无论发生在世界坐标系的 $(0,0)$ 还是 $(50, -30)$、朝向东还是朝向北，其刚体动力学与地面接触力学完全等价。如果让世界模型直接输入世界系坐标，数据复杂度会随场地范围指数爆炸。
- **实现**（[`supermimic/state.py`](file:///home/sunteng/carbon-test/supermimic/state.py)）：
  - 提取骨盆（root）绕世界 $Z$ 轴的偏航角 $\psi$（通过 `extract_yaw_cos_sin` 解析提取 $\cos\psi, \sin\psi$），将 14 个关键刚体的位置、6D 旋转矩阵、线速度、角速度统一变换到以当前 root $(x, y, 0)$ 为原点、消去偏航角 $\psi$ 但**保留真实高度 $z$ 与重力倾角（roll/pitch）**的规范坐标系 $\tilde{s}_t \in \mathbb{R}^{278}$。
  - 世界模型 $f_\omega(\tilde{s}_t, a_t, \xi)$ 仅预测规范系下的状态残差 $\Delta \tilde{s}_t$，再通过 `apply_canonical_delta` 积分回世界系 `RobotState`，从而在结构上严格保证了水平位置与朝向的 $\text{SE}(2)$ 等变性，同时保留地面接触高度 $z$ 的绝对物理意义。

### 2. $\text{SO}(3)$ 旋转流形可微算子与小角度梯度奇点防护
- **遇到的问题**：在 BPTT 反向传播中，若使用标准反余弦 $\theta = \arccos((\text{tr}(R_1^\top R_2) - 1)/2)$ 计算姿态跟踪奖励或世界模型姿态损失，当预测姿态与目标姿态重合（$\text{tr} \to 3, x \to 1^-$）时，$\frac{d}{dx}\arccos(x) = -\frac{1}{\sqrt{1 - x^2}} \to -\infty$，会直接产生 `NaN` 梯度炸穿整个 32 步计算图。
- **思考与修改**（[`supermimic/math_utils.py`](file:///home/sunteng/carbon-test/supermimic/math_utils.py)）：
  - 旋转预测统一采用连续无奇点的 6D 旋转表示（`rot6d_to_matrix` 可微 Gram-Schmidt 正交化）；
  - 在 `geodesic_rot_error_sq` 中，对 $u = 1 - x < 10^{-4}$ 的小角度邻域采用解析泰勒展开 $\theta^2 \approx 2u + \frac{1}{3}u^2 + \frac{4}{45}u^3$，并通过 `torch.where` 与标准 `acos` 分支无缝拼接，保证在 $\theta = 0$ 处一阶与二阶导数平滑有界。

### 3. 解析观测头与解析奖励头（避免“奖励模型幻觉”）
- **思考**：Model-Based RL 常见失败模式是策略利用学习的奖励网络漏洞刷出虚高回报。依据论文 Appendix C.1，由于状态 $\hat{s}_t$ 包含了全部 14 个跟踪刚体的位置、姿态、速度与 29 个关节状态：
  1. 跟踪奖励 $r_t$（Table A.2：身体位姿、速度、动作平滑度、关节软限位惩罚）可以直接写成关于 $\hat{s}_t$ 与动作 $a_t$ 的**纯解析可微函数** `compute_tracking_reward`；
  2. 策略输入的 93 维单步本体感觉观测（Table A.1：关节位置/速度、root 重力投影、root 角速度、上一帧动作）也可由 $\hat{s}_t$ 通过 `compute_clean_obs` 解析算出，并在想象展开中叠加同分布传感器噪声 $\eta_t \sim \rho_\eta$ 滚动更新 10 帧观测历史 $o_{t-9:t}$。
- **效果**：策略无法通过欺骗奖励头获利，只能真正改善预测物理轨迹与参考轨迹的几何对齐度。

### 4. 非对称特权物理上下文 $\xi$
- **实现**（[`supermimic/env.py`](file:///home/sunteng/carbon-test/supermimic/env.py)、[`supermimic/models.py`](file:///home/sunteng/carbon-test/supermimic/models.py)）：
  - 训练环境中对脚底摩擦系数（1 维）、29 关节编码器零点偏置（29 维）、躯干质心偏移（3 维）做域随机化（Table A.3），拼成 $\xi \in \mathbb{R}^{33}$ 传给世界模型 $f_\omega$ 与价值网络 $V_\psi$，使确定性世界模型在不同物理参数下仍能精确预测下一步状态；
  - 部署/评估策略 $\pi_\theta$ 仅接收带噪历史观测 $o_{t-9:t}$ 与未来 $K=10$ 帧参考目标序列 $g_t \in \mathbb{R}^{730}$，保证零特权信息可部署性。

---

## 二、 吞吐率瓶颈与算子级优化（Phase 2: `01:44 – 02:00`, Commit `0a400b3`）

### 遇到的问题
首版训练循环跑通后发现：每次参数更新需要执行 10 个 batch 的 $k=8$ 步自回归世界模型训练 + $B=4096, H=32$ 步策略 BPTT 展开 + 20 个 minibatch 的 Critic GAE 拟合。Python 层频繁调用大量小张量旋转变换算子，且日志统计中存在 `.item()` 隐式同步，拖慢了单步迭代速度并推高了峰值显存。

### 思考与实验修改
1. **TorchScript 算子融合**：将 [`math_utils.py`](file:///home/sunteng/carbon-test/supermimic/math_utils.py) 中所有高频几何核（`quat_to_matrix`, `matrix_to_rot6d`, `rot6d_to_matrix`, `geodesic_rot_error_sq`, `rotate_vec_z`, `rotate_mat_z`, `compute_relative_body_poses_rot`）全部用 `@torch.jit.script` 编译融合。
2. **启用 TF32 Tensor Core 与无同步累积**：与论文 Figure 5(a) 一致开启 `allow_tf32 = True`，将训练循环内所有标量 loss 改为 GPU Tensor 累加，仅在每步更新结束后做一次 `.item()` 读取。
3. **BPTT 微批次梯度累积**：将 `B=4096, H=32` 的想象展开按 `microbatch=2048` 分块反传累加梯度，在数学严格等价的前提下将单卡显存稳定控制在 `~23.6 GB`（适配单张 RTX 5090 32GB）。

---

## 三、 评估状态隔离与回放池边界污染修复（Phase 3: `02:00 – 02:09`, Commit `582e8eb`）

### 遇到的问题
1. **评估干扰训练采样分布**：快速全集分段评估 `evaluate_window_segments` 为了节省显存复用了 4,096 并行度的训练环境 `env`。若评估后简单调用 `reset_envs`，会导致所有 4,096 个训练环境每 500 步被强制同步清零重置，破坏长周期轨迹探索与回放池时间连续性。
2. **回放池跨 Episode 伪转移**：环形缓冲区 `ReplayBuffer` 在环境发生摔倒重置或环形写指针 `ptr` 覆盖边界处，如果世界模型 $k=8$ 步采样窗口或策略 9 帧历史观测跨越了两个不同回合，世界模型就会把“从摔倒姿态瞬移到新参考帧”误认为物理动力学一步转移，产生巨大伪残差。

### 思考与实验修改
1. **零副作用环境快照**（[`supermimic/env.py`](file:///home/sunteng/carbon-test/supermimic/env.py)）：编写 `save_snapshot()` 与 `load_snapshot()`，完整保存并恢复 MuJoCo Warp 的 `qpos, qvel, qacc, act, mocap`、域随机化物理参数、观测历史队列与回合统计张量，实测快照恢复前后状态最大绝对误差为 **`0.0`**。
2. **Episode ID 严格守卫与独立终止帧池**（[`supermimic/buffer.py`](file:///home/sunteng/carbon-test/supermimic/buffer.py)）：
   - 为每次环境重置分配全局唯一单调递增的 `episode_id`；
   - 当某步触发 `terminated` 或 `timed_out`（环境随后自动重置），将重置前的真实物理下一帧 `next_state` 存入专用的 `terminal_bank`；
   - 在 `sample_wm_windows` 中一旦遇到终止步或跨 `ptr` 断点，自动将后续步 `valid_mask` 置零；在 `sample_imagination_starts` 重建过去 9 帧观测历史时，按 `episode_id` 精确识别该回合起点并用首帧观测填充越界步，彻底杜绝跨回合状态泄漏。

---

## 四、 策略梯度尖峰解剖与非论文重置扰动清理（Phase 4: `02:10 – 04:00`, Commit `8f28404`）

这是整个复现过程中最关键的一次深度诊断。

### 问题现象
在 `XN-5090-GPU3` 上训练推进到 `Iter 7350`（单卡）和 `Iter 8000 -> 8300`（四卡，`seg_ret` 刚达 `0.6863`）时，观察到策略回报偶发突降（`0.686 -> 0.597`），且世界模型 8 步误差 `wm_8step` 长期停留在 `0.0185 – 0.0210`，未能降入论文 Figure 3(c) 报告的 `0.007 – 0.011` 区间。

### 深度诊断与根因定位
1. **根因一：终止预测头锐度过高导致边界轨迹产生 32 倍梯度尖峰**
   - 我们在检查点上抽取一个真实想象 batch（`B=1024, H=32`），逐条轨迹计算策略目标函数对初始状态和动作的梯度范数 $\|\nabla_{\hat{s}_0} \mathcal{J}\|$。
   - **实验发现**：Batch 中 99% 分位样本的梯度范数仅为 **`7.090`**，但单样本最大值高达 **`232.555`**（超出正常样本 **32 倍**）！
   - **数学机理**：`WorldModel.predict_termination_logit` 中可学习的边界锐度参数 $k_z, k_t$ 当时上限设为 `100.0`。一旦 batch 中哪怕只有 1–2 条想象轨迹刚好擦过终止边界（$|z_{\text{ref}} - z_{\text{rob}}| \approx 0.25\text{ m}$ 或倾角误差 $\approx 0.8$），Sigmoid 导数 $\hat{d}(1-\hat{d}) \cdot k_z \approx 25$ 再乘上后续步累积价值 $V_\psi(\hat{s}_H) \approx 6.5$，就会产生超过 `230` 的极端梯度尖峰，瞬间打满 `max_norm = 1.0` 的全局梯度裁剪，导致其余 4,095 条正常轨迹的有效学习步长被压缩数十倍且方向被边界噪声主导！
   - **实验验证**：将 `kz, kt` 上限约束到 `15.0`（[`supermimic/models.py#L175-L176`](file:///home/sunteng/carbon-test/supermimic/models.py#L175-L176)）后重新测量同一 batch，单样本最大梯度直接从 **`232.555` 降至 `14.599`**，梯度分布完全平稳。

2. **根因二：`reset_envs` 中混入了非论文的初速度/关节随机跳变**
   - 检查为何 `wm_8step` 卡在 `0.019` 时发现，首版 `env.reset_envs` 参考了常规模型无关 PPO 的习惯，在每次重置到参考帧后额外叠加了 `±0.5 m/s` 根部线速度、`±0.78 rad/s` 角速度与 `±0.1 rad` 关节角度随机扰动。
   - 对照论文 **Appendix A（第 13 页）** 原文核查发现，作者专门强调省略了此类根速度随机推力（*"such pushes inject state discontinuities that the deterministic world model cannot capture, and we find randomization over the physical parameters alone suffices"*），不仅注入了非物理的接触穿透与初态不连续，还破坏了首步动力学一致性。
   - **实验效果**：移除该非论文初态速度/关节突变、保留全部 Table A.3 物理参数随机化与 Table A.1 观测噪声（Commit `8f28404`）后：
     - `wm_8step` 从 `0.0187` 立刻下降 **40.6%** 至 **`0.0105 – 0.0111`**，**精准落入论文 Figure 3(c) 报告的 `[0.007, 0.011]` 区间**！
     - 仅用 9.6 分钟（`Iter 12500 -> 13500`），`seg_ret` 从 `0.7042 (fail=85)` 跳涨至 **`0.7403 (fail=51)`**；
     - 随后无人值守运行至 `Iter 53500–55000`，`seg_ret` 推进至 **`0.8050`**，全集片段失败数从 85 降至最低 **8 / 862**（**99.07% 存活**），全长轨迹从头测试 `FULL_EVAL` 升至 **`0.7455`（加权 `0.7587`，33/40 条完整数分钟长轨迹零跌倒跑完）**！

---

## 五、 收尾冲刺阶段：40 条长轨迹逐帧解剖与悬空重置过滤（Phase 5: `11:30 – 当前`）

### 1. 40 条完整长轨迹逐帧评估解剖（`Iter 55000` 最优检查点）
对 `GPU 0` 在 `Iter 55000` 保存的 `best_seg.pt`（`seg_ret = 0.8050`）执行全部 40 条完整长轨迹确定性一镜到底评估，得到 **`FULL_EVAL = 0.7520`，34 / 40 条完整数分钟长轨迹零跌倒跑完**：
- **12 / 12 条 `walk` 全数通关**（`ret = 0.8258 – 0.9341`）；
- **6 / 6 条 `run` & `sprint` 全数通关**（`ret = 0.7713 – 0.8625`）；
- **5 / 5 条 `fight` & `fightAndSports` 全数通关**（`ret = 0.7492 – 0.8337`）；
- **3 / 6 条 `fallAndGetUp`（反复倒地爬起近 3 分钟）已完整通关**；
- **仅剩的 6 条未通关轨迹及其精确失败帧误差解剖**：
  1. `Clip 9 (fallAndGetUp1_subject4)`：在 `frame 815 / 8410` 撑地起身时 `err_z = 0.252 m`（仅超出 `0.250 m` 阈值 **2 毫米**，若不触发终止机器人后续实际成功站起恢复！）；
  2. `Clip 12 (fallAndGetUp2_subject3)`：在 `frame 2015 / 8195` 地面翻滚时 `err_tilt = 0.817`（仅超出 `0.800` 阈值 `0.017`）；
  3. `Clip 6 (dance2_subject4)`：在 `frame 2964 / 11284` 地板街舞翻滚时 `err_z = 0.023 m`，`err_tilt = 0.830`（仅超出 `0.800` 阈值 `0.030`）；
  4. `Clip 20 (jumps1_subject2)`：连续成功跳跃 **239 秒（11,950 帧）**，仅在最后 5 秒（`11950 / 12222`）落地倾角擦边；
  5. `Clip 13 (fallAndGetUp3_subject1)`：`FAIL @ 3422 / 5109`（`ret = 0.5005`）；
  6. `Clip 0 (dance1_subject1)`：`FAIL @ 5350 / 6574`（`ret = 0.6591`）。

### 2. 为什么这最后 6 个边界动作差 2 毫米没被彻底修平？
将优先重置分布 `probs` 按采样权重从高到低排序后发现：
- 排名前列的最高概率分箱（`bin=2896, 2310, 2304, 8080, 8077` 等，合计共 **46 / 8802 个分箱**，仅占总数的 `0.52%`）全部是原始动捕中演员**坐在隐形箱子/椅子上双脚悬空静止**的区段（`min_body_z > 0.12 m` 且根部速度 `< 0.5 m/s`，例如 `walk3_subject3` 连续 5.76 秒坐在 22 cm 高的箱子上）。
- 当从第 0 帧正常走过去时，机器人双脚踩在平地上保持骨盆高度即可通关（因此 `walk3_subject3` 在 `FULL_EVAL` 中实际拿到了 `0.8258` 通关）；但若在训练中**直接把机器人重置到半空蜷腿坐姿帧**，平地无支撑会导致机器人在 11 帧（`0.22s`）内自由落体跌落（`G_b ≈ 0.12`）。
- 指数困难采样 `softmax(-25 * G_b / G_max)` 因此给这 46 个不可行悬空分箱赋予了高达 **150–366 倍**的重置权重，不仅抢占了本该分配给上述 6 个真实地面起身/翻滚难点的训练预算，还周期性向回放池注入半空坠落轨迹，配合当时仍处于峰值 `89%`（`1.77e-4`）的高学习率，造成边界微调超调。

### 3. 实验修改（Phase 5 收尾迭代）
1. **平地重置可行性门控**（[`supermimic/data.py`](file:///home/sunteng/carbon-test/supermimic/data.py)，对应论文 Appendix E *Feasibility Filter*）：在构建 1 秒重置分箱时标记并屏蔽悬空静止分箱（`min_body_z > 0.12 m` 且 `||v_root|| < 0.5 m/s` 占比超半数的 46 个分箱，保留 `8,756 / 8,802` 个可行分箱），让 `softmax(-25 * G_b / G_max)` 将 100% 的困难采样火力集中到 `Clip 9 / 12 / 6 / 13 / 0 / 20` 的真实地面接触难点上。
2. **精细学习率微调与精简 GPU 占用**：停掉冗余的四卡与旧单卡进程（释放 6 张空闲 GPU），仅保留 `GPU 0` 原基线对照，并在 `GPU 4` 上从 `Iter 55000`（`0.8050 / 0.7520`）检查点以 `--lr 8e-5` 启动可行性过滤后的精细收敛任务，直接消除那 `2 mm – 0.03` 的边界误差以冲刺 `FULL_EVAL >= 0.82`。

