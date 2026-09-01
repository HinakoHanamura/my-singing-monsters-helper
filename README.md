# MSM Helper

[English](README.en.md) | **简体中文**

![Python](https://img.shields.io/badge/Python-3.11-3776ab)
![PySide6](https://img.shields.io/badge/GUI-PySide6%206.9-41cd52)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV%204.11-5c3ee8)
![Tests](https://img.shields.io/badge/tests-472%20passed-2f6df6)
![Platform](https://img.shields.io/badge/platform-Windows-0078d4)

《My Singing Monsters》（Steam / Windows）的后台自动化助手，基于 Python + PySide6 + OpenCV 构建。

在自动完成部分游戏活动的同时，不移动物理鼠标、不抢占前台焦点。游戏窗口即使被完全遮挡，也依然可以正常运行，不影响对电脑的正常使用。

> **免责声明**：本项目为个人技术练习项目，旨在探索 Windows 后台自动化、计算机视觉与参数实测标定的工程实践。游戏自动化可能违反《My Singing Monsters》的服务条款，使用风险自负。本项目与 Big Blue Bubble 没有任何关联，亦未获得其官方授权。
>
> **开发模式**：本项目采用现代人机协作（Human-in-the-Loop）开发模式。开发者负责制定开发路径、总体架构决策、提出最坏构造与数学证明、真实游戏环境录制与实机验收；AI 负责异步图形界面构建、算法状态机代码实现、运行异常排查追踪、全量无头测试套件编写与感知参数统计测绘。

---

## 快速开始

### 运行环境要求

| 依赖环境 | 要求 / 推荐配置 | 说明 |
| :--- | :--- | :--- |
| **操作系统** | Windows 10 / 11 (64-bit) | 深度依赖 Win32 底层消息与截图机制 |
| **Python** | 3.11+ | 核心开发与运行环境 |
| **游戏客户端** | Steam 版《My Singing Monsters》 | 窗口化运行，支持自适应缩放（不支持窗口最小化） |

### 安装与启动

```powershell
# 1. 克隆代码仓库
git clone https://github.com/HinakoHanamura/my-singing-monsters-helper.git
cd my-singing-monsters-helper

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动助手
python main.py
```

### 操作快捷键

| 功能操作 | 全局快捷键 | 使用说明 |
| :--- | :--- | :--- |
| **收集金币** | `F9` | 在游戏岛屿启动金币自动化收集 |
| **记忆小游戏** | `F11` | 需先手动进入记忆游戏后启动 |
| **停止运行** | `F10` | 安全终止当前工作线程 |

>勾选「加速配对」将启用记忆游戏的两阶段并发翻牌策略（推荐使用。同处在最坏情况下，开启加速配对比不开启的失误次数仅多一次）。

---

## 效果演示

### 1. 软件主界面
深色现代化 UI，提供收集金币、记忆小游戏及其加速入口，配备实时运行与诊断日志。

![UI 界面](docs/images/ui_window.jpg)

### 2. 岛屿金币收集
单帧多目标多尺度模板匹配，带置信度排序与防重叠逐层剥离机制。

![金币检测](docs/images/coin_detection.jpg)

### 3. 记忆翻牌小游戏
跨尺度卡背色彩分割，通过行带聚类自适应任意非网格动态布局，自动还原阅读顺序。

![记忆小游戏盘面](docs/images/memory_board.jpg)

---

## 核心功能与性能

| 功能模块 | 运行状态 | 实测性能表现 | 核心技术亮点 |
| :--- | :--- | :--- | :--- |
| **岛屿金币收集** | 实机完全可用 | 约 **0.4~0.5 秒/个**（最初 2.1 秒） | 模板匹配降采样加速、重叠候选逐层剥离、点击后回读验证 |
| **记忆翻牌小游戏** | 实机连续全通关 | 约 **1.10 秒/张**，平均每关 21.1 秒 | 阅读顺序聚类还原、无模板卡面聚类、并发双卡读取优化 |

> 注：实测失配后翻回动画耗时约 1.6 秒，期间游戏不接收新的翻牌输入。

---

## 技术架构与工程设计

本项目严格遵循高内聚、低耦合的分层架构与实测驱动方法：

| 架构层级 | 核心模块 | 职责说明 |
| :--- | :--- | :--- |
| **UI 展示层** | `ui/main_window.py` | PySide6 现代化深色异步界面，只读日志与状态展示 |
| **逻辑控制层** | `core/bot_engine.py` | `QThread` 异步主调度循环，不阻塞 UI 线程 |
| **防误触守卫层** | `core/validators.py`<br>`core/click_guard.py` | 环境校验规则链、跨帧位置追踪与阶梯冷却退避 |
| **算法求解层** | `core/minigames/` | 行带聚类还原阅读顺序、卡面动态聚类与求解状态机 |
| **视觉感知层** | `core/vision_agent.py` | `BaseVisionAgent` 统一感知契约，多尺度模板匹配 |
| **动作执行层** | `core/action_agent.py` | Win32 消息级后台点击，内置高斯位置与延迟抖动 |
| **几何换算层** | `core/geometry.py` | 参考分辨率（1024×768）与实时窗口尺度的自适应映射 |
| **窗口接入层** | `core/game_window.py` | GDI 截图管理、DPI 线程上下文隔离与黑边裁剪 |

**[点击查阅完整《MSM Helper 设计文档》（docs/DESIGN.md）](docs/DESIGN.md)**

---

## 测试与质量保障

项目配备了完善的测试套件，**无需依赖真实游戏与 Qt 窗口**，支持在无头环境下毫秒级运行：

```powershell
python -m pytest tests/unit   # 运行快速单元测试 (410 项)
python tools/run_suite.py     # 运行全量测试套件 (472 项全部通过)
```

- **472 项测试全部通过**，覆盖状态机证明、几何换算、卡面聚类与编码恢复。
- 基于纯函数抽离与依赖注入（FakeBoard、FakeClock），保证业务逻辑的高确定性。

---

## 已知限制

| 限制维度 | 现状与影响说明 | 应对建议 / 后续规划 |
| :--- | :--- | :--- |
| **固定镜头视角** | 金币图标为屏幕空间 UI 精灵（尺寸固定），但镜头缩放会改变视野可见范围 | 运行前建议将游戏镜头拉至最大视野 |
| **多资源遮挡** | 钻石与食物图标会与金币发生层叠遮挡，导致局部模板匹配分下降 | 当前版本按优先级逐层剥离，多资源联合识别将在后续版本支持 |
| **平台限制** | 深度依赖 Win32 消息队列与 GDI/DirectX 截图机制 | 仅支持 Windows，暂无跨平台计划 |

---

## 目录结构

| 目录 / 文件 | 说明 |
| :--- | :--- |
| `config.py` | 集中配置中心（所有可调参数与标定依据） |
| `main.py` | 程序主入口 |
| `core/` | 系统核心分层（窗口、几何、感知、动作、防误触、引擎） |
| `core/minigames/` | 记忆小游戏完整状态机与求解器 |
| `ui/main_window.py` | PySide6 图形界面 |
| `tools/` | 标定测量、测试运行与诊断工具集 |
| `tests/` | 472 项无游戏依赖的单元与集成测试 |
| `docs/DESIGN.md` | 技术架构与设计文档 |
| `.agents/rules/` | 决策记录、实测结论、模型经验与开发者观察 |

---

## 版权

代码版权归作者所有。本项目未授予任何开源许可，未经明确书面授权，严禁任何形式的商业化、二次分发或代码复用。

`assets/templates/coin.png` 与 `docs/images/` 下的相关游戏截图版权归 Big Blue Bubble 所有，在此仅作技术展示与说明用途。
