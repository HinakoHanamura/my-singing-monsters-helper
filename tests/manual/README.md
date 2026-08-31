# tests/manual — 需要真实游戏的手动脚本

这里的脚本**不是自动化测试**，pytest 不会收集它们（见 `pytest.ini` 的
`--ignore=tests/manual`）。它们需要《My Singing Monsters》正在运行，会真的截图、
可能真的点击，因此只能由人手动执行、手动看结果。

| 脚本 | 用途 | 会点击吗 |
| --- | --- | --- |
| `test_eyes.py` | 验证 `PrintWindow` 能截取未聚焦、被遮挡的窗口 | 否 |
| `test_hands.py` | 验证 `SendMessage` 能投递点击而不移动物理鼠标 | **会，且无任何保护** |
| `test_memory_minigame.py` | memory game 的实机验证入口 | **默认不点**，加 `--play` 才点 |

前两个是项目最早的可行性验证，演化成了 `core/game_window.py` 与
`core/action_agent.py`。保留它们有两个用途：记录整套框架的出发点；当感知或点击整体
失灵时，用最少的依赖判断问题出在框架之内还是之外。

## 用法

```powershell
python tests/manual/test_eyes.py     # 输出 test_background.png 到本目录
python tests/manual/test_hands.py    # 在客户区 (200, 200) 点一下
```

memory game 的实机验证，**先跑观察模式**：

```powershell
python tests/manual/test_memory_minigame.py             # 只观察，不点击
python tests/manual/test_memory_minigame.py --seconds 30
python tests/manual/test_memory_minigame.py --play --levels 1   # 只打一关
python tests/manual/test_memory_minigame.py --play             # 九关
```

**跑之前**：手动把小游戏打开，停在**某一关的开局界面**，所有卡牌背面朝下。
槽位表只能从完整盘面构建，而"已翻开偶数张"与"完整盘面"从单帧无法区分，
中途启动会让脚本对漏掉的牌视而不见。它会检测到并停下，但从开局启动能绕开整个问题。

窗口可以被遮挡、可以不在前台，但**不能最小化**（最小化后客户区停止渲染，截不到东西）。

标注图输出到 `reports/manual_memory/`：`board.png`（含槽位编号）、
`rejected.png`（门禁不通过时的画面）、`stopped.png`（实战模式停止时的画面）。

## 注意

- `test_hands.py` **会真的点击**，且没有任何目标校验或防误触保护，(200, 200)
  上是什么就点什么。需要带保护的探测请用 `tools/probe_click.py`。
- `test_memory_minigame.py --play` 会点击卡牌。失败结算后会弹出「花 2 钻石重玩」，
  循环在结构上无法点到非卡牌的东西（每个目标都来自当帧的卡背检出，送出前再确认它
  仍落在某个检出框内；而结算页产不出卡背框）。**但那是单元测试断言的性质，
  实机头一回面对没预料过的画面，所以先跑观察模式。**
- `test_eyes.py` 没有做 `core/game_window.py` 里的 DPI 修正。在缩放显示器上它会
  报告放大后的客户区尺寸，并且只截到画面左上角——当初的 DPI bug 就是这样暴露的。
  这个差异是有意保留的。
- 三个脚本的主体都放在 `main()` 里并由 `if __name__ == "__main__"` 守卫。文件名
  匹配 `test_*.py`，一旦哪天被误收集，import 也不会触发真实截图或点击。
- 生成的 `test_background.png` 是本机游戏画面，已在 `.gitignore` 中排除。
  `reports/` 整个目录也已排除。
