# 环境陷阱与操作规程

这份文档记录的坑**每一个都实际发生过并造成过损失**。开工前请通读。

## 运行环境

- Python **3.11.9（Microsoft Store 版）**，用户级安装，无虚拟环境。
- PySide6 6.9.1 ｜ opencv-python 4.11.0.86 ｜ numpy 2.2.5 ｜ pywin32 312 ｜ pytest 8.4.2
- Windows，PowerShell。命令分隔符用 `;`，**不要用 `&&`**。
- 环境变量用 `$env:TEMP` 形式，**不要用 `%TEMP%`**（PowerShell 不展开，会变成字面文件名）。
- 显示器 **1 块**，系统缩放 **150%**。
- 游戏：**窗口化**（非全屏），游戏内分辨率 **1024×768**。
- 系统 ANSI 代码页是 **GB18030**（不是 GBK）。这一条是下面整节的根因。

多显示器假设已被排除，不要再往那个方向猜。

## 陷阱一：中文文件的编码会被写坏

**这是本项目最容易踩、后果最脏、且已反复发生的坑。** 以下全部实测验证过。

### 1a. 铁律：含中文的文件只能用 `fs_write` 整体重写

**禁止对含中文的文件使用 `str_replace` 或 `fs_append`。**

实测证据：对一个 UTF-8 的中文 md 用 `str_replace` 替换一段，结果 `utf-8 FAIL at 30`，
用 gb18030 能解出来，但**新写入的段落正常、原有的 UTF-8 中文全变成乱码**。
产出的是真正的**混合编码**文件——用任何单一编码都还原不了。

`fs_append` 同理，也已真实踩中：往含中文的测试文件追加内容后，从 position 2001 起损坏。

**这条规则已经被违反过四次，包括在把它写进本文档之后。**
**看到"只改一行"这个念头时，正是最该警惕的时刻。**

### 1b. 「这个文件是纯 ASCII」是会过期的事实

**第四次违规是这么发生的。** 本节原来点名说
`tests/unit/test_minigame_runner_control_loop.py` 是纯 ASCII、可以放心 `str_replace`。
那句话在写下时是对的。后来往那个文件里加了一条 `assert "秒" in result.message`，
**它就不再是纯 ASCII 了，而文档没跟着改。** 下一轮照着文档用 `str_replace`，
那三个字节当场变成 `\xe7\xbb\x89\xef\xbf\xbd`（"绉" + U+FFFD），测试报
`assert '绉\ufffd' in '...'` 失败。

教训有两条：

1. **不要相信任何文档里"某文件是纯 ASCII"的记载，动手前自己验：**

   ```powershell
   python -c "import pathlib;print(sum(1 for b in pathlib.Path('<文件>').read_bytes() if b>=0x80))"
   ```

   输出 0 才能用 `str_replace`。这条命令一秒就跑完，没有理由省。

2. **测试里根本不该出现中文断言。** 项目规则早就写了「断言 ASCII `code`，
   不要断言中文显示文案」，那条断言违反了规则，然后规则的违反反过来把文件写坏。
   现在那处改成了 `timings = [word for word in result.message.split() if "." in word]`，
   文件回到 0 个非 ASCII 字节。

### 1c. 整体重写在大文件上会失败，所以有了 `tools/apply_doc_edits.py`

**两条已确认的事实在这里对撞：**

- 含中文的文件做局部文本编辑 → 编码被重写，可能产出混合编码（1a）
- 于是整体重写是唯一安全的编辑方式 → **但它在 25KB 的 steering 文档上会失败**，
  连续五次 `The connection was interrupted`

结果是「给大段中文文档做小改」没有任何安全路径，而保持 steering 文档新鲜正好需要它。

`tools/apply_doc_edits.py` 就是那条路径。它把改写放进一个**纯 ASCII** 的脚本里，
中文只存在于它读的数据文件中，编码写死成 UTF-8：

```powershell
python tools/apply_doc_edits.py .kiro/steering/workflow.md reports/_edit1.md
```

数据文件用三行标记分隔（三个左尖括号 + `OLD` / `NEW` / `END`，**只在行首生效**），
可以放多个块。格式与注意事项见该脚本的 docstring。

**每个块必须恰好命中一次**，否则**一个字都不写**并在报告里指出是哪个块——
猜锚点会静默损坏文档，而这个工具存在的全部理由就是避免那件事。
（这道闸门第一次实战命中，就是拦住了「把本节写进文档」时示例标记被解析器
当成真标记的乌龙。标记因此改成只在行首生效。）

目标文件按 UTF-8 → GB18030 → GBK 顺序读，**一律按 UTF-8 写回**，
写完再解码一遍作为证明。报告落在 `reports/apply_doc_edits.txt`。

**数据文件是一次性的**：写在 `reports/` 下，用完删掉。
每个数据文件只用一次 `fs_write` 写完——
**不要对它 `fs_append`**，那条铁律对临时文件同样成立（已经因此毁过一个）。

**它的目标不限于文档。** 注释英语化那一轮用它改的是 `ui/main_window.py`：
一次 25 个块、全部恰好命中一次、写回 UTF-8。凡是「改完仍含中文的代码文件」
都该走它，而不是整体 `fs_write`（那会掉进 1e 的编码轮盘）。
带长划线的分节注释（`# ------ 构建界面`）要先量出精确的划线根数再做锚点，
数错就是失配——**但失配是安全的**，工具会一个字都不写并指出是哪一块。

### 1d. 修混合编码文件：按字节改，不要按文本改

上面那次损坏只有一处、6 个字节，而整个文件另外 31KB 全是 ASCII。
**这种情况不要整体重写**（900 行的测试文件，转录风险远大于收益），
也**不要用 `str_replace`**（它会把整个文件重新编码一遍）。

做法是写一个一次性脚本按字节替换：

```python
data = target.read_bytes()
old = b'    assert "\xe7\xbb\x89\xef\xbf\xbd" in result.message\n'
new = b'    ...ASCII 替代...\n'
assert data.count(old) == 1
target.write_bytes(data.replace(old, new))
```

改完立刻确认非 ASCII 字节数归零，然后**删掉那个一次性脚本**。
字节级替换不碰其他任何位置，这是它比文本编辑安全的全部理由。

定位损坏位置也用脚本：遍历字节、记下 `>= 0x80` 的连续段和行号、结果写进
`reports/`。上面那次就是这么定位的——一个 run，line 728。

### 1e. `fs_write` 会把中文写成 GB18030，而且不稳定

**`fs_write` 不保证 UTF-8，行为不确定。** 同一批操作里有的文件写出 UTF-8、
有的写出 GB18030；同一个文件重写两次结果可能不同。
**所以每次写完都必须验，不能靠"上次没事"推断。**

**是 GB18030，不是 GBK。** 上标字符会被编成 4 字节扩展序列
`\x81\x30\x85\x35`，**GBK 解码直接失败**，表现为"既不是 UTF-8 也不是 GBK"，
看起来像文件损坏，其实用 gb18030 能无损还原。

根因确认：本文件与 `minigame_memory.md` 里含数学符号等字符，**根本无法编成 GBK**，
写入器只能退到 GB18030。

### 1f. 混合编码文件绝不能自动转换

**这是最危险的情形。** 混合文件能被 gb18030 "成功"解码，所以任何自动转换都会
"修复成功"同时把原本正确的 UTF-8 字符全部变成乱码——**比不修更坏**。

`ensure_utf8.py` 已加入混合编码识别，遇到就报 `MIXED-ENCODING` 并**拒绝改动**，
需要人手处理（见 1d）。两个独立判据（任一命中即判混合）：

1. **合法前缀里有真正的 UTF-8 非 ASCII 字符 ≥ 4 个。** 整体单一编码的文件，
   第一个非 ASCII 字节就已经是非法 UTF-8，所以合法前缀必然是纯 ASCII，
   无论 ASCII 头有多长。
2. **能按 UTF-8 正确解出的汉字数 ≥ 0.25 × 替换字符数。** 实测分离度：
   混合文件 0.86，整体 GB18030 长文档 0.003，整体 GBK 0.007——差两个数量级。

判据 1 是为**代码文件**加的：源码大多是 ASCII、中文只在注释里，比值判据抓不到它，
已经因此漏判过一次。这套守卫的第一次实战命中，就是拦住了本文件被自己毁掉。

注意 1b 那次损坏**只有一个汉字**，两条判据都抓不到——
守卫是给"大面积损坏"设的，单字符损坏只能靠跑测试或数非 ASCII 字节发现。

### 1g. 不要依赖钩子，手动跑

`.kiro/hooks/ensure-utf8.json` 是 `PostToolUse` 钩子，匹配
`fs_write|str_replace|fs_append` 后跑 `ensure_utf8.py`。**它并不总是触发**——
实测有写完文件后钩子没跑、文件保持 GB18030 直到手动执行才被修的情况。

**规则：写完含中文的文件，自己验一遍。**

```powershell
python tools/ensure_utf8.py          # 就地修复
python tools/ensure_utf8.py --check  # 只报告，脏则非零退出
python -m compileall -q config.py main.py core ui tools tests
```

**`ensure_utf8.py` 的输出经常被终端吞掉**（见陷阱二），这时用一条短命令直接验字节：

```powershell
python -c "import pathlib;pathlib.Path('<文件>').read_bytes().decode('utf-8');print('utf8-ok')"
```

严格 UTF-8 解码不抛异常就说明这个文件是干净的 UTF-8。这个办法**从不被终端吞掉**，
比读 `ensure_utf8.py` 的报告可靠。

`ensure_utf8.py` 现在的覆盖范围与回退顺序：

- 扫 `.kiro/steering`（以前 `.kiro` 整棵树被 `SKIP_DIRS` 跳过，
  判定是**按路径分量**做的，所以连 `--root .kiro` 也静默返回空——别把那当成「干净」）
- 仍跳过 `.kiro/hooks`、`.kiro/settings`、`__pycache__`、`.git`、`assets`、`node_modules`
- 回退顺序 `gbk → cp936 → gb18030 → big5`。gbk 排在 gb18030 前面是因为它更严格，
  先试它能让报告出的编码名更具体
- UTF-16 **只认 BOM**，绝不试解码。试解码会把中文 ANSI 误判成 UTF-16 然后毁掉文件，
  这个错误犯过一次

行为由 `tests/unit/test_ensure_utf8_encoding_recovery.py` 的 12 项测试锁住，
包含两个「能被回退编码解出的混合样本」——这类样本的字节是否对得上不能靠肉眼判断，
是搜出来的，改动阈值前先看那个文件。

### 1h. 验证内容时的三个假信号

- **`grep_search` 默认跳过点开头的目录。** 对 `.kiro/steering/*.md` 检索会返回
  「无匹配」，即使内容确实存在。**别当成文件损坏的证据。**
- **`grep_search` 的输出会把中文显示成乱码。** 曾据此误以为
  `core/validators.py` 的 `label = "下方无怪物特征"` 被破坏，实际文件完全正常。
  **验证中文内容一律用读文件工具。**
- **PowerShell 里反引号是转义符。** 传给 `python -c` 的检索串里带反引号会被吃掉，
  于是命中失败。

### 1i. VS Code 可能用错编码打开，而保存就会毁掉文件

**实测**：VS Code 打开 steering 文档时，需要手动切到 UTF-8 才能正常显示中文。

**已核对：文件本身没问题。** 所有 steering 文档都是干净的 UTF-8，严格解码通过。

仓库里也**没有** `.vscode/settings.json`。所以**是编辑器的读取设置不对，不是文件坏了**。

**这不影响 git 与 GitHub。** git 存的是字节，字节是 UTF-8，GitHub 按 UTF-8 渲染。
**上传不会造成任何损坏**，也不需要为此做任何转换。

**但它是一个真实的本机隐患，而且后果正是本节最怕的那种：**
如果 VS Code 用 GBK 打开一个 UTF-8 文件，**而你按了保存，它会按 GBK 写回去**——
文件当场被毁，毁法与 1a 完全同类，用任何单一编码都恢复不了。

修法在 VS Code 的**用户级**设置里（不能放工作区，`.vscode/` 被 gitignore）：

```json
"files.encoding": "utf8",
"files.autoGuessEncoding": false
```

`autoGuessEncoding` 关掉是故意的：猜测会误判（jschardet 对中文 UTF-8 与 GB18030 常常分不清），
**固定成 UTF-8 比让它猜更可靠**——这与 1g「UTF-16 只认 BOM，绝不试解码」是同一条道理。

**已处理**：`files.encoding` 已设为 `utf8`，
而 `autoGuessEncoding` 本来就没开（VS Code 默认 false）。
之前需要手动切编码的原因就是 `files.encoding` 不是 utf8，这个隐患已解除。

**规则：不要保存一个显示为乱码的文件。** 先把编码切对、确认内容正常，再保存。
这条对任何编辑器都成立。

### 1j. 非 ASCII 字节数**不能**用来判断「中文注释清没清完」

1b 立的规矩是「动手前自己数非 ASCII 字节」。那条对
**「这个文件能不能用 `str_replace`」**是完全正确的判据，**别改**。

但注释英语化那一轮暴露了它的边界：**字节数分不清「中文注释」和
「按语言约定必须保留的中文文案」。** 按字节数看，20 个 `.py` 文件里有约 1.1 万
非 ASCII 字节，像是一场大工程；按 token 分类看，**中文注释只在一个文件里**
（`ui/main_window.py`，17 条注释 + 8 处 docstring），其余 291 处全是 UI 文案、
日志、异常消息和测试里的中文断言。**按字节数估工作量会高估一个数量级。**

要分开就得按 token 看：`tokenize` 取 COMMENT，`ast` 标出哪些 STRING 是 docstring，
剩下的 STRING 一律算用户可见文案。这套逻辑固化在
`tools/audit_comment_language.py`。判据是**注释与 docstring 里的中文只允许出现在
ASCII 双引号内**——英文讲道理，中文只做原文引用。
（合法例子：`grid.describe_grid` 的 docstring 举它自己的返回值
`"3 行 (4/4/2)，共 10 张"`。**把那个例子翻成英文会让文档描述一个不存在的输出。**）

**分类脚本自己也要做会计核对。** 把所有 COMMENT 与 STRING token 里的非 ASCII
字节数加起来，与文件总非 ASCII 字节数对上，才能确定没有中文藏在 token 之外
（标识符、格式说明符……）。17 个文件全部对上、0 字节漏出，才敢说"扫全了"。

**闸门要拿负控制验过再信。** `audit_comment_language.py` 第一次报 PASS 时，
它是否真的抓得住违规是未知的——所以造了一个含中文注释 + 中文 docstring +
中文日志字符串的文件：前两者报 findings、后者放过、整体 FAIL，然后删掉。
**没验过的闸门等于没有闸门**，而"0 findings"这种通过条件恰恰最容易假成立。

两条工具向的坑，记下来省时间：

- **Python 3.11 没有 `tokenize.FSTRING_MIDDLE`**（那是 3.12 才有的）。
  3.11 里 f-string 是单个 STRING token，处理起来反而更简单。
- 要造一个含中文的临时样本文件，**别用 `python -c`**（引号会被 PowerShell 破坏，
  见陷阱二）。把中文写成 `\uXXXX` 转义、放进一个**纯 ASCII 的临时 `.py`** 里跑——
  这样脚本自身的编码就影响不到样本，与 1c 让 `apply_doc_edits.py` 保持纯 ASCII
  是同一条道理。

## 陷阱二：终端极不稳定

这台机器上 `execute_pwsh` 的表现：

- 输出经常为空，或回显被逐字符重复搞得面目全非。
- 退出码经常是 `-1`，**即使命令实际执行成功了**。不能用退出码判断成败。
- **重定向到文件也可能整个失效。** `python x.py > out.txt 2>&1` 跑完后 `out.txt`
  根本不存在。**要落文件就让脚本自己写，不要靠 shell 重定向。**
- **超过约 1 秒的命令会被掐断**，连带杀死子进程。`--collect-only`（0.2 秒）能出结果，
  跑完整测试套件（约 80 秒）不能。批量读几百张 PNG 也不能。
- **后台进程被复用时可能根本不重跑**，直接返回上一条命令的过期输出。实测踩中过：
  `isReused: true` 的进程没有重新执行，读到的是上一轮的 `suite_result.txt`。
  **启动后台进程时若返回 `isReused: true`，先 stop 再重开。**
- **三段链式命令会吞掉最后一步。** `ensure_utf8; compileall; run_suite` 这种写法
  前两步执行、第三步无声失败，`suite_result.txt` 根本不生成。**已稳定复现三次。**
  规则：`run_suite.py` **单独起一条命令**，不要和别的命令串在一起。
- 三引号、反引号在传给 `python -c` 时会被 shell 破坏。复杂脚本写成临时 `.py` 文件再跑，
  用完删掉。

**唯一可靠的做法：让结果落到文件，再用读文件的工具读回来。**

```powershell
python tools/run_suite.py            # 全量
python tools/run_suite.py tests/unit # 子集
```

结果写进 `reports/suite_result.txt`（`exit=` 与 `summary=` 两行），完整输出在
`reports/suite_output.txt`。这个脚本会在开始前自动清掉旧文件。

**实测有效的组合**：用后台进程启动 `run_suite.py`，终端回显全是乱码、退出码 `-1`、
`get_process_output` 也看不到结果，但约 50~160 秒后 `reports/suite_result.txt` 正常出现。
**别因为终端没动静就判定失败，去看文件。** `Get-ChildItem` 查文件存在性也会莫名返回空，
同样不能作为「文件没生成」的证据。

### 2a. 同时只能有一个 `run_suite.py` 在跑

**已造成实际损失。** 一个早前启动的 `run_suite.py` 一直挂着没退出，
新起的一条跑完写出了 `suite_result.txt`，随后**旧的那条把它删掉了**
（脚本开头会 `unlink` 自己的输出）。于是反复读到「文件不存在」，
看起来像新的那次跑失败了，其实它早就成功了。

**规则：起 `run_suite.py` 之前先 `list_processes`**，把状态还是 `running` 的
旧 suite 进程 stop 掉。除此之外，**绝不在 pytest 跑完前 stop 自己那条终端。**

短命令（`python -c ...` 级别）是可用的，回显虽乱但 stdout 能读到，可以拿来做编码检查
这类即时验证。

**新工具一律遵循这个形状**：干活写文件、只 print 一行 `wrote <路径>`。
`tools/` 下的 `survey_frames`、`probe_card_backs`、`probe_board_tracking`、
`calibrate_face_similarity`、`apply_doc_edits` 都是这么写的。
一次性的诊断脚本也照这个形状写，放 `reports/` 下用完即删。

## 陷阱三：读报告前必须先删旧文件

曾经因为读到上一轮的过期报告，得出过完全错误的结论并据此汇报。

`tools/run_suite.py` 已经内建了清理。如果你手写 `--junitxml=...`，**必须先删目标文件**。
另外不要整读 junit xml —— 它的第一行超过 30000 字符，用 grep 搜 `testsuite name`
取 `tests=` / `failures=` 即可。

新写的工具也要照做：都在开头 `unlink(missing_ok=True)` 掉自己的输出。

## 陷阱四：DPI 感知不一致

`QApplication` 会让进程变成 DPI-aware，于是在 150% 缩放下 `GetClientRect` 返回
**1536×1152**，而游戏实际只渲染 **1024×768** 并画在左上角。

修法：截图和点击**都要**通过 `dpi_unaware_thread()` 切到游戏的 DPI 上下文。
两边必须一致——只改一边会导致坐标系错位，那正是当初误触的一个来源。

安全网：`strip_black_padding` 只在"内容起于 (0,0) 且黑边超过 2%"时才裁剪，避免误伤
正常画面。

`tests/manual/test_eyes.py` **故意不做**这个修正，保留了当初暴露 bug 的那个差异。

## 陷阱五：`tests/manual/` 里的脚本会真的操作游戏

`tests/manual/test_eyes.py` 和 `test_hands.py` 的文件名匹配 `python_files = test_*.py`，
而它们会真的截图、真的点击客户区 (200, 200)。

两层防护已就位，**不要拆掉任何一层**：

1. 脚本主体在 `main()` 里，由 `if __name__ == "__main__"` 守卫，import 无副作用。
2. `pytest.ini` 的 `addopts` 含 `--ignore=tests/manual`。

## 陷阱六：手动执行的命令必须带上切目录

录制／标定脚本在终端里手动运行，而终端默认开在**主目录**，
不是项目根目录。曾因此报错 `can't open file '<主目录>\tools\grab_frame.py'`——
脚本相对路径被解析到了主目录下面。

终端是 **cmd**（不是 PowerShell），而项目不在系统盘，
所以跨盘切换**必须带 `/d`**：

```
cd /d "<仓库根目录>"
```

**规则：任何手动执行的命令段，都要把切目录那一行一起写出**，
并确认提示符已经变成仓库根目录再继续。只给脚本命令等于埋一个坑。
命令里用真实根目录，但**不要把它写进文档**。

### 每一段都要带，不是第一次带

新开的终端窗口会回到主目录，所以「上一段已经在正确目录」不构成省略的理由。
在主目录执行 `git init` 会在家目录建仓库，紧接着的 `git add .` 会开始把整个家目录
往里塞——实测写出 **17,907 个文件、303.6 MB** 的松散对象。

这类损失是可完全复原的，因为 `git add` 只读取并复制内容，**不修改、不移动、不删除文件**。
删掉那个 `.git` 就回到原点。真正的麻烦不是磁盘：**主目录下有 `.git` 会让此后
在家目录任何位置执行的 git 命令都以为自己在那个仓库里。**

### 自检必须标注停止条件

命令段里放一条能证明「位置正确」的命令，例如

```
git check-ignore -v captures reports
```

它在正确目录下会打印命中的 `.gitignore` 规则行，**在错误目录下什么都不输出**。

**规则：`check-ignore` 无输出 = 不在项目根目录，立即停止，不要执行后面的命令。**
只要命令段里包含一条自检，就要同时写明**它的失败表现是什么、以及失败时必须停下**。
一个没有标注停止条件的自检等于没有自检。

录制类命令还要一并交代：先手动把游戏界面点到目标状态；游戏窗口可被遮挡、可不在前台，
但**不能最小化**（最小化后客户区停止渲染）；`--duration 0` 靠 `Ctrl+C` 停止。

## 陷阱七：录制素材里混着退化帧与无关场景

一次录制会横跨多个无关画面，还会在窗口关闭瞬间产出畸形帧。
`captures/memory` 的第 262 帧只有 689 字节、客户区塌成一条细缝。

**规则：任何遍历 `captures/` 的分析都要先过滤退化帧**（尺寸过小、标准差接近 0、
读不出来），并确认自己处理的片段确实是目标场景。`tools/survey_frames.py` 就是为此存在的。

另一个坑：`grab_frame.py` 默认 `--dedup 1.5`，静止画面不存盘，
所以时间轴上十几秒的空隙是**正常的**，不是丢帧，别据此判断录制失败。

## 陷阱八：合成图只能验逻辑，不能定阈值，而且纯色会骗人

合成图**不允许**用来标定任何感知阈值。合成图不像真实的半透明动画精灵，
从它身上得到的任何阈值都是假的。真实标定素材在 `captures/`。

更隐蔽的一条：**用纯色块当合成素材会得到错误的行为，不只是错误的数字。**
归一化相关系数要除以各自的标准差，两块近似纯色的图之间它在数学上未定义，
OpenCV 返回的值不是测量结果。写决策循环的测试时用了纯色假卡面，
结果所有卡面互相"匹配"、循环把不成对的牌判成配对，排查了两轮才定位。

**规则：合成的图像素材一律带纹理。** 相关代码已加守卫
（`fingerprint.similarity()` 对标准差低于 3 的裁片返回 `UNCOMPARABLE`）。

另外：**假环境里注入的 `sleep` 什么都不做，所以墙上时间在单元测试里没有意义。**
要在测试里比较"哪种做法更省"，数**截图次数**——循环里的每一次等待都是一串截图，
那是唯一与机器无关的时钟。`Arena.captures` 就是为此加的。

## 陷阱九：前置断言可能断言了不成立的事

写测试的「前置条件」时当心。合成的混合编码样本**不一定**能被 gb18030 解出，
曾因此让两项测试在前置断言处就失败，而真正要考察的逻辑根本没跑到。
需要「能被回退编码解出的混合样本」时，是写脚本搜出来的，不是猜的。

## 陷阱十：`compileall` 抓不到坏掉的 import

`python -m compileall` 只做语法检查，**不执行 import**。删掉一个模块级常量之后，
别处 `from ... import 那个常量` 仍然能"编译通过"，直到 pytest 收集阶段才炸。

实际发生过：`memory_runner.py` 删掉 `CODE_GROUP_OVERFLOW` / `CODE_SOLVER_STUCK`，
`compileall` 干净通过，全量套件却在 collection 就 `1 error in 0.39s` 停住，
`tests/integration/test_ui_engine_lifecycle.py` 连一项都没跑。

**规则：删或改任何模块级公开符号后，跑全量套件，不要只看 `compileall`。**
`grep` 一下那个符号名也是两秒钟的事。

## 测试规程

- 全量应为 **472 项通过**，约 80 秒。构成：

```
既有（窗口/几何/视觉/校验/引擎/UI）  251
minigame 决策层（求解器 66 + 阅读顺序 18） 84
卡背检测与槽位追踪（25 + 14 + 3）    42
决策循环                             46
卡面聚类                             34
编码工具                             12
```

  历史基线：搬到 E 盘后 322，加编码测试 334，加感知层 371，加决策循环 392，
  加聚类 421，卡背补测 424，加速度优化与策略开关 448，加翻牌阶段优化 453，
  加关卡切换竞态 455，加机会数预算与杂框区分 462，加分组合并与失配黑名单 468，
  加卡背比对与读取推后 **472**。

- 每次改动后：`ensure_utf8.py` → `compileall` → `run_suite.py`（**三条分开跑**）。
  改过注释或 docstring 之后再加一条 `python tools/audit_comment_language.py`，
  必须 0 findings。**不要拿非 ASCII 字节数当这件事的判据**，理由见 1j。
- 测试要能在**没有 Qt、没有游戏**的情况下跑。决策逻辑抽成模块级纯函数并注入时钟，
  依赖注入替掉窗口、时钟和鼠标（见 `test_minigame_runner_control_loop.py` 的 `FakeBoard`）。
- `tests/real_frames/` 目前只有 README，全量测试**不依赖** `captures/`。
- 断言一律用稳定的 ASCII `code` 字段，**不要断言中文显示文案**。
  曾因整段日志做子串匹配（撞上"拉黑"二字）误报过失败；
  也曾因一条 `assert "秒" in ...` 把整个测试文件的编码带坏（见 1b）。
  **现状是这条规则有 23 处未遵守**，分布在 4 个测试文件里，全部是对中文日志
  或中文 summary 做子串匹配。它们让那 4 个文件继续留在编码陷阱内。
  没有顺手改，因为**不是每处都有 ASCII `code` 可断言**：`Verdict` / `GuardBlock`
  有 `code`，而 `bot_engine` 的日志行只是中文字符串，要断言 ASCII 就得先给日志
  加稳定标识——那是设计改动。清单与取舍记在 `roadmap.md` 的待办清单里。
- UI 集成测试会**真的起工作线程**，所以固定用一个不存在的窗口标题，
  让它们只能走"找不到窗口"那条分支。别把那个标题改成真的。

## 素材现状

`captures/` 已 gitignore。各目录帧数：

```
island1 118 ｜ island2 107 ｜ island3 35 ｜ small 94 ｜ calib_1024 86 ｜ calib_mixed 50
memory 263（完整 9 关 + 结算收尾，分析结论见 minigame_memory.md）
```

`assets/templates/coin.png`（50×38）是从真实帧裁出的金币图标，**不含石框**，
是目前唯一不可再生的资源。

`reports/card_backs/` 的 30 张标注图**保留**，是给人眼核对检测框用的，
运行时代码不读取 `reports/`。
