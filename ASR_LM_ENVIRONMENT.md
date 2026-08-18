# ASR 语言模型课程：安装与排错

本课程的 Notebook 内核运行在 Windows Python 中，并通过 `wsl.exe` 调用 Linux 里的 OpenFst、KenLM 和 `kaldilm`。这种方式便于在 Windows 上交互学习，同时使用成熟的 Linux ASR 工具链。

本说明以 Windows 11、WSL2 和名为 `Ubuntu` 的发行版为准。第 1～2 课无需 WSL；第 3～8 课需要本页工具。

## 1. 安装 Python 课程环境

先安装 [uv](https://docs.astral.sh/uv/)，然后在 PowerShell 中运行：

```powershell
Set-Location -LiteralPath '<你的 learn-asr 仓库目录>'
uv sync --locked
uv run python --version
uv run jupyter lab
```

`uv.lock` 会锁定 Python 依赖。不要在 Notebook 中逐个 `pip install`，否则很难复现实验。

## 2. 安装并确认 WSL Ubuntu

在管理员 PowerShell 中运行：

```powershell
wsl --install -d Ubuntu
wsl --list --verbose
```

首次安装后可能需要重启并创建 Linux 用户。本课程第 3～8 课的辅助函数明确调用 `wsl -d Ubuntu`，因此 `wsl --list --quiet` 中必须出现完全相同的 `Ubuntu`。如果你的发行版叫 `Ubuntu-24.04`，可以另装名为 `Ubuntu` 的发行版，或把 Notebook 中 `['wsl', '-d', 'Ubuntu', '--', ...]` 的名称改成实际名称。

## 3. 安装 OpenFst 和构建依赖

打开 Ubuntu 终端：

```bash
sudo apt-get update
sudo apt-get install -y \
  libfst-tools graphviz git build-essential cmake \
  libboost-system-dev libboost-thread-dev \
  libboost-program-options-dev libboost-test-dev \
  libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev \
  python3-venv
```

Ubuntu 24.04 的仓库通常提供 OpenFst 1.7.9；本课程已用这一版本执行通过。OpenFst 官网截至 2026-08-18 的源码版本是 1.8.5，但学习本课程不必为了版本号自行升级，因为这里使用的基础命令在 1.7.9 上已经满足要求。

验证：

```bash
fstcompile --help >/dev/null
fstcompose --help >/dev/null
fstshortestpath --help >/dev/null
command -v fstcompile fstprint fstinfo fstarcsort fstcompose
```

## 4. 把 KenLM 构建到课程固定路径

第 5 课使用以下固定路径：

- `/opt/kenlm/build/bin/lmplz`
- `/opt/kenlm/build/bin/build_binary`
- `/opt/kenlm/build/bin/query`

在 Ubuntu 中运行：

```bash
sudo git clone https://github.com/kpu/kenlm.git /opt/kenlm
sudo chown -R "$USER":"$USER" /opt/kenlm
cmake -S /opt/kenlm -B /opt/kenlm/build
cmake --build /opt/kenlm/build --parallel 4
```

如果 `/opt/kenlm` 已存在，不要再次 clone；先运行验证命令。KenLM 官方目前推荐 CMake 的 out-of-tree build，上述依赖和命令与官方 `BUILDING` 一致。

验证：

```bash
/opt/kenlm/build/bin/lmplz --help | head
/opt/kenlm/build/bin/build_binary 2>&1 | head
printf 'hello world\n' | /opt/kenlm/build/bin/lmplz -o 2 >/tmp/tiny.arpa
test -s /tmp/tiny.arpa && echo 'KenLM OK'
```

## 5. 安装 `kaldilm`

第 5 课调用 `/opt/kaldilm-venv/bin/python -m kaldilm` 把 ARPA 转成 OpenFst `G.fst`。在 Ubuntu 中运行：

```bash
sudo python3 -m venv /opt/kaldilm-venv
sudo chown -R "$USER":"$USER" /opt/kaldilm-venv
/opt/kaldilm-venv/bin/python -m pip install --upgrade pip
/opt/kaldilm-venv/bin/python -m pip install 'kaldilm==1.15.4'
/opt/kaldilm-venv/bin/python -m kaldilm --help
```

课程锁定已验证的 `kaldilm 1.15.4`，避免之后的接口变化影响 Notebook。

## 6. 从 PowerShell 做一次总检查

```powershell
wsl --list --quiet
wsl -d Ubuntu -- which fstcompile
wsl -d Ubuntu -- test -x /opt/kenlm/build/bin/lmplz
wsl -d Ubuntu -- test -x /opt/kenlm/build/bin/build_binary
wsl -d Ubuntu -- /opt/kaldilm-venv/bin/python -m kaldilm --help
uv run python scripts/validate_lm_course.py
```

这些命令都成功后，再从项目根目录启动 Jupyter，并先运行第 3 课的环境检查 cell。

## 7. Windows 路径怎样映射到 WSL

Notebook 会通过 `wslpath` 把当前 Windows 仓库路径转换成对应的 `/mnt/<盘符>/...` 路径。因此：

- 项目必须在 WSL 能挂载的 Windows 盘符下；
- 文件名可包含中文，但写 OpenFst 文本时必须使用 UTF-8 和 LF；
- 不要在 PowerShell 中手工拼 `/mnt/g/...`，Notebook 的 `to_wsl_path()` 会统一处理。

## 常见故障

### `WslRegisterDistribution failed` 或没有 `Ubuntu`

先运行 `wsl --status` 和 `wsl --list --verbose`。确保已启用虚拟化和 WSL2，并确认发行版名称与 Notebook 的 `Ubuntu` 完全一致。

### `which fstcompile` 没有输出

在 Ubuntu 中重新运行 `sudo apt-get update` 和 `sudo apt-get install -y libfst-tools`。不要把 Windows 上另一个同名程序加入 PATH 来冒充 OpenFst。

### `apt-get` 出现 404

包索引过期时先运行 `sudo apt-get update`。仍失败时检查 Ubuntu 版本是否已停止支持，以及 `/etc/apt/sources.list*` 是否指向有效镜像。

### 第 5 课显示 `/opt/kenlm/... MISSING`

运行：

```bash
ls -l /opt/kenlm/build/bin/{lmplz,build_binary,query}
cmake --build /opt/kenlm/build --parallel 4
```

若路径不同，最稳妥的做法是按本页重建到 `/opt/kenlm`，而不是在每个 Notebook 中改三处路径。

### `No module named _kaldilm`

PyPI 的排错建议是删除损坏安装后显示详细编译日志：

```bash
/opt/kaldilm-venv/bin/python -m pip uninstall -y kaldilm
/opt/kaldilm-venv/bin/python -m pip install -v --no-cache-dir 'kaldilm==1.15.4'
```

### 第 6～8 课提示缺少 `lesson05` 文件

这些课不是独立运行的。先完整运行第 5 课，确认 `openfst_lab/lesson05/` 中存在 `words.txt`、`tiny.1gram.arpa`、`tiny.2gram.arpa`、`tiny.3gram.arpa` 和生成的 FST。

### `Symbol table`、`label` 或 compose 后图为空

先检查：epsilon 是否为编号 0；两张图要连接的 symbol table 是否逐项一致；`L` 是否按输出标签排序、`G` 是否按输入标签排序；`#0` 自环是否仍存在。不要先尝试提高 beam，这类错误属于图接口错误。

### Jupyter 找不到项目根目录

关闭 Jupyter，回到你克隆的 `learn-asr` 仓库根目录重新运行 `uv run jupyter lab`。不要从临时目录或下载目录打开单个 Notebook。

## 官方资料

- [OpenFst 官方下载页](https://www.openfst.org/twiki/bin/view/FST/FstDownload)
- [KenLM 官方仓库与构建说明](https://github.com/kpu/kenlm)
- [kaldilm PyPI 页面](https://pypi.org/project/kaldilm/)
- [Kaldi 测试时 HCLG 图构建说明](https://kaldi-asr.org/doc/graph_recipe_test.html)
