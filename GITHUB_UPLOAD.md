# 上传 GitHub

本目录已经初始化为 `main` 分支的 Git 仓库，但尚未创建 commit，也没有绑定远程仓库。

## 1. 检查许可证和署名

仓库已采用 Apache-2.0（软件）与 CC BY 4.0（原创课程内容）。FSDD 数据仍为上游 CC BY-SA 4.0。发布时必须保留 `LICENSE`、`LICENSE-CONTENT`、`LICENSE-SCOPE.md`、`NOTICE` 和 `DATA_SOURCES.md`。

## 2. 本地最终检查

```powershell
uv sync --locked
uv run python scripts/validate_course.py
git status
```

确认 `.env`、密钥、个人录音和无授权数据没有被列入。

## 3. 创建第一次提交

```powershell
git add .
git commit -m "course: publish ASR lessons 1-41"
```

## 4. 创建并推送 GitHub 仓库

如果安装并登录了 GitHub CLI：

```powershell
gh auth login
gh repo create learn-asr --public --source . --remote origin --push
```

也可以先在 GitHub 网页创建空仓库，然后运行：

```powershell
git remote add origin https://github.com/YOUR_NAME/learn-asr.git
git push -u origin main
```

将 `YOUR_NAME` 替换成自己的 GitHub 用户名。不要把 token 写进 remote URL、Notebook 或配置文件。

## 5. 推送后检查

- README 是否正确显示；
- 中文 Notebook 文件名和公式是否正常；
- Actions 中 `course-check` 是否通过；
- `DATA_SOURCES.md` 和选择的许可证是否存在；
- 仓库中没有 `.venv`、`.env` 或个人数据。
