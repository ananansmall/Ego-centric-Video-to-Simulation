# Git 使用指南 — Ego-Video-to-SIM 项目

## 一、新设备配置（首次使用必做）

### 1.1 安装 Git

```bash
# Ubuntu/Debian
sudo apt install git

# macOS
brew install git
```

### 1.2 配置 Git 用户信息

**必须和现有设备保持一致**，否则贡献者会分裂：

```bash
git config --global user.name "ananansmall"
git config --global user.email "1573880504@qq.com"
```

### 1.3 生成 SSH 密钥

```bash
# 生成密钥（用你的邮箱）
ssh-keygen -t ed25519 -C "1573880504@qq.com"
# 一路回车即可

# 查看公钥
cat ~/.ssh/id_ed25519.pub
```

### 1.4 添加 SSH 公钥到 GitHub

1. 复制上面输出的公钥内容
2. 打开 https://github.com/settings/keys
3. 点击 **New SSH key**
4. Title 填设备名（如 "Lab-PC"），Key 粘贴公钥
5. 点击 **Add SSH key**

### 1.5 验证 SSH 连接

```bash
ssh -T git@github.com
# 输出: Hi ananansmall! You've successfully authenticated...
```

### 1.6 克隆项目

```bash
# 克隆主仓库（含子模块）
git clone --recurse-submodules git@github.com:ananansmall/Ego-Video-to-SIM.git
cd Ego-Video-to-SIM

# 如果忘记 --recurse-submodules，补救：
git submodule update --init --recursive
```

---

## 二、项目仓库结构

```
Ego-Video-to-SIM/              ← 主仓库 (git@github.com:ananansmall/Ego-Video-to-SIM.git)
├── ReplicateAnyScene/          ← 子模块 (git@github.com:ananansmall/Ego-centric-Video-to-Simulation.git)
├── HaWoR/                      ← 子模块 (git@github.com:ananansmall/HaWoR.git)
├── combination/                ← 子模块
├── pv_retargeting/
├── libs/
└── SUBMODULE_GUIDE.md
```

每个子模块是**独立的 git 仓库**，有自己的 remote 和提交历史。

---

## 三、日常操作速查

### 3.1 修改 ReplicateAnyScene 代码并推送

```bash
# 进入子模块目录
cd ReplicateAnyScene/

# 修改代码...
vim src/some_file.py

# 提交
git add src/some_file.py
git commit -m "描述你的修改"

# 推送到 GitHub
git push origin main
```

### 3.2 修改 HaWoR 代码并推送

```bash
cd HaWoR/

# 修改代码...
vim demov2.py

# 提交
git add demov2.py
git commit -m "描述你的修改"

# 推送到你的 fork（注意 remote 名是 myfork）
git push myfork main
```

### 3.3 更新主仓库的子模块引用

子模块推送后，主仓库还指向旧版本，需要手动更新：

```bash
cd Ego-Video-to-SIM/

# 拉取子模块最新代码
cd ReplicateAnyScene/ && git pull origin main && cd ..
cd HaWoR/ && git pull origin main && cd ..

# 提交子模块引用变更
git add ReplicateAnyScene HaWoR
git commit -m "Update submodules"
git push origin main
```

---

## 四、两台设备协同工作

### 场景：设备 A 修改了代码，设备 B 要同步

```bash
# === 设备 B 操作 ===

# 1. 同步主仓库
cd Ego-Video-to-SIM/
git pull origin main

# 2. 同步子模块
git submodule update --remote

# 现在设备 B 拥有和设备 A 一样的代码了
```

### 场景：两台设备都修改了代码（冲突处理）

```bash
# === 设备 B 操作 ===

# 1. 先拉取远程最新代码
cd ReplicateAnyScene/
git pull origin main

# 2. 如果有冲突，Git 会提示
#    打开冲突文件，手动选择保留哪部分
#    冲突标记格式：
#    <<<<<<< HEAD
#    你的修改
#    =======
#    远程的修改
#    >>>>>>> origin/main

# 3. 解决冲突后
git add .
git commit -m "Merge: 解决冲突"
git push origin main
```

### 避免冲突的最佳实践

1. **每次开始工作前先 pull**：`git pull origin main`
2. **频繁提交推送**：改完一个功能就 push，不要攒太多
3. **不要同时改同一个文件的同一个地方**

---

## 五、完整工作流（从修改到推送）

### ReplicateAnyScene

```bash
cd ReplicateAnyScene/

# 1. 拉取最新代码（避免冲突）
git pull origin main

# 2. 修改代码
vim src/some_file.py

# 3. 查看修改
git status              # 看哪些文件改了
git diff                # 看具体改了什么

# 4. 提交
git add src/some_file.py    # 添加指定文件
# 或
git add -A                  # 添加所有修改（谨慎使用）

git commit -m "Fix: 修复xxx问题"

# 5. 推送
git push origin main

# 6. 更新主仓库引用
cd ../Ego-Video-to-SIM/
cd ReplicateAnyScene/ && git pull origin main && cd ..
git add ReplicateAnyScene
git commit -m "Update ReplicateAnyScene"
git push origin main
```

### HaWoR

```bash
cd HaWoR/

# 1. 拉取最新代码
git pull myfork main

# 2. 修改代码
vim demov2.py

# 3. 提交
git add demov2.py
git commit -m "Enhance: 改进xxx"

# 4. 推送
git push myfork main

# 5. 更新主仓库引用
cd ../Ego-Video-to-SIM/
cd HaWoR/ && git pull origin main && cd ..
git add HaWoR
git commit -m "Update HaWoR"
git push origin main
```

---

## 六、Remote 说明

| 仓库 | Remote 名 | URL | 说明 |
|---|---|---|---|
| ReplicateAnyScene | `origin` | `git@github.com:ananansmall/Ego-centric-Video-to-Simulation.git` | 你的仓库，直接 push |
| ReplicateAnyScene | `new-origin` | `git@github.com:ananansmall/Ego-Video-to-SIM.git` | 主仓库（一般不用） |
| HaWoR | `origin` | `https://github.com/ThunderVVV/HaWoR.git` | 上游仓库，**不能 push** |
| HaWoR | `myfork` | `git@github.com:ananansmall/HaWoR.git` | 你的 fork，push 到这里 |
| Ego-Video-to-SIM | `origin` | `git@github.com:ananansmall/Ego-Video-to-SIM.git` | 主仓库 |

**关键**：
- ReplicateAnyScene 推送用 `git push origin main`
- HaWoR 推送用 `git push myfork main`（不是 `origin`！）

---

## 七、常见问题

### Q1: push 时提示 Permission denied

```bash
# 检查 SSH 连接
ssh -T git@github.com

# 如果失败，检查密钥是否添加
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### Q2: push 时提示 failed to push some refs

```bash
# 远程有新提交，先拉取
git pull origin main    # ReplicateAnyScene
# 或
git pull myfork main    # HaWoR

# 然后再推送
git push origin main    # ReplicateAnyScene
# 或
git push myfork main    # HaWoR
```

### Q3: 不小心改了不想提交的文件

```bash
# 丢弃单个文件的修改
git checkout -- 文件名

# 丢弃所有修改（危险！）
git checkout .

# 暂存修改（安全，可以恢复）
git stash
# 恢复暂存
git stash pop
```

### Q4: 提交信息写错了

```bash
# 修改最近一次提交信息
git commit --amend -m "新的提交信息"

# 注意：如果已经 push 了，需要 force push
git push origin main --force
```

### Q5: 想撤销最近一次提交

```bash
# 撤销提交，保留修改
git reset --soft HEAD~1

# 撤销提交，丢弃修改（危险！）
git reset --hard HEAD~1
```

### Q6: 子模块目录为空

```bash
cd Ego-Video-to-SIM/
git submodule update --init --recursive
```

### Q7: 新设备上 git log 显示其他贡献者

确保新设备的 git 配置使用相同的邮箱：
```bash
git config --global user.email "1573880504@qq.com"
```

---

## 八、一键脚本

### 快速提交推送 ReplicateAnyScene

```bash
#!/bin/bash
# 文件: push_ras.sh
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene
git add -A
git commit -m "$1"
git push origin main
echo "ReplicateAnyScene pushed!"
```

用法：`bash push_ras.sh "修复xxx问题"`

### 快速提交推送 HaWoR

```bash
#!/bin/bash
# 文件: push_hawor.sh
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR
git add -A
git commit -m "$1"
git push myfork main
echo "HaWoR pushed!"
```

用法：`bash push_hawor.sh "改进xxx功能"`

### 同步所有子模块

```bash
#!/bin/bash
# 文件: sync_all.sh
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene
git pull origin main
cd ../HaWoR
git pull myfork main
echo "All synced!"
```
