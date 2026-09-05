# media_code 项目长期约定

## commit 规范（用户明确要求，2026-09-01）
- 中文，格式 `<算法名>：简短描述`
- **默认只写标题行，不写正文**；正文最多 1-2 行（规范原文在 media_code/AGENTS.md 第 8 节）
- 历史教训：406153d 写了多段正文，被用户纠正

## 先读规范再动手（2026-09-05，项目工作纪律）
- **操作 git 前先读 AGENTS.md 的目录结构 / 命名 / commit 规范章节**，不要凭印象套用旧路径或旧约定
- **跑 WSL 训练前先读 README_wsl.md 的路径策略章节**（训练输出写 `~/output/` Linux fs，跑完用 08 搬到 `/mnt/d/`；不要直接写 drvfs/9p）
- 通用原则：动手前先读相关 README 章节，再下命令
- 历史教训（vggt_human）：
  - 09-05 误删整个 vggt_human 目录：没读 AGENTS.md 目录规范就 git rm + git restore，触发 WSL 9p 与 git 索引竞态，工作区整目录被识别为删除（git restore 救回）
  - 09-05 训练输出写错位置：没读 README_wsl.md 路径策略就把 RESULTS_DIR 指向 /mnt/d（drvfs），违反"训练写 Linux fs"规范（已搬运修正）
