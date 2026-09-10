"""修复 thesis_final.md 中新章节的标题层级（统一降一级）"""
import re

PATH = "/home/lby/projects/bishe/docs/thesis_final.md"
with open(PATH, encoding="utf-8") as f:
    text = f.read()

# 针对新插入内容的特定标题修复
fixes = [
    ("# 第一章 1.3 本文主要工作（更新版）", "### 1.3 本文主要工作"),
    ("# 第二章 2.3 补充：大语言模型与传感器/生理信号建模\n", ""),
    ("## 2.3.4 传感器语言对齐：SensorLM", "### 2.3.4 传感器语言对齐：SensorLM"),
    ("## 2.3.5 语义瓶颈与强化学习：TimeSRL", "### 2.3.5 语义瓶颈与强化学习：TimeSRL"),
    ("## 2.3.6 对本文的启示", "### 2.3.6 对本文的启示"),
    ("# 第六章 基于大语言模型的血压估计", "## 第六章 基于大语言模型的血压估计"),
    ("# 第七章 语义抽象增强", "## 第七章 语义抽象增强"),
    ("# 第八章 总结与展望", "## 第八章 总结与展望"),
    ("# 参考文献（更新版）", "## 参考文献"),
]

for old, new in fixes:
    if old in text:
        text = text.replace(old, new)
        print(f"修复: {old[:40]}...")
    else:
        print(f"未找到: {old[:40]}...")

# 第六章、第七章、第八章内的节标题：## → ###（仅新章节范围）
# 这些标题是 "## 6.1"、"## 7.1"、"## 8.1" 等
for pat, rep in [
    (r"^## (6\.\d)", r"### \1"),
    (r"^## (7\.\d)", r"### \1"),
    (r"^## (8\.\d)", r"### \1"),
]:
    text = re.sub(pat, rep, text, flags=re.MULTILINE)

# 小节标题：### 6.2.1 → #### 6.2.1
for pat, rep in [
    (r"^### (6\.\d+\.\d+)", r"#### \1"),
    (r"^### (7\.\d+\.\d+)", r"#### \1"),
]:
    text = re.sub(pat, rep, text, flags=re.MULTILINE)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(text)

print("\n✅ 标题层级修复完成")
