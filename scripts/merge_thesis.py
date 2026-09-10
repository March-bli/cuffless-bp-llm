"""合并旧稿 thesis.md 与新章节，生成最终论文 thesis_final.md"""
import re

BASE = "/home/lby/projects/bishe/docs"

with open(f"{BASE}/thesis.md", encoding="utf-8") as f:
    old = f.read()

with open(f"{BASE}/thesis_new_chapters.md", encoding="utf-8") as f:
    new = f.read()

# 从新文件中提取各部分
def extract_section(text, start_marker, end_marker=None):
    """提取从 start_marker 开始到 end_marker（不含）的文本。"""
    si = text.find(start_marker)
    if si == -1:
        return None
    if end_marker:
        ei = text.find(end_marker, si)
        return text[si:ei] if ei != -1 else text[si:]
    return text[si:]

# ── 从新文件提取各章节 ────────────────────────────────────────────────
new_abstract = extract_section(new, "# 摘要", "# 第一章")
new_1_3 = extract_section(new, "# 第一章 1.3", "# 第二章")
new_ch2_supp = extract_section(new, "# 第二章 2.3 补充", "# 第六章")
new_ch6 = extract_section(new, "# 第六章", "# 第七章")
new_ch7 = extract_section(new, "# 第七章", "# 第八章")
new_ch8 = extract_section(new, "# 第八章", "# 参考文献")
new_refs = extract_section(new, "# 参考文献")

for name, sec in [("abstract", new_abstract), ("1.3", new_1_3), ("ch2", new_ch2_supp),
                  ("ch6", new_ch6), ("ch7", new_ch7), ("ch8", new_ch8), ("refs", new_refs)]:
    print(f"{name}: {'OK' if sec else 'MISSING'} ({len(sec) if sec else 0} chars)")

# ── 替换旧稿对应章节 ───────────────────────────────────────────────────
def replace_between(text, start_marker, end_marker, replacement):
    """替换 text 中从 start_marker 到 end_marker（不含）之间的内容。"""
    si = text.find(start_marker)
    if si == -1:
        return text, False
    ei = text.find(end_marker, si)
    if ei == -1:
        return text, False
    return text[:si] + replacement + text[ei:], True

result = old

# 1. 替换摘要（"## 摘要" 到 "## 第一章"）
result, ok = replace_between(result, "## 摘要", "## 第一章", new_abstract)
print("替换摘要:", ok)

# 2. 替换 1.3（"### 1.3 本文主要工作" 到 "### 1.4"）
result, ok = replace_between(result, "### 1.3 本文主要工作", "### 1.4", new_1_3)
print("替换 1.3:", ok)

# 3. 插入第二章补充（"### 2.4 本章小结" 之前插入）
marker = "### 2.4 本章小结"
if marker in result:
    result = result.replace(marker, new_ch2_supp + "\n" + marker)
    print("插入第二章补充: OK")

# 4. 替换第六章（"## 第六章" 到 "## 第七章"）
result, ok = replace_between(result, "## 第六章", "## 第七章", new_ch6)
print("替换第六章:", ok)

# 5. 替换第七章（"## 第七章" 到 "## 第八章"）
result, ok = replace_between(result, "## 第七章", "## 第八章", new_ch7)
print("替换第七章:", ok)

# 6. 替换第八章（"## 第八章" 到 "## 致谢"）
result, ok = replace_between(result, "## 第八章", "## 致谢", new_ch8)
print("替换第八章:", ok)

# 7. 替换参考文献（"## 参考文献" 到 "## 附录"）
result, ok = replace_between(result, "## 参考文献", "## 附录", new_refs)
print("替换参考文献:", ok)

# 8. 更新标题
result = result.replace(
    "# 基于 PPG 信号的无袖带血压估计与 LLM 智能健康报告系统",
    "# Novel Blood Pressure Monitoring Using Wearable Devices and Artificial Intelligence (AI)\n# 基于可穿戴设备与人工智能的新型血压监测")

with open(f"{BASE}/thesis_final.md", "w", encoding="utf-8") as f:
    f.write(result)

print(f"\n✅ 合并完成 → thesis_final.md ({len(result.splitlines())} 行)")
