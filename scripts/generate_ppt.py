"""
Generate defense PPT for graduation thesis:
  PPG-based Cuffless Blood Pressure Estimation with LLM Health Report
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
import os

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── Color scheme ──
BG_DARK = RGBColor(0x1A, 0x1A, 0x2E)       # Dark navy
BG_LIGHT = RGBColor(0xF8, 0xF9, 0xFA)        # Light gray
ACCENT_RED = RGBColor(0xE7, 0x4C, 0x3C)      # Red accent
ACCENT_BLUE = RGBColor(0x2E, 0x86, 0xC1)     # Blue accent
ACCENT_GREEN = RGBColor(0x27, 0xAE, 0x60)    # Green accent
ACCENT_ORANGE = RGBColor(0xF3, 0x9C, 0x12)   # Orange accent
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TEXT_DARK = RGBColor(0x2C, 0x3E, 0x50)
TEXT_GRAY = RGBColor(0x7F, 0x8C, 0x8D)
GOLD = RGBColor(0xF3, 0x9C, 0x12)

# ── Helper functions ──────────────────────────────────────────────────────

def add_bg(slide, color=BG_DARK):
    """Set slide background color."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_text(slide, left, top, width, height, text, font_size=24,
             color=WHITE, bold=False, alignment=PP_ALIGN.LEFT, font_name='Arial'):
    """Add a text box to a slide."""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top),
                                      Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return tf

def add_bullet_list(slide, left, top, width, height, items, font_size=18,
                    color=WHITE, spacing=Pt(6)):
    """Add bullet points."""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top),
                                      Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = 'Arial'
        p.space_after = spacing
        p.level = 0
    return tf

def add_section_header(slide, title, subtitle=""):
    """Add a section header bar at top."""
    # Top accent bar
    shape = slide.shapes.add_shape(
        1,  # Rectangle
        Inches(0), Inches(0), prs.slide_width, Inches(1.2)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT_BLUE
    shape.line.fill.background()

    add_text(slide, 0.8, 0.15, 11.7, 0.9, title,
             font_size=36, color=WHITE, bold=True)

    if subtitle:
        add_text(slide, 0.8, 0.9, 11.7, 0.4, subtitle,
                 font_size=16, color=RGBColor(0xBD, 0xC3, 0xC7))


def add_image_safe(slide, path, left, top, width=None, height=None):
    """Add image if it exists."""
    if os.path.exists(path):
        kwargs = {}
        if width: kwargs['width'] = Inches(width)
        if height: kwargs['height'] = Inches(height)
        slide.shapes.add_picture(path, Inches(left), Inches(top), **kwargs)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SLIDE 1: Title                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
add_bg(slide)

add_text(slide, 1.5, 1.5, 10.3, 1.0,
         "基于 PPG 信号的无袖带血压估计\n与 LLM 智能健康报告系统",
         font_size=40, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

add_text(slide, 1.5, 3.2, 10.3, 0.6,
         "Cuffless Blood Pressure Estimation from PPG Signals\nwith LLM-Driven Health Report Generation",
         font_size=20, color=TEXT_GRAY, alignment=PP_ALIGN.CENTER)

# Divider line
shape = slide.shapes.add_shape(1, Inches(5.5), Inches(4.0), Inches(2.3), Inches(0.03))
shape.fill.solid()
shape.fill.fore_color.rgb = ACCENT_BLUE
shape.line.fill.background()

add_text(slide, 4.0, 4.3, 5.3, 0.4, "毕业答辩", font_size=28, color=ACCENT_BLUE,
         bold=True, alignment=PP_ALIGN.CENTER)
add_text(slide, 4.0, 4.9, 5.3, 0.4, "指导教师：孙少雄  教授", font_size=18,
         color=TEXT_GRAY, alignment=PP_ALIGN.CENTER)
add_text(slide, 4.0, 5.4, 5.3, 0.4, "2026 年 7 月", font_size=16,
         color=TEXT_GRAY, alignment=PP_ALIGN.CENTER)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SLIDE 2: Outline                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, BG_LIGHT)
add_section_header(slide, "汇报提纲", "Outline")

items = [
    "1. 研究背景与动机",
    "2. 系统总体架构",
    "3. PulseDB 数据集与信号处理",
    "4. 多维 PPG 特征工程（55 维）",
    "5. 集成学习血压估计实验结果",
    "6. LLM 智能健康报告生成",
    "7. 消融实验与对比分析",
    "8. 总结与展望",
]
add_bullet_list(slide, 1.2, 1.8, 10.8, 5.0, items,
                font_size=24, color=TEXT_DARK, spacing=Pt(14))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SLIDE 3: Background & Motivation                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, BG_LIGHT)
add_section_header(slide, "研究背景与动机", "Background & Motivation")

# Left column: Problem
add_text(slide, 1.0, 1.8, 5.5, 0.4, "📌  传统血压测量的痛点",
         font_size=22, color=ACCENT_RED, bold=True)

items_left = [
    "• 心血管疾病：全球首要死因（32%），高血压影响 12.8 亿成年人",
    "• 不连续：袖带式单次测量无法捕捉昼夜波动",
    "• 不便携：需专门设备和操作环境",
    "• 不适感：充气压迫不适合夜间/频繁监测",
    "• 46% 患者不知晓自身高血压状况",
]
add_bullet_list(slide, 1.0, 2.5, 5.5, 3.5, items_left,
                font_size=16, color=TEXT_DARK, spacing=Pt(8))

# Right column: Opportunity
add_text(slide, 7.0, 1.8, 5.5, 0.4, "💡  PPG + LLM 的技术机遇",
         font_size=22, color=ACCENT_GREEN, bold=True)

items_right = [
    "• PPG 传感器：体积小、功耗低、成本低",
    "• 可穿戴设备普及（Apple Watch / Fitbit / 华为手环）",
    "• PPG 形态学特征与血压存在统计相关性",
    "• LLM 可将结构化数据转化为自然语言",
    "• 现有工作缺失：「信号 → 数值 → 语义」闭环",
]
add_bullet_list(slide, 7.0, 2.5, 5.5, 3.5, items_right,
                font_size=16, color=TEXT_DARK, spacing=Pt(8))

# Key insight box at bottom
shape = slide.shapes.add_shape(1, Inches(1.0), Inches(5.8), Inches(11.3), Inches(0.8))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0xEB, 0xF5, 0xFB)
shape.line.color.rgb = ACCENT_BLUE
add_text(slide, 1.2, 5.85, 10.9, 0.7,
         "核心问题：如何从 PPG 信号精确估计血压，并用自然语言呈现结果？",
         font_size=20, color=ACCENT_BLUE, bold=True, alignment=PP_ALIGN.CENTER)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SLIDE 4: System Architecture                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, BG_LIGHT)
add_section_header(slide, "系统总体架构", "Four-Stage Pipeline Architecture")

fig_path = "docs/figures/fig6_architecture.png"
if os.path.exists(fig_path):
    add_image_safe(slide, fig_path, left=0.5, top=1.5, width=12.3)
else:
    add_text(slide, 2, 3, 9, 2, "[系统架构图]",
             font_size=24, color=TEXT_GRAY, alignment=PP_ALIGN.CENTER)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SLIDE 5: Dataset & Signal Processing                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, BG_LIGHT)
add_section_header(slide, "PulseDB 数据集与信号处理", "Data & Signal Preprocessing")

# Left
add_text(slide, 1.0, 1.8, 5.5, 2.5,
         "PulseDB 数据集\n\n"
         "• 来源：MIMIC-III + VitalDB\n"
         "• 规模：5,361 人 / 524 万段 (完整版)\n"
         "• 本文：70 文件 → 3,591 段\n"
         "• 信号：PPG + ECG + ABP (125 Hz, 10s)\n"
         "• 标签：SegSBP / SegDBP",
         font_size=18, color=TEXT_DARK)

# Right
add_text(slide, 7.0, 1.8, 5.5, 4.5,
         "五阶段预处理流水线\n\n"
         "① 质量门控 (SNR + 覆盖率)\n"
         "② 基线漂移校正 (0.5 Hz 高通)\n"
         "③ 工频陷波 (50/60 Hz Notch)\n"
         "④ 运动伪影检测 (滑动窗口 Z-score)\n"
         "⑤ 尖峰剔除 (中值滤波 MAD)\n\n"
         "峰值检测: NeuroKit2 + SciPy 双策略",
         font_size=18, color=TEXT_DARK)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SLIDE 6: Feature Engineering                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, BG_LIGHT)
add_section_header(slide, "55 维 PPG 特征工程", "Feature Engineering")

categories = [
    ("时域形态学 (16)", "收缩峰幅度、脉幅、\n脉搏间期、收缩上升时间、\n脉宽25/50/75%、AIx、SI", ACCENT_RED),
    ("导数 VPG+APG (18) ⭐", "VPG 极值/均值/标准差；\nAPG a-b-c-d-e 波均值/标准差；\nb/a、c/a、d/a、e/a 比值", ACCENT_GREEN),
    ("全局统计 (14)", "均值、标准差、偏度、峰度、\n百分位数 p5/p25/p75/p95、\nRMS、信息熵", ACCENT_BLUE),
    ("频域 (7)", "频谱总能量、心率频带功率比、\n主频/主频功率、频谱质心、\nLF/HF 比", ACCENT_ORANGE),
]

for idx, (title, desc, color) in enumerate(categories):
    x = 0.8 + idx * 3.1
    # Title box
    shape = slide.shapes.add_shape(1, Inches(x), Inches(1.8), Inches(2.8), Inches(0.7))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    add_text(slide, x + 0.1, 1.9, 2.6, 0.5, title,
             font_size=18, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

    # Desc box
    shape = slide.shapes.add_shape(1, Inches(x), Inches(2.65), Inches(2.8), Inches(3.5))
    shape.fill.solid()
    shape.fill.fore_color.rgb = WHITE
    shape.line.color.rgb = color
    add_text(slide, x + 0.2, 2.85, 2.4, 3.1, desc,
             font_size=15, color=TEXT_DARK)

# Bottom key insight
add_text(slide, 1.0, 6.4, 11.3, 0.5,
         "⭐ APG 二阶导数特征是最重要的特征组（SHAP 分析验证）",
         font_size=18, color=ACCENT_RED, bold=True, alignment=PP_ALIGN.CENTER)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SLIDE 7: ML Results - Main Table                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, BG_LIGHT)
add_section_header(slide, "血压估计实验结果", "ML Model Comparison (5-Fold CV)")

# Table
from pptx.util import Emu
rows_data = [
    ["模型", "SBP MAE", "DBP MAE", "SBP R²", "DBP R²", "SBP BHS", "DBP BHS"],
    ["RF", "9.04", "5.48", "0.625", "0.570", "D", "B"],
    ["XGBoost ⭐", "8.02", "5.00", "0.689", "0.628", "C", "A"],
    ["SVR", "9.46", "5.48", "0.543", "0.528", "D", "B"],
    ["KNN (k=5)", "8.18", "5.11", "0.632", "0.557", "D", "B"],
    ["MLP", "9.83", "6.02", "0.563", "0.511", "D", "B"],
]

n_rows = len(rows_data)
n_cols = len(rows_data[0])
table = slide.shapes.add_table(n_rows, n_cols, Inches(1.5), Inches(1.8),
                                Inches(10.3), Inches(3.5)).table

for r in range(n_rows):
    for c in range(n_cols):
        cell = table.cell(r, c)
        cell.text = rows_data[r][c]
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(16) if r == 0 else Pt(15)
            p.font.name = 'Arial'
            p.alignment = PP_ALIGN.CENTER
            if r == 0:
                p.font.bold = True
                p.font.color.rgb = WHITE
            elif r == 2:  # XGBoost row
                p.font.bold = True
                p.font.color.rgb = ACCENT_GREEN
            else:
                p.font.color.rgb = TEXT_DARK

        # Header color
        if r == 0:
            cell.fill.solid()
            cell.fill.fore_color.rgb = ACCENT_BLUE
        elif r == 2:
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xE8, 0xF8, 0xF5)
        # Highlight BHS A
        if c == 6 and rows_data[r][c] == 'A':
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0x27, 0xAE, 0x60)
            for p in cell.text_frame.paragraphs:
                p.font.color.rgb = WHITE
                p.font.bold = True

# Key metrics highlight
add_text(slide, 1.5, 5.6, 10.3, 0.5,
         "🏆 XGBoost: DBP MAE=5.00 mmHg | BHS Grade A | AAMI 通过 ✅",
         font_size=22, color=ACCENT_GREEN, bold=True, alignment=PP_ALIGN.CENTER)
add_text(slide, 1.5, 6.2, 10.3, 0.4,
         "📊 SBP MAE=8.02 mmHg | BHS Grade C | 仍有提升空间",
         font_size=18, color=TEXT_DARK, alignment=PP_ALIGN.CENTER)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SLIDE 8: BHS + Bland-Altman                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, BG_LIGHT)
add_section_header(slide, "BHS 评级与 Bland-Altman 分析", "BHS Grade & Consistency Analysis")

# Left: BHS table
add_text(slide, 1.0, 1.8, 5.0, 0.4, "XGBoost BHS 分级详情",
         font_size=20, color=TEXT_DARK, bold=True)

bhs_rows = [
    ["指标", "≤5mmHg", "≤10mmHg", "≤15mmHg", "BHS"],
    ["SBP", "45.8%", "72.2%", "85.5%", "C"],
    ["DBP", "61.2%", "87.1%", "95.3%", "A"],
]
table = slide.shapes.add_table(3, 5, Inches(1.0), Inches(2.4),
                                Inches(5.0), Inches(2.0)).table
for r in range(3):
    for c in range(5):
        cell = table.cell(r, c)
        cell.text = bhs_rows[r][c]
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(14)
            p.alignment = PP_ALIGN.CENTER
        if r == 0:
            cell.fill.solid()
            cell.fill.fore_color.rgb = ACCENT_BLUE
            for p in cell.text_frame.paragraphs:
                p.font.color.rgb = WHITE
                p.font.bold = True
        if r == 2 and c == 4:
            cell.fill.solid()
            cell.fill.fore_color.rgb = ACCENT_GREEN
            for p in cell.text_frame.paragraphs:
                p.font.color.rgb = WHITE
                p.font.bold = True

# Right: Bland-Altman key numbers
add_text(slide, 7.0, 1.8, 5.5, 4.0,
         "Bland-Altman 分析（XGBoost）\n\n"
         "SBP:\n"
         "  Bias: +0.08 mmHg (无偏)\n"
         "  LoA: [−22.1, +22.3] mmHg\n"
         "  ±5 mmHg: 45.8%\n\n"
         "DBP:\n"
         "  Bias: −0.01 mmHg (无偏)\n"
         "  LoA: [−14.9, +14.9] mmHg\n"
         "  ±5 mmHg: 61.2%\n\n"
         "✅ DBP 满足 AAMI 标准 (ME<5, SDE<8)",
         font_size=18, color=TEXT_DARK)

# Figure
fig_path = "docs/figures/fig1_bland_altman.png"
if os.path.exists(fig_path):
    add_image_safe(slide, fig_path, left=7.0, top=4.5, width=5.5)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SLIDE 9: SHAP Feature Importance                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, BG_LIGHT)
add_section_header(slide, "SHAP 特征重要性分析", "SHAP Feature Importance")

fig_path = "docs/figures/fig2_shap_importance.png"
if os.path.exists(fig_path):
    add_image_safe(slide, fig_path, left=0.3, top=1.5, width=12.7)

add_text(slide, 1.0, 6.5, 11.3, 0.6,
         "核心发现：APG 二阶导数特征（a/b/c/d/e 波）是 SBP 和 DBP 估计中最重要的特征组",
         font_size=18, color=ACCENT_BLUE, bold=True, alignment=PP_ALIGN.CENTER)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SLIDE 10: Ablation Studies                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, BG_LIGHT)
add_section_header(slide, "消融实验", "Ablation Studies")

fig_path = "docs/figures/fig4_ablation.png"
if os.path.exists(fig_path):
    add_image_safe(slide, fig_path, left=0.5, top=1.5, width=12.3)

add_text(slide, 1.0, 6.5, 11.3, 0.6,
         "APG 导数特征独立 SBP R²=0.592 | 全特征融合 R²=0.689 | 数据量增加持续提升性能",
         font_size=18, color=ACCENT_BLUE, bold=True, alignment=PP_ALIGN.CENTER)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SLIDE 11: LLM Health Report                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, BG_LIGHT)
add_section_header(slide, "LLM 智能健康报告生成", "LLM-Driven Health Report Generation")

# Left
add_text(slide, 0.8, 1.8, 5.8, 5.0,
         "设计思路\n\n"
         "• 临床知识层：AHA/ESC/NICE 指南\n"
         "  → 5 级风险分级 (danger/warning/caution/healthy)\n\n"
         "• Prompt 工程：三段式设计\n"
         "  ① 系统角色（专业健康助手）\n"
         "  ② 结构化数据编码（风险排序）\n"
         "  ③ 输出规范（语言/长度/免责声明）\n\n"
         "• 多后端支持\n"
         "  DeepSeek / xAI Grok / OpenAI 兼容\n\n"
         "• 兜底策略：API 不可用时自动回退至模板",
         font_size=17, color=TEXT_DARK)

# Right: Comparison table
add_text(slide, 7.3, 1.8, 5.3, 0.4, "LLM vs 模板 质量对比",
         font_size=18, color=TEXT_DARK, bold=True)

comp_rows = [
    ["评估维度", "模板 NL", "LLM NL"],
    ["自然流畅度", "机械拼接", "流畅叙述 ✅"],
    ["多指标关联", "逐项罗列", "综合分析 ✅"],
    ["个性化建议", "固定模板", "指标组合定制 ✅"],
    ["病理机理解释", "仅告知状态", "解释原因 ✅"],
    ["报告长度", "~1112 字符", "~83 字符（精炼）"],
    ["语气共情度", "清单式", "温和共情 ✅"],
]
table = slide.shapes.add_table(7, 3, Inches(7.3), Inches(2.4),
                                Inches(5.3), Inches(3.5)).table
for r in range(7):
    for c in range(3):
        cell = table.cell(r, c)
        cell.text = comp_rows[r][c]
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(12)
            p.alignment = PP_ALIGN.CENTER
        if r == 0:
            cell.fill.solid()
            cell.fill.fore_color.rgb = ACCENT_BLUE
            for p in cell.text_frame.paragraphs:
                p.font.color.rgb = WHITE
                p.font.bold = True


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SLIDE 12: Comparison with Existing Work                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, BG_LIGHT)
add_section_header(slide, "与现有工作对比", "Comparison with State-of-the-Art")

comp_rows = [
    ["方法", "数据来源", "信号", "SBP MAE", "DBP MAE"],
    ["Wang 2023 (RF)", "PulseDB 5,361人", "PPG", "~12.1", "~5.6"],
    ["Liu 2024 (LLM-BP)", "PulseDB 子集", "PPG→LLM", "7.1", "5.3"],
    ["本文 XGBoost", "PulseDB 70文件", "PPG", "8.02", "5.00 ⭐"],
]
table = slide.shapes.add_table(4, 5, Inches(1.5), Inches(2.2),
                                Inches(10.3), Inches(2.5)).table
for r in range(4):
    for c in range(5):
        cell = table.cell(r, c)
        cell.text = comp_rows[r][c]
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(18)
            p.alignment = PP_ALIGN.CENTER
        if r == 0:
            cell.fill.solid()
            cell.fill.fore_color.rgb = ACCENT_BLUE
            for p in cell.text_frame.paragraphs:
                p.font.color.rgb = WHITE
                p.font.bold = True
        if r == 3:
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xE8, 0xF8, 0xF5)

add_text(slide, 1.5, 5.5, 10.3, 1.5,
         "DBP 优于 Wang 2023 (5.00 vs 5.6) 和 Liu 2024 (5.00 vs 5.3)\n"
         "SBP 接近 Liu 2024 (8.02 vs 7.1)，显著优于 Wang 2023 (8.02 vs 12.1)\n"
         "本文方法仅使用 3,591 段数据，推理延迟 < 1ms，便于端侧部署",
         font_size=18, color=TEXT_DARK, alignment=PP_ALIGN.CENTER)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SLIDE 13: Summary & Contributions                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, BG_LIGHT)
add_section_header(slide, "工作总结与贡献", "Summary & Contributions")

contributions = [
    ("1", "PPG 信号处理流水线",
     "五阶段清洗 + 双策略峰值检测 + 55 维特征工程\n覆盖时域/导数/统计/频域四个维度"),
    ("2", "集成学习血压估计",
     "5 模型系统对比 → XGBoost 最优\nDBP: BHS Grade A + AAMI 通过 ✅"),
    ("3", "LLM 健康报告生成",
     "临床知识层 + 三段式 Prompt 工程\n多后端支持 + 自动模板兜底"),
    ("4", "双模态闭环系统",
     "「信号 → 数值 → 语义」完整流水线\n模块化设计，代码开源可复用"),
]

for idx, (num, title, desc) in enumerate(contributions):
    y = 1.8 + idx * 1.4
    # Number badge
    shape = slide.shapes.add_shape(1, Inches(1.0), Inches(y), Inches(0.6), Inches(0.6))
    shape.fill.solid()
    shape.fill.fore_color.rgb = ACCENT_BLUE
    shape.line.fill.background()
    add_text(slide, 1.05, y + 0.05, 0.5, 0.5, num,
             font_size=20, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER)

    add_text(slide, 2.0, y - 0.05, 3.5, 0.5, title,
             font_size=20, color=TEXT_DARK, bold=True)
    add_text(slide, 2.0, y + 0.5, 10.0, 0.6, desc,
             font_size=15, color=TEXT_GRAY)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SLIDE 14: Limitations & Future Work                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide, BG_LIGHT)
add_section_header(slide, "局限性与未来工作", "Limitations & Future Directions")

items = [
    "📊  数据规模：仅 PulseDB 70 文件子集 → 完整版 524 万段有望大幅提升 SBP 精度",
    "",
    "🎯  SBP 精度瓶颈：当前 BHS Grade C → 端到端深度学习 / 个体校准机制",
    "",
    "🤖  LLM 可靠性：定性评估 → 临床医生双盲评分 / 医学知识图谱 RAG 增强",
    "",
    "🔬  多模态融合：仅 PPG 信号 → 融合加速度计/温度等多传感器数据",
    "",
    "📱  端侧部署：云端 API 依赖 → 小参数 LLM 量化 + 端侧推理框架",
]
add_bullet_list(slide, 1.2, 1.8, 10.8, 5.0, items,
                font_size=20, color=TEXT_DARK, spacing=Pt(10))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SLIDE 15: Thank You                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide)

add_text(slide, 2.0, 2.0, 9.3, 1.5, "感谢聆听",
         font_size=56, color=WHITE, bold=True, alignment=PP_ALIGN.CENTER,
         font_name='SimHei')

add_text(slide, 2.0, 3.5, 9.3, 0.6,
         "Thank You — Questions & Discussion Welcome",
         font_size=24, color=TEXT_GRAY, alignment=PP_ALIGN.CENTER)

shape = slide.shapes.add_shape(1, Inches(5.5), Inches(4.3), Inches(2.3), Inches(0.03))
shape.fill.solid()
shape.fill.fore_color.rgb = ACCENT_BLUE
shape.line.fill.background()

add_text(slide, 3.0, 4.8, 7.3, 0.4,
         "基于 PPG 信号的无袖带血压估计与 LLM 智能健康报告系统",
         font_size=18, color=TEXT_GRAY, alignment=PP_ALIGN.CENTER)
add_text(slide, 3.0, 5.3, 7.3, 0.4,
         "GitHub: [项目链接]  |  Email: [邮箱]",
         font_size=14, color=TEXT_GRAY, alignment=PP_ALIGN.CENTER)


# ── Save ──────────────────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)
output_path = "output/defense.pptx"
prs.save(output_path)
print(f"\n✅ PPT saved to: {output_path}")
print(f"   Total slides: {len(prs.slides)}")
