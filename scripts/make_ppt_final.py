"""按导师意见重做的 8 页项目展示 PPT"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

BLUE = RGBColor(0x04, 0x5E, 0xCE)
DARK = RGBColor(0x1F, 0x3A, 0x5F)
LIGHT_BLUE = RGBColor(0xE8, 0xF0, 0xFE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x22, 0x22, 0x22)
RED = RGBColor(0xD6, 0x27, 0x28)
GREEN = RGBColor(0x2E, 0x7D, 0x32)

FIG = "/home/lby/projects/bishe/docs/figures"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_text(slide, text, left, top, width, height, size=18, bold=False,
             color=BLACK, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.text = text
    for p in tf.paragraphs:
        p.alignment = align
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = color
        p.font.name = "Calibri"
    return box


def add_rect(slide, left, top, width, height, fill_color):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    return shape


def add_title_bar(slide, title):
    add_rect(slide, 0, 0, 13.333, 1.1, BLUE)
    add_text(slide, title, 0.5, 0.25, 12.3, 0.6, size=30, bold=True, color=WHITE)


# ═══════════════════════════════════════════════════════════════════
# 第 1 页：标题
# ═══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_rect(s, 0, 0, 13.333, 7.5, BLUE)
add_text(s, "Novel Blood Pressure Monitoring Using\nWearable Devices and Artificial Intelligence",
         1.0, 1.6, 11.3, 2.2, size=34, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_text(s, "Boyuan Li", 1.0, 4.0, 11.3, 0.6, size=26, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_text(s, "Supervisor: Dr. Shaoxiong Sun", 1.0, 4.7, 11.3, 0.5, size=20, color=WHITE, align=PP_ALIGN.CENTER)
add_text(s, "MSc Artificial Intelligence  |  Department of Computer Science  |  University of Sheffield",
         1.0, 5.4, 11.3, 0.5, size=16, color=LIGHT_BLUE, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════════
# 第 2 页：为什么需要无袖带 BP 监测
# ═══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Why Cuffless Blood Pressure Monitoring?")

add_text(s, "Hypertension is the leading modifiable risk factor\nfor cardiovascular disease worldwide.",
         0.8, 1.6, 6.5, 1.4, size=22, bold=True, color=BLUE)

add_text(s, "The problem with cuff-based measurement:", 0.8, 3.2, 6.5, 0.5, size=19, bold=True, color=DARK)
add_text(s, "• Intermittent — only occasional snapshots\n"
             "• Inconvenient — requires a cuff and quiet rest\n"
             "• Misses night-time and daily fluctuations",
         1.0, 3.8, 6.3, 1.6, size=17, color=BLACK)

add_rect(s, 7.4, 1.5, 5.2, 4.3, LIGHT_BLUE)
add_text(s, "The opportunity", 7.6, 1.7, 4.8, 0.5, size=19, bold=True, color=DARK)
add_text(s, "PPG — the optical signal already\n"
             "in smartwatches for heart rate —\n"
             "enables continuous, cuffless,\n"
             "everyday BP monitoring.",
         7.6, 2.4, 4.8, 2.0, size=16, color=BLACK)
add_text(s, "Goal: clinical-grade BP\nfrom wrist-worn PPG",
         7.6, 4.6, 4.8, 1.0, size=17, bold=True, color=GREEN)

# ═══════════════════════════════════════════════════════════════════
# 第 3 页：现有方案 + LLM 动机
# ═══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Existing Solutions and the Motivation for LLMs")

add_text(s, "Existing cuffless BP approaches:", 0.8, 1.4, 6.0, 0.5, size=19, bold=True, color=DARK)
add_text(s, "• PTT-based (pulse transit time)\n"
             "   — needs ECG + PPG, frequent recalibration\n"
             "• Traditional ML on PPG features\n"
             "   — strong within-subject, weak cross-subject",
         1.0, 2.0, 6.2, 2.0, size=16, color=BLACK)

add_text(s, "Why LLMs?", 7.4, 1.4, 5.2, 0.5, size=19, bold=True, color=BLUE)
add_text(s, "• In-context learning: adapts from a\n"
             "   few examples without fine-tuning\n"
             "• Pre-trained with broad reasoning ability\n"
             "• Recent work: SensorLM, TimeSRL\n"
             "   apply LLMs to physiological signals",
         7.6, 2.0, 5.2, 2.4, size=15, color=BLACK)
add_rect(s, 7.4, 4.7, 5.2, 1.2, LIGHT_BLUE)
add_text(s, "Idea: use the LLM's in-context learning\n"
             "for PERSONALISED calibration",
         7.6, 4.9, 4.8, 0.9, size=15, bold=True, color=DARK)

# ═══════════════════════════════════════════════════════════════════
# 第 4 页：数据集
# ═══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Dataset: PulseDB (MIMIC-III + VitalDB)")

add_text(s, "PulseDB — a large, cleaned dataset for cuffless BP estimation", 0.8, 1.5, 12.0, 0.6, size=20, bold=True, color=BLUE)

add_text(s, "Subjects", 0.8, 2.4, 2.5, 0.5, size=17, bold=True, color=DARK)
add_text(s, "5,159", 0.8, 2.95, 2.5, 0.8, size=30, bold=True, color=BLUE)
add_text(s, "PPG segments", 4.0, 2.4, 3.0, 0.5, size=17, bold=True, color=DARK)
add_text(s, "5.55 million", 4.0, 2.95, 3.0, 0.8, size=26, bold=True, color=BLUE)
add_text(s, "Segment length", 8.0, 2.4, 2.5, 0.5, size=17, bold=True, color=DARK)
add_text(s, "10 seconds\n@ 125 Hz", 8.0, 2.95, 2.5, 1.0, size=18, bold=True, color=BLUE)
add_text(s, "Signals", 11.0, 2.4, 2.0, 0.5, size=17, bold=True, color=DARK)
add_text(s, "PPG / ECG / ABP", 11.0, 2.95, 2.0, 0.8, size=15, bold=True, color=BLUE)

add_text(s, "Features & protocol:", 0.8, 4.3, 12.0, 0.5, size=18, bold=True, color=DARK)
add_text(s, "• 55 hand-crafted PPG features (morphology, APG, statistics, frequency) + age & gender\n"
             "• Strict subject-level split: 20 test subjects, the rest for training\n"
             "• Calibration = first K chronological segments of the target subject (K = 1/3/5/10/20)",
         1.0, 4.8, 11.5, 1.7, size=16, color=BLACK)

# ═══════════════════════════════════════════════════════════════════
# 第 5 页：瓶颈论证 + BHS 标准
# ═══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "The Bottleneck: Individual Physiology, Not Model Capacity")

add_text(s, "Evidence from our experiments:", 0.8, 1.4, 6.0, 0.5, size=19, bold=True, color=DARK)
add_text(s, "XGBoost trained on 5,139 subjects\n"
             "→ SBP MAE still 14.9 mmHg\n"
             "→ data and model are NOT the problem",
         1.0, 2.0, 6.0, 1.7, size=17, color=BLACK)
add_text(s, "Bias correction with only K=20 samples\n"
             "of the target subject\n"
             "→ SBP MAE drops to 7.0 mmHg\n"
             "→ the individual OFFSET is the key",
         1.0, 3.9, 6.0, 1.7, size=17, color=GREEN)

# 右栏：BHS 标准 + XGBoost global 结果
add_text(s, "BHS grading standard", 7.4, 1.35, 5.2, 0.4, size=15, bold=True, color=DARK)
bhs_std = [
    ("Grade", "≤5", "≤10", "≤15", True),
    ("A", "60%", "85%", "95%", False),
    ("B", "50%", "75%", "90%", False),
    ("C", "40%", "65%", "85%", False),
]
y = 1.75
for g, p5, p10, p15, hl in bhs_std:
    add_rect(s, 7.4, y, 5.2, 0.36, BLUE if hl else WHITE)
    c = WHITE if hl else BLACK
    add_text(s, g, 7.5, y + 0.03, 1.1, 0.3, size=11, bold=hl, color=c)
    add_text(s, p5, 8.7, y + 0.03, 1.1, 0.3, size=11, bold=hl, color=c)
    add_text(s, p10, 9.9, y + 0.03, 1.1, 0.3, size=11, bold=hl, color=c)
    add_text(s, p15, 11.1, y + 0.03, 1.1, 0.3, size=11, bold=hl, color=c)
    y += 0.38

add_text(s, "XGBoost global → fails all grades", 7.4, 3.42, 5.2, 0.4, size=14, bold=True, color=RED)
xgb_rows = [
    ("", "≤5", "≤10", "≤15", "Grade", True),
    ("SBP", "30.6%", "48.2%", "60.0%", "✗", False),
    ("DBP", "21.2%", "50.6%", "65.9%", "✗", False),
]
y2 = 3.82
for name, p5, p10, p15, grade, hl in xgb_rows:
    add_rect(s, 7.4, y2, 5.2, 0.36, RED if hl else LIGHT_BLUE)
    c = WHITE if hl else BLACK
    add_text(s, name, 7.45, y2 + 0.03, 0.8, 0.3, size=11, bold=hl, color=c)
    add_text(s, p5, 8.25, y2 + 0.03, 1.1, 0.3, size=11, color=c)
    add_text(s, p10, 9.45, y2 + 0.03, 1.1, 0.3, size=11, color=c)
    add_text(s, p15, 10.65, y2 + 0.03, 1.1, 0.3, size=11, color=c)
    add_text(s, grade, 11.8, y2 + 0.03, 0.8, 0.3, size=11, bold=True, color=c)
    y2 += 0.38

# ═══════════════════════════════════════════════════════════════════
# 第 6 页：方法（两种 LLM）
# ═══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Method: Two LLM Calibration Approaches")

# 流程
steps = [
    ("PPG signal\n(10s, 125Hz)", 0.6),
    ("55 features\n+ age & gender", 3.4),
    ("LLM\n(Qwen3-8B)", 6.2),
    ("SBP / DBP\nprediction", 9.0),
]
for text, x in steps:
    add_rect(s, x, 1.5, 2.6, 1.3, LIGHT_BLUE)
    add_text(s, text, x + 0.1, 1.65, 2.4, 1.0, size=15, bold=True, color=DARK, align=PP_ALIGN.CENTER)
for x in [3.25, 6.05, 8.85]:
    add_text(s, "→", x, 1.75, 0.35, 0.8, size=28, bold=True, color=BLUE)

# 左：数值 few-shot
add_rect(s, 0.6, 3.3, 5.9, 3.4, WHITE)
add_rect(s, 0.6, 3.3, 5.9, 0.55, BLUE)
add_text(s, "Approach 1: Numeric few-shot", 0.8, 3.38, 5.5, 0.4, size=17, bold=True, color=WHITE)
add_text(s, "Feed raw numbers as in-context examples:\n"
             "“HR 72, AIx 75%, b/a −0.35 … → BP 130/85”\n\n"
             "The LLM learns the individual mapping\n"
             "from the first K samples.",
         0.8, 4.0, 5.5, 2.4, size=15, color=BLACK)

# 右：语义描述
add_rect(s, 6.85, 3.3, 5.9, 3.4, WHITE)
add_rect(s, 6.85, 3.3, 5.9, 0.55, GREEN)
add_text(s, "Approach 2: Semantic description (ours)", 7.05, 3.38, 5.5, 0.4, size=17, bold=True, color=WHITE)
add_text(s, "Translate features into clinical language,\n"
             "RETAINING quantitative anchors:\n\n"
             "“moderate arterial stiffness (AIx 75%);\n"
             " normal vascular resistance (b/a −0.35)”",
         7.05, 4.0, 5.5, 2.4, size=15, color=BLACK)

# ═══════════════════════════════════════════════════════════════════
# 第 7 页：结果
# ═══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Results: Semantic Description Wins (K=20)")

# 左：校准曲线
s.shapes.add_picture(f"{FIG}/fig1_calibration_curve.png", Inches(0.35), Inches(1.35), width=Inches(5.7))

# 右上：K=20 对比表
add_text(s, "All methods at K=20 (mmHg)", 6.35, 1.35, 6.6, 0.45, size=18, bold=True, color=DARK)
rows = [
    ("Method", "SBP", "DBP", True),
    ("XGBoost global", "14.9", "11.8", False),
    ("XGBoost bias-corr", "7.0", "4.2", False),
    ("LLM numeric", "8.4", "6.7", False),
    ("LLM semantic ★", "6.5", "3.6", True),
]
y = 1.85
for name, sbp, dbp, hl in rows:
    add_rect(s, 6.35, y, 6.6, 0.42, BLUE if hl else (LIGHT_BLUE if name != "Method" else WHITE))
    c = WHITE if hl else BLACK
    add_text(s, name, 6.45, y + 0.04, 3.4, 0.34, size=13 if name != "Method" else 12,
             bold=hl or name == "Method", color=c)
    add_text(s, sbp, 9.9, y + 0.04, 1.2, 0.34, size=13, bold=hl, color=c)
    add_text(s, dbp, 11.6, y + 0.04, 1.2, 0.34, size=13, bold=hl, color=c)
    y += 0.45

# 右下：BHS 分布
add_text(s, "BHS grades — semantic K=20", 6.35, 4.25, 6.6, 0.4, size=14, bold=True, color=DARK)
bhs_rows = [
    ("", "≤5", "≤10", "≤15", "Grade", True),
    ("SBP", "53.7%", "83.6%", "91.0%", "B", False),
    ("DBP", "76.1%", "89.6%", "98.5%", "A", True),
]
by = 4.7
for name, p5, p10, p15, grade, hl in bhs_rows:
    add_rect(s, 6.35, by, 6.6, 0.42, GREEN if (grade == "A" and hl) else (BLUE if hl else LIGHT_BLUE))
    c = WHITE if hl else BLACK
    add_text(s, name, 6.45, by + 0.04, 1.2, 0.34, size=12, bold=hl, color=c)
    add_text(s, p5, 7.7, by + 0.04, 1.2, 0.34, size=12, bold=hl, color=c)
    add_text(s, p10, 8.9, by + 0.04, 1.2, 0.34, size=12, bold=hl, color=c)
    add_text(s, p15, 10.1, by + 0.04, 1.2, 0.34, size=12, bold=hl, color=c)
    add_text(s, grade, 11.6, by + 0.04, 1.2, 0.34, size=12, bold=hl, color=c)
    by += 0.45

# 底部
add_rect(s, 0.35, 6.35, 12.6, 0.85, GREEN)
add_text(s, "DBP meets Grade A (76% ≤5)  |  SBP meets Grade B  |  Semantic abstraction beats numeric learning",
         0.55, 6.42, 12.2, 0.4, size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_text(s, "BHS rule:  Grade A 60/85/95%   ·   Grade B 50/75/90%   ·   Grade C 40/65/85%   (≤5 / ≤10 / ≤15 mmHg)",
         0.55, 6.85, 12.2, 0.32, size=12, color=WHITE, align=PP_ALIGN.CENTER)

# ═══════════════════════════════════════════════════════════════════
# 第 8 页：结论
# ═══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
add_title_bar(s, "Conclusions")

findings = [
    ("1", "Cross-subject generalisation is the real bottleneck",
     "The individual offset, not data or model capacity, is the key."),
    ("2", "The LLM needs NO fine-tuning",
     "Pre-trained and ready to use — only prompt design, unlike XGBoost trained on 5,139 subjects."),
    ("3", "Semantic description = interpretable + accurate",
     "Transparent clinical language, not a black box — DBP meets BHS Grade A (6.5/3.6)."),
]
y = 1.5
for num, title, desc in findings:
    add_rect(s, 0.6, y, 0.6, 0.6, BLUE)
    add_text(s, num, 0.6, y + 0.05, 0.6, 0.5, size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text(s, title, 1.4, y, 11.3, 0.5, size=20, bold=True, color=DARK)
    add_text(s, desc, 1.4, y + 0.5, 11.3, 0.6, size=15, color=BLACK)
    y += 1.5

add_rect(s, 0.6, 6.1, 12.1, 0.8, LIGHT_BLUE)
add_text(s, "Future: reduce calibration cost  •  on-device deployment  •  learned (not hand-crafted) semantic rules",
         0.9, 6.3, 11.5, 0.4, size=15, bold=True, color=DARK, align=PP_ALIGN.CENTER)

prs.save("/home/lby/projects/bishe/docs/project_presentation_final.pptx")
print("✅ 8 页 PPT 已生成")
