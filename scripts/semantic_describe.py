"""
Semantic PPG Descriptor
=======================
把 55 维 PPG 数值特征映射为层次化语义描述（借鉴 SensorLM）。
三层：统计层 / 结构层 / 语义层。

规则基于心血管生理学常识（用于 BP 估计）。
"""

# 特征名 → 医学术语
TERMS = {
    "heart_rate": "heart rate",
    "pulse_interval_mean": "pulse interval",
    "aix": "augmentation index",
    "stiffness_index": "arterial stiffness index",
    "upstroke_time_mean": "upstroke time",
    "diastolic_time_mean": "diastolic time",
    "pulse_width_75pct": "pulse width (75%)",
    "pulse_width_50pct": "pulse width (50%)",
    "pulse_width_25pct": "pulse width (25%)",
    "apg_a_mean": "APG a-wave",
    "apg_b_mean": "APG b-wave",
    "apg_c_mean": "APG c-wave",
    "apg_d_mean": "APG d-wave",
    "apg_e_mean": "APG e-wave",
    "apg_b_a_ratio": "APG b/a ratio",
    "apg_c_a_ratio": "APG c/a ratio",
    "apg_d_a_ratio": "APG d/a ratio",
    "apg_e_a_ratio": "APG e/a ratio",
    "sys_amp_mean": "systolic amplitude",
    "pulse_amp_mean": "pulse amplitude",
    "ppg_skew": "PPG skewness",
    "ppg_kurtosis": "PPG kurtosis",
    "dominant_freq": "dominant frequency",
    "lf_hf_ratio": "LF/HF ratio",
}


def describe_heart_rate(hr):
    if hr < 60:
        return f"bradycardia (HR {hr:.0f} bpm)"
    elif hr <= 100:
        return f"normal heart rate (HR {hr:.0f} bpm)"
    else:
        return f"tachycardia (HR {hr:.0f} bpm)"


def describe_aix(aix):
    if aix < 60:
        return f"low arterial stiffness (AIx {aix:.0f}%)"
    elif aix <= 80:
        return f"moderate arterial stiffness (AIx {aix:.0f}%)"
    else:
        return f"high arterial stiffness (AIx {aix:.0f}%)"


def describe_b_a_ratio(r):
    if r < -0.4:
        return f"low vascular resistance (b/a {r:.2f})"
    elif r <= -0.2:
        return f"normal vascular resistance (b/a {r:.2f})"
    else:
        return f"high vascular resistance (b/a {r:.2f})"


def describe_stiffness(s):
    if s is None or (isinstance(s, float) and s != s):
        return None
    if s < 6:
        return f"soft arteries (stiffness {s:.1f})"
    elif s <= 8:
        return f"moderate arteries (stiffness {s:.1f})"
    else:
        return f"stiff arteries (stiffness {s:.1f})"


def describe_upstroke(t):
    if t is None or (isinstance(t, float) and t != t):
        return None
    if t < 150:
        return f"strong cardiac contractility (upstroke {t:.0f} ms)"
    elif t <= 250:
        return f"normal contractility (upstroke {t:.0f} ms)"
    else:
        return f"weak contractility (upstroke {t:.0f} ms)"


def describe_pulse_width(w):
    if w is None or (isinstance(w, float) and w != w):
        return None
    if w < 250:
        return f"narrow pulse width ({w:.0f} ms)"
    elif w <= 400:
        return f"normal pulse width ({w:.0f} ms)"
    else:
        return f"wide pulse width ({w:.0f} ms)"


def build_semantic_description(feats):
    """把数值特征 dict 转成三层语义描述字符串。"""
    stat_lines = []
    struct_lines = []
    sem_lines = []

    hr = feats.get("heart_rate")
    if hr is not None and hr == hr:
        stat_lines.append(f"heart rate {hr:.1f} bpm")
    pi = feats.get("pulse_interval_mean")
    if pi is not None and pi == pi:
        stat_lines.append(f"pulse interval {pi:.1f} ms")
    pw = feats.get("pulse_width_50pct")
    if pw is not None and pw == pw:
        stat_lines.append(f"pulse width {pw:.1f} ms")

    # 结构层：波形形态
    for k in ["apg_b_a_ratio", "apg_c_a_ratio", "sys_amp_mean"]:
        v = feats.get(k)
        if v is not None and v == v:
            struct_lines.append(f"{TERMS[k]} {v:.3f}")
    sk = feats.get("ppg_skew")
    ku = feats.get("ppg_kurtosis")
    if sk is not None and sk == sk:
        struct_lines.append(f"PPG skewness {sk:.2f}")
    if ku is not None and ku == ku:
        struct_lines.append(f"PPG kurtosis {ku:.2f}")

    # 语义层：生理推断
    if hr is not None and hr == hr:
        sem_lines.append(describe_heart_rate(hr))
    aix = feats.get("aix")
    if aix is not None and aix == aix:
        sem_lines.append(describe_aix(aix))
    bar = feats.get("apg_b_a_ratio")
    if bar is not None and bar == bar:
        sem_lines.append(describe_b_a_ratio(bar))
    st = feats.get("stiffness_index")
    ds = describe_stiffness(st)
    if ds:
        sem_lines.append(ds)
    ut = feats.get("upstroke_time_mean")
    du = describe_upstroke(ut)
    if du:
        sem_lines.append(du)

    text = ""
    if stat_lines:
        text += "[Statistical] " + "; ".join(stat_lines) + "\n"
    if struct_lines:
        text += "[Structural] " + "; ".join(struct_lines) + "\n"
    if sem_lines:
        text += "[Semantic] " + "; ".join(sem_lines)
    return text


if __name__ == "__main__":
    # 示例
    demo = {
        "heart_rate": 72.5, "pulse_interval_mean": 827.6,
        "pulse_width_50pct": 280.0, "aix": 75.0,
        "apg_b_a_ratio": -0.35, "stiffness_index": 7.2,
        "upstroke_time_mean": 180.0,
        "apg_c_a_ratio": 0.1, "sys_amp_mean": 1.2,
        "ppg_skew": 0.3, "ppg_kurtosis": 3.5,
    }
    print(build_semantic_description(demo))
