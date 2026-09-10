"""
Clinical reference ranges and project configuration.
Based on AHA/ESC/NICE guidelines.
"""

# AHA/ESC Clinical Reference Ranges
CLINICAL_RANGES = {
    "heart_rate": {
        "unit": "bpm",
        "ranges": [
            ("bradycardia", 0, 60, "danger"),
            ("low_normal", 60, 65, "caution"),
            ("normal", 65, 100, "healthy"),
            ("tachycardia", 100, 200, "danger"),
        ],
        "suggestion_map": {
            "bradycardia": "Consult a cardiologist if this persists, especially if accompanied by dizziness or fatigue.",
            "low_normal": "Your rate is at the low end of normal. This is fine, especially if you are physically active.",
            "normal": "Your heart is functioning within a healthy resting range.",
            "tachycardia": "Elevated resting heart rate detected. Consider reducing caffeine, stress, and consult a GP if persistent.",
        },
    },
    "systolic_bp": {
        "unit": "mmHg",
        "ranges": [
            ("hypotension", 0, 90, "danger"),
            ("normal", 90, 120, "healthy"),
            ("elevated", 120, 130, "caution"),
            ("stage1_hypertension", 130, 140, "warning"),
            ("stage2_hypertension", 140, 300, "danger"),
        ],
        "suggestion_map": {
            "hypotension": "Low blood pressure. Stay hydrated and rise slowly from sitting. Consult a GP if you feel faint.",
            "normal": "Your systolic pressure is in the healthy range.",
            "elevated": "Slightly elevated. Consider reducing sodium intake and increasing physical activity.",
            "stage1_hypertension": "Stage 1 hypertension. Lifestyle changes recommended: reduce salt, exercise regularly, manage stress.",
            "stage2_hypertension": "Stage 2 hypertension. Please consult your GP for a management plan.",
        },
    },
    "diastolic_bp": {
        "unit": "mmHg",
        "ranges": [
            ("hypotension", 0, 60, "danger"),
            ("normal", 60, 80, "healthy"),
            ("elevated", 80, 90, "caution"),
            ("stage1_hypertension", 90, 100, "warning"),
            ("stage2_hypertension", 100, 200, "danger"),
        ],
        "suggestion_map": {
            "hypotension": "Diastolic pressure is low. Monitor for symptoms like lightheadedness.",
            "normal": "Your diastolic pressure is in the healthy range.",
            "elevated": "Diastolic pressure is slightly elevated. Reduce sodium and manage stress.",
            "stage1_hypertension": "Stage 1 hypertension indicated by diastolic reading. Lifestyle modification is advised.",
            "stage2_hypertension": "Stage 2 hypertension on diastolic. Please seek medical advice.",
        },
    },
    "spo2": {
        "unit": "%",
        "ranges": [
            ("critical", 0, 90, "danger"),
            ("low", 90, 95, "caution"),
            ("normal", 95, 101, "healthy"),
        ],
        "suggestion_map": {
            "critical": "CRITICAL: Low oxygen saturation. Seek immediate medical attention.",
            "low": "Oxygen saturation is below normal. If you have respiratory symptoms, consult a GP.",
            "normal": "Blood oxygen level is within normal range.",
        },
    },
    "hrv_sdnn": {
        "unit": "ms",
        "ranges": [
            ("low", 0, 50, "caution"),
            ("moderate", 50, 100, "healthy"),
            ("high", 100, 300, "healthy"),
        ],
        "suggestion_map": {
            "low": "Low heart rate variability may indicate elevated stress or fatigue. Consider relaxation techniques.",
            "moderate": "Heart rate variability is in a healthy moderate range.",
            "high": "High heart rate variability — typically a sign of good cardiovascular fitness and autonomic balance.",
        },
    },
}

METRIC_NAMES_DISPLAY = {
    "heart_rate": "Heart Rate",
    "systolic_bp": "Systolic Blood Pressure",
    "diastolic_bp": "Diastolic Blood Pressure",
    "spo2": "Blood Oxygen Saturation (SpO2)",
    "hrv_sdnn": "Heart Rate Variability (SDNN)",
}

INTRO_TEMPLATE = "Here is your health summary based on a {duration}-second measurement:"
OUTRO_TEMPLATE = (
    "\nReminder: This is not a medical diagnosis. "
    "Please consult a healthcare professional for clinical assessment."
)
