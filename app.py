from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import gradio as gr
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

SYMPTOMS_FILE = DATA_DIR / "DiseaseAndSymptoms.csv"
PRECAUTIONS_FILE = DATA_DIR / "Disease precaution.csv"

MODEL_FILE = MODELS_DIR / "disease_prediction_model.pkl"
FEATURES_FILE = MODELS_DIR / "feature_columns.pkl"
CLASSES_FILE = MODELS_DIR / "class_names.pkl"


def clean_text(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    x = " ".join(x.split())
    return x


def load_raw_data():
    if not SYMPTOMS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {SYMPTOMS_FILE}")
    if not PRECAUTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {PRECAUTIONS_FILE}")

    symptoms_df = pd.read_csv(SYMPTOMS_FILE)
    precautions_df = pd.read_csv(PRECAUTIONS_FILE)
    return symptoms_df, precautions_df


def preprocess_symptom_data(symptoms_df):
    df = symptoms_df.copy()
    df["Disease"] = df["Disease"].apply(clean_text)

    symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]
    for col in symptom_cols:
        df[col] = df[col].apply(clean_text)

    all_symptoms = sorted(
        {
            symptom
            for col in symptom_cols
            for symptom in df[col].dropna().unique()
        }
    )

    rows = []
    for _, row in df.iterrows():
        disease = row["Disease"]
        selected = {row[col] for col in symptom_cols if pd.notna(row[col])}

        one_hot = {"disease": disease}
        for symptom in all_symptoms:
            one_hot[symptom] = 1 if symptom in selected else 0
        rows.append(one_hot)

    processed_df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    X = processed_df.drop(columns=["disease"])
    y = processed_df["disease"]

    return X, y, list(X.columns)


def train_and_save_model():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    symptoms_df, _ = load_raw_data()
    X, y, feature_columns = preprocess_symptom_data(symptoms_df)

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(feature_columns, FEATURES_FILE)
    joblib.dump(list(model.classes_), CLASSES_FILE)

    return model, feature_columns, list(model.classes_)


def load_or_train_model():
    if MODEL_FILE.exists() and FEATURES_FILE.exists() and CLASSES_FILE.exists():
        model = joblib.load(MODEL_FILE)
        feature_columns = joblib.load(FEATURES_FILE)
        class_names = joblib.load(CLASSES_FILE)
        return model, feature_columns, class_names

    print("Model files not found. Training model from included dataset...")
    return train_and_save_model()


def load_precautions():
    _, precautions_df = load_raw_data()

    df = precautions_df.copy()
    if "Disease" in df.columns:
        df["Disease"] = df["Disease"].apply(clean_text)
        df.rename(columns={"Disease": "disease"}, inplace=True)
    else:
        df["disease"] = df["disease"].apply(clean_text)

    precaution_cols = [c for c in df.columns if "Precaution" in c or "precaution" in c]
    for col in precaution_cols:
        df[col] = df[col].apply(clean_text)

    disease_to_precautions = {}
    for _, row in df.iterrows():
        items = []
        for col in precaution_cols:
            if pd.notna(row[col]) and str(row[col]).strip() != "":
                items.append(str(row[col]))
        disease_to_precautions[row["disease"]] = items

    return disease_to_precautions


model, feature_columns, class_names = load_or_train_model()
disease_to_precautions = load_precautions()

symptom_choices = sorted(feature_columns)


def pretty_symptom_name(symptom):
    return symptom.replace("_", " ").title()


dropdown_choices = [(pretty_symptom_name(s), s) for s in symptom_choices]


def get_top3_predictions(selected_symptoms):
    input_df = pd.DataFrame([[0] * len(feature_columns)], columns=feature_columns)

    valid_selected = []
    for symptom in selected_symptoms:
        if symptom in input_df.columns:
            input_df.at[0, symptom] = 1
            valid_selected.append(symptom)

    probs = model.predict_proba(input_df)[0]
    classes = model.classes_
    top3_idx = np.argsort(probs)[-3:][::-1]

    results = []
    for idx in top3_idx:
        results.append({
            "disease": classes[idx],
            "confidence": round(float(probs[idx] * 100), 2)
        })

    return results, valid_selected


def get_precautions(disease_name):
    disease_name = clean_text(disease_name)
    return disease_to_precautions.get(disease_name, ["no precautions found for this disease."])


def build_explanation(selected_symptoms, top_disease):
    selected_symptoms = [s.replace("_", " ") for s in selected_symptoms[:6]]
    top_disease = top_disease.replace("_", " ")

    if not selected_symptoms:
        return "No symptoms were selected."

    symptom_text = ", ".join(selected_symptoms)
    return (
        "The prediction was made by comparing the selected symptoms with patterns "
        f"learned from the training data. In this case, symptoms such as {symptom_text} "
        f"matched most strongly with {top_disease}."
    )


def build_warning(selected_symptoms):
    red_flags = {
        "chest_pain": "Chest pain may need urgent medical attention.",
        "shortness_of_breath": "Breathing difficulty may need urgent medical attention.",
        "vomiting": "Persistent vomiting can lead to dehydration."
    }

    found = []
    for symptom in selected_symptoms:
        if symptom in red_flags:
            found.append(f"- {symptom.replace('_', ' ')}: {red_flags[symptom]}")

    if not found:
        return ""

    return (
        "### Warning\n"
        + "\n".join(found)
        + "\n\nPlease consult a doctor urgently if symptoms are severe or worsening."
    )


def render_prediction_cards(results):
    if not results:
        return """
        <div class="empty-state">
            <div class="empty-title">No prediction yet</div>
            <div class="empty-text">Select symptoms and click <b>Predict Conditions</b>.</div>
        </div>
        """

    medals = ["🥇", "🥈", "🥉"]
    html = '<div class="pred-grid">'

    for i, item in enumerate(results[:3]):
        disease = str(item["disease"]).replace("_", " ").title()
        confidence = float(item["confidence"])
        badge = medals[i] if i < len(medals) else "•"

        html += f"""
        <div class="pred-card">
            <div class="pred-top">
                <div class="pred-rank">{badge} Rank {i+1}</div>
                <div class="pred-score">{confidence:.2f}%</div>
            </div>
            <div class="pred-disease">{disease}</div>
            <div class="progress-wrap">
                <div class="progress-bar" style="width: {confidence}%;"></div>
            </div>
        </div>
        """

    html += "</div>"
    return html


def render_precautions_html(top_disease, precautions, warning_text):
    disease_name = str(top_disease).replace("_", " ").title()

    precaution_items = ""
    for item in precautions:
        precaution_items += f"<li>{str(item).capitalize()}</li>"

    warning_block = ""
    if warning_text.strip():
        warning_block = f"""
        <div class="warning-box">
            {warning_text.replace('### Warning', '<div class="warning-title">Warning</div>').replace(chr(10), '<br>')}
        </div>
        """

    return f"""
    <div class="content-card-inner precautions-text">
        <div class="content-heading">Basic Precautions for {disease_name}</div>
        <ul class="clean-list">
            {precaution_items}
        </ul>
        {warning_block}
    </div>
    """


def render_explanation_html(explanation_text):
    return f"""
    <div class="content-card-inner">
        <div class="content-heading">Why this prediction?</div>
        <p class="content-paragraph">{explanation_text}</p>
    </div>
    """


def render_disclaimer_html():
    return """
    <div class="content-card-inner">
        <div class="content-heading">Safety Disclaimer</div>
        <p class="content-paragraph">
            This tool is for <b>educational purposes only</b>. It is <b>not a medical diagnosis</b>
            and does not replace a licensed doctor. Always consult a healthcare professional for
            proper diagnosis and treatment. If symptoms are severe, worsening, or urgent, seek medical care immediately.
        </p>
    </div>
    """


def symptom_checker_ui(selected_symptoms):
    if selected_symptoms is None:
        selected_symptoms = []

    if len(selected_symptoms) == 0:
        empty_html = """
        <div class="empty-state">
            <div class="empty-title">No symptoms selected</div>
            <div class="empty-text">Please choose one or more symptoms to continue.</div>
        </div>
        """
        return empty_html, empty_html, empty_html, render_disclaimer_html()

    results, valid_selected = get_top3_predictions(selected_symptoms)
    top_disease = results[0]["disease"]
    precautions = get_precautions(top_disease)
    explanation_text = build_explanation(valid_selected, top_disease)
    warning_text = build_warning(valid_selected)

    prediction_html = render_prediction_cards(results)
    precautions_html = render_precautions_html(top_disease, precautions, warning_text)
    explanation_html = render_explanation_html(explanation_text)
    disclaimer_html = render_disclaimer_html()

    return prediction_html, precautions_html, explanation_html, disclaimer_html


def make_example(symptoms):
    valid = [s for s in symptoms if s in symptom_choices]
    return [valid] if valid else None


examples_data = []
for ex in [
    ["itching", "skin_rash", "nodal_skin_eruptions"],
    ["vomiting", "diarrhoea", "abdominal_pain"],
    ["chest_pain", "shortness_of_breath"],
    ["headache", "nausea", "vomiting"],
]:
    item = make_example(ex)
    if item:
        examples_data.append(item)


custom_css = """
:root {
  --bg1: #f5f7ff;
  --bg2: #eef4ff;
  --surface: rgba(255, 255, 255, 0.88);
  --surface-2: rgba(255, 255, 255, 0.94);
  --border: rgba(148, 163, 184, 0.18);
  --text: #0f172a;
  --muted: #475569;
  --title: #111827;

  --brand1: #5b4ce6;
  --brand2: #7c3aed;
  --brand3: #8b5cf6;
  --brand4: #4f46e5;

  --warn-bg: #fff1f2;
  --warn-border: #fecdd3;
  --warn-text: #991b1b;

  --shadow-soft: 0 18px 45px rgba(15, 23, 42, 0.08);
  --shadow-strong: 0 24px 60px rgba(91, 76, 230, 0.24);
}

body.dark-mode {
  --bg1: #0b1020;
  --bg2: #111827;
  --surface: rgba(15, 23, 42, 0.88);
  --surface-2: rgba(17, 24, 39, 0.94);
  --border: rgba(148, 163, 184, 0.16);
  --text: #e5eefb;
  --muted: #b6c2d1;
  --title: #ffffff;

  --warn-bg: rgba(69, 10, 10, 0.40);
  --warn-border: rgba(248, 113, 113, 0.35);
  --warn-text: #fecaca;

  --shadow-soft: 0 18px 45px rgba(0, 0, 0, 0.30);
  --shadow-strong: 0 24px 60px rgba(0, 0, 0, 0.40);
}

.gradio-container {
  max-width: 1280px !important;
  margin: 0 auto !important;
  padding: 18px 12px 36px 12px !important;
  background:
    radial-gradient(circle at top left, rgba(91,76,230,0.10), transparent 28%),
    radial-gradient(circle at bottom right, rgba(124,58,237,0.08), transparent 24%),
    linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%);
  font-family: 'Inter', sans-serif !important;
  color: var(--text) !important;
}

.app-shell {
  min-height: 100vh;
}

.sidebar-panel {
  position: sticky;
  top: 16px;
  background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface) 100%);
  border: 1px solid var(--border);
  border-radius: 28px;
  box-shadow: var(--shadow-soft);
  backdrop-filter: blur(12px);
  padding: 22px;
}

.sidebar-brand {
  display: flex;
  gap: 14px;
  align-items: center;
  margin-bottom: 18px;
}

.brand-icon-wrap {
  width: 56px;
  height: 56px;
  border-radius: 18px;
  background: linear-gradient(135deg, var(--brand1), var(--brand2));
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 14px 30px rgba(91, 76, 230, 0.25);
  flex-shrink: 0;
}

.brand-title {
  font-family: 'Poppins', sans-serif;
  color: var(--title);
  font-size: 1.08rem;
  font-weight: 700;
  line-height: 1.25;
}

.brand-sub {
  color: var(--muted);
  font-size: 0.92rem;
  line-height: 1.55;
  margin-top: 4px;
}

.sidebar-card {
  background: rgba(255,255,255,0.06);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 16px;
  margin-top: 14px;
}

body.dark-mode .sidebar-card {
  background: rgba(255,255,255,0.03);
}

.sidebar-card-title {
  font-family: 'Poppins', sans-serif;
  color: var(--title);
  font-size: 0.98rem;
  font-weight: 600;
  margin-bottom: 6px;
}

.sidebar-card-text {
  color: var(--muted);
  line-height: 1.65;
  font-size: 0.92rem;
}

.info-pill-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 12px;
}

.info-pill {
  padding: 7px 11px;
  border-radius: 999px;
  background: linear-gradient(135deg, rgba(91,76,230,0.12), rgba(124,58,237,0.10));
  color: var(--brand4);
  border: 1px solid rgba(91,76,230,0.14);
  font-size: 0.84rem;
  font-weight: 600;
}

body.dark-mode .info-pill {
  color: #d9ccff;
  background: rgba(124,58,237,0.18);
  border-color: rgba(139,92,246,0.26);
}

.toggle-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
}

.toggle-label-wrap {
  min-width: 0;
}

.toggle-title {
  font-family: 'Poppins', sans-serif;
  color: var(--title);
  font-size: 0.96rem;
  font-weight: 600;
}

.toggle-sub {
  color: var(--muted);
  font-size: 0.88rem;
  margin-top: 3px;
}

.theme-switch {
  position: relative;
  display: inline-block;
  width: 54px;
  height: 30px;
  flex-shrink: 0;
}

.theme-switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.theme-slider {
  position: absolute;
  inset: 0;
  border-radius: 999px;
  background: #dbe4f5;
  transition: 0.25s ease;
  cursor: pointer;
}

.theme-slider:before {
  content: "";
  position: absolute;
  width: 22px;
  height: 22px;
  left: 4px;
  top: 4px;
  border-radius: 999px;
  background: white;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.15);
  transition: 0.25s ease;
}

.theme-switch input:checked + .theme-slider {
  background: linear-gradient(135deg, var(--brand1), var(--brand2));
}

.theme-switch input:checked + .theme-slider:before {
  transform: translateX(24px);
}

.main-hero {
  background: linear-gradient(135deg, var(--brand1) 0%, var(--brand2) 55%, var(--brand4) 100%);
  color: white;
  border-radius: 30px;
  padding: 28px 32px;
  box-shadow: var(--shadow-strong);
  position: relative;
  overflow: hidden;
  margin-bottom: 18px;
}

.main-hero::before {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.00));
  pointer-events: none;
}

.main-hero::after {
  content: "";
  position: absolute;
  right: -70px;
  top: -70px;
  width: 220px;
  height: 220px;
  border-radius: 999px;
  background: rgba(255,255,255,0.10);
}

.hero-top {
  display: flex;
  gap: 14px;
  align-items: center;
  position: relative;
  z-index: 2;
}

.hero-icon {
  width: 54px;
  height: 54px;
  border-radius: 18px;
  background: rgba(255,255,255,0.16);
  display: flex;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(8px);
  flex-shrink: 0;
}

.hero-title {
  font-family: 'Poppins', sans-serif;
  font-size: 2rem;
  font-weight: 700;
  line-height: 1.2;
  margin: 0;
}

.hero-subtitle {
  position: relative;
  z-index: 2;
  margin-top: 12px;
  max-width: 820px;
  color: rgba(255,255,255,0.95);
  line-height: 1.8;
  font-size: 0.98rem;
}

.main-grid-card {
  background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface) 100%);
  border: 1px solid var(--border);
  border-radius: 24px;
  box-shadow: var(--shadow-soft);
  padding: 20px;
  backdrop-filter: blur(12px);
}

.section-title {
  font-family: 'Poppins', sans-serif;
  font-size: 1.08rem;
  font-weight: 600;
  color: var(--title);
  margin-bottom: 6px;
}

.section-subtitle {
  color: var(--muted);
  line-height: 1.65;
  font-size: 0.94rem;
}

.pred-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 14px;
}

.pred-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.94) 0%, rgba(248,250,255,0.98) 100%);
  border: 1px solid var(--border);
  border-radius: 22px;
  padding: 16px 18px;
  box-shadow: 0 8px 24px rgba(15,23,42,0.05);
}

body.dark-mode .pred-card {
  background: linear-gradient(180deg, rgba(17,24,39,0.92) 0%, rgba(15,23,42,0.96) 100%);
}

.pred-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.pred-rank {
  color: var(--brand4);
  font-weight: 700;
  font-size: 0.95rem;
}

body.dark-mode .pred-rank {
  color: #d9ccff;
}

.pred-score {
  color: var(--title);
  font-weight: 700;
  font-size: 1rem;
}

.pred-disease {
  font-family: 'Poppins', sans-serif;
  color: var(--title);
  font-size: 1.08rem;
  margin-bottom: 10px;
}

.progress-wrap {
  width: 100%;
  height: 10px;
  background: #e2e8f0;
  border-radius: 999px;
  overflow: hidden;
}

body.dark-mode .progress-wrap {
  background: #243244;
}

.progress-bar {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--brand1), var(--brand2), var(--brand3));
}

.content-card-inner {
  padding: 2px 2px 4px 2px;
}

.content-heading {
  font-family: 'Poppins', sans-serif;
  font-size: 1.02rem;
  color: var(--title);
  font-weight: 600;
  margin-bottom: 10px;
}

.content-paragraph {
  color: var(--text);
  line-height: 1.8;
  margin: 0;
}

.clean-list {
  margin: 0;
  padding-left: 20px;
  color: var(--text);
  line-height: 1.8;
}

.precautions-text,
.precautions-text .content-heading,
.precautions-text .clean-list,
.precautions-text .clean-list li {
  color: #000000 !important;
}

body.dark-mode .precautions-text,
body.dark-mode .precautions-text .content-heading,
body.dark-mode .precautions-text .clean-list,
body.dark-mode .precautions-text .clean-list li {
  color: var(--text) !important;
}

.warning-box {
  margin-top: 14px;
  border-radius: 16px;
  background: var(--warn-bg);
  border: 1px solid var(--warn-border);
  padding: 14px 16px;
  color: var(--warn-text);
  line-height: 1.7;
}

.warning-title {
  font-family: 'Poppins', sans-serif;
  font-weight: 600;
  margin-bottom: 6px;
  color: var(--warn-text);
}

.empty-state {
  border-radius: 18px;
  background: rgba(248,250,252,0.72);
  border: 1px dashed #cbd5e1;
  padding: 26px 18px;
  text-align: center;
}

body.dark-mode .empty-state {
  background: rgba(17,24,39,0.72);
  border-color: rgba(148,163,184,0.24);
}

.empty-title {
  font-family: 'Poppins', sans-serif;
  color: var(--title);
  font-weight: 600;
  margin-bottom: 6px;
}

.empty-text {
  color: var(--muted);
}

button.primary-btn {
  background: linear-gradient(135deg, var(--brand1), var(--brand2)) !important;
  color: white !important;
  border: none !important;
  border-radius: 14px !important;
  min-height: 48px !important;
  font-weight: 600 !important;
  box-shadow: 0 10px 25px rgba(91,76,230,0.22) !important;
}

button.secondary-btn {
  border-radius: 14px !important;
  min-height: 48px !important;
  font-weight: 600 !important;
}

footer {
  visibility: hidden !important;
}

@media (max-width: 900px) {
  .main-hero {
    padding: 24px 24px;
  }
  .hero-title {
    font-size: 1.65rem;
  }
  .sidebar-panel {
    position: static;
  }
}
"""

custom_head = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700&display=swap" rel="stylesheet">

<script>
(function () {
  function applyDarkMode(isDark) {
    document.body.classList.toggle("dark-mode", isDark);
    try {
      localStorage.setItem("msc_dark_mode", isDark ? "1" : "0");
    } catch (e) {}
  }

  document.addEventListener("change", function (e) {
    if (e.target && e.target.id === "darkModeSwitch") {
      applyDarkMode(e.target.checked);
    }
  });

  window.addEventListener("load", function () {
    setTimeout(function () {
      let saved = false;
      try {
        saved = localStorage.getItem("msc_dark_mode") === "1";
      } catch (e) {}

      const toggle = document.getElementById("darkModeSwitch");
      if (toggle) {
        toggle.checked = saved;
      }
      applyDarkMode(saved);
    }, 350);
  });
})();
</script>
"""

medical_svg = """
<svg width="28" height="28" viewBox="0 0 24 24" fill="none" aria-hidden="true">
  <path d="M19 14a4 4 0 0 1-4 4h-1v2a2 2 0 1 1-4 0v-2H9a4 4 0 0 1-4-4V5a3 3 0 0 1 6 0v3h2V5a3 3 0 0 1 6 0v9Z" stroke="white" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M9 11h6M12 8v6" stroke="white" stroke-width="1.8" stroke-linecap="round"/>
</svg>
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, head=custom_head) as demo:
    with gr.Row(elem_classes=["app-shell"]):
        with gr.Column(scale=3, elem_classes=["sidebar-panel"]):
            gr.HTML(f"""
            <div class="sidebar-brand">
                <div class="brand-icon-wrap">{medical_svg}</div>
                <div>
                    <div class="brand-title">Medical Symptom Checker</div>
                    <div class="brand-sub">Professional ML-based educational screening interface</div>
                </div>
            </div>

            <div class="sidebar-card">
                <div class="sidebar-card-title">Project capabilities</div>
                <div class="sidebar-card-text">
                    Search symptoms, get top ranked conditions, review confidence, and read basic precautions in one place.
                </div>
                <div class="info-pill-row">
                    <div class="info-pill">Top 3 Results</div>
                    <div class="info-pill">Confidence</div>
                    <div class="info-pill">Precautions</div>
                    <div class="info-pill">Disclaimer</div>
                </div>
            </div>

            <div class="sidebar-card">
                <div class="toggle-row">
                    <div class="toggle-label-wrap">
                        <div class="toggle-title">Dark mode</div>
                        <div class="toggle-sub">Switch to a low-glare interface</div>
                    </div>
                    <label class="theme-switch">
                        <input type="checkbox" id="darkModeSwitch">
                        <span class="theme-slider"></span>
                    </label>
                </div>
            </div>

            <div class="sidebar-card">
                <div class="sidebar-card-title">Select symptoms</div>
                <div class="sidebar-card-text">Search and choose one or more symptoms below.</div>
            </div>
            """)

            symptom_input = gr.Dropdown(
                choices=dropdown_choices,
                multiselect=True,
                value=[],
                label="Symptoms",
                info="Search symptoms and choose multiple if needed",
                allow_custom_value=False,
                filterable=True
            )

            with gr.Row():
                predict_btn = gr.Button("Predict Conditions", elem_classes=["primary-btn"])
                clear_btn = gr.Button("Clear", elem_classes=["secondary-btn"])

            if examples_data:
                gr.Examples(examples=examples_data, inputs=[symptom_input], label="Quick Examples")

            gr.HTML("""
            <div class="sidebar-card">
                <div class="sidebar-card-title">How to use</div>
                <div class="sidebar-card-text">
                    1. Pick symptoms<br>
                    2. Click <b>Predict Conditions</b><br>
                    3. Review results, explanation, precautions, and warning notes
                </div>
            </div>
            """)

        with gr.Column(scale=9):
            gr.HTML(f"""
            <div class="main-hero">
                <div class="hero-top">
                    <div class="hero-icon">{medical_svg}</div>
                    <div>
                        <div class="hero-title">AI-Powered Medical Symptom Checker</div>
                    </div>
                </div>
                <div class="hero-subtitle">
                    Select symptoms, review the <b>top 3 likely conditions</b>, check confidence scores,
                    and read basic precautions through a cleaner clinical-style interface with a professional
                    layout and optional dark mode.
                </div>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=6):
                    gr.HTML("""
                    <div class="main-grid-card">
                        <div class="section-title">Prediction Results</div>
                        <div class="section-subtitle">Top 3 ranked conditions with confidence bars.</div>
                    </div>
                    """)
                    prediction_output = gr.HTML("""
                    <div class="empty-state">
                        <div class="empty-title">No prediction yet</div>
                        <div class="empty-text">Select symptoms and click <b>Predict Conditions</b>.</div>
                    </div>
                    """)

                with gr.Column(scale=6):
                    gr.HTML("""
                    <div class="main-grid-card">
                        <div class="section-title">Precautions</div>
                        <div class="section-subtitle">Basic care guidance from the top predicted condition.</div>
                    </div>
                    """)
                    precautions_output = gr.HTML("""
                    <div class="empty-state">
                        <div class="empty-title">No precautions yet</div>
                        <div class="empty-text">Prediction results will appear here.</div>
                    </div>
                    """)

            with gr.Row():
                with gr.Column(scale=6):
                    gr.HTML("""
                    <div class="main-grid-card">
                        <div class="section-title">Explanation</div>
                        <div class="section-subtitle">Simple reason behind the prediction.</div>
                    </div>
                    """)
                    explanation_output = gr.HTML("""
                    <div class="empty-state">
                        <div class="empty-title">No explanation yet</div>
                        <div class="empty-text">Prediction explanation will appear here.</div>
                    </div>
                    """)

                with gr.Column(scale=6):
                    gr.HTML("""
                    <div class="main-grid-card">
                        <div class="section-title">Disclaimer</div>
                        <div class="section-subtitle">Important safety information.</div>
                    </div>
                    """)
                    disclaimer_output = gr.HTML(render_disclaimer_html())

    predict_btn.click(
        fn=symptom_checker_ui,
        inputs=[symptom_input],
        outputs=[prediction_output, precautions_output, explanation_output, disclaimer_output]
    )

    clear_btn.click(
        fn=lambda: (
            [],
            """
            <div class="empty-state">
                <div class="empty-title">No prediction yet</div>
                <div class="empty-text">Select symptoms and click <b>Predict Conditions</b>.</div>
            </div>
            """,
            """
            <div class="empty-state">
                <div class="empty-title">No precautions yet</div>
                <div class="empty-text">Prediction results will appear here.</div>
            </div>
            """,
            """
            <div class="empty-state">
                <div class="empty-title">No explanation yet</div>
                <div class="empty-text">Prediction explanation will appear here.</div>
            </div>
            """,
            render_disclaimer_html()
        ),
        inputs=[],
        outputs=[symptom_input, prediction_output, precautions_output, explanation_output, disclaimer_output]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
