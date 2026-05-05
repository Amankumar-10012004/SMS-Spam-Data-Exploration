# ⚙️ Low Level Design (LLD)
## SMS Spam Data Exploration — SpamShield

> **Authors:** Alok Chauhan (251810700318) · Aman Kumar (251810700231) · Batch 2C

---

## 1. Project File Structure

```
SMS-Spam-Data-Exploration/
│
├── 📓 01_data_cleaning.ipynb        ← Step 1: Clean raw data, extract 9 features
├── 📓 02_eda_distribution.ipynb     ← Step 2: Exploratory analysis charts
├── 📓 03_text_statistics.ipynb      ← Step 3: Word frequency & n-gram analysis
├── 📓 04_segmentation.ipynb         ← Step 4: Segment analysis & rule mining
│
├── 🐍 train_model.py                ← Step 5: Train 4 ML classifiers, save best
├── 🐍 05_dashboard.py               ← Step 6: Streamlit 6-page dashboard
├── 🐍 save_charts.py                ← Utility: Pre-render all charts as PNG
│
├── 🌐 index.html                    ← Static web app (SpamShield frontend)
│
├── 📁 netlify/
│   └── functions/
│       └── predict.js               ← Serverless prediction API
│
├── 📁 outputs/
│   ├── spam_model.pkl               ← Saved best ML pipeline
│   ├── ml_results.json              ← All model metrics + ROC data
│   ├── 01_quality_chart.png         ← Data quality bar chart
│   └── previews/                    ← Pre-rendered chart PNGs
│       ├── 01_pie_chart.png
│       ├── 02_length_histogram.png
│       ├── 03_wordcount_boxplot.png
│       ├── 04_feature_bars.png
│       ├── 05_top_spam_words.png
│       ├── 06_top_ham_words.png
│       ├── 07_segments_bar.png
│       └── 08_signal_breakdown.png
│
├── 📁 docs/                         ← Design documentation
│   ├── HLD.md
│   ├── LLD.md
│   ├── CFD.md
│   └── DFD.md
│
├── spam.csv                         ← Raw UCI dataset
├── spam_cleaned.csv                 ← Cleaned + feature-enriched dataset
├── requirements.txt                 ← Python deps (version-pinned)
├── netlify.toml                     ← Netlify build config
├── package.json                     ← Project metadata
├── run_dashboard.bat                ← Windows launcher script
└── README.md
```

---

## 2. Module-Level Design

### 2.1 `01_data_cleaning.ipynb` — Data Cleaning Module

```mermaid
flowchart TD
    A["Load spam.csv\nencoding=latin-1\nusecols=[0,1]"] --> B["Rename columns\nlabel · message"]
    B --> C{"Duplicate\nrows?"}
    C -->|"Yes — 403 found"| D["Drop duplicates\n5572 → 5169"]
    C -->|"No"| E["Continue"]
    D --> E
    E --> F["Drop nulls in\n'message' column"]
    F --> G["Extract Feature 1\nchar_count = len(message)"]
    G --> H["Extract Feature 2\nword_count = len(split)"]
    H --> I["Extract Feature 3\nhas_url = 'http'/'www' in msg"]
    I --> J["Extract Feature 4\nhas_phone = 10+ digit seq"]
    J --> K["Extract Feature 5\nhas_currency = £/$/ in msg"]
    K --> L["Extract Feature 6\nhas_free = 'free' in words"]
    L --> M["Extract Feature 7\nhas_call = 'call' in words"]
    M --> N["Extract Feature 8\nhas_txt = 'txt/text' in words"]
    N --> O["Extract Feature 9\nspam_signals = sum of above"]
    O --> P["label_num = 1 if spam, 0 if ham"]
    P --> Q["Save → spam_cleaned.csv"]

    style A fill:#1a1f2e,stroke:#5bffa8,color:#e2e8f8
    style Q fill:#1a1f2e,stroke:#5bffa8,color:#e2e8f8
```

**Extracted Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `label` | str | `spam` or `ham` |
| `message` | str | Original SMS text |
| `char_count` | int | Number of characters |
| `word_count` | int | Number of whitespace-separated tokens |
| `has_url` | bool | Contains `http://`, `https://`, or `www.` |
| `has_phone` | bool | Contains ≥10 consecutive digits |
| `has_currency` | bool | Contains `£`, `$`, `€`, or `₹` |
| `has_free` | bool | Word `free` appears |
| `has_call` | bool | Word `call` appears |
| `has_txt` | bool | Word `txt` or `text` appears |
| `spam_signals` | int | Count of True boolean features |
| `label_num` | int | `1` = spam, `0` = ham |

---

### 2.2 `train_model.py` — ML Training Module

```mermaid
flowchart TD
    A["Load spam.csv\nlatent-1 encoding"] --> B["Drop nulls\nCreate label_num"]
    B --> C["Train/Test Split\n80/20 · stratified\nrandom_state=42"]
    C --> D["Build 4 Pipelines\neach with fresh TfidfVectorizer"]

    D --> E1["Pipeline 1\nTF-IDF → MultinomialNB\nalpha=0.1"]
    D --> E2["Pipeline 2\nTF-IDF → LogisticRegression\nC=5, max_iter=1000"]
    D --> E3["Pipeline 3\nTF-IDF → CalibratedClassifierCV\n wrapping LinearSVC C=1.0"]
    D --> E4["Pipeline 4\nTF-IDF → DecisionTreeClassifier\nmax_depth=20"]

    E1 --> F["pipe.fit(X_train, y_train)"]
    E2 --> F
    E3 --> F
    E4 --> F

    F --> G["Evaluate on X_test\nAccuracy · Precision · Recall\nF1 · ROC-AUC · Confusion Matrix"]
    G --> H["5-Fold Cross-Val F1\ncross_val_score(pipe, X, y)"]
    H --> I{{"Best by F1?"}}
    I -->|"Linear SVM ✓"| J["joblib.dump → outputs/spam_model.pkl"]
    I -->|"Others"| K["Metrics saved only"]
    J --> L["Write outputs/ml_results.json\nAll metrics + ROC curves + meta"]
    K --> L

    style A fill:#1a1f2e,stroke:#4d8fff,color:#e2e8f8
    style L fill:#1a1f2e,stroke:#5bffa8,color:#e2e8f8
```

**TfidfVectorizer Parameters:**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `max_features` | 6,000 | Limit vocabulary size for performance |
| `ngram_range` | `(1, 2)` | Capture phrases like "call now", "free prize" |
| `sublinear_tf` | `True` | Log-scale TF to reduce impact of high-frequency words |
| `strip_accents` | `"unicode"` | Normalize international characters |
| `min_df` | `2` | Ignore rare tokens (< 2 occurrences) |

---

### 2.3 `05_dashboard.py` — Streamlit Dashboard Module

```mermaid
flowchart TD
    START["streamlit run 05_dashboard.py"] --> INIT["Page Config\ntitle · icon · wide layout"]
    INIT --> LOAD["@st.cache_data\nload_data() → spam_cleaned.csv\nload_ml_results() → ml_results.json\n@st.cache_resource\nload_model() → spam_model.pkl"]
    LOAD --> CHK{"data\nloaded?"}
    CHK -->|"No"| ERR["st.error + st.stop()"]
    CHK -->|"Yes"| NAV["Sidebar Radio Navigation"]

    NAV --> P1["🏠 Overview\nMetrics · About · Insights\nData Quality Chart · Sample Data"]
    NAV --> P2["📊 EDA Charts\nPie Chart · Length Histogram\nBoxplot · Feature Bar Chart"]
    NAV --> P3["🔤 Word Analysis\nTop N Spam Words · Top N Ham Words\nWord Comparison Table"]
    NAV --> P4["📏 Segmentation\nSpam Rate by Length Group\nSpam Rate by Signal Score"]
    NAV --> P5["🤖 ML Model Results\nMetrics Table · F1 Bar Chart\nROC Curves · Confusion Matrices"]
    NAV --> P6["🔍 Check a Message\nExample Selector · Text Input\nSignal Breakdown Chart · Stats"]

    P6 --> VERDICT{"ML Model\nloaded?"}
    VERDICT -->|"Yes"| ML["predict_proba → probability"]
    VERDICT -->|"No"| RULES["check_signals() → sum\nspam_verdict() → SPAM/LIKELY/SAFE"]
    ML --> SHOW["Display Verdict Banner\nSignal Chart · Message Stats"]
    RULES --> SHOW

    style START fill:#1a1f2e,stroke:#5bffa8,color:#e2e8f8
    style ERR fill:#1a1f2e,stroke:#ff4d6a,color:#e2e8f8
    style SHOW fill:#1a1f2e,stroke:#5bffa8,color:#e2e8f8
```

**Key Functions:**

| Function | Input | Output | Notes |
|----------|-------|--------|-------|
| `find_file(*paths)` | path strings | first existing path or `None` | Enables cross-machine portability |
| `load_data()` | — | `(DataFrame, error_str)` | `@st.cache_data` — loads once |
| `load_ml_results()` | — | `dict` or `None` | Reads `ml_results.json` |
| `load_model()` | — | `(model, status, msg)` | `@st.cache_resource` |
| `clean_words(message)` | str | `List[str]` | Strips punctuation, removes stopwords |
| `check_signals(message)` | str | `Dict[str, bool]` | 9-rule signal checker |
| `spam_verdict(signals, model, msg)` | dict, model, str | `(verdict, score, method)` | ML-first, rule-based fallback |

---

### 2.4 `index.html` — Static Frontend Module

```mermaid
flowchart TD
    LOAD["Page Load\nDOMContentLoaded"] --> PROBE["fetch(/.netlify/functions/predict)\nAPI health probe"]
    PROBE -->|"200 or 400"| LIVE["USE_DEMO = false\nDot: 🟢 Connected"]
    PROBE -->|"Network error"| DEMO["USE_DEMO = true\nDot: 🟡 Demo mode"]

    USER["User Action"] --> EX["setExample(i)\nFill textarea"] 
    USER --> TYPE["onInput()\nUpdate char counter\nEnable/disable button"]
    USER --> ANALYSE["analyse()\nCtrl+Enter or button click"]

    ANALYSE --> CHK{"USE_DEMO?"}
    CHK -->|"Yes"| DPRED["demoPredict(msg)\nClient-side rule engine\n8 SPAM_SIGNALS\n700-1200ms simulated delay"]
    CHK -->|"No"| APRED["apiPredict(msg)\nPOST /.netlify/functions/predict\n{message: msg}"]
    APRED -->|"Error"| DPRED

    DPRED --> SHOW["showResult(result, message)\nUpdate result-panel classes\nAnimate confidence bar\nAnimate SVG ring\nRender signal chips"]
    APRED --> SHOW
    SHOW --> HIST["addToHistory(result, message)\nPrepend to history array\nSave to localStorage\nrenderHistory()"]

    style LOAD fill:#1a1f2e,stroke:#4d8fff,color:#e2e8f8
    style SHOW fill:#1a1f2e,stroke:#5bffa8,color:#e2e8f8
```

**JavaScript Functions:**

| Function | Purpose |
|----------|---------|
| `onInput()` | Updates char counter; enables Analyse button only if ≥4 chars and not loading |
| `setExample(i)` | Fills textarea with pre-written example message |
| `clearAll()` | Resets textarea, hides result panel, clears `lastResult` |
| `analyse()` | Main async handler — calls API or demo engine |
| `demoPredict(msg)` | Client-side 8-signal rule engine with simulated network delay |
| `apiPredict(msg)` | Fetch POST to Netlify function endpoint |
| `showResult(result, msg)` | Renders result panel with animations (confidence bar + SVG ring) |
| `addToHistory(result, msg)` | Prepends to `localStorage` history (max 30 items) |
| `renderHistory()` | Re-renders full history list with click-to-reload |
| `shareResult()` | Uses Web Share API or clipboard fallback |
| `showApiHelp()` | Prompt to change API endpoint; handles Streamlit URL edge case |
| `animateBars(sel)` | CSS width animation trigger for data visualization bars |

---

### 2.5 `netlify/functions/predict.js` — Serverless API Module

```mermaid
flowchart TD
    REQ["HTTP Request\nPOST /.netlify/functions/predict"] --> OPT{"OPTIONS\nPreflight?"}
    OPT -->|"Yes"| CORS["Return 204\nCORS headers"]
    OPT -->|"No"| METHOD{"POST?"}
    METHOD -->|"No"| ERR405["Return 405\nMethod Not Allowed"]
    METHOD -->|"Yes"| PARSE["JSON.parse(event.body)"]
    PARSE -->|"Parse error"| ERR400A["Return 400\nInvalid JSON"]
    PARSE -->|"OK"| MSG["Extract message string\nString(payload.message).trim()"]
    MSG --> LEN{"Length\ncheck"}
    LEN -->|"< 1"| ERR400B["Return 400\nMessage required"]
    LEN -->|"> 10000"| ERR400C["Return 400\nToo long"]
    LEN -->|"1–10000"| SCAN["Run 8 SPAM_PATTERNS\nfilter pattern.test(message)"]
    SCAN --> SCORE["score = foundSignals.length"]
    SCORE --> PROB["Calculate spamProbability\nbased on score + phone presence"]
    PROB --> RESP["Return 200 JSON\n{label, is_spam, confidence,\nscore, found_signals}"]

    style REQ fill:#1a1f2e,stroke:#4d8fff,color:#e2e8f8
    style RESP fill:#1a1f2e,stroke:#5bffa8,color:#e2e8f8
```

**Spam Probability Scoring Logic:**

| Condition | Spam Probability |
|-----------|-----------------|
| Has phone AND score ≥ 2 | 0.94 |
| score ≥ 4 | 0.91 |
| score = 3 | 0.78 |
| score = 2 | 0.58 |
| score = 1 | 0.29 |
| score = 0 | 0.07 |

---

## 3. Data Schema

### `spam_cleaned.csv` Schema

```
label        : string  → "spam" | "ham"
message      : string  → raw SMS text
char_count   : int     → len(message)
word_count   : int     → len(message.split())
has_url      : bool    → 'http' or 'www.' present
has_phone    : bool    → ≥10 consecutive digit sequence
has_currency : bool    → £, $, €, ₹ present
has_free     : bool    → word 'free' present
has_call     : bool    → word 'call' present
has_txt      : bool    → word 'txt' or 'text' present
spam_signals : int     → sum of boolean features
label_num    : int     → 1=spam, 0=ham
```

### `ml_results.json` Schema

```json
{
  "Model Name": {
    "accuracy": float,
    "precision": float,
    "recall": float,
    "f1": float,
    "roc_auc": float,
    "cv_f1": float,
    "confusion_matrix": [[TN, FP], [FN, TP]],
    "roc_fpr": [float, ...],
    "roc_tpr": [float, ...]
  },
  "_meta": {
    "best_model": string,
    "train_size": int,
    "test_size": int,
    "total_rows": int,
    "spam_count": int,
    "ham_count": int,
    "model_names": [string, ...],
    "tfidf_params": { ... }
  }
}
```

---

## 4. Error Handling Matrix

| Component | Error Condition | Handling Strategy |
|-----------|----------------|-------------------|
| `load_data()` | File not found | Return `(None, error_msg)` → `st.error` + `st.stop()` |
| `load_ml_results()` | File not found / bad JSON | Return `None` → show warning + `st.stop()` on ML page |
| `load_model()` | File not found | Return `(None, "not_found", msg)` → rule-based fallback |
| `load_model()` | joblib load fails | Return `(None, "load_failed", msg)` → sidebar error |
| `spam_verdict()` | `predict_proba` exception | `except Exception: pass` → falls through to rules |
| `05_dashboard.py` | Missing feature columns | `if col in data.columns` guards throughout |
| `index.html` | API network error | `catch()` → `demoPredict()` fallback |
| `index.html` | Clipboard denied | `.catch()` → user-facing toast message |
| `predict.js` | Invalid JSON body | `try/catch JSON.parse` → 400 response |
| `predict.js` | Message > 10,000 chars | Explicit length check → 400 response |
| `train_model.py` | `spam.csv` missing | `os.path.exists` check → `FileNotFoundError` |
| `save_charts.py` | `spam_cleaned.csv` missing | `next(candidates)` check → `FileNotFoundError` |

---

*Document generated: 2026-05-06 · SMS Spam Data Exploration Project*
