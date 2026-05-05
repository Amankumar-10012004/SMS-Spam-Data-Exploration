# 🔁 Data Flow Diagram (DFD)
## SMS Spam Data Exploration — SpamShield

> **Authors:** Alok Chauhan (251810700318) · Aman Kumar (251810700231) · Batch 2C

---

## 1. DFD Notation Guide

| Symbol | Meaning |
|--------|---------|
| 🗄️ `[( )]` | **Data Store** — persistent storage (files, databases) |
| ⚙️ `[ ]` | **Process** — transformation or computation |
| 👤 `(( ))` | **External Entity** — user, dataset source, cloud service |
| `→` | **Data Flow** — direction data moves, labeled with what moves |

---

## 2. Level-0 DFD — Context Diagram

*The entire system as a single process with external entities.*

```mermaid
graph LR
    UCI(("🌐 UCI\nRepository\n[External]"))
    USER(("👤 End User\n[Browser]"))
    DEV(("🎓 Developer /\nReviewer\n[Local]"))
    SYSTEM["⚙️\n\nSMS Spam\nDetection\nSystem"]

    UCI -->|"spam.csv\nraw dataset"| SYSTEM
    USER -->|"SMS text message"| SYSTEM
    SYSTEM -->|"spam / ham prediction\n+ confidence score"| USER
    DEV -->|"run dashboard\nview analytics"| SYSTEM
    SYSTEM -->|"EDA charts, ML metrics\nsignal breakdown"| DEV

    style SYSTEM fill:#1a1f2e,stroke:#5bffa8,color:#e2e8f8,font-size:16px
    style UCI fill:#0d1117,stroke:#4d8fff,color:#e2e8f8
    style USER fill:#0d1117,stroke:#5bffa8,color:#e2e8f8
    style DEV fill:#0d1117,stroke:#ffb84d,color:#e2e8f8
```

---

## 3. Level-1 DFD — Major Processes

*The system broken into its 6 major processes.*

```mermaid
graph TD
    UCI(("🌐 UCI Dataset\nspam.csv"))
    USER(("👤 End User"))
    DEV(("🎓 Reviewer /\nDeveloper"))

    P1["⚙️ P1\nData Cleaning\n& Feature\nExtraction"]
    P2["⚙️ P2\nExploratory\nData Analysis"]
    P3["⚙️ P3\nML Model\nTraining\n& Evaluation"]
    P4["⚙️ P4\nStreamlit\nDashboard"]
    P5["⚙️ P5\nStatic Web App\nSpamShield"]
    P6["⚙️ P6\nNetlify Function\npredict.js"]

    DS1[("🗄️ DS1\nspam_cleaned.csv")]
    DS2[("🗄️ DS2\nml_results.json")]
    DS3[("🗄️ DS3\nspam_model.pkl")]
    DS4[("🗄️ DS4\noutputs/previews\nChart PNGs")]
    DS5[("🗄️ DS5\nlocalStorage\nHistory")]

    UCI -->|"raw CSV"| P1
    P1 -->|"cleaned + features"| DS1
    DS1 -->|"DataFrame"| P2
    P2 -->|"chart PNGs"| DS4
    DS1 -->|"X_train, X_test"| P3
    P3 -->|"metrics JSON"| DS2
    P3 -->|"pipeline pkl"| DS3
    DS1 -->|"load data"| P4
    DS2 -->|"load metrics"| P4
    DS3 -->|"load model"| P4
    DS4 -->|"load images"| P4
    P4 -->|"analytics\nvisualizations"| DEV
    P5 -->|"SMS message\n{message: str}"| P6
    P6 -->|"prediction\n{is_spam, confidence}"| P5
    P5 -->|"spam/ham result\nconfidence bar\nsignal chips"| USER
    USER -->|"paste SMS text"| P5
    P5 -->|"analysis result"| DS5
    DS5 -->|"history items"| P5
    DEV -->|"open dashboard"| P4

    style DS1 fill:#0d1117,stroke:#4d8fff,color:#e2e8f8
    style DS2 fill:#0d1117,stroke:#4d8fff,color:#e2e8f8
    style DS3 fill:#0d1117,stroke:#4d8fff,color:#e2e8f8
    style DS4 fill:#0d1117,stroke:#4d8fff,color:#e2e8f8
    style DS5 fill:#0d1117,stroke:#4d8fff,color:#e2e8f8
```

---

## 4. Level-2 DFD — Process P1: Data Cleaning & Feature Extraction

*Detailed data flows inside the cleaning pipeline.*

```mermaid
flowchart LR
    IN(("spam.csv\nRaw Input"))

    P1A["P1.1\nLoad CSV\nlatin-1 encoding\ncols 0,1 only"]
    P1B["P1.2\nRename Columns\nv1→label\nv2→message"]
    P1C["P1.3\nDrop Duplicates\n403 removed\n5572→5169"]
    P1D["P1.4\nDrop Null Messages\nhandle NaN"]
    P1E["P1.5\nExtract Numeric\nFeatures\nchar_count\nword_count"]
    P1F["P1.6\nExtract Signal\nFeatures\nhas_url · has_phone\nhas_currency · has_free\nhas_call · has_txt"]
    P1G["P1.7\nCompute Scores\nspam_signals = sum\nlabel_num = 0/1"]
    P1H["P1.8\nExport CSV\nspam_cleaned.csv"]

    DS1[("spam_cleaned.csv\n+12 columns")]

    IN -->|"5572 rows\n2 cols"| P1A
    P1A -->|"DataFrame"| P1B
    P1B -->|"labeled DF"| P1C
    P1C -->|"5169 rows"| P1D
    P1D -->|"clean DF"| P1E
    P1E -->|"DF + char/word count"| P1F
    P1F -->|"DF + 6 bool cols"| P1G
    P1G -->|"DF + 12 cols"| P1H
    P1H -->|"writes"| DS1

    style IN fill:#0d1117,stroke:#5bffa8,color:#e2e8f8
    style DS1 fill:#0d1117,stroke:#4d8fff,color:#e2e8f8
```

---

## 5. Level-2 DFD — Process P3: ML Training & Evaluation

*Detailed data flows inside the ML training pipeline.*

```mermaid
flowchart LR
    DS1[("spam_cleaned.csv")]

    P3A["P3.1\nLoad & Split\n80/20 stratified"]
    P3B["P3.2\nBuild Pipelines\n4× TF-IDF + Model"]
    P3C["P3.3\nFit Pipelines\npipe.fit(X_train, y_train)"]
    P3D["P3.4\nPredict & Score\nAccuracy · P · R · F1\nROC-AUC · CM"]
    P3E["P3.5\nCross-Validate\n5-fold CV F1\ncross_val_score()"]
    P3F["P3.6\nSelect Best\nmax(results, key=f1)"]
    P3G["P3.7\nSave Best Pipeline\njoblib.dump()"]
    P3H["P3.8\nSave All Metrics\njson.dump()"]

    DS2[("spam_model.pkl\nBest Pipeline")]
    DS3[("ml_results.json\nAll Metrics")]

    DS1 -->|"X=message\ny=label_num"| P3A
    P3A -->|"X_train/test\ny_train/test"| P3B
    P3B -->|"4 pipelines"| P3C
    P3C -->|"fitted models"| P3D
    P3D -->|"y_pred, y_prob\nmetrics dict"| P3E
    P3E -->|"cv_f1 scores"| P3F
    P3F -->|"best model name"| P3G
    P3F -->|"results dict"| P3H
    P3G -->|"writes"| DS2
    P3H -->|"writes"| DS3

    style DS1 fill:#0d1117,stroke:#4d8fff,color:#e2e8f8
    style DS2 fill:#0d1117,stroke:#5bffa8,color:#e2e8f8
    style DS3 fill:#0d1117,stroke:#5bffa8,color:#e2e8f8
```

---

## 6. Level-2 DFD — Process P5/P6: Real-Time Prediction

*Data flow when a user submits a message on the SpamShield web app.*

```mermaid
sequenceDiagram
    actor User
    participant Browser as 🌐 Browser (index.html)
    participant LS as 🗄️ localStorage
    participant NF as ⚡ Netlify Function (predict.js)

    User->>Browser: Types / pastes SMS message
    Browser->>Browser: onInput() → update char counter<br/>enable Analyse button if len ≥ 4

    User->>Browser: Clicks "⚡ Analyse Message"<br/>(or Ctrl+Enter)
    Browser->>Browser: btn.classList.add('loading')
    Browser->>Browser: USE_DEMO check

    alt USE_DEMO = false (Netlify live)
        Browser->>NF: POST /.netlify/functions/predict<br/>{"message": "FREE prize call 0800..."}
        NF->>NF: 1. Validate input (len 1–10000)
        NF->>NF: 2. Run 8 SPAM_PATTERNS (regex filter)
        NF->>NF: 3. score = foundSignals.length
        NF->>NF: 4. Lookup spamProbability by score
        NF-->>Browser: 200 OK<br/>{"is_spam": true, "confidence": 0.94,<br/>"score": 3, "found_signals": [...]}
    else USE_DEMO = true (no API)
        Browser->>Browser: demoPredict(msg)<br/>Client-side 8-signal check<br/>Simulated 700–1200ms delay
    end

    Browser->>Browser: showResult(result, message)<br/>→ Update result-panel CSS classes<br/>→ Animate confidence bar (CSS transition)<br/>→ Animate SVG ring (double rAF)<br/>→ Render signal chips

    Browser->>LS: addToHistory(result, message)<br/>localStorage["ss_history"] = JSON

    Browser->>Browser: renderHistory()<br/>→ Rebuild history list UI
    Browser-->>User: Result displayed with confidence & signals
```

---

## 7. Level-2 DFD — Process P4: Streamlit Dashboard

*Data flow when the dashboard is opened and navigated.*

```mermaid
flowchart TD
    DEV(("👤 Developer\nruns dashboard"))

    P4A["P4.1\nPage Load\nstreamlit run 05_dashboard.py"]
    P4B["P4.2\nCache Miss?\nload_data()\nload_ml_results()\nload_model()"]
    P4C["P4.3\nSidebar Navigation\nRadio button selection"]

    P4D1["P4.4a\n🏠 Overview Page\nMetrics + Data Quality"]
    P4D2["P4.4b\n📊 EDA Charts Page\nPie + Hist + Bars"]
    P4D3["P4.4c\n🔤 Word Analysis\nCounter + Comparison"]
    P4D4["P4.4d\n📏 Segmentation\nGroupby + Bar Charts"]
    P4D5["P4.4e\n🤖 ML Results\nTable + ROC + CM"]
    P4D6["P4.4f\n🔍 Check Message\nPredict + Signal Chart"]

    DS1[("🗄️ spam_cleaned.csv")]
    DS2[("🗄️ ml_results.json")]
    DS3[("🗄️ spam_model.pkl")]
    OUT[("📊 Rendered\nStreamlit Pages")]

    DEV -->|"HTTP request"| P4A
    P4A --> P4B
    DS1 -->|"DataFrame"| P4B
    DS2 -->|"dict"| P4B
    DS3 -->|"Pipeline obj"| P4B
    P4B -->|"cached data"| P4C
    P4C -->|"page=Overview"| P4D1
    P4C -->|"page=EDA"| P4D2
    P4C -->|"page=Words"| P4D3
    P4C -->|"page=Segments"| P4D4
    P4C -->|"page=ML"| P4D5
    P4C -->|"page=Check"| P4D6
    P4D1 --> OUT
    P4D2 --> OUT
    P4D3 --> OUT
    P4D4 --> OUT
    P4D5 --> OUT
    P4D6 --> OUT
    OUT -->|"HTML/PNG response"| DEV

    style DS1 fill:#0d1117,stroke:#4d8fff,color:#e2e8f8
    style DS2 fill:#0d1117,stroke:#4d8fff,color:#e2e8f8
    style DS3 fill:#0d1117,stroke:#4d8fff,color:#e2e8f8
    style OUT fill:#0d1117,stroke:#5bffa8,color:#e2e8f8
```

---

## 8. Complete Data Inventory

### Data Stores

| ID | Name | Format | Created By | Read By | Contents |
|----|------|--------|-----------|---------|----------|
| DS1 | `spam.csv` | CSV | UCI Repository | `01_data_cleaning.ipynb`, `train_model.py` | Raw 5,572 SMS messages |
| DS2 | `spam_cleaned.csv` | CSV | `01_data_cleaning.ipynb` | `05_dashboard.py`, `save_charts.py`, `train_model.py` | Cleaned data + 9 features |
| DS3 | `outputs/spam_model.pkl` | Pickle | `train_model.py` | `05_dashboard.py` | Best trained TF-IDF + LinearSVM pipeline |
| DS4 | `outputs/ml_results.json` | JSON | `train_model.py` | `05_dashboard.py`, `index.html` (indirectly) | All model metrics + ROC data |
| DS5 | `outputs/previews/*.png` | PNG | `save_charts.py` | `05_dashboard.py` | Pre-rendered chart images |
| DS6 | `localStorage["ss_history"]` | JSON string | `index.html` (browser) | `index.html` (browser) | Up to 30 recent predictions |

### Data Flows Summary

| Flow | Source | Data | Destination |
|------|--------|------|-------------|
| F1 | UCI Repository | `spam.csv` | `01_data_cleaning.ipynb` |
| F2 | `01_data_cleaning.ipynb` | `spam_cleaned.csv` | project root |
| F3 | `spam_cleaned.csv` | DataFrame | `02_eda_distribution.ipynb` |
| F4 | `spam_cleaned.csv` | DataFrame | `03_text_statistics.ipynb` |
| F5 | `spam_cleaned.csv` | DataFrame | `04_segmentation.ipynb` |
| F6 | `spam_cleaned.csv` | X, y arrays | `train_model.py` |
| F7 | `train_model.py` | `spam_model.pkl` | `outputs/` |
| F8 | `train_model.py` | `ml_results.json` | `outputs/` |
| F9 | `spam_cleaned.csv` | DataFrame | `05_dashboard.py` |
| F10 | `ml_results.json` | dict | `05_dashboard.py` |
| F11 | `spam_model.pkl` | Pipeline | `05_dashboard.py` |
| F12 | User browser | SMS text string | `index.html` |
| F13 | `index.html` | `{message: str}` JSON | `predict.js` |
| F14 | `predict.js` | `{is_spam, confidence, score}` JSON | `index.html` |
| F15 | `index.html` | Prediction result | `localStorage` |
| F16 | `localStorage` | History array | `index.html` |

---

*Document generated: 2026-05-06 · SMS Spam Data Exploration Project*
