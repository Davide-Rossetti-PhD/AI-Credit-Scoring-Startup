# 🌍 WeTrust – Sustainable Credit Scoring for Microfinance
**i4C Challenge 2024 – In collaboration with Banca Sella**

---

## 🚀 Project Overview

**WeTrust** is an innovative digital solution designed for the **i4C Challenge 2024**, in collaboration with **Banca Sella**, aiming to expand **sustainable access to microcredit** for migrant communities.

The project leverages **machine learning–based credit scoring** to help evaluate creditworthiness using **behavioral and alternative data**, enabling migrants who send **international remittances** to access small credit lines and become part of the **bankable population**.

> 💡 *Our vision:* Turn remittances into a bridge for financial inclusion.

---

## 🎯 Challenge Context

Traditional credit scoring models often exclude individuals without a formal credit history.  
WeTrust addresses this gap by developing a **data-driven merit system** that:

- Uses behavioral and mobile data to assess reliability.
- Enables **microcredit on top of remittance transactions**.
- Encourages responsible financial behavior through gamified merit classes.
- Promotes financial inclusion aligned with **Banca Sella’s ESG & impact innovation goals**.

---

## 🧠 The Credit Scoring Model

The **Machine Learning component** was developed in Python and simulates the full lifecycle of a credit scoring system:

- **Synthetic dataset generation** (behavioral and financial features)
- **Supervised classification model** using:
  - Logistic Regression (multinomial)
  - Random Forest baseline
- **5-Fold Cross Validation** for robustness
- **Evaluation metrics:** Accuracy, F1-score, Macro ROC–AUC, Average Precision (AP)
- **Explainability layer:** Feature importance analysis and interpretability through coefficients
- **Visual analytics:**
  - Normalized Confusion Matrix
  - ROC and Precision–Recall Curves
  - Feature Importance visualization

### 🔍 Core Technologies
| Category | Tools & Frameworks |
|-----------|--------------------|
| Data Generation | NumPy, Pandas |
| Machine Learning | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Documentation | LaTeX (technical report), Markdown |
| Video Demo | Figma, Loom, AVI Video Editor |

---

## 📊 Results

The model achieves:
- **Macro ROC–AUC ≈ 0.9**, indicating excellent discriminative ability.
- **High per-class precision (>0.9)** on most credit merit categories.
- **Clear feature interpretability**, where positive behaviors (routine, remittance regularity) raise the score and negative behaviors (betting apps, irregular mobility) reduce it.

### 🧩 Key Visuals
| Visualization | Description |
|----------------|-------------|
| [Confusion Matrix] | Normalized Confusion Matrix showing per-class accuracy |
| [Feature Importance] | Feature weights from Logistic Regression |
| [ROC PR Curves] | ROC and PR Curves demonstrating strong separability |

All detailed explanations and formulas are included in the **LaTeX technical report** under `/docs/WeTrust_Report.pdf`.

---

## 📱 The WeTrust App – Business Concept

The **WeTrust App** provides a seamless ecosystem for migrants to send remittances and access microcredit responsibly.

### 🧩 Key Features
- **Remittance & Credit Integration:** Each remittance builds a track record for microcredit eligibility.  
- **Merit Class System:** 5 merit tiers based on repayment consistency and financial behavior.  
- **Zero-to-Low Interest Microloans:** Dynamic interest based on user merit.  
- **Financial Education Layer:** Rewards users for responsible behavior.  
- **Dashboard for Transparency:** Clear visual of credit level, repayments, and merit evolution.

### 💼 Business Plan Summary
- **Target:** Migrants sending cross-border remittances (initially Europe → Africa/Asia).  
- **Revenue Model:** Transaction fees + microcredit interest margins.  
- **Partnerships:** Banca Sella (banking infrastructure), Ente Nazionale Microcredito, remittance operators.  
- **Impact Goal:** Increase the number of financially included individuals by creating trust through data.

---

## 🎬 Demo and UX Prototype

A complete **interactive prototype** of the WeTrust App was designed in **Figma**  
and demonstrated via **Loom video**, later edited with **AVI Video Editor**.

- 🎥 **Video Demo (Loom):** [Watch the WeTrust Demo]
- 🎨 **Figma Prototype:** [Try the Interactive Prototype](https://www.figma.com/proto/fEPwkDrNwq7LFiKHeiOB6O/Alpha-App-WeTrust?node-id=1-91&t=UKF1eHJ6UBGeraLm-1)

---

## 📁 Repository Structure

The following repository layout is presented in **bash-style format** for clarity and visual consistency.

```bash
# Repository Structure (bash view)

WeTrust/
│
├── Notebooks_WT/                     # Synthetic datasets and ML algorithms
│   ├── wetrust_synthetic_dataset.csv # Generated dataset with behavioral and financial features
│   ├── Synthetic_Generator.py        # Synthetic data generation script
│   └── Model_WT.py                   # Credit scoring model training and evaluation
│
├── BancaSella_FinalReport.pdf        # Official report for the i4C Challenge with Banca Sella
│
├── Results_WeTrust.pdf               # Analytical results and LaTeX explanations
│
├── Wetrustt_final_video.mp4          # Final project demo video (Figma + Loom presentation)
│
├── Images/                           # Model plots and visuals for documentation
│   ├── Confusion_Matrix.png
│   ├── ROC_Precision_Curve.png
│   └── Feature_Importance.png
│
└── README.md                         # Project documentation (this file)
