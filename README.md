# Health Insurance Cost Analysis and Price Prediction

Python implementation of an interpretable insurance-cost modeling workflow using pandas and scikit-learn.

## Business Problem & Overview
Medical insurance providers face significant challenges in accurately pricing policy risk. Miscalculating healthcare costs can lead to underpriced premiums (resulting in financial losses) or overpriced plans (losing customers to competitors). 

This project analyzes individual healthcare charges using the US Insurance dataset (1,338 policyholders) to identify key cost drivers—such as age, BMI, and smoking habits—and builds an interpretable predictive regression model to estimate individual medical charges.

---

## Key Results & Business Impact
* **Best Model:** Log-transformed Multiple Linear Regression with non-linear terms ($Age^2$, $BMI^2$) and key interaction terms ($BMI \times Smoker$, $Age \times Smoker$).
* **Model Performance:** The Python implementation achieved a **validation R2 of 0.836** and a **training adjusted R2 of 0.833** with nonlinear terms and interactions.
* **Core Takeaways:**
  * **Smoker Impact:** Smoking is the single largest driver of high insurance charges. For a 27-year-old male with a BMI of 28, becoming a smoker increases predicted charges from **$2,757** to **$22,148**—a nearly **8x increase**.
  * **Interaction Effects:** BMI has a compounding effect on medical costs *specifically for smokers*—non-smokers show a much flatter cost curve as BMI increases compared to smokers.

---

## Data & Feature Preprocessing
* **Dataset:** 1,338 insurance records containing demographic and lifestyle attributes (`age`, `sex`, `bmi`, `children`, `smoker`, `region`, and `charges`).
* **Transformations & Cleaning:**
  1. Handled missing values and verified unique data rows.
  2. Applied a **Log Transformation** (`log(charges)`) to handle extreme right-skewness in medical costs and normalize error residuals.
  3. Formatted categorical features (`sex`, `smoker`, `region`) as factor variables.
  4. Added quadratic terms ($Age^2$, $BMI^2$) to capture non-linear relationships.

---

## Methodology & Model Comparison

Multiple regression specifications were evaluated using **AIC**, **Adjusted $R^2$**, and **70/30 Train/Test RMSE** via cross-validation:

1. **Baseline Model (`lm_full`):** Standard linear regression across all 6 features.
2. **Stepwise Model (`lm1`):** Stepwise selection evaluating core main effects and $BMI \times Smoker$.
3. **Final Complex Model (`lm2` / `Model 1`):** Complex model incorporating non-linear age/BMI trends alongside critical interaction pairs ($Age \times Smoker$, $BMI \times Smoker$, $Sex \times Smoker$).

### Performance Evaluation

| Model | Log Transformed | Adj. $R^2$ | Key Features / Interaction Terms |
| :--- | :---: | :---: | :--- |
| **Baseline Linear** | ❌ No | ~0.747 | Basic linear terms without interactions |
| **Stepwise (`lm1`)** | ✅ Yes | ~0.782 | `age`, `bmi`, `smoker`, `bmi:smoker`, `children`, `sex`, `region` |
| **Final Model (`lm2`)** | ✅ Yes | **~0.838** | $Age^2$, $BMI^2$, $BMI \times Smoker$, $Age \times Smoker$, $Sex \times Smoker$ |

---

## Python Workflow

Run the reproducible analysis with:

```bash
pip install -r requirements.txt
python health_insurance_analysis.py
```

The workflow removes duplicate rows, log-transforms charges, creates nonlinear numeric features, one-hot encodes categorical variables, fits a regression model, and reports RMSE, R2, adjusted R2, and an example prediction.

## Original R Materials

The original coursework materials are preserved in `legacy_r/` for reference; the primary implementation in this repository is Python.

---

## 👥 Contributors
This project was conducted as part of our statistical modeling coursework:

* **Myungje Kim**
* **Nicholas Fisher**
* **Felix Arroyo Viglino**

---

## 📁 Project Structure
```text
├── insurance.csv            # Raw healthcare cost dataset
├── health_insurance_analysis.py # Reproducible Python analysis
├── requirements.txt         # Python dependencies
├── legacy_r/                # Original R coursework materials
└── README.md                # Project documentation
```

---

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `scikit-learn`
