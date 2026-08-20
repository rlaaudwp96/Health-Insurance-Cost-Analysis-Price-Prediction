# 🏥 Health Insurance Cost Analysis & Price Prediction

## Business Problem & Overview
Medical insurance providers face significant challenges in accurately pricing policy risk. Miscalculating healthcare costs can lead to underpriced premiums (resulting in financial losses) or overpriced plans (losing customers to competitors). 

This project analyzes individual healthcare charges using the US Insurance dataset (1,338 policyholders) to identify key cost drivers—such as age, BMI, and smoking habits—and builds an interpretable predictive regression model to estimate individual medical charges.

---

## Key Results & Business Impact
* **Best Model:** Log-transformed Multiple Linear Regression with non-linear terms ($Age^2$, $BMI^2$) and key interaction terms ($BMI \times Smoker$, $Age \times Smoker$).
* **Model Performance:** Achieved an **Adjusted $R^2$ of ~0.838**, significantly outperforming the baseline linear model (~0.782).
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

## 💻 Core R Code Snippets

### 1. Key Interactions & Log Regression Model
```R
# Final Multiple Linear Regression Specification with Interactions
lm_final <- lm(
  log(charges) ~ age + I(age^2) + sex + bmi + I(bmi^2) + children + smoker + region + 
    age:sex + age:children + age:smoker + age:region + 
    sex:smoker + bmi:smoker + children:smoker + smoker:region, 
  data = insurance
)

summary(lm_final)
```

### 2. Evaluating Prediction Intervals (Risk Assessment)
```R
# Estimating cost increase for a 27yo non-smoker vs. smoker
new_person <- data.frame(
  age = 27, bmi = 28, children = 0,
  sex = factor("male", levels = levels(insurance$sex)),
  smoker = factor("no", levels = levels(insurance$smoker)),
  region = factor("southwest", levels = levels(insurance$region))
)

# Non-smoker baseline vs Smoker cost calculation
pred_non_smoker <- exp(predict(lm_final, newdata = new_person))
smoker_person <- new_person; smoker_person$smoker <- "yes"
pred_smoker <- exp(predict(lm_final, newdata = smoker_person))
```

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
├── Final Project.qmd        # Quarto / R Markdown script containing EDA and model selection
└── README.md                # Project documentation
```

---

## 🚀 How to Reproduce
1. **Clone this repository:**
   ```bash
   git clone https://github.com/rlaaudwp96/Medical-Insurance-Cost-Prediction.git
   cd Medical-Insurance-Cost-Prediction
   ```
2. **Open Project:** Open `Final Project.qmd` in **RStudio**.
3. **Install Packages:** Run the following command in the R console:
   ```R
   install.packages(c("tidyverse", "car", "MASS", "MuMIn", "caret", "pacman"))
   ```
4. **Run Analysis:** Render the Quarto document or run the script blocks sequentially to reproduce all visualizations, diagnostic plots, and regression tables.

---

## 🛠️ Tech Stack
* **Language:** R
* **Libraries:** `tidyverse` (`ggplot2`, `dplyr`), `car`, `MASS`, `caret`, `MuMIn`
