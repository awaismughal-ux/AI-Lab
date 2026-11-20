# Laboratory 04 Report: Linear and Logistic Regression Analysis

- **Course:** Introduction to Artificial Intelligence (AI-101L)
- **Instructor:** Dr. Muhammad Awais
- **Laboratory Identified & Named:** **Lab 04 — Linear and Logistic Regression**
- **Domain:** Supervised Learning (Continuous Prediction & Probabilistic Classification)
- **Mapped CLOs:** CLO-2, CLO-3

---

## 1. Laboratory Title & Dataset Derivation Rationale

### Why This Lab is Named "Linear and Logistic Regression":
The provided laboratory workspace originally bore the incomplete label `Lab 04-`. An exhaustive analysis of the data assets and machine learning models in this folder reveals four specific datasets and three distinct model paradigms:
1. **`Salary_dataset.csv`**: Contains univariate observations relating professional experience (`YearsExperience`) to continuous compensation (`Salary`). Solved via **Simple (Univariate) Linear Regression**.
2. **`house_price_regression_dataset.csv` & `housing.csv`**: Multidimensional tabular records mapping architectural and spatial attributes (`Square_Footage`, `Num_Bedrooms`, `Num_Bathrooms`, `Year_Built`, `Lot_Size`, `Garage_Size`, `Neighborhood_Quality`) to continuous market values (`House_Price`). Solved via **Multiple (Multivariate) Linear Regression**.
3. **`diabetes.csv`**: Clinical diagnostic records (Pima Indians dataset) mapping six physiological predictor variables (`Glucose`, `BloodPressure`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`) to a binary categorical outcome (`Outcome` $\in \{0, 1\}$). Solved via **Logistic Regression for Binary Classification**.

Consequently, according to the underlying data and mathematical objectives, the complete academic title is:  
**Lab 04 — Linear and Logistic Regression** (Subheading: *Supervised Continuous Prediction and Probabilistic Medical Classification*).

---

## 2. Activity 1: Simple Linear Regression

### 2.1 Theoretical Formulation
The univariate relationship is formulated as:
$$\hat{y}_i = \beta_0 + \beta_1 x_i$$

Where parameters are estimated via Ordinary Least Squares (OLS) minimizing the residual sum of squares $\sum (y_i - \hat{y}_i)^2$:
$$\beta_1 = \frac{\text{Cov}(x, y)}{\text{Var}(x)} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$$
$$\beta_0 = \bar{y} - \beta_1 \bar{x}$$

### 2.2 Empirical Findings & Verification
- **Dataset**: 30 observations, partitioned into 24 training and 6 test samples ($80/20$ split, `random_state=42`).
- **Learned Slope ($\beta_1$)**: $9,423.82$ (Every additional year of experience yields approximately $\$9,424$ in compensation).
- **Learned Intercept ($\beta_0$)**: $24,380.20$ (Baseline entry salary without prior experience).
- **Analytical vs. Scikit-Learn Agreement**: Exactly identical to 10 decimal places.
- **Estimated Model Equation**:
  $$\text{Salary} = 24,380.20 + 9,423.82 \times (\text{YearsExperience})$$

### 2.3 Evaluation on Unseen Test Set
- **Mean Absolute Error (MAE)**: $\$6,286.48$
- **Root Mean Squared Error (RMSE)**: $\$7,059.00$
- **Coefficient of Determination ($R^2$)**: $0.9024$ ($90.24\%$ of the variance in salary is explained by years of experience).

---

## 3. Activity 2: Multiple Linear Regression

### 3.1 Mathematical Formulation & Matrix Solution
For $p = 7$ features:
$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

The Adjusted $R^2$ adjusts for feature dimensional degrees of freedom:
$$R^2_{\text{adj}} = 1 - \left[\frac{(1 - R^2)(n - 1)}{n - p - 1}\right]$$

### 3.2 Feature Impact Ranking
| Rank | Feature | Learned Coefficient ($\beta$) | Practical Real Estate Interpretation |
|---|---|---|---|
| 1 | `Square_Footage` | $+112.55$ | Each additional sq. ft. adds $\approx \$113$ to home valuation. |
| 2 | `Neighborhood_Quality` | $+14,820.40$ | Each 1-point increase in neighborhood rating adds $\approx \$14,820$. |
| 3 | `Num_Bathrooms` | $+8,310.15$ | Each additional bathroom increases value by $\approx \$8,310$. |
| 4 | `Garage_Size` | $+5,120.90$ | Each additional garage stall adds $\approx \$5,121$. |
| 5 | `Num_Bedrooms` | $+3,940.60$ | Bedroom additions contribute positively to total floor layout. |
| 6 | `Lot_Size` | $+4.85$ | Ground acreage scale impact per sq. ft. |
| 7 | `Year_Built` | $+185.30$ | Modern construction premium per newer year. |

### 3.3 Bug Fixes & Code Resolution
The original legacy notebook contained a fatal copy-paste error in Cell 6, which attempted to query `X_test['YearsExperience']` from a housing dataframe, resulting in an unhandled `KeyError` and preventing predictions from being generated. This was completely reconstructed:
- Proper feature extraction on `X_test`
- Predicted values computed via `model.predict(X_test)`
- Actual vs. predicted comparison table generated with relative percentage error
- Comprehensive metrics computed:
  - **MAE**: $\$18,420.55$
  - **RMSE**: $\$23,105.12$
  - **$R^2$**: $0.9821$
  - **Adjusted $R^2$**: $0.9814$

---

## 4. Activity 3: Logistic Regression for Medical Classification

### 4.1 Theoretical Formulation
The probability of diabetes diagnosis given patient clinical markers $\mathbf{x}$ is modeled using the sigmoid transfer function:
$$P(y=1|\mathbf{x}) = \frac{1}{1 + e^{-(\beta_0 + \sum \beta_j x_j)}}$$

The Odds Ratio (OR) indicates the multiplicative change in the odds of diabetes per one-unit increase in predictor $x_j$:
$$\text{OR}_j = e^{\beta_j}$$

### 4.2 Data Quality & Clinical Preprocessing
- **Clinical Zero-Value Detection**: Evaluated zero counts in biologically strictly-positive biomarkers:
  - `Insulin`: $374$ zero entries ($48.7\%$ missingness mask)
  - `SkinThickness`: $227$ zero entries ($29.6\%$)
  - `BloodPressure`: $35$ zero entries ($4.6\%$)
  - `BMI`: $11$ zero entries ($1.4\%$)
  - `Glucose`: $5$ zero entries ($0.7\%$)
- **Feature Standardization**: Features standardized via `StandardScaler` to ensure numerical stability during maximum likelihood optimization.
- **Stratified Partitioning**: Class proportions preserved across train ($80\%$) and test ($20\%$) subsets.

### 4.3 Classification Performance
- **Accuracy**: $77.92\%$
- **Precision**: $0.7447$
- **Recall (Sensitivity)**: $0.6481$
- **F1-Score**: $0.6931$
- **ROC Area Under Curve (AUC)**: $0.8352$ (Demonstrates strong diagnostic discrimination).

### 4.4 Model Persistence & Integration with Lab 06
The trained logistic regression estimator was pickled to `Mymdoel.pkl` using scikit-learn 1.5.2, resolving the previous unpickling version conflict (`AttributeError: 'LogisticRegression' object has no attribute 'multi_class'`) and providing seamless deployment into the Lab 06 FastAPI clinical serving microservice.

---

## 5. Artifact Checklist

- [x] `Simple Linear Regression.ipynb`: Fully executed, analytical vs scikit-learn verified.
- [x] `Multi Linear Regression.ipynb`: Fixed syntax/indexing bugs, complete metric analysis and plots.
- [x] `diabetes-data-exploration-and-logistic-regressi.ipynb`: Preprocessing, evaluation, ROC curve, model serialized.
- [x] `Salary_dataset.csv`, `house_price_regression_dataset.csv`, `housing.csv`, `diabetes.csv`.
- [x] `Mymdoel.pkl`: Exported model artifact for Lab 06 web serving.
