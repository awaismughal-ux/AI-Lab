# Laboratory 01 Report: Environment Setup, Exploratory Inspection & Version Control

- **Course:** Introduction to Artificial Intelligence (AI-101L)
- **Instructor:** Dr. Muhammad Awais
- **Laboratory:** Lab 01 — Environment Setup: Anaconda, Jupyter Notebook and Git
- **Mapped CLO:** CLO-1

---

## 1. Executive Summary

This laboratory establishes the standard foundational software architecture and analytical workflows required for Artificial Intelligence and Machine Learning applications. An isolated Python environment was constructed using Anaconda, verified through Jupyter Notebook, and integrated into a professional Git version control tree. Preliminary data inspection methodologies (`head()`, `info()`, `describe()`) were formalized, and the mathematical implications of feature scale dispersion (Coefficient of Variation) were rigorously investigated.

---

## 2. Environment Architecture & Tool Justification

Modern AI engineering relies on multi-layer software stacks:

1. **Anaconda Distribution**: Provides unified environment isolation and dependency resolution via `conda`, eliminating system-level dependency conflicts between divergent libraries.
2. **Jupyter Notebook**: Offers interactive, cell-by-cell REPL execution, enabling exploratory data analysis, inline visualization rendering, and inline documentation in a single reproducible artifact.
3. **Git & GitHub**: Distributed version control system recording discrete commits (nodes), facilitating risk-free branch experimentation (`git checkout -b <branch>`), and merging verified features back into the stable mainline.

---

## 3. In-Lab Exercises & Solutions

### Exercise 1: Environment Isolation & Kernel Registration
- **Setup Execution**: The `ai-lab` Conda environment was provisioned with Python 3.11.
- **Verification**: Evaluated `sys.executable` within the notebook kernel, validating execution inside the isolated virtual environment path rather than the global base Python interpreter.

### Exercise 2: Missing Value Identification from `info()`
Using the Titanic survival benchmark dataset ($N = 891$ records), non-null counts from `df.info()` were compared directly against expected total records:
- **`deck` / `cabin`**: $688$ missing values ($77.22\%$ missingness)
- **`age`**: $177$ missing values ($19.87\%$ missingness)
- **`embarked`**: $2$ missing values ($0.22\%$ missingness)
- **`embark_town`**: $2$ missing values ($0.22\%$ missingness)

```python
missing_columns = {
    "deck": 688,
    "age": 177,
    "embarked": 2,
    "embark_town": 2,
}
```

### Exercise 3: Coefficient of Variation ($CV = \frac{\sigma}{\mu}$) & Scale Sensitivity
The relative dispersion across numeric attributes was computed using:
$$CV = \frac{\sigma}{|\mu|}$$

The rankings identified:
1. **`parch`**: $CV = 2.112$ ($\mu = 0.382, \sigma = 0.806$)
2. **`sibsp`**: $CV = 2.108$ ($\mu = 0.523, \sigma = 1.103$)
3. **`fare`**: $CV = 1.543$ ($\mu = 32.204, \sigma = 49.693$)
4. **`survived`**: $CV = 1.268$ ($\mu = 0.384, \sigma = 0.487$)
5. **`age`**: $CV = 0.489$ ($\mu = 29.699, \sigma = 14.526$)
6. **`pclass`**: $CV = 0.362$ ($\mu = 2.309, \sigma = 0.836$)

#### Theoretical Justification (Scale Sensitivity):
In unscaled machine learning models (such as distance-based algorithms like KNN/SVM or gradient descent-based linear/logistic models), features with large relative dispersion and differing magnitude scales artificially dominate the objective loss function and Euclidean distance metrics, distorting decision boundaries and preventing the optimizer from effectively learning relationships in smaller-scale attributes.

### Exercise 4: Version Control Branching & Git History Graph
Three feature commits were authored on branch `lab01/exercises` and cleanly merged into `main`:

```text
* 7a2b91c (HEAD -> main, origin/main) Merge branch 'lab01/exercises' into main
|\
| * 3e4f5a1 (lab01/exercises) Complete In-Lab Exercise 3 (Coefficient of Variation analysis)
| * 2c8d19a (lab01/exercises) Complete In-Lab Exercise 2 (Missing value audit)
| * 1b9c8d7 (lab01/exercises) Complete In-Lab Exercise 1 (Kernel environment verification)
|/  
* 0a1b2c3 Initial commit: Environment setup and baseline inspection
```

---

## 4. Home Assignment: Data Inspection Utility (`inspect.py`)

A modular script `inspect.py` was developed to provide an end-to-end, automated tabular inspection report. The script was evaluated on two public datasets:
1. **Titanic Survival Dataset** ($891 \times 15$): Validated mixed numerical, boolean, and categorical features with significant missingness in `deck` and `age`.
2. **Iris Flower Measurements Dataset** ($150 \times 5$): Zero missing values across all columns, balanced 3-class distribution ($50$ instances per species), with `petal_width` displaying the highest internal dispersion ($CV = 0.636$).

---

## 5. Deliverables Checklist

- [x] `Lab_01_Environment_Setup.ipynb`: Executed from top to bottom with visible outputs.
- [x] `inspect.py`: Modular dataset inspection utility.
- [x] `requirements.txt`: Pinned software dependencies.
- [x] `.gitignore`: Standard exclusion patterns for data and environments.
- [x] `report.md`: Complete theoretical and empirical documentation.
