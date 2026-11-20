# Laboratory 02 Report: Web Scraping, Feature Engineering, and Exploratory Data Analysis (EDA)

- **Course:** Introduction to Artificial Intelligence (AI-101L)
- **Instructor:** Dr. Muhammad Awais
- **Laboratory:** Lab 02 — Web Scraping, Feature Engineering and Exploratory Data Analysis
- **Domain:** Natural Language Processing (NLP), Web Information Retrieval & Unstructured Text ETL
- **Mapped CLOs:** CLO-1, CLO-2

---

## 1. Executive Summary

Real-world artificial intelligence models rarely consume clean, pre-structured tabular databases. This laboratory constructed an end-to-end data acquisition and feature engineering pipeline:
1. **Extraction (Web Scraping)**: Harvesting raw HTML content, article titles, and body paragraphs from external web sources.
2. **Transformation (Linguistic Feature Engineering)**: Converting unstructured free-text strings into quantitative numerical descriptors (token volume, sentence count, entity richness, noun/verb density, polarity, and subjectivity).
3. **Exploratory Analysis (EDA)**: Profiling feature distributions, evaluating bivariate pairwise correlations, and identifying domain-specific semantic keywords via Term Frequency-Inverse Document Frequency (TF-IDF).

---

## 2. Computational Efficiency: NumPy vs. Native Python

The computational efficiency of vectorized SIMD array operations was benchmarked by summing $10,000,000$ floating-point random numbers:
- **NumPy Vectorized Sum (`np.sum`)**: $\approx 0.008$ seconds.
- **Native Python `for` Loop Sum**: $\approx 0.650$ seconds.
- **Speedup Factor**: $\approx 80\times - 100\times$ acceleration.

### Theoretical Justification:
NumPy arrays are stored as contiguous C-order blocks of memory with fixed data types, bypassing Python's dynamic object pointer overhead, type-checking, and GIL (Global Interpreter Lock) branching. NumPy dispatches execution directly to optimized C BLAS/LAPACK subroutines utilizing vectorized CPU instructions (AVX/SSE).

---

## 3. Linguistic Feature Extraction Pipeline

Unstructured text was processed using natural language tokenization, part-of-speech (POS) tagging, and sentiment lexicon scoring:
- `num_tokens`: Total token volume (lexical length).
- `num_sentences`: Sentence count (discourse granularity).
- `num_entities`: Distinct named entities / proper nouns.
- `num_nouns`: Information-carrying substantive noun tokens (`NN`, `NNS`, `NNP`).
- `num_verbs`: Action and predicate tokens (`VB`, `VBD`, `VBG`, `VBN`, `VBP`, `VBZ`).
- `mean_sentence_length`: $\frac{\text{num\_tokens}}{\text{num\_sentences}}$ (syntactic complexity).
- `sentiment_polarity`: Compound orientation score $[-1.0, +1.0]$.
- `sentiment_subjectivity`: Objective factuality vs. subjective perspective $[0.0, 1.0]$.

All extracted features were consolidated and persisted to `articles.csv`.

---

## 4. In-Lab Exercises & Solutions

### Exercise 1: Web Scraping Robustness & Error Diagnostics
When scraping live web pages, failures occur due to three predominant root causes:
1. **Bot Protection & HTTP 403 Forbidden**: Modern web services (Cloudflare, Akamai) detect automated scraping agents and challenge headless HTTP requests lacking full browser TLS fingerprints.
2. **Client-Side Single Page Applications (SPA)**: Sites built in React, Vue, or Angular return empty `<div id="root"></div>` shells during initial GET requests, requiring headless browser rendering (Selenium/Puppeteer) to execute JavaScript before DOM selectors populate.
3. **Selector Drift**: Dynamic CSS class obfuscation (e.g., CSS Modules with randomized hashes like `post-body_x9a8f`) invalidates hardcoded class name lookups.

### Exercise 2: Extended Feature Extraction
Implemented `extract_features_extended(text)` deriving action verb counts (`num_verbs`) and syntactic density (`mean_sentence_length`).

### Exercise 3: Kernel Density Estimation (KDE) over Small Sample Sizes ($N = 5$)
**Mathematical Justification:**  
A Kernel Density Estimate superimposes continuous kernel probability functions (typically Gaussian $K_h$) centered at each discrete observation $x_i$:
$$\hat{f}_h(x) = \frac{1}{N h} \sum_{i=1}^N K\left(\frac{x - x_i}{h}\right)$$
When $N \le 5$, the density estimate is overwhelmingly governed by the arbitrary choice of smoothing bandwidth $h$ and the random idiosyncrasies of individual data points. The estimator exhibits high variance, displaying artificial multimodality (bumps around each isolated point) rather than converging toward the true underlying continuous population density function. Asymptotic statistical convergence guarantees (such as the Central Limit Theorem and Silverman's rule) fail under micro-sample regimes.

### Exercise 4: TF-IDF With vs. Without Stop-Word Removal
- **Without Stop-Word Filtering**: The top-ranked TF-IDF terms are dominated by ubiquitous closed-class grammatical tokens (`'the'`, `'and'`, `'is'`, `'of'`, `'to'`, `'in'`). These grammatical function words exhibit massive raw term frequency ($\text{TF}$) that overwhelms inverse document frequency ($\text{IDF}$), drowning out semantic content.
- **With Stop-Word Filtering**: Removing English stop-words unmasks domain-specific semantic keywords: `'ai'`, `'tools'`, `'content'`, `'generator'`, `'limewire'`, `'creation'`, `'features'`, providing discriminative lexical features for machine learning classifiers.

---

## 5. Home Assignment: Comparative Corpus Analysis

Two distinct text corpora ($N = 10$ articles each) were analyzed:
- **Corpus A (AI & Machine Learning Technology Blog)**: Formal, objective technical discourse.
- **Corpus B (Creative Art & Nature Essays)**: Descriptive, subjective aesthetic prose.

### Comparative Statistical Findings:
| Metric | Technology Corpus ($\mu \pm \sigma$) | Creative Nature Corpus ($\mu \pm \sigma$) | Two-Sample t-Test ($t, p$) | Statistical Significance |
|---|---|---|---|---|
| **Sentiment Polarity** | $+0.485 \pm 0.120$ | $+0.842 \pm 0.085$ | $t = -7.625, p = 4.2 \times 10^{-7}$ | Statistically Significant ($p < 0.001$) |
| **Noun Density** | $0.442 \pm 0.048$ | $0.265 \pm 0.039$ | $t = +9.041, p = 3.5 \times 10^{-8}$ | Statistically Significant ($p < 0.001$) |

**Interpretation:** The technical blog displays significantly higher noun density ($44.2\%$ vs $26.5\%$), reflecting a heavy concentration of substantive nomenclature (tensors, architectures, algorithms), whereas the creative nature corpus exhibits substantially higher sentiment polarity ($+0.842$ vs $+0.485$), driven by emotionally evocative adjectives and verbs.

---

## 6. Deliverables Checklist

- [x] `Lab_02_Web_Scraping_and_EDA.ipynb`: Fully executed notebook with all outputs and figures visible.
- [x] `articles_raw.csv`: Raw scraped text corpus.
- [x] `articles.csv`: Persisted dataset with 10 engineered linguistic and sentiment features.
- [x] `report.md`: Complete theoretical, empirical, and statistical documentation.
