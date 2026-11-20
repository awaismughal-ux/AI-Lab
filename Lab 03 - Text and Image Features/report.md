# Laboratory 03 Report: Transforming Textual and Image Data into Machine-Understandable Formats

- **Course:** Introduction to Artificial Intelligence (AI-101L)
- **Instructor:** Dr. Muhammad Awais
- **Laboratory:** Lab 03 — Transform Textual and Image Data into Machine Understandable Format
- **Domain:** Feature Engineering & Representation Learning for Unstructured Modalities
- **Mapped CLOs:** CLO-2, CLO-3

---

## 1. Executive Summary

Traditional machine learning algorithms are strictly numerical vector estimators; they possess no intrinsic capacity to parse raw textual strings or high-dimensional visual matrices. This laboratory rigorously explored two fundamental modalities of unstructured data:
1. **Text Modality**: Converting variable-length qualitative prose into structured numeric document-term vectors via the **Bag-of-Words (BoW)** model (`CountVectorizer`).
2. **Visual Modality**: Transforming multi-channel spatial images into normalized, flat 1D feature arrays using **luminance grayscale conversion** and **pixel intensity flattening**.

---

## 2. Part A: Feature Engineering for Text Data (Bag-of-Words)

### 2.1 Theoretical Framework
The Bag-of-Words (BoW) paradigm models a document $d$ as an unordered multiset of words, intentionally discarding syntax, grammar, and sequential positioning. The corpus vocabulary $V = \{w_1, w_2, \dots, w_{|V|}\}$ forms an orthogonal basis for a vector space $\mathbb{R}^{|V|}$, where each document is mapped to a vector:
$$\mathbf{x}_d = \big[ \text{tf}(w_1, d), \text{tf}(w_2, d), \dots, \text{tf}(w_{|V|}, d) \big]^T$$

### 2.2 Discussion Answers & Findings
- **Column Semantics**: Each column represents a distinct token in the learned lexicographically sorted vocabulary.
- **Machine-Understandable Transformation**: Unstructured variable-length text is mapped to a fixed-dimensional matrix $\mathbf{X} \in \mathbb{N}^{N \times |V|}$, enabling dot products $\mathbf{w}^T \mathbf{x}$ and metric distance evaluations.
- **Sparsity & Counts**: Zero indicates word absence; positive integers denote word frequency. The resulting matrix exhibits high sparsity, typical of NLP vector spaces.
- **Silent Heuristics in `CountVectorizer`**: Lowercasing is enforced automatically, punctuation is stripped, and tokens with length $< 2$ (e.g. `'a'`) are removed by default regex `(?u)\b\w\w+\b`.

### 2.3 In-Lab Exercises (Text)
- **Unigrams vs. Bigrams (`ngram_range=(1,2)`)**: Evaluated vocabulary expansion from 7 unigrams to 14 total features (unigrams + bigrams: `'good day'`, `'good the'`, `'is bright'`, `'is good'`, `'is shining'`, `'shining sun'`, `'shining today'`, `'sun is'`, `'weather is'`). Bigrams preserve local phrase semantics at the cost of exponential feature space expansion and increased sparsity.
- **Document Length Normalization**: Demonstrated how long documents with repetitive words inflate dot products, and implemented $L_1$ relative frequency and $L_2$ Euclidean unit-norm scaling to level document magnitude.

---

## 3. Part B: Feature Engineering for Image Data

### 3.1 Grayscale Dimensionality Reduction & Flattening
Raw RGB images exist in $\mathbb{R}^{H \times W \times 3}$. Grayscale transformation reduces channels to 1 using ITU-R BT.601 luminance weighting:
$$L = 0.299R + 0.587G + 0.114B$$
Flattening unrolls the 2D spatial matrix row-wise into a 1D vector $\mathbf{v} \in \mathbb{R}^{HW}$.

### 3.2 Discussion Answers & Findings
- **Grayscale as Feature Engineering**: Compresses channel dimensionality by $66.7\%$, removing superficial illumination variations while retaining structural boundaries, textures, and shape geometry.
- **Dimensionality Impact**: An image of size $H \times W \times 3$ has $3HW$ features; post-grayscale flattening it has exactly $HW$ features.
- **Loss of Spatial Locality**: 1D flattening destroys 2D spatial topology; vertical adjacent pixels $(r, c)$ and $(r+1, c)$ are separated by $W$ elements. Standard dense linear weights treat each pixel independently, losing translation invariance. This structural loss motivates Convolutional Neural Networks (CNNs).
- **Pixel Intensity Normalization ($/255.0$)**: Scaling $[0, 255] \to [0.0, 1.0]$ prevents high-intensity pixels from creating exploding gradients, standardizes learning rates, and accelerates gradient descent convergence.

---

## 4. Home Assignment: High-Dimensional Image Dataset & The Curse of Dimensionality

A structured dataset of 20 images across two classes was synthesized:
- **Class 0 (`data/class_a/`)**: 10 geometric circles (varying radiuses, centers, and grayscale gradients).
- **Class 1 (`data/class_b/`)**: 10 geometric rectangles/boxes.

Using `image_to_features(path, size=(64, 64))`, each image was mapped to a 4,096-dimensional vector, forming design matrix:
$$\mathbf{X} \in \mathbb{R}^{20 \times 4096}, \quad \mathbf{y} \in \{0, 1\}^{20}$$

### Analysis of the Curse of Dimensionality ($p = 4096 \gg N = 20$)
Situating 20 training samples in a 4,096-dimensional space creates extreme vulnerability:
1. **Severe Data Sparsity**: The volume of the unit hypersphere grows exponentially with dimension ($V \propto r^p$), leaving the space virtually empty. The 20 data points exist as isolated vertices in a cosmic void.
2. **Distance Metric Collapse (Distance Concentration)**: In high dimensions, Euclidean distances between arbitrary pairs of points converge to a uniform constant:
   $$\lim_{p \to \infty} \frac{\text{dist}_{\max} - \text{dist}_{\min}}{\text{dist}_{\min}} = 0$$
   This degrades distance-based classifiers (KNN) into random chance, as every point appears equidistant from every other point.
3. **Catastrophic Overfitting**: Any linear classifier has $4,096$ free parameters to fit only $20$ data points, leading to perfect empirical training separation by memorizing background noise rather than geometric concepts.
4. **Remediation**: This failure demonstrates why global flattening is suboptimal and why dimensionality reduction (PCA) or spatial convolutional weight-sharing (CNNs in Lab 08) are mandatory.

---

## 5. Artifact Checklist

- [x] `Lab_03_Text_and_Image_Features.ipynb`: Executed from top to bottom with visible outputs and plots.
- [x] `image_utils.py`: Reusable, tested module for image transformation and batch matrix creation.
- [x] `data/class_a/` & `data/class_b/`: 20 real synthesized images on disk.
- [x] `report.md`: Complete theoretical and empirical documentation.
