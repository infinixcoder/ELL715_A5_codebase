# Part 1: Viola-Jones Face Detector

You are tasked with implementing the famous Viola-Jones face detector from scratch. The use of external libraries for implementing sub-functionalities and matrix manipulations is permitted, but direct usage of functions/classes that directly implement the Viola-Jones algorithm is strictly prohibited (except for reading images).

## Deliverables:

1.  **Final test accuracy.**
2.  **Face detection results on a couple of images with multiple faces.** (You can use images from the internet.)
3.  **A well-documented codebase and an informal report.** The report should include all the results.

## Implementation Steps:

### 1. Dataset Generation (20 marks)

The dataset consists of three folders: `maleStaff` and `female` for training the Viola-Jones classifier, and `male` for testing.

* **'face' class:** For ground truths, select a 16x16 patch from the center of each image.
* **'not-a-face' class:** Extract 5 other 16x16 random patches from each image.
* Repeat this process for all images in the training and testing sets.

### 2. Haar Features (20 marks)

Extract Haar features. Consider horizontal, vertical, and diagonal Haar filters of multiple scales.

### 3. Integral Image (20 marks)

To extract Haar features efficiently, implement Integral Image based feature extraction. Refer to the Viola-Jones paper for more details.

### 4. Adaboost Algorithm (40 marks)

Implement the Adaboost algorithm from scratch to classify an image as 'face' or 'not-a-face'.

### 5. Cascade of Classifiers (20 marks)

Finally, arrange these classifiers in a cascade, as described in the original paper.



The following is a detailed procedure for implementing the specified components of the **Viola-Jones face detector**, based on the provided paper.

---

## 2. Haar Features 🎨

[cite_start]The features used in the Viola-Jones detector are **rectangle features** [cite: 75][cite_start], which are reminiscent of Haar basis functions[cite: 34, 75]. [cite_start]The value of a rectangle feature is the difference between the sum of the pixels within white rectangular regions and the sum of pixels in the grey rectangular regions[cite: 71].

### Feature Types:

The paper specifies three kinds of features:

* [cite_start]**Two-rectangle features:** The difference between the sum of pixels in two adjacent rectangular regions of the same size and shape (either horizontally or vertically adjacent)[cite: 76, 77]. 
* [cite_start]**Three-rectangle features:** The sum of pixels in the two outside rectangles is subtracted from the sum in the center rectangle[cite: 78]. 
* [cite_start]**Four-rectangle features:** Computes the difference between diagonal pairs of rectangles[cite: 79]. 

### Extraction:

[cite_start]These features should be extracted at **multiple scales and locations** within the image sub-window[cite: 37, 83]. [cite_start]Note that the total set of rectangle features is **overcomplete** (not a complete basis)[cite: 81, 88]. [cite_start]For a base resolution of a $24 \times 24$ sub-window, the exhaustive set of these rectangle features is over 180,000[cite: 80, 122].

---

## 3. Integral Image 🖼️

[cite_start]The **Integral Image** representation (or Summed-Area Table [cite: 89][cite_start]) is introduced to compute the rectangle features very rapidly[cite: 31, 83]. [cite_start]It allows any Haar-like feature to be computed at any scale or location in **constant time**[cite: 37].

### Definition and Computation:

1.  [cite_start]**Definition:** The Integral Image, $ii(x, y)$, at a location $(x, y)$ contains the sum of all pixels $i(x', y')$ in the original image that are **above and to the left** of $(x, y)$, inclusive[cite: 84]:
    $$ii(x,y) = \sum_{x' \le x, y' \le y} i(x', y')$$

2.  [cite_start]**Recurrence Relations (for efficient computation):** The Integral Image can be computed in a single pass over the original image using the following recurrences[cite: 104]:
    [cite_start]$$s(x,y) = s(x, y-1) + i(x, y) \quad (1) [cite: 101, 102]$$
    [cite_start]$$ii(x,y) = ii(x-1, y) + s(x, y) \quad (2) [cite: 101, 103]$$
    [cite_start]where $s(x, y)$ is the cumulative row sum, $s(x, -1) = 0$, and $ii(-1, y) = 0$[cite: 104].

### Feature Value Calculation:

[cite_start]The sum of pixels within any arbitrary rectangular region (like region D in the diagram) can be computed with only **four array references** to the Integral Image[cite: 98, 105].



For a rectangle defined by points 1, 2, 3, and 4, the sum of pixels in region D is:
[cite_start]$$\text{Sum}(D) = ii(4) + ii(1) - (ii(2) + ii(3)) [cite: 99]$$

* [cite_start]The difference between two adjacent rectangular sums (a **two-rectangle feature**) can be computed in **six array references**[cite: 107].
* [cite_start]A **three-rectangle feature** requires **eight array references**[cite: 107].
* [cite_start]A **four-rectangle feature** requires **nine array references**[cite: 107].

---

## 4. Adaboost Algorithm 🤖

[cite_start]A variant of **Adaboost** is used both to train the classifier and to select a small set of important features from the large initial set[cite: 38, 116]. [cite_start]The goal is to classify an image sub-window as 'face' or 'not-a-face'[cite: 115].

### Weak Classifier (Single Feature Selection):

1.  [cite_start]The weak learning algorithm is **constrained** so that each weak classifier returned depends on only a **single feature**[cite: 41, 43]. [cite_start]This converts each boosting stage into a feature selection process[cite: 43].
2.  [cite_start]For the selected feature, $f_{j}$, the weak learner determines an **optimal threshold classification function**, $h_{j}(x)$, to minimize the number of misclassified examples[cite: 127].
3.  [cite_start]The weak classifier $h_{j}(x)$ outputs 1 (face) or 0 (not-a-face) based on the feature value, $f_{j}(x)$, a threshold, $\theta_{j}$, and a parity, $p_{j}$ (direction of the inequality sign)[cite: 128]:
    [cite_start]$$h_{j}(x) = \begin{cases} 1 & \text{if } p_{j}f_{j}(x) < p_{j}\theta_{j} \\ 0 & \text{otherwise} \end{cases} [cite: 129, 130, 131]$$

### Boosting Procedure (Classifier Learning):

[cite_start]The Adaboost algorithm constructs a strong classifier $h(x)$ by combining a set of weak classifiers $h_t(x)$ selected over $T$ rounds (where $t=1, \dots, T$)[cite: 157, 159].

1.  [cite_start]**Initialization:** Initialize weights $w_{1,i}$ to each training example $x_i$: $\frac{1}{2m}$ for negative examples ($y_i=0$, $m$ is total negatives) and $\frac{1}{2l}$ for positive examples ($y_i=1$, $l$ is total positives)[cite: 145, 146].
2.  **For $t=1$ to $T$ rounds:**
    * [cite_start]**Normalize Weights:** Normalize $w_{t,i}$ so that $w_t$ is a probability distribution: $w_{t,i}\leftarrow\frac{w_{t,i}}{\sum_{j=1}^{n}w_{t,j}}$[cite: 147, 148, 149, 150].
    * [cite_start]**Train/Select Weak Classifier:** For each feature $j$, train a single-feature classifier $h_{j}$ and compute its error $\epsilon_{j}$ with respect to the current weights $w_{t}$: $\epsilon_{j} = \sum_{i}w_{t,i}|h_{j}(x_{i}) - y_{i}|$[cite: 151, 152]. [cite_start]Choose the classifier $h_{t}$ with the lowest error $\epsilon_{t}$[cite: 153].
    * [cite_start]**Update Weights:** Update the weights for the next round: $w_{t+1,i} = w_{t,i}\beta_{t}^{1-e_{i}}$ [cite: 154][cite_start], where $e_{i}$ is the classification error (0 or 1), and $\beta_{t}=\frac{\epsilon_{t}}{1-\epsilon_{t}}$[cite: 155].
3.  **Final Strong Classifier:** The final strong classifier $h(x)$ is a weighted majority vote of the weak classifiers $h_{t}(x)$:
    [cite_start]$$h(x) = \begin{cases} 1 & \sum_{t=1}^{T}\alpha_{t}h_{t}(x) \ge \frac{1}{2}\sum_{t=1}^{T}\alpha_{t} \\ 0 & \text{otherwise} \end{cases} [cite: 157]$$
    [cite_start]where $\alpha_{t} = \log\frac{1}{\beta_{t}}$[cite: 158].

---

## 5. Cascade of Classifiers 🚀

[cite_start]The **cascade of classifiers** is a structure for combining successively more complex classifiers[cite: 12, 45, 176]. [cite_start]It acts as an object-specific focus-of-attention mechanism to dramatically increase the speed of the detector by quickly discarding non-object regions[cite: 12, 45].

### Structure and Operation:

1.  [cite_start]The detection process is a sequence of classifiers, arranged like a **degenerate decision tree**[cite: 56, 179]. 
2.  [cite_start]**All sub-windows** are applied to the first classifier[cite: 187, 198].
3.  [cite_start]The **initial classifier** is extremely simple and efficient (e.g., two features) [cite: 52, 203][cite_start], and is adjusted (by its threshold) to achieve a **very high detection rate** (close to 100% true positives)[cite: 177, 204].
4.  [cite_start]If any classifier in the cascade yields a **negative outcome** (F), the sub-window is immediately rejected, and no further processing is performed on it[cite: 182, 196, 210].
5.  [cite_start]A **positive result** (T) from one classifier triggers the evaluation of the next, slightly more complex classifier[cite: 54, 180, 181].
6.  [cite_start]The cascade structure ensures that the majority of negative sub-windows are rejected at the earliest possible stage[cite: 199, 210].

### Training the Cascade:

1.  [cite_start]Each stage in the cascade is trained using **AdaBoost**[cite: 183].
2.  [cite_start]The default AdaBoost threshold is adjusted to **minimize false negatives** (maximize detection rate)[cite: 177, 183].
3.  [cite_start]A target is selected for the minimum reduction in false positives and the maximum decrease in detection rate for each stage[cite: 225].
4.  [cite_start]Features are added to a stage until its target detection and false positive rates are met (tested on a validation set)[cite: 226].
5.  [cite_start]The negative examples used to train subsequent stages are the **false positives** collected by scanning the *partial* cascade (all previous stages) across the set of non-face images[cite: 266]. [cite_start]These are considered "harder" examples[cite: 214].
6.  [cite_start]Stages are added until the overall target for the final false positive and detection rate is met[cite: 227].