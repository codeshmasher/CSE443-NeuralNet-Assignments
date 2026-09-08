# Problem Set 02 – Bank Marketing Classification

## Abstract
This assignment applies a feedforward neural network to the UCI Bank Marketing dataset to predict whether a client subscribes to a term deposit. The workflow includes dataset loading, preprocessing, model training for 10 epochs, evaluation, and visualization of training history. Results show strong overall accuracy (~0.89) while highlighting class imbalance challenges for the positive class.

---

## Dataset
- **Source:** UCI Bank Marketing Dataset (`bank-full.csv`)
- **Separator:** `;`
- **Target column:** **y** (term deposit subscription: yes/no)
- **Features:** 16 attributes — age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome

---

## Preprocessing
- Converted target **y** into binary values (`yes` → `1`, `no` → `0`).
- Encoded categorical features using `LabelEncoder`.
- Scaled numerical features with `StandardScaler`.
- Train-test split: **80/20** with stratification on the target to preserve class balance.
- Validation split during training: **20%** of the training set.

---

## Model
- **Architecture:** Feedforward neural network (MLP)
  - Input: 16 features
  - Hidden layers: `Dense(32, relu)`, `Dense(16, relu)`
  - Output: `Dense(1, sigmoid)`
- **Optimizer:** Adam
- **Loss:** Binary crossentropy
- **Epochs:** 10
- **Batch size:** 64
- **Validation split:** 20%

---

## Results

### Accuracy
Final test accuracy: **0.89**

### Confusion Matrix
[[7686  299]
 [ 618  440]]

### Classification Report
              precision    recall  f1-score   support

           0       0.93      0.96      0.95      7985
           1       0.60      0.42      0.49      1058

    accuracy                           0.89      9043
   macro avg       0.77      0.69      0.72      9043
weighted avg       0.88      0.89      0.88      9043

**Interpretation:**
- Class 0 (no subscription) is predicted with high precision and recall.
- Class 1 (subscription) is harder to predict due to imbalance; precision and recall are lower, producing an F1‑score around 0.49.
- Overall accuracy is strong, but additional techniques (class weighting, resampling) could improve positive-class performance.

---

## Training Plots
- **Accuracy vs Epoch** (Train vs Validation)
- **Loss vs Epoch** (Train vs Validation)
---

## Deliverables
- `banking_model.py` — final training and evaluation code  
- `Banking_Training_Accuracy_And_Loss.png` — accuracy and loss plot  
- `README.md` — documentation of Problem Set 2
