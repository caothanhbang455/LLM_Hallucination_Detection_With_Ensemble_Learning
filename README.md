# LLMS_1% - Hallucination detection

This project is a solution for the UIT Data Science Challenge, focusing on the detection of hallucinations in large language models. The task is to classify generated text into one of three categories: **"no"** (no hallucination), **"intrinsic"** (hallucination contradicting the provided context), or **"extrinsic"** (hallucination introducing new information not verifiable from the context).

Our methodology is based on a **2-level stacking ensemble**, which combines the predictions of several fine-tuned transformer models using a meta-learner.

---

## Project structure

The repository is organized as follows:

project/
├── old/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   ├── preprocess.py
│   ├── train_utils.py
│   └── predict_utils.py
├── vncorenlp/
│   └── (thư mục chứa file jar và resource của vncorenlp)
├── README.md
├── run_train_and_predict.ipynb
├── train.py
├── vihallu-public-test.csv
└── vihallu-train.csv

---

## Methodology

Our approach treats this problem as a **text-pair classification (NLI-style)** task. The model takes the context as the *premise* and the response as the *hypothesis* to make a classification.

---

### 1. Data preprocessing

The input consists of a context and a response, which can be very long. To handle this, we use a **sliding window approach (`HalluDataset`)**:

- The context and response are concatenated.  
- This combined text is tokenized into overlapping chunks (windows) with a specified **max_len** (e.g., 512 tokens) and **doc_stride** (e.g., 96 tokens).  
- The model makes a prediction for each window. During inference, these window-level predictions are aggregated (**mean-pooled**) to produce a single prediction for the entire example.

---

### 2. Level 1: Base models

We fine-tune several pre-trained transformer models independently on the training data. Each model serves as a *base learner* in our ensemble.

**Architecture:**  
`HalluModel` consists of a pre-trained encoder (e.g., mDeBERTa) followed by a `mean_pooling` layer and a **2-layer MLP** classification head with **GELU activation** and **Dropout**.

**Models used:**
- `MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli`  
- `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`  
- `intfloat/multilingual-e5-base`

**Training:**  
Each model is trained using `CrossEntropyLoss` with **label smoothing**, optimized by **AdamW**. The best checkpoint is saved based on the **macro F1 score** on a held-out validation set.

---

### 3. Level 2: Stacking ensemble

Instead of simple averaging, we use **stacking** to combine the predictions from the base models.

- **OOF predictions:** After training, each base model is used to generate out-of-fold (OOF) predictions (probabilities) for the validation set.  
- **Meta-learner:** These OOF probabilities are concatenated to form a new feature set (**meta-features**).  
- A **LogisticRegression** model (the “stacker” or “meta-learner”) is then trained on these meta-features, using the true labels from the validation set. This stacker learns the optimal weights to assign to each base model’s prediction.

---

### 4. Post-processing: Temperature scaling

Before generating OOF predictions, we apply **temperature scaling** (`fit_temperature_from_logits`) to each base model.  
This calibration step adjusts the model’s output probabilities to be more reliable and less over-confident, improving the quality of the inputs for the level-2 stacker.

---

## How to run

### 1. Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
