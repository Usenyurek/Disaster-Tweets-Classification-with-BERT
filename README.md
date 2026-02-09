# Disaster-Tweets-Classification-with-BERT

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue)

## 📌 Project Overview
This project is an End-to-End Natural Language Processing (NLP) solution designed to classify tweets into two categories:
1.  **Real Disasters:** Tweets reporting actual emergencies (e.g., "Forest fire near La Ronge").
2.  **Not Disasters:** Tweets using disaster-related words metaphorically (e.g., "I'm on fire today!").

The project compares a **Baseline Model (TF-IDF + Logistic Regression)** against a State-of-the-Art **Deep Learning Model (BERT)**.

## 📂 Dataset
The dataset is provided by the [Kaggle NLP Getting Started Competition](https://www.kaggle.com/c/nlp-getting-started).
* **Train Set:** ~7,600 tweets (labelled)
* **Test Set:** ~3,200 tweets (unlabelled)

## 🛠️ Methodology

### 1. Exploratory Data Analysis (EDA)
* Analyzed tweet lengths and word counts.
* Visualized high-frequency words using **Word Clouds** for both classes.
* Identified that real disaster tweets tend to be longer and contain specific keywords (e.g., *fire, storm, police*).

### 2. Data Preprocessing
* **Cleaning:** Removed URLs, HTML tags, emojis, and punctuation using Regex.
* **Tokenization:**
    * For Baseline: Standard text splitting.
    * For BERT: Used `BertTokenizer` (Input IDs & Attention Masks).

### 3. Model 1: Baseline (Traditional ML)
* **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency).
* **Algorithm:** Logistic Regression.
* **Score:** F1 Score ~71%.

### 4. Model 2: BERT (Transfer Learning)
* **Architecture:** Fine-tuned `bert-base-uncased` model using TensorFlow/Keras.
* **Optimization:** Adam Optimizer with a learning rate of `2e-5`.
* **Training:** Trained for 2 epochs to prevent overfitting.
* **Score:** F1 Score ~80%+.

## 🚧 Challenges & Solutions
During the development, several technical challenges were addressed:

* **Keras 3 vs. Transformers Compatibility:** Encountered `KeyError: '__class__'` due to TensorFlow 2.16+ updates.
    * *Solution:* Configured the environment to use `tf-keras` (Legacy Mode) and pinned compatible library versions.
* **SafeTensors Error:** Issues with loading the BERT model weights in Python 3.12 environment.
    * *Solution:* Forced the model to load standard PyTorch/H5 weights using `use_safetensors=False`.
* **OOM (Out of Memory) Error:** GPU memory exhaustion during prediction on the test set.
    * *Solution:* Implemented **batch processing** for predictions instead of processing the entire dataset at once.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/disaster-tweets-bert.git](https://github.com/yourusername/disaster-tweets-bert.git)
    cd disaster-tweets-bert
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Notebook:**
    Open `Disaster_Tweets_BERT.ipynb` in Jupyter Notebook, Google Colab, or Kaggle Kernels. Make sure to enable **GPU Acceleration**.

## 📊 Results
| Model | F1 Score | Accuracy |
|-------|----------|----------|
| TF-IDF + Logistic Regression | 0.71 | ~79% |
| **BERT (Fine-Tuned)** | **0.80+** | **~83%** |

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License
[MIT](https://choosealicense.com/licenses/mit/)
