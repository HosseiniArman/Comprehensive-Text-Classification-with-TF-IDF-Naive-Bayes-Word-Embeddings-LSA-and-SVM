
# Text Classification Project

## Overview

This project implements text classification techniques to classify documents based on sentiment analysis using machine learning models. The focus is on leveraging Naive Bayes and Support Vector Machine (SVM) classifiers enhanced with Latent Semantic Analysis (LSA) to improve classification performance.

## Features

- **Data Preprocessing**: Tokenization, removal of stopwords, and handling of punctuation.
- **Word Embeddings**: Support for Word2Vec, GloVe, and FastText for creating meaningful word representations.
- **TF-IDF Calculation**: Implementation of term frequency-inverse document frequency to represent document relevance.
- **Classification Models**:
  - **Naive Bayes**
  - **SVM with Latent Semantic Analysis (SELSA)**

## Data Input

The project reads training and testing data from specified CSV files. It allows users to specify a limit on the number of documents to process. The text is tokenized, and stopwords are removed during preprocessing.

### Example

```python
search_engine.input(dataFrameTrain, dataFrameTest, buildPostingList=True, limit=100)
```

## Performance Results

### Naive Bayes Classifier

The results of the Naive Bayes classifier are as follows:

- **True Positives (TP)**: 9630
- **True Negatives (TN)**: 10999
- **False Positives (FP)**: 1501
- **False Negatives (FN)**: 2870

**Evaluation Metrics**:

- **Precision**: 0.865
- **F1 Score**: 1.058

### SVM with Latent Semantic Analysis (SELSA)

The results of the SVM classifier using LSA are as follows:

- **True Positives (TP)**: 8398
- **True Negatives (TN)**: 7822
- **False Positives (FP)**: 4677
- **False Negatives (FN)**: 4102

**Evaluation Metrics**:

- **Accuracy**: 0.649
- **Recall**: 0.672
- **Precision**: 0.642
- **F1 Score**: 0.978

## Usage

### Training the Model

You can train the Naive Bayes or SVM classifier using the following commands:

```python
# Example usage
search_engine = searchEngineLSA_SVM()
search_engine.input(dataFrameTrain, dataFrameTest, buildPostingList=True)
naive_bayes_result = search_engine.train(mode="naiveBayes")
svm_result = search_engine.train(mode="word2vecModel")  # or gloveModel / fastTextModel
```

### Evaluating the Model

After training, the models can be evaluated using the `Test` method:

```python
# Evaluate Naive Bayes
naive_bayes_evaluation = search_engine.Test(naive_bayes_classifier, test_data)

# Evaluate SVM
svm_evaluation = search_engine.Test(svm_classifier, test_data)
```

## Conclusion

The results show that the Naive Bayes classifier achieved a high precision and F1 score, while the SVM with LSA method provides a balanced accuracy and recall. These findings suggest that different models can be more effective depending on the specific characteristics of the dataset and the evaluation metrics of interest.
