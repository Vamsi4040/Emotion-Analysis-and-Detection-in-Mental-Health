# Emotion Analysis and Detection in Mental Health

## Abstract
Mental health issues, particularly depression, have become a global concern affecting millions of lives. Early identification and intervention are crucial for mitigating their impact. With the rise of social media platforms, text-based expressions of emotions have become a valuable resource for mental health detection. This study explores the application of Natural Language Processing (NLP) techniques to detect depression from textual data. Using data collected from platforms like Twitter, the research employs preprocessing steps—such as tokenization and lemmatization—and advanced machine learning models, including CNN-LSTM and transformer-based models (BERT and DistilBERT). The results indicate that while deep learning models show promise, performance is constrained by data imbalances and complex emotional nuances, highlighting the need for further improvements in model architecture and preprocessing techniques.

### Keywords
Emotion Analysis, Natural Language Processing, Mental Health Detection, Depression, Social Media, CNN-LSTM, BERT, DistilBERT, Preprocessing, Tokenization, Word Embeddings, Real-time Applications

## Introduction
Depressive disorder, commonly known as depression, significantly impairs individuals’ lives worldwide. According to various studies and reports, depression affects millions—hindering daily functioning and in severe cases leading to self-harm. Adolescence and middle-age are particularly vulnerable periods. With traditional diagnostic methods being time-consuming and resource-intensive, there is an increasing interest in leveraging the linguistic patterns found in social media text to enable early detection of depression. This project investigates how NLP techniques can extract meaningful emotional cues from text data to serve as early indicators of mental health issues.

## Related Work
Prior research has explored text-based depression detection by analyzing linguistic markers using methods such as sentiment analysis, topic modeling, and affective computing. Early work employed traditional machine learning algorithms like Support Vector Machines, Random Forests, and Naive Bayes, while recent studies have shifted towards deep learning architectures. Transformer-based models (e.g., BERT) and hybrid models (e.g., CNN-LSTM) have been shown to capture both local and global features of text. However, challenges remain due to class imbalance and the complexity of human language—especially in informal social media texts.

## Methodology

### Dataset Collection and Preprocessing
- **Data Source:**  
  The dataset used (`tweet_emotions.csv`) was created in-house using social media data (Twitter) where users express a wide range of emotions.
  
- **Text Cleaning:**  
  - Removal of URLs, mentions, hashtags, punctuation, and numbers  
  - Conversion to lowercase
  
- **Tokenization and Lemmatization:**  
  - Splitting text into tokens  
  - Using NLTK’s lemmatizer and stopword removal to reduce words to their root forms
  
- **Exploratory Data Analysis (EDA):**  
  - Generating word clouds to visualize prominent terms  
  - Plotting word frequency and sentence length distributions

### Model Development
Three main models were implemented:

1. **Emotion Classifier (CNN-LSTM based / Feedforward Neural Network):**
   - **Architecture:**  
     - Input layer: 10,000 features (from CountVectorizer)
     - Fully Connected Layer with 128 neurons and ReLU activation
     - Dropout layer (rate 0.3) to mitigate overfitting
     - Output layer: 13 classes corresponding to different emotions
   - **Training:**  
     The model was trained over 10 epochs. While the training loss decreased steadily, the validation loss indicated overfitting.

2. **BERT Model:**
   - Utilizes a pretrained BERT (bert-base-uncased) model with a custom classification head.
   - Fine-tuned on the emotion dataset.  
   - Achieved an approximate accuracy of 35%, though with low macro precision and recall due to imbalanced class distribution.

3. **DistilBERT Model:**
   - A lighter and faster variant of BERT designed for real-time applications.
   - Demonstrates a balance between performance and inference speed, making it suitable for deployment in real-world settings.

### Training and Evaluation
- **Loss and Accuracy Monitoring:**  
  Both training and validation metrics (loss and accuracy) were tracked over epochs.  
  ![Loss Plot](images/loss_plot.png)  
  *Figure 1: Training and Validation Loss over Epochs.*

  ![Accuracy Plot](images/accuracy_plot.png)  
  *Figure 2: Training and Validation Accuracy over Epochs.*

- **Evaluation Metrics:**  
  The models were evaluated using metrics such as precision, recall, and F1-score. Despite achieving weighted F1-scores around 28–29%, the macro metrics were low due to issues with underrepresented classes.

- **Confusion Matrix:**  
  A confusion matrix was generated to visualize misclassifications among the 13 emotion categories.

## Results and Analysis
- **Emotion Classifier:**  
  - **Training:** Loss decreased from 2.0078 to 0.6292 over 10 epochs.  
  - **Validation:** Loss increased from 1.8974 to 2.9749, indicating overfitting.  
  - **Overall Accuracy:** ~30%

- **BERT Model:**  
  - **Training & Validation:** Gradual reduction in training loss with validation accuracy around 35%.  
  - **Performance Issues:** Low macro precision, recall, and F1-score due to difficulty in capturing minority classes.

- **Model Comparison:**

  | Model               | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
  |---------------------|----------|-----------------|--------------|----------------|-------------------|
  | BERT                | 35%      | 8%              | 12%          | 9%             | 28%               |
  | Emotion Classifier  | 30%      | 19%             | 17%          | 17%            | 29%               |

- **Real-Time Deployment:**  
  The Gradio interface was used to create an interactive demo for real-time emotion prediction, allowing users to input text and receive immediate feedback.

## Conclusion and Future Work
This study demonstrates the potential of NLP techniques for detecting depression and other emotions from text data. While advanced models like BERT and DistilBERT show promise in capturing nuanced emotional signals, challenges such as data imbalance and overfitting persist. Future work should focus on improving preprocessing methods, exploring ensemble strategies, and incorporating multimodal data (e.g., images, audio) to enhance model performance and robustness.


## Installation
Clone the repository and install the dependencies with:
```bash
pip install -r requirements.txt
```

### Prerequisites
- Python 3.8 or later

### Setup
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/emotion-analysis.git
   cd emotion-analysis
   ```
