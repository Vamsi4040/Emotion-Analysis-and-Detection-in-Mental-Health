# Emotion Analysis and Detection in Mental Health Using Natural Language Processing Techniques

## Abstract
Mental health issues—particularly depression—pose a significant global challenge, impacting millions of lives. Early detection and intervention are critical to reduce the negative outcomes associated with depression. With the proliferation of social media platforms, text-based expressions of emotions have emerged as invaluable indicators for mental health detection. This project explores the use of Natural Language Processing (NLP) techniques to detect depression from textual data. By leveraging data collected from platforms such as Twitter and Reddit, the study applies extensive preprocessing (including tokenization, stopword removal, and lemmatization) and develops advanced machine learning models. These models include a CNN-LSTM based Emotion Classifier as well as transformer-based architectures such as BERT and DistilBERT. Although the deep learning models show promise, challenges related to data imbalance and the complexity of emotional nuances persist, highlighting areas for future improvement in both model architecture and preprocessing strategies.

## Keywords
Emotion Analysis, Natural Language Processing, Mental Health Detection, Depression, Social Media, CNN-LSTM, BERT, DistilBERT, Preprocessing, Tokenization, Word Embeddings, Real-time Applications

## Introduction
Depressive disorder, commonly referred to as depression, is a widespread mental illness that drastically affects individuals' lives across the globe. According to the World Health Organization (WHO), depression can severely impair one's ability to function, and in extreme cases, may even lead to self-harm. Adolescents and middle-aged adults are especially vulnerable, with rising global rates observed between 2005 and 2022.

Traditional diagnostic methods—relying on clinical interviews and self-report surveys—are not only time-consuming but also subject to human error. Recent advances in NLP have demonstrated that the linguistic patterns in a person’s text (e.g., social media posts) can offer crucial insights into their mental state. This project harnesses these techniques to automatically detect depression by analyzing the text shared on social media, thereby potentially enabling early intervention and better long-term health outcomes.

## Problem Statement
Depression is a pervasive mental health issue that significantly impacts lives worldwide. Traditional diagnostic methods are time-consuming and error-prone. This project addresses the need for an automated, data-driven approach to detect depression from social media text, which can help in early intervention and support mental health diagnostics.

## Objectives
- **Automated Emotion Detection:** Detect depression and related emotional states from social media text using NLP.
- **Model Comparison:** Implement and compare multiple models (Emotion Classifier, BERT, DistilBERT).
- **Real-Time Prediction:** Develop an interactive interface for real-time emotion classification using Gradio.
- **Data Analysis:** Perform extensive EDA to understand emotional trends and text characteristics.
- **Identify Challenges:** Address issues such as data imbalance and overfitting while suggesting potential improvements.

## Related Work
Depression detection via text analysis has attracted significant attention in recent years due to the vast amounts of available social media data. Early studies employed sentiment analysis to classify text based on emotional tone, while subsequent research incorporated more complex linguistic and statistical methods—such as topic modeling and affective computing. Traditional machine learning algorithms (e.g., Support Vector Machines, Random Forests, Naive Bayes) have been used in earlier efforts, but they often struggled with the inherent complexity of human language, particularly in informal contexts.

More recent work has turned to deep learning models:
- **CNNs and LSTMs:** These architectures capture local and sequential features of text, respectively.
- **Transformer Models (BERT and DistilBERT):** These models, with their bidirectional attention mechanisms, are highly effective at capturing nuanced context from text. However, challenges such as data imbalance and subtle emotional cues continue to limit their overall performance.

The literature indicates that while advanced models can outperform traditional methods in some respects, improvements in model tuning, data preprocessing, and handling imbalanced datasets are necessary to further enhance performance.

## Methodology

### Dataset Collection and Preprocessing
The dataset used in this study was collected from social media platforms—primarily Twitter and Reddit—where users express a broad spectrum of emotions. Posts are labeled according to various emotional states (e.g., sadness, anger, joy) and are further categorized based on whether they indicate depressive symptoms.

### Dataset

The dataset (`tweet_emotions.csv`) was created in-house and contains real-world social media posts with the following columns:

- **tweet_id:** Unique identifier.
- **Emotion:** Labeled emotional state (e.g., sadness, anger, neutral, etc.).
- **Text:** Original tweet content.

### Preprocessing Steps

- **Text Cleaning:** Remove URLs, mentions, hashtags, punctuation, and special characters; convert text to lowercase.
- **Tokenization:** Split text into tokens.
- **Stopword Removal:** Remove common words (e.g., "the", "and", "of").
- **Lemmatization:** Reduce words to their base forms.


#### Data Preparation Steps:
1. **Text Cleaning:**  
   - **Removal of Noise:** URLs, mentions, hashtags, punctuation, and special characters are removed.
   - **Lowercasing:** All text is converted to lowercase for consistency.

2. **Tokenization:**  
   - Text is split into tokens (words or subwords) to prepare for embedding and further analysis.

3. **Stopword Removal:**  
   - Common words (e.g., "the", "and", "of") that do not carry significant meaning are removed to reduce dimensionality.

4. **Lemmatization:**  
   - Words are reduced to their root forms to ensure consistency and to manage vocabulary size.

5. **Exploratory Data Analysis (EDA):**  
   - **Emotion Distribution:** Visualization of the frequency of each emotion reveals that neutral, worry, and happiness are the most prevalent, while emotions such as anger, boredom, and enthusiasm appear less frequently.
   - **Word Cloud Generation:** Word clouds illustrate the most common words and contextual themes in the dataset.
   - **Sentence Length Distribution:** Histograms of sentence lengths show a right-skewed distribution, indicating that most sentences are short and concise—a characteristic common in online communications.

### Model Development
Three main models were developed to perform emotion detection from the preprocessed text:

#### 1. Emotion Classifier
A simple feedforward neural network designed to classify text into 13 emotional categories.  
- **Architecture:**
  - **Input Layer:** Processes 10,000 features extracted via CountVectorizer.
  - **Hidden Layer:** A fully connected layer with 128 neurons followed by a ReLU activation function to introduce non-linearity.
  - **Dropout Layer:** A dropout rate of 0.3 is applied to prevent overfitting.
  - **Output Layer:** A final linear layer outputs predictions across 13 emotion classes.
- **Training:**  
  The model was trained for 10 epochs. While the training loss consistently decreased (from 2.0078 to 0.6292), the validation loss increased (from 1.8974 to 2.9749), indicating issues with overfitting.

#### 2. BERT Model
A transformer-based model leveraging the pretrained `bert-base-uncased` architecture, fine-tuned for emotion detection.
- **Key Characteristics:**
  - Uses bidirectional attention to capture context from both directions in a sentence.
  - Shows better contextual understanding but is computationally expensive.
- **Performance:**  
  Achieves approximately 35% accuracy on the validation set, with challenges in macro-level metrics (precision, recall, F1-score) due to imbalanced class distributions.

#### 3. DistilBERT Model
A lightweight and faster version of BERT, optimized for real-time applications.
- **Advantages:**
  - Retains 97% of BERT’s performance while significantly reducing computational requirements.
  - Ideal for scenarios requiring quick, real-time predictions.
- **Deployment:**  
  Integrated into a Gradio interface to provide immediate emotion predictions on user-supplied text.

### Model Training and Evaluation
The training pipeline included:
- **Label Encoding:** Converting categorical emotion labels into numerical values.
- **Data Splitting:** Dividing the dataset into training (80%), validation (10%), and test (10%) sets using stratified sampling to maintain class balance.
- **Vectorization:** Applying CountVectorizer with a maximum feature limit of 10,000 to convert text to numerical arrays.
- **Conversion to Tensors:** Transforming vectorized data into PyTorch tensors for model input.
- **Training Loop:** Monitoring training and validation losses and accuracies over epochs.
- **Evaluation Metrics:** Accuracy, precision, recall, and F1-scores were calculated to assess model performance. A confusion matrix was also generated to visualize misclassifications among the 13 classes.

/////////////- HERE Example Visuals:

Loss Plot:

Figure 1: Training and Validation Loss over Epochs.

Accuracy Plot:

Figure 2: Training and Validation Accuracy over Epochs.///////////////////

## Results and Analysis

### Performance Metrics
The models were compared based on various evaluation metrics:

| Model               | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
|---------------------|----------|-----------------|--------------|----------------|-------------------|
| **BERT**                | 35%      | 8%              | 12%          | 9%             | 28%               |
| **Emotion Classifier**  | 30%      | 19%             | 17%          | 17%            | 29%               |

- **Emotion Classifier:**  
  Although it achieved a decent training performance (loss reduced to 0.6292), the increasing validation loss suggested overfitting. Its weighted F1-score of 29% indicates that it struggles particularly with minority classes.
  
- **BERT Model:**  
  Provided slightly higher overall accuracy (35%) but exhibited very low macro precision and recall. This shortfall is likely due to its inability to adequately learn from imbalanced data, resulting in poor performance on underrepresented classes.
  
- **DistilBERT:**  
  Balances performance and speed, making it particularly attractive for real-time applications. Although detailed metrics are similar to BERT in some respects, its computational efficiency gives it an edge in deployment scenarios.

### Visualizations
The project includes several visualizations to better understand model performance and data characteristics:

- **Loss Plot:**  
  ![Loss Plot](images/loss_plot.png)  
  *Figure 1: Training and Validation Loss over Epochs.*

- **Accuracy Plot:**  
  ![Accuracy Plot](images/accuracy_plot.png)  
  *Figure 2: Training and Validation Accuracy over Epochs.*

- **Confusion Matrix:**  
  A detailed confusion matrix illustrates the distribution of correct and incorrect classifications across the 13 emotion classes, highlighting the challenge of correctly predicting minority classes.

## Conclusion and Future Work
This study demonstrates the potential of NLP techniques in detecting depression and other emotions from text data collected from social media platforms. The application of deep learning models—ranging from simple feedforward networks to complex transformer architectures like BERT and DistilBERT—shows that while significant progress has been made, there remain notable challenges:
- **Data Imbalance:** Underrepresented emotions result in low macro performance metrics.
- **Overfitting:** Evident in the Emotion Classifier, where validation loss increases despite improvements in training loss.
- **Computational Costs:** Transformer-based models, particularly BERT, demand substantial computational resources, which can limit their real-time applicability.

Future research directions include:
- Enhancing preprocessing methods and experimenting with different embedding techniques.
- Exploring ensemble methods and more robust architectures.
- Incorporating multimodal data (e.g., images, audio) to capture richer contextual information.
- Optimizing hyperparameters and fine-tuning transformer models further to handle class imbalance more effectively.

Ultimately, the integration of NLP-based emotion detection into mental health diagnostics holds promise for earlier intervention, reduced diagnostic burden, and improved patient outcomes.

## Installation
Clone the repository and install the dependencies with:
```bash
pip install -r requirements.txt
```

### Prerequisites
- Python 3.8 or later

### Setup Instructions
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/emotion-analysis.git
cd emotion-analysis
pip install -r requirements.txt
```

## References

1. A. Li, D. Jiao, and T. Zhu, “Detecting depression stigma on social media: A linguistic analysis,” *Journal of Affective Disorders*, vol. 232, pp. 358–362, 2018.
2. L. Squarcina, F. M. Villa, M. Nobile, E. Grisan, and P. Brambilla, “Deep learning for the prediction of treatment response in depression,” *Journal of Affective Disorders*, vol. 281, pp. 618–622, 2021.
3. T. Zhang, A. M. Schoene, S. Ji, and S. Ananiadou, “Natural language processing applied to mental illness detection: A narrative review,” *NPJ Digital Medicine*, vol. 5, no. 1, pp. 1–13, 2022.
4. A. Leis, F. Ronzano, M. A. Mayer, L. I. Furlong, and F. Sanz, “Detecting signs of depression in tweets in Spanish: Behavioral and linguistic analysis,” *Journal of Medical Internet Research*, vol. 21, no. 6, p. e14199, 2019.
5. L. S. Jones, E. Anderson, M. Loades, R. Barnes, and E. Crawley, “Can linguistic analysis be used to identify whether adolescents with a chronic illness are depressed?,” *Clinical Psychology & Psychotherapy*, vol. 27, no. 2, pp. 179–192, 2020.
6. A. Picardi et al., “A randomized controlled trial of the effectiveness of a program for early detection and treatment of depression in primary care,” *Journal of Affective Disorders*, vol. 198, pp. 96–101, 2016.
7. K. Rost, J. L. Smith, and M. Dickinson, “The effect of improving primary care depression management on employee absenteeism and productivity: A randomized trial,” *Medical Care*, vol. 42, no. 12, pp. 1202–1210, 2004.
8. Statista, “Number of worldwide social network users,” 2023. [Online]. Available: [https://www.statista.com/statistics/278414/number-of-worldwide-socialnetwork-users/](https://www.statista.com/statistics/278414/number-of-worldwide-socialnetwork-users/). [Accessed: 08-Dec-2024].
9. DataReportal, “Social media users,” 2023. [Online]. Available: [https://datareportal.com/social-media-users](https://datareportal.com/social-media-users). [Accessed: 08-Dec-2024].
10. A. Dhand, D. A. Luke, C. E. Lang, and J. M. Lee, “Social networks and neurological illness,” *Nature Reviews Neurology*, vol. 12, no. 10, pp. 605–612, 2016.

