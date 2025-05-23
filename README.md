# SMS Spam Classifier

## Project Description
This project is an SMS Spam Classifier that uses advanced machine learning techniques to classify SMS messages as spam or not spam (ham). It includes comprehensive data preprocessing, feature extraction, visualization, model training, evaluation, and saving the final model pipeline for deployment.

## Features
- Loads and preprocesses SMS spam dataset with enhanced cleaning
- Custom text preprocessing including:
  - Lowercasing, emoji removal, URL, email, and phone number removal
  - Tokenization, stopword removal, stemming, and optional lemmatization
- Extracts additional text features such as:
  - Word count, character count, average word length, uppercase count
  - Exclamation and question mark counts, currency symbol presence
  - Sentence count, average sentence length, and POS tagging counts (nouns, verbs)
- Data visualization with:
  - Distribution plots, boxplots, word clouds for spam and ham messages
  - Correlation heatmap of extracted features
- Trains multiple models including:
  - Naive Bayes (ComplementNB), Logistic Regression, Random Forest, SVM (LinearSVC), Gradient Boosting
  - Calibrated versions of SVM models for probability estimates
- Evaluates models with:
  - Cross-validation F1 scores, classification reports, confusion matrices, ROC curves
- Builds a final pipeline combining:
  - Text preprocessing and TF-IDF vectorization (1-3 grams)
  - Numeric feature extraction and scaling
  - SMOTE for handling class imbalance
  - Logistic Regression classifier with balanced class weights
- Optional hyperparameter tuning using RandomizedSearchCV
- Predicts new messages with:
  - Spam/ham classification, confidence scores, and top feature indicators
- Saves the entire trained pipeline and components separately for deployment

## Installation
Clone the repository:

```bash
git clone https://github.com/yourusername/sms-spam-classifier.git
cd sms-spam-classifier
```

Create and activate a virtual environment (recommended):

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

If you don't have a requirements.txt file, install the packages directly:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk joblib imbalanced-learn emoji wordcloud
```

Download NLTK data:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

## Training the Model
Run the main script to train and evaluate the models:

```bash
python spam_classifier.py
```

This will:

- Load and preprocess the data with advanced cleaning and balancing
- Visualize data distributions and feature correlations
- Train and evaluate multiple machine learning models including ensemble and calibrated models
- Build and train the final pipeline with SMOTE and Logistic Regression
- Optionally perform hyperparameter tuning (disabled by default)
- Test example messages for spam prediction with confidence scores and feature insights
- Save the trained pipeline and components for deployment

## Example Messages Tested
The script tests the final model on example messages such as:

- "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005."
- "Hey, how are you doing today?"
- "Congratulations! You've won a $1000 Walmart gift card. Click here to claim."
- "Can we meet tomorrow to discuss the project?"
- "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!"
- "Hi Mom, I'll be home for dinner tonight. See you around 7pm!"
- "Your account has been compromised. Click here to secure it now!"
- "Meeting reminder: Tomorrow at 3pm in conference room B"
- "Last chance to claim your prize! Text STOP to opt out"
- "The project deadline has been extended to Friday"

## Model Saving
The final trained model pipeline and related artifacts are saved as:

- `spam_classifier_pipeline.pkl` (entire pipeline)
- `spam_classifier_vectorizer.pkl` (TF-IDF vectorizer)
- `spam_classifier_classifier.pkl` (Logistic Regression classifier)

These files can be loaded later for deployment or further predictions.

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk
- joblib
- imbalanced-learn
- emoji
- wordcloud

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- SMS Spam Collection Dataset from Kaggle
- Scikit-learn, NLTK, imbalanced-learn, and other open-source libraries
#   S p a m - S M S - D e t e c t i o n 
 
 
