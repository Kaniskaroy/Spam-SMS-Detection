# SMS Spam Classifier - Complete Working Version
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, roc_curve, auc, f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import re
import joblib
import time
import emoji
from wordcloud import WordCloud
from collections import Counter
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Custom text preprocessor that can be used in sklearn pipelines"""
    def __init__(self, stem=True, lemmatize=False, remove_emoji=True):
        self.stem = stem
        self.lemmatize = lemmatize
        self.remove_emoji = remove_emoji
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Preprocess individual text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove emojis if specified
        if self.remove_emoji:
            text = emoji.replace_emoji(text, replace='')
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        words = word_tokenize(text)
        
        # Remove stopwords and stem/lemmatize
        processed_words = []
        for word in words:
            if word not in self.stop_words and len(word) > 2:
                if self.lemmatize:
                    word = self.lemmatizer.lemmatize(word, pos='v')
                if self.stem:
                    word = self.stemmer.stem(word)
                processed_words.append(word)
        
        return ' '.join(processed_words)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [self.preprocess_text(text) for text in X]

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts additional text features"""
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def extract_features(self, text):
        """Extract features from a single text"""
        features = {}
        
        # Basic features
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text else 0
        features['uppercase_count'] = sum(1 for char in text if char.isupper())
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['has_currency'] = int(any(symbol in text for symbol in ['$', '£', '€']))
        
        # Advanced features
        sentences = sent_tokenize(text)
        features['sentence_count'] = len(sentences)
        features['avg_sentence_length'] = np.mean([len(sent) for sent in sentences]) if sentences else 0
        
        # POS tagging features
        pos_tags = nltk.pos_tag(word_tokenize(text))
        pos_counts = Counter(tag for word, tag in pos_tags)
        features['noun_count'] = pos_counts.get('NN', 0) + pos_counts.get('NNS', 0)
        features['verb_count'] = pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + \
                                pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + \
                                pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)
        
        return features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            text_features = self.extract_features(text)
            features.append(list(text_features.values()))
        return np.array(features)

def load_and_prepare_data():
    """Load and prepare the SMS spam dataset with enhanced cleaning"""
    # Load the dataset
    data = pd.read_csv('spam.csv', encoding='latin-1')
    
    # Clean the data
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    
    # Remove duplicates
    data = data.drop_duplicates(subset=['message'])
    
    # Balance the dataset (optional - could also use SMOTE)
    spam_count = data['label'].sum()
    ham_data = data[data['label'] == 0].sample(n=min(2*spam_count, len(data[data['label'] == 0])), random_state=42)
    balanced_data = pd.concat([ham_data, data[data['label'] == 1]])
    
    # Extract basic features for visualization
    balanced_data['word_count'] = balanced_data['message'].apply(lambda x: len(x.split()))
    balanced_data['char_count'] = balanced_data['message'].apply(len)
    balanced_data['uppercase_count'] = balanced_data['message'].apply(lambda x: sum(1 for char in x if char.isupper()))
    balanced_data['exclamation_count'] = balanced_data['message'].apply(lambda x: x.count('!'))
    balanced_data['question_count'] = balanced_data['message'].apply(lambda x: x.count('?'))
    balanced_data['has_currency'] = balanced_data['message'].apply(lambda x: int(any(symbol in x for symbol in ['$', '£', '€'])))
    
    return balanced_data

def visualize_data_enhanced(data):
    """Create enhanced visualizations of the dataset"""
    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 18))
    
    # 1. Label distribution
    plt.subplot(3, 2, 1)
    ax = sns.countplot(x='label', data=data)
    plt.title('Distribution of Spam vs Ham Messages', fontsize=12)
    plt.xlabel('Message Type (0=Ham, 1=Spam)')
    plt.ylabel('Count')
    
    # Add percentage annotations
    total = float(len(data))
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 20,
                '{:1.2f}%'.format((height/total)*100),
                ha='center', fontsize=10)
    
    # 2. Word count distribution
    plt.subplot(3, 2, 2)
    sns.boxplot(x='label', y='word_count', data=data)
    plt.title('Word Count Distribution by Message Type', fontsize=12)
    plt.xlabel('Message Type (0=Ham, 1=Spam)')
    plt.ylabel('Word Count')
    
    # 3. Uppercase count distribution
    plt.subplot(3, 2, 3)
    sns.boxplot(x='label', y='uppercase_count', data=data)
    plt.title('Uppercase Count Distribution by Message Type', fontsize=12)
    plt.xlabel('Message Type (0=Ham, 1=Spam)')
    plt.ylabel('Uppercase Count')
    
    # 4. Word clouds for spam and ham
    plt.subplot(3, 2, 4)
    spam_words = ' '.join(data[data['label'] == 1]['message'])
    wordcloud = WordCloud(width=600, height=400, background_color='white').generate(spam_words)
    plt.imshow(wordcloud)
    plt.title('Frequent Words in Spam Messages', fontsize=12)
    plt.axis('off')
    
    plt.subplot(3, 2, 5)
    ham_words = ' '.join(data[data['label'] == 0]['message'])
    wordcloud = WordCloud(width=600, height=400, background_color='white').generate(ham_words)
    plt.imshow(wordcloud)
    plt.title('Frequent Words in Ham Messages', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Correlation heatmap of features
    numeric_cols = ['word_count', 'char_count', 'uppercase_count', 'exclamation_count', 
                   'question_count', 'has_currency']
    if all(col in data.columns for col in numeric_cols):
        plt.figure(figsize=(10, 8))
        corr = data[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap', fontsize=14)
        plt.show()

def evaluate_model(model, X_train, X_test, y_train, y_test, cv=5):
    """Enhanced model evaluation with cross-validation"""
    start_time = time.time()
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Time taken
    time_taken = time.time() - start_time
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model.__class__.__name__}')
    plt.show()
    
    # Plot ROC curve if probabilities are available
    if y_pred_proba is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
    
    return {
        'model_name': model.__class__.__name__,
        'cv_mean_f1': np.mean(cv_scores),
        'cv_std_f1': np.std(cv_scores),
        'test_accuracy': accuracy,
        'test_f1': f1,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'time_taken': time_taken
    }

def train_models_enhanced(X_train, X_test, y_train, y_test):
    """Train and evaluate different models with enhanced features"""
    # Define models with initial parameters
    models = {
        'Naive Bayes': ComplementNB(alpha=0.1),
        'Logistic Regression': LogisticRegression(C=10, penalty='l2', max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced_subsample'),
        'SVM': LinearSVC(C=1.0, class_weight='balanced', max_iter=10000),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    }
    
    # Add calibrated versions of models that don't support predict_proba
    calibrated_models = {
        'Calibrated SVM': CalibratedClassifierCV(models['SVM'], cv=3),
        'Calibrated LinearSVC': CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=10000), cv=3)
    }
    
    models.update(calibrated_models)
    
    results = []
    
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        try:
            model_result = evaluate_model(model, X_train, X_test, y_train, y_test)
            results.append(model_result)
            
            # Print results
            print(f"\nCross-validation F1: {model_result['cv_mean_f1']:.3f} ± {model_result['cv_std_f1']:.3f}")
            print(f"Test Accuracy: {model_result['test_accuracy']:.3f}")
            print(f"Test F1 Score: {model_result['test_f1']:.3f}")
            print(f"Time Taken: {model_result['time_taken']:.2f} seconds")
            print("\nClassification Report:")
            print(model_result['classification_report'])
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    # Create voting classifier from top models
    top_models = [(name, model) for name, model in models.items() 
                 if name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting']]
    if top_models:
        voting_clf = VotingClassifier(
            estimators=top_models,
            voting='soft',
            n_jobs=-1
        )
        print("\n=== Training Voting Classifier ===")
        voting_result = evaluate_model(voting_clf, X_train, X_test, y_train, y_test)
        results.append(voting_result)
        
        # Print results
        print(f"\nCross-validation F1: {voting_result['cv_mean_f1']:.3f} ± {voting_result['cv_std_f1']:.3f}")
        print(f"Test Accuracy: {voting_result['test_accuracy']:.3f}")
        print(f"Test F1 Score: {voting_result['test_f1']:.3f}")
        print(f"Time Taken: {voting_result['time_taken']:.2f} seconds")
        print("\nClassification Report:")
        print(voting_result['classification_report'])
    
    return pd.DataFrame(results)

def build_final_pipeline():
    """Build the final pipeline with optimal parameters"""
    # Text features pipeline
    text_pipeline = Pipeline([
        ('preprocessor', TextPreprocessor(stem=True, lemmatize=False)),
        ('vectorizer', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95
        ))
    ])
    
    # Numeric features pipeline
    numeric_pipeline = Pipeline([
        ('feature_extractor', FeatureExtractor()),
        ('scaler', StandardScaler())
    ])
    
    # Combine features
    feature_union = FeatureUnion([
        ('text_features', text_pipeline),
        ('numeric_features', numeric_pipeline)
    ])
    
    # Final pipeline with SMOTE and classifier
    pipeline = make_imb_pipeline(
        feature_union,
        SMOTE(random_state=42),
        LogisticRegression(
            C=10,
            penalty='l2',
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear',
            random_state=42
        )
    )
    
    return pipeline

def hyperparameter_tuning(pipeline, X_train, y_train):
    """Perform hyperparameter tuning using RandomizedSearchCV"""
    param_dist = {
        'text_features__vectorizer__max_features': [5000, 10000, 15000],
        'text_features__vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'logisticregression__C': [0.1, 1, 10, 100],
        'logisticregression__penalty': ['l1', 'l2'],
        'logisticregression__solver': ['liblinear', 'saga']
    }
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("\nStarting hyperparameter tuning...")
    search.fit(X_train, y_train)
    
    print("\nBest parameters found:")
    print(search.best_params_)
    print(f"Best F1 score: {search.best_score_:.3f}")
    
    return search.best_estimator_

def predict_new_messages_enhanced(pipeline, messages, threshold=0.5):
    """Predict whether new messages are spam with enhanced output"""
    results = []
    
    for msg in messages:
        # Get prediction probabilities
        proba = pipeline.predict_proba([msg])[0]
        spam_prob = proba[1]
        
        # Make prediction based on threshold
        prediction = 'SPAM' if spam_prob >= threshold else 'NOT SPAM'
        
        # Get top features if possible
        top_features = []
        try:
            if hasattr(pipeline.named_steps['logisticregression'], 'coef_'):
                vectorizer = pipeline.named_steps['text_features'].named_steps['vectorizer']
                classifier = pipeline.named_steps['logisticregression']
                
                feature_names = vectorizer.get_feature_names_out()
                coef = classifier.coef_[0]
                
                # Get top 5 spam indicators
                top_spam_indices = np.argsort(coef)[-5:][::-1]
                top_spam_features = [(feature_names[i], coef[i]) for i in top_spam_indices]
                
                # Get top 5 ham indicators
                top_ham_indices = np.argsort(coef)[:5]
                top_ham_features = [(feature_names[i], coef[i]) for i in top_ham_indices]
                
                top_features = {
                    'spam_indicators': top_spam_features,
                    'ham_indicators': top_ham_features
                }
        except Exception as e:
            pass
        
        results.append({
            'message': msg,
            'prediction': prediction,
            'spam_probability': spam_prob,
            'ham_probability': proba[0],
            'top_features': top_features
        })
    
    # Display results
    for result in results:
        print(f"\nMessage: {result['message']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Spam Confidence: {result['spam_probability']:.2%}")
        print(f"Ham Confidence: {result['ham_probability']:.2%}")
        
        if result['top_features']:
            print("\nTop Spam Indicators:")
            for feature, weight in result['top_features']['spam_indicators']:
                print(f"  - {feature}: {weight:.3f}")
            
            print("\nTop Ham Indicators:")
            for feature, weight in result['top_features']['ham_indicators']:
                print(f"  - {feature}: {weight:.3f}")
    
    return results

def save_model_artifacts(pipeline, filename='spam_classifier'):
    """Save the model and related artifacts"""
    print("\nSaving model artifacts...")
    
    # Save the entire pipeline
    joblib.dump(pipeline, f'{filename}_pipeline.pkl')
    
    # Save components separately for inspection
    try:
        vectorizer = pipeline.named_steps['text_features'].named_steps['vectorizer']
        joblib.dump(vectorizer, f'{filename}_vectorizer.pkl')
        
        classifier = pipeline.named_steps['logisticregression']
        joblib.dump(classifier, f'{filename}_classifier.pkl')
    except Exception as e:
        print(f"Could not save components separately: {str(e)}")
    
    print("Model artifacts saved successfully!")

def main():
    # Step 1: Load and prepare data
    print("Loading and preparing data...")
    data = load_and_prepare_data()
    
    # Step 2: Visualize data
    print("\nVisualizing data...")
    visualize_data_enhanced(data)
    
    # Step 3: Split data into training and test sets
    print("\nSplitting data into training and test sets...")
    X = data['message']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Step 4: Train and evaluate baseline models
    print("\nTraining and evaluating baseline models...")
    baseline_results = train_models_enhanced(X_train, X_test, y_train, y_test)
    
    # Display results comparison
    print("\n=== Model Performance Comparison ===")
    print(baseline_results[['model_name', 'test_accuracy', 'test_f1', 'time_taken']]
          .sort_values('test_f1', ascending=False))
    
    # Step 5: Build and tune final pipeline
    print("\nBuilding and tuning final pipeline...")
    final_pipeline = build_final_pipeline()
    
    # Optional: Perform hyperparameter tuning
    if False:  # Set to True to enable tuning (takes longer)
        final_pipeline = hyperparameter_tuning(final_pipeline, X_train, y_train)
    
    # Train final pipeline
    print("\nTraining final pipeline...")
    final_result = evaluate_model(final_pipeline, X_train, X_test, y_train, y_test)
    
    print("\n=== Final Model Performance ===")
    print(f"Test Accuracy: {final_result['test_accuracy']:.3f}")
    print(f"Test F1 Score: {final_result['test_f1']:.3f}")
    
    # Step 6: Test with example messages
    print("\nTesting with example messages:")
    test_messages = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
        "Hey, how are you doing today?",
        "Congratulations! You've won a $1000 Walmart gift card. Click here to claim.",
        "Can we meet tomorrow to discuss the project?",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!",
        "Hi Mom, I'll be home for dinner tonight. See you around 7pm!",
        "Your account has been compromised. Click here to secure it now!",
        "Meeting reminder: Tomorrow at 3pm in conference room B",
        "Last chance to claim your prize! Text STOP to opt out",
        "The project deadline has been extended to Friday"
    ]
    
    results = predict_new_messages_enhanced(final_pipeline, test_messages, threshold=0.4)
    
    # Step 7: Save the model
    save_model_artifacts(final_pipeline)
    
    # Final summary
    print("\n=== Final Summary ===")
    print(f"Best performing model: {final_pipeline.named_steps['logisticregression'].__class__.__name__}")
    print(f"Test Accuracy: {final_result['test_accuracy']:.2%}")
    print(f"Test F1 Score: {final_result['test_f1']:.3f}")
    print("\nModel ready for deployment!")

if __name__ == "__main__":
    main()