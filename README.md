📚 Kindle Review Sentiment Analysis
📌 Overview
This project performs Sentiment Analysis on Amazon Kindle Store product reviews using Natural Language Processing (NLP) and Machine Learning. The goal is to classify reviews as positive or negative based on their text content.

📂 About the Dataset
Source: Amazon Product Data – Julian McAuley (UCSD)

Domain: Kindle Store reviews

Total Entries: 982,619

Timeframe: May 1996 – July 2014

Format: CSV

🔑 Columns:
Column	Description
asin	Product ID (e.g., B000FA64PK)
helpful	Helpfulness rating of the review (e.g., 2/3)
overall	Original rating of the product (1–5)
reviewText	Full review text
reviewTime	Time of review
reviewerID	Unique reviewer ID
reviewerName	Name of the reviewer
summary	Summary of the review
unixReviewTime	Unix timestamp

🎯 Objectives
Binary sentiment classification (positive vs. negative)

Text preprocessing and cleaning

Feature extraction using Bag-of-Words (BoW) and TF-IDF

Model training using Naive Bayes

Performance evaluation using confusion matrix and accuracy

🧰 Libraries Used
pandas, numpy

nltk

re (regular expressions)

BeautifulSoup (HTML tag removal)

scikit-learn (CountVectorizer, TfidfVectorizer, Naive Bayes, train_test_split)

⚙️ Setup Instructions
Clone this repo or download the code.

Install dependencies:


pip install pandas numpy nltk scikit-learn beautifulsoup4
Download NLTK resources:

python
Copy
Edit
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
🧼 Preprocessing Steps
Select relevant columns (reviewText, rating)

Convert ratings to binary sentiment:

Ratings < 3 → Negative (0)

Ratings ≥ 3 → Positive (1)

Clean review text:

Lowercasing

Remove special characters, URLs, and HTML tags

Remove stopwords

Lemmatization

✨ Feature Engineering
Bag of Words (BoW) using CountVectorizer

TF-IDF (Term Frequency-Inverse Document Frequency) using TfidfVectorizer

🤖 Model Training
Classifier: GaussianNB (Naive Bayes)

Input: BoW and TF-IDF vectors

Evaluation: Accuracy and Confusion Matrix

🧪 Results
Vectorizer	Accuracy
BoW	~Your Output
TF-IDF	~Your Output

Replace with your actual model output after training.

📌 Notes
This is a subset of the full Amazon dataset (5-core).

lxml parser is replaced with html.parser to avoid external dependency issues.

Consider replacing GaussianNB with MultinomialNB for better results in text classification.

🔎 Further Improvements
Use MultinomialNB, LogisticRegression, or deep learning (e.g., LSTM, BERT)

Add cross-validation and hyperparameter tuning

Use advanced embeddings (e.g., Word2Vec, GloVe)

Handle class imbalance using SMOTE or resampling

🏷 License & Credits
Dataset: © UCSD / Julian McAuley (for academic use)

License for dataset: Refer to http://jmcauley.ucsd.edu/data/amazon/
