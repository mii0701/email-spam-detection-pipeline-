import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def run_spam_pipeline():
    # Sample Dataset: Raw Email Messages
    raw_data = {
        'text': [
            "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
            "Hey, are we still meeting for group study at UNAM library today?",
            "WINNER!! As a valued network customer you have been selected to receive £900 prize reward!",
            "Can you please review my SQL query and send feedback when free?",
            "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!",
            "Hi, please check the attached assignment schedule for this week."
        ],
        'label': [1, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Safe
    }

    # Data Preprocessing & Vectorization
    df = pd.DataFrame(raw_data)
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    # Model Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluation
    predictions = model.predict(X_test)
    print("Pipeline Execution Successful!")
    print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

if __name__ == "__main__":
    run_spam_pipeline()