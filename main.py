# Importing required libraries
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from matplotlib import style
from sklearn.feature_extraction.text import CountVectorizer

style.use('ggplot')
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from nltk.corpus import stopwords

# Create a set of stopwords from NLTK's English stopwords list
stop_words = set(stopwords.words('english'))

from wordcloud import WordCloud  # For generating word clouds
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.linear_model import LogisticRegression  # Logistic regression classifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay  # Metrics to evaluate model performance

# Load the CSV file, fixing encoding issues
df = pd.read_csv('vaccination_tweets.csv', encoding='latin1')

# Initialising the stemmer
stemmer = PorterStemmer()

# Function to view and process data
def view_data():
    # Filter out unwanted columns from the dataframe
    text_df = filter_unwanted_columns()

    # Process the text data by applying data processing function
    text_df['text'] = text_df['text'].apply(data_processing)

    # Remove duplicate rows based on the 'text' column
    text_df = text_df.drop_duplicates(subset='text')

    # Apply stemming to the text column
    text_df['text'] = text_df['text'].apply(lambda x: stemming(x))

    # Add new columns for polarity and sentiment
    text_df['polarity'] = text_df['text'].apply(polarity)
    text_df['sentiment'] = text_df['polarity'].apply(sentiment)

    # Create a count plot to show the distribution of sentiments
    fig = plt.figure(figsize=(5, 5))
    sns.countplot(x="sentiment", data=text_df)
    plt.title('Count of Sentiments')

    # Display the count plot
    plt.show()

    # Create a pie chart showing sentiment distribution
    fig = plt.figure(figsize=(7, 7))
    colors = ('yellowgreen', 'gold', "red")
    wp = {'linewidth': 2, 'edgecolor': "black"}  # Proper edge colouring for pie chart
    tags = text_df['sentiment'].value_counts()
    explode = (0.1, 0.1, 0.1)  # Slice explosion for visual effect
    tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors, startangle=90, wedgeprops=wp, explode=explode,
              label='')
    plt.title('Distribution of Sentiments')

    # Display the pie chart
    plt.show()

    # Display all positive tweets
    pos_tweets = text_df[text_df.sentiment == 'Positive']
    pos_tweets = pos_tweets.sort_values(['polarity'], ascending=False)
    pos_tweets.head()

    # Generate word cloud for positive tweets
    text = ' '.join([word for word in pos_tweets['text']])
    plt.figure(figsize=(20, 15), facecolor='None')
    wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Most frequent words in positive tweets', fontsize=19)
    plt.show()

    # Display all negative tweets
    neg_tweets = text_df[text_df.sentiment == 'Negative']
    neg_tweets = neg_tweets.sort_values(['polarity'], ascending=False)
    neg_tweets.head()

    # Generate word cloud for negative tweets
    text = ' '.join([word for word in neg_tweets['text']])
    plt.figure(figsize=(20, 15), facecolor='None')
    wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Most frequent words in negative tweets', fontsize=19)
    plt.show()

    # Display all neutral tweets
    neutral_tweets = text_df[text_df.sentiment == 'Neutral']
    neutral_tweets = neutral_tweets.sort_values(['polarity'], ascending=False)
    neutral_tweets.head()

    # Generate word cloud for neutral tweets
    text = ' '.join([word for word in neutral_tweets['text']])
    plt.figure(figsize=(20, 15), facecolor='None')
    wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Most frequent words in neutral tweets', fontsize=19)
    plt.show()

    # Getting bigrams (pairs of words) from the tweets
    vect = CountVectorizer(ngram_range=(1, 2)).fit(text_df['text'])
    feature_names = vect.get_feature_names_out()

    # Prepare the data for machine learning
    X = text_df['text']
    Y = text_df['sentiment']
    X = vect.transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

    # Initialising the Logistic Regression model
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    logreg_pred = logreg.predict(x_test)
    logreg_acc = accuracy_score(logreg_pred, y_test)

    # Displaying the confusion matrix for Logistic Regression model
    style.use('classic')
    cm = confusion_matrix(y_test, logreg_pred, labels=logreg.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix= cm, display_labels=logreg.classes_)
    disp.plot()

    # Hyperparameter tuning for Logistic Regression
    from sklearn.model_selection import GridSearchCV
    param_grid = {'C': [0.001, 0.01, 0.1, 10]}
    grid = GridSearchCV(LogisticRegression(), param_grid)
    grid.fit(x_train, y_train) #trains the gridSearchCv on the training data
    print("Best parameters: ", grid.best_params_)

    # Predicting using the tuned model and calculating accuracy
    y_pred = grid.predict(x_test)
    logreg_acc = accuracy_score(y_pred, y_test)
    print("Test accuracy: {:.2f}".format(logreg_acc*100))

    # Evaluating the model's performance with confusion matrix and classification report
    print(confusion_matrix(y_test, logreg_pred))
    print("\n")
    print(classification_report(y_test, logreg_pred))

    # Support Vector Classifier (SVC) model, which may give better accuracy
    from sklearn.svm import LinearSVC
    SVCmodel = LinearSVC()
    SVCmodel.fit(x_train, y_train)

    # Predicting using the SVC model and calculating accuracy
    svc_pred = SVCmodel.predict(x_test)
    svc_acc = accuracy_score(svc_pred, y_test)
    print("Test accuracy: {:.2f}%".format(svc_acc*100))

    # Displaying the confusion matrix and classification report for the SVC model
    print(confusion_matrix(y_test, svc_pred))
    print("\n")
    print(classification_report(y_test, svc_pred))

    # Hyperparameter tuning for SVC model
    grid = {
        'C': [0.1, 0.1, 1, 10],
        'kernel' : ["linear", "poly", "rb"],
        'degree' : [1,3,5,7],
        'gamma' : [0.01, 1]
    }
    grid = GridSearchCV(SVCmodel, param_grid)
    grid.fit(x_train, y_train)
    print("Best parameter: ", grid.best_params_)
    y_pred = grid.predict(x_test)

    # Calculating and displaying the model's accuracy
    y_pred = grid.predict(x_test)
    logreg_acc = accuracy_score(y_pred, y_test)
    print("Test accuracy: {:.2f}%".format(logreg_acc*100))

    print(confusion_matrix(y_test, y_pred))
    print("\n")
    print(classification_report(y_test, y_pred))

# Function to filter unwanted columns from the dataframe
def filter_unwanted_columns():
    text_df = df.drop([
        'id', 'user_name', 'user_location', 'user_description', 'user_created',
        'user_followers', 'user_friends', 'user_favourites', 'user_verified',
        'date', 'hashtags', 'source', 'retweets', 'favorites', 'is_retweet'
    ], axis=1)
    return text_df

# Function to process the text data
def data_processing(text):
    # Convert all text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https\S+|www\S+", '', text, flags=re.MULTILINE)
    # Remove hashtags and punctuation
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenise and remove stopwords
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if w not in stop_words]

    return " ".join(filtered_text)

# Function to apply stemming to the text
def stemming(text):
    # Tokenise the text
    text_tokens = word_tokenize(text)
    # Stem each word in the text
    text = [stemmer.stem(word) for word in text_tokens]
    return " ".join(text)

# Function to calculate polarity of the text
def polarity(text):
    return TextBlob(text).sentiment.polarity

# Function to classify sentiment based on polarity
def sentiment(label):
    if label < 0:
        return "Negative"
    elif label == 0:
        return "Neutral"
    elif label > 0:
        return "Positive"

# Main execution block
if __name__ == '__main__':
    view_data()
