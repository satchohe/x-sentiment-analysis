Vaccination Tweet Sentiment Analysis
====================================

This project performs sentiment analysis on a dataset of tweets related to vaccinations. It processes the text data, visualises the sentiment distribution, and employs machine learning models to classify tweets as positive, negative, or neutral.

Features
--------

*   **Data Preprocessing:** Cleans and processes tweet text by converting it to lowercase, removing URLs, hashtags, punctuation, and common **stopwords**.
    
*   **Stemming:** Applies the **Porter Stemmer** to reduce words to their root form, aiding in text normalisation.
    
*   **Sentiment Analysis:** Utilises the TextBlob library to calculate the **polarity** of each tweet, subsequently classifying it as **Positive**, **Negative**, or **Neutral**.
    
*   **Data Visualisation:** Generates several visualisations to represent the data and insights:
    
    *   **Count Plot and Pie Chart:** Displays the overall distribution of sentiments within the dataset.
        
    *   **Word Clouds:** Shows the most frequent words in positive, negative, and neutral tweets, offering a quick visual summary of common themes.
        
*   **Machine Learning Models:** Trains and evaluates two popular classification models for sentiment prediction:
    
    *   **Logistic Regression:** A robust baseline model for sentiment classification.
        
    *   **Support Vector Classifier (SVC):** A more advanced model often providing superior accuracy in text classification tasks.
        
*   **Model Evaluation:** Provides comprehensive metrics including **accuracy scores**, **classification reports**, and **confusion matrices** for both models to thoroughly assess their performance.
    
*   **Hyperparameter Tuning:** Employs GridSearchCV to systematically find the optimal parameters for both the Logistic Regression and SVC models, thereby enhancing their predictive accuracy.
    

Prerequisites
-------------

To run this project successfully, you'll need the following Python libraries. You can install them using pip:

`   pip install pandas matplotlib seaborn scikit-learn nltk textblob wordcloud   `

Additionally, you'll need to download the NLTK stopwords and punkt tokenizer. You can do this by running the following commands in a Python interpreter or script:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   import nltk  nltk.download('stopwords')  nltk.download('punkt')   `

Project Structure
-----------------

*   vaccination\_tweets.csv: This is the primary dataset containing the tweets used for sentiment analysis.
    
*   your\_script\_name.py: This Python script contains all the code for data processing, sentiment analysis, visualisation, and machine learning model training.
    

Usage
-----

1.  Ensure both the vaccination\_tweets.csv file and the Python script (your\_script\_name.py) are located in the same directory.
    
2.  python your\_script\_name.py
    
3.  Upon execution, the script will automatically carry out all the defined processing steps, display the various plots (sentiment distribution, word clouds), and output the machine learning model performance metrics directly in your terminal.
    

Sample Output
-------------

After running the script, you will observe a series of visualisations and terminal outputs.

### Sentiment Distribution

### Word Clouds

### Model Evaluation

The terminal output will display the accuracy scores, classification reports, and confusion matrices for both the Logistic Regression and SVC models, providing a detailed breakdown of their performance.
