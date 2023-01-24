"""
    _summary_
"""
from typing import Union
import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO, BytesIO
import base64
from wordcloud import WordCloud, STOPWORDS
import nltk
from string import punctuation
from heapq import nlargest

try:
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from collections import Counter
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
except ImportError:
    nltk.download("all")
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from collections import Counter
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import spacy
try:
    from spacy.lang.en.stop_words import STOP_WORDS
except ImportError:
    print(f"python -m spacy download en_core_web_sm")
    from spacy.lang.en.stop_words import STOP_WORDS

column_names: [str] = [
    "ID",
    "ACADYEAR",
    "SEMESTER",
    "SETID",
    "STAFFID",
    "COURSECODE",
    "DEPARTMENT",
    "FACULTY",
    "COLLEGE",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "Q5",
    "Q6",
    "Q7",
    "Q8",
    "Q9",
    "Q10",
    "Q11",
    "Q12",
    "Q13",
    "Q14",
    "Q15",
    "COMMENT",
]

filter_column_names: [str] = [
    "STAFFID",
    "COURSECODE",
    "DEPARTMENT",
    "FACULTY",
    "COLLEGE",
]

statistics_column_names: [str] = [
    "LECTURES",
    "FACULTY",
    "COURSES",
    "DEPARTMENTS",
    "COLLEGES",
    "COMMENTS"
]

response_categories: [str] = ["GOOD", "FAIR", "POOR"]
sentiments_categories: [str] = ["POSITIVE", "NEUTRAL", "NEGATIVE"]


def get_unique_values(dataframe: pd.DataFrame, column: str) -> any:
    values = dataframe[column].replace(
        r'^\s*$',
        np.nan,
        regex=True,
    ).dropna().unique()
    return values


def group_by(dataframe: any, terms=Union[str, list]) -> any:
    grouped_data = dataframe.groupby(terms)
    return grouped_data


def get_group(grouped_data: any, required_group: Union[str, tuple]) -> any:
    target = grouped_data.get_group(required_group)
    return target


# # # remove all stop words
STOPWORDS = set(stopwords.words('english'))


def remove_stopwords(txt):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(txt).split() if word not in STOPWORDS])


# # # lemnetization
lemmatizer = WordNetLemmatizer()
wordnet_map = {
    "N": wordnet.NOUN, "V": wordnet.VERB,
    "J": wordnet.ADJ, "R": wordnet.ADV
}


def lemmatize_words(txt):
    """_summary_

    Args:
        txt (_type_): _description_

    Returns:
        _type_: _description_
    """
    pos_tagged_text = nltk.pos_tag(txt.split())
    return " ".join(
        [lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])


stemmer = PorterStemmer()


def stem_words(txt):
    """_summary_

    Args:
        txt (_type_): _description_

    Returns:
        _type_: _description_
    """
    return " ".join([stemmer.stem(word) for word in txt.split()])


# # # getting the status


def classifier(polar):
    """_summary_

    Args:
        polar (_type_): _description_

    Returns:
        _type_: _description_
    """
    if polar > 0:
        return 'positive'
    elif polar == 0:
        return 'neutral'
    else:
        return 'negative'


def generate_excel_download_link(df_, name="Data"):
    """_summary_

    Args:
        df_ (_type_): _description_
        name (_type_): _description_

    Returns:
        _type_: _description_
    """
    towrite = BytesIO()
    df_.to_excel(
        towrite,
        encoding="utf-8",
        index=False,
        header=True
    )  # write to BytesIO buffer
    towrite.seek(0)  # reset pointer
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a style="color:white; text-decoration: none" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="data_download.xlsx">Download {name} Excel File ⬇</a>'
    return st.sidebar.markdown(href, unsafe_allow_html=True)


def generate_html_download_link(fig, name="plot"):
    """_summary_

    Args:
        fig (_type_): _description_
        name (_type_): _description_

    Returns:
        _type_: _description_
    """
    towrite = StringIO()
    fig.write_html(towrite, include_plotlyjs="cdn")
    towrite = BytesIO(towrite.getvalue().encode())
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a style="color:white; text-decoration: none" href="data:text/html;charset=utf-8;base64, {b64}" download="plot.html">Download {name} ⬇</a>'
    return st.sidebar.markdown(href, unsafe_allow_html=True)


def get_question_dataframe(data_frame: any, start: str = "Q1", end: str = "Q13") -> any:
    """_summary_

    Args:
        data_frame (any): _description_
        start (str, optional): _description_. Defaults to "Q1".
        end (str, optional): _description_. Defaults to "Q13".

    Returns:
        any: _description_
    """
    # select all the columns from start - end
    question = pd.DataFrame(data_frame.loc[:, start:end])
    # make sure all floating point values are now integers (i.e 1.0 = 1)
    question = question.replace(
        r'^\s*$',
        np.nan,
        regex=True,
    ).dropna().convert_dtypes()
    return question


def get_comments_dataframe(data_frame: any) -> any:
    """_summary_

    Args:
        data_frame (any): _description_

    Returns:
        any: _description_
    """
    comments_data = data_frame["COMMENT"].replace(
        r'^\s*$',
        np.nan,
        regex=True,
    ).dropna()
    return comments_data


def count_entries(dataset: any) -> any:
    """
        This function get the columns that holds the responses
        to the questions, the responses may sometime have null|None values
        as values. in such case handle it by returning True flag otherwise
        return False
        :param dataset:
        :return
    """
    count = Counter(dataset.values.ravel())
    # print(count)
    return count


def response_counts(
        count_dict: dict,
) -> any:
    """
        return a dataframe of the responses and their counts
        :param count_dict:
        :return:
    """
    good_values = 0
    poor_values = 0
    fair_values = 0
    for key, value in count_dict.items():
        if key == 1:
            good_values += value
        elif key == 2:
            fair_values += value
        elif key == 3:
            fair_values = value
        else:
            poor_values += value
    count_dict = {
        "GOOD": good_values,
        "FAIR": fair_values,
        "POOR": poor_values,
    }
    return count_dict


def create_comment_data(array: list) -> list:
    """_summary_

    Args:
        array (list): _description_

    Returns:
        list: _description_
    """
    data_set = []
    for comment in array:
        if type(comment) is not str or len(comment) <= 0 or comment == '' or comment == 'nan' or comment == 'null':
            pass
        else:
            data_set.append(str(comment))
    return data_set


def create_comment_dataframe(comments_array: any):
    """_summary_

    Args:
        comments_array (_type_): _description_

    Returns:
        _type_: _description_
    """
    student_comments_dataset = create_comment_data(comments_array)
    student_comments_dataset_for_dataframe = np.array(
        student_comments_dataset, dtype=str
    )
    student_comments_dataset_dataframe = pd.DataFrame(
        student_comments_dataset_for_dataframe, columns=["Comments"])
    return student_comments_dataset_dataframe


def get_word_cloud(txt: any):
    """_summary_

    Args:
        txt (_type_): _description_

    Returns:
        _type_: _description_
    """
    comment_words = ' '
    _stopwords = set(STOPWORDS)
    for val in txt:
        # typecast each val to string
        val = str(val)
        # split the value
        tokens = val.split()
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        for words in tokens:
            comment_words = comment_words + words + ' '

    wordcloud = WordCloud(
        width=500, height=400,
        background_color='white',
        stopwords=_stopwords,
        min_font_size=10
    ).generate(comment_words)
    return wordcloud


def get_common_word_cloud(comments_array: any):
    """_summary_

    Args:
        comments_array (_type_): _description_
    """
    if len(comments_array) < 0:
        return None
    student_comments_dataset_dataframe = create_comment_dataframe(
        comments_array
    )
    _text = student_comments_dataset_dataframe.Comments.values
    return get_word_cloud(_text)


# Function to perform sentiment analysis


def get_comment_for_analysis(comment_text: any) -> any:
    """_summary_

    Args:
        comment_text (any): _description_

    Returns:
        any: _description_
    """
    _comments = create_comment_data(comment_text)
    _comments_dataframe = create_comment_dataframe(_comments)
    _comments_dataframe["Pro-Comment"] = _comments_dataframe["Comments"].str.lower()
    _comments_dataframe["Pro-Comment"] = _comments_dataframe["Pro-Comment"].apply(
        lemmatize_words
    )
    _comments_dataframe["Pro-Comment"] = _comments_dataframe["Pro-Comment"].apply(
        remove_stopwords
    )
    _comments_dataframe["Sentiment"] = _comments_dataframe["Pro-Comment"].apply(
        lambda x: TextBlob(x).sentiment
    )
    _comments_dataframe["Polarity"] = _comments_dataframe["Pro-Comment"].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )
    _comments_dataframe["Subjectivity"] = _comments_dataframe["Pro-Comment"].apply(
        lambda x: TextBlob(x).sentiment[1]
    )
    _comments_dataframe["Subjectivity"] = _comments_dataframe["Polarity"].apply(
        classifier
    )
    return _comments_dataframe


def get_sentiment_stats(sentiment_dataframe: any) -> any:
    """_summary_

    Args:
        sentiment_dataframe (_type_): _description_

    Returns:
        _type_: _description_
    """
    _counts = count_entries(sentiment_dataframe)
    positive_value = 0
    neutral_value = 0
    negative_value = 0
    for key, value in _counts.items():
        if key == 'positive':
            positive_value = value
        elif key == 'neutral':
            neutral_value = value
        elif key == 'negative':
            negative_value = value
        else:
            neutral_value += value
    count_dict = {
        "POSITIVE": positive_value,
        "NEUTRAL": neutral_value,
        "NEGATIVE": negative_value,
    }
    return count_dict


def summarize_comment(text: str) -> str:
    stop_words = list(STOP_WORDS)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    tokens = [token.text for token in doc]
    punc = punctuation + '\n'
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stop_words:
            if word.text.lower() not in punc:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency
    sentence_tokens = [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
    select_length = int(len(sentence_tokens) * 0.3)
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    return summary
