from pdfminer.high_level import extract_text
import re
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import spacy
from spacy_download import load_spacy


def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)


def preprocess(text):
    '''
    To do all basic preprocessing of text
    :param text:
    :return text:
    '''

    # converting into lowercase
    text = text.lower()

    # Removing punctuations
    text = re.sub(r'[^\w\s]', '', text)

    # remove urls
    pattern = re.compile('https?://S+|www.S+')
    text = pattern.sub(r'', text)

    # Remove html tags
    pattern = re.compile('[<,*?>]')
    text = pattern.sub(r'', text)

    # spelling correction
    text = str(TextBlob(text))

    # Tokenisation
    nlp = load_spacy("en_core_web_sm", exclude=["parser", "tagger"])
    tokens = nlp(text)
    text = []
    for i in tokens:
        text.append(i.text)

    # remove unwated
    temp = []
    for i in text:
        if " " not in i and "\n" not in i:
            temp.append(i)
    text = temp

    # removing stop words
    stopwords = nltk.corpus.stopwords.words('english')
    text = [i for i in text if i not in stopwords]

    # Lemmetization
    # defining the object for Lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]

    return lemm_text


def score(path):
    '''
    To calculate score of each attributes
    :param path of resume:
    :return dictionary:
    '''
    # Keywords for each category
    decision_making = ["decision", "analytical", "leader", "research", "mediation", "critical", "think", "problem",
                       "solve", "assessed"]
    communication = ["outline", "edit", "read", "comprehension", "time", "listen", "vocabulary", "patience", "goal",
                     "verbal"]
    delegation = ["conference", "foreign", "help", "forum", "right", "climate", "community", "communicate", "assembly",
                  "groups"]
    team_work = ["morale", "inspire", "empathy", "helps", "support", "recognize", "interact", "present", "translate",
                 "influence"]
    adaptability = ["adopt", "replace", "prioritize", "delegate", "guidance", "provided", "support", "evaluate",
                    "recommend", "pbserve"]
    problem_solving = ["logical", "thought", "efficient", "patience", "analytical", "plan", "troubleshoot", "critical",
                       "think", "problem", "solve"]
    trust = ["time", "manage", "trust", "relaible", "plan", "emotion", "social", "team", "responsible", "organize",
             "solve", "charge", "help"]
    tech_savy = ["technology", "word", "program", "python", "java", "c", "html", "geek", "test", "sql", "software",
                 "excel", "ux"]

    attributes = ["Decision Making", "Communication", "Delegation", "Team Work", "Adaptability", "Problem Solving",
                  "Trustworthiness", "Tech Saviness"]

    # Reading the pdf file
    text = extract_text_from_pdf(path)

    # preprocessing the text
    preprocessed_text = preprocess(text)

    # Score for each category
    score_decision_making = 0
    score_communication = 0
    score_delegation = 0
    score_team_work = 0
    score_adaptability = 0
    score_problem_solving = 0
    score_trust = 0
    score_tech_savy = 0

    # Checking for keywords

    for i in range(len(decision_making)):
        if decision_making[i] in preprocessed_text:
            score_decision_making += 1
        if communication[i] in preprocessed_text:
            score_communication += 1
        if delegation[i] in preprocessed_text:
            score_delegation += 1
        if team_work[i] in preprocessed_text:
            score_team_work += 1
        if adaptability[i] in preprocessed_text:
            score_adaptability += 1
        if problem_solving[i] in preprocessed_text:
            score_problem_solving += 1
        if trust[i] in preprocessed_text:
            score_trust += 1
        if tech_savy[i] in preprocessed_text:
            score_tech_savy += 1

    scores = [score_decision_making, score_communication, score_delegation, score_team_work, score_adaptability,
              score_problem_solving, score_trust, score_tech_savy]
    for i in range(len(scores)):
        scores[i] = round(scores[i] * 100 / len(decision_making), 2)

    args = dict(zip(attributes, scores))
    #print(preprocessed_text)
    return args
