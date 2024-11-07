import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import time
import re
import pandas as pd
import skfuzzy.control as ctrl
import skfuzzy as fuzz

nltk.download('vader_lexicon')

SAVE_AUX_DATASET = False
# Ruta del archivo CSV
DATASET_PATH = "data/test_data.csv"
# Ruta del archivo CSV de resultados
RESULT_PATH = "data/result.csv"

global data

def preprocess_text(tweet):
    tweet = re.sub(r'http\S+|www.\S+', '', tweet)
    tweet = re.sub(r'@', '', tweet)
    contractions = {
        "idk": "I do not know",
        "imo": "in my opinion",
        "imho": "in my humble opinion",
        "fyi": "for your information",
        "omg": "oh my god",
        "lol": "laughing out loud",
        "btw": "by the way",
        "brb": "be right back",
        "lmao": "laughing my ass off",
        "nvm": "never mind",
        "tbh": "to be honest",
        "smh": "shaking my head",
        "dm": "direct message",
        "afaik": "as far as I know",
        "ikr": "I know right",
        "wtf": "what the fuck",
        "wysiwyg": "what you see is what you get",
        "texn": "technology",
        "lt": "less than",
        "rds": "relational database system",
        "hmu": "hit me up",
        "bff": "best friends forever",
        "ftw": "for the win",
        "irl": "in real life",
        "jk": "just kidding",
        "np": "no problem",
        "rofl": "rolling on the floor laughing",
        "tba": "to be announced",
        "tbd": "to be determined",
        "afk": "away from keyboard",
        "bbl": "be back later",
        "bfn": "bye for now",
        "omw": "on my way",
        "thx": "thanks",
        "ttyl": "talk to you later",
        "gg": "good game",
        "g2g": "got to go",
        "atm": "at the moment",
        "gr8": "great",
        "b4": "before",
        "ur": "your",
        "u": "you",
        "cya": "see you",
        "txt": "text",
        "plz": "please",
        "cu": "see you",
        "bday": "birthday",
        "can't": "can not",
        "cannot": "can not",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'t": " not",
        "'ve": " have",
        "'m": " am",
        "it's": "it is",
        "i'm": "i am",
        "you're": "you are",
        "they're": "they are",
        "we're": "we are",
        "let's": "let us",
        "that's": "that is",
        "who's": "who is",
        "what's": "what is",
        "here's": "here is",
        "there's": "there is",
        "where's": "where is",
        "how's": "how is",
        "cant": "can not",
        "wont": "will not",
        "dont": "do not",
        "doesnt": "does not",
        "didnt": "did not",
        "isnt": "is not",
        "arent": "are not",
        "wasnt": "was not",
        "werent": "were not",
        "havent": "have not",
        "hasnt": "has not",
        "hadnt": "had not",
        "youre": "you are",
        "theyre": "they are",
        "were": "we are",
        "lets": "let us",
        "thats": "that is",
        "whos": "who is",
        "whats": "what is",
        "heres": "here is",
        "theres": "there is",
        "wheres": "where is",
        "hows": "how is",
        "im": "i am",
        "can t": "can not",
        "won t": "will not",
        " n t": " not",
        " re": " are",
        " s": " is",
        " d": " would",
        " ll": " will",
        " t": " not",
        " ve": " have",
        " m": " am",
        "it s": "it is",
        "i m": "i am",
        "you re": "you are",
        "they re": "they are",
        "we re": "we are",
        "let s": "let us",
        "that s": "that is",
        "who s": "who is",
        "what s": "what is",
        "here s": "here is",
        "there s": "there is",
        "where s": "where is",
        "how s": "how is",
    }

    def replace_contractions(text):
        contractions_pattern = re.compile('({})'.format('|'.join(contractions.keys())), flags=re.IGNORECASE | re.DOTALL)
        def expand_match(contraction):
            match = contraction.group(0)
            expanded_contraction = contractions.get(match.lower())
            return expanded_contraction
        return contractions_pattern.sub(expand_match, text)

    tweet = replace_contractions(tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'(.)\1{2,}', r'\1\1', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet


def preprocess():
    data['original_sentence'] = data['sentence'].copy()

    def process_row(row):
        start_time = time.perf_counter()
        row['sentence'] = preprocess_text(row['sentence'])
        end_time = time.perf_counter()
        row['preprocess_time'] = end_time - start_time
        return row

    data[['sentence', 'preprocess_time']] = data.apply(process_row, axis=1)[['sentence', 'preprocess_time']]


def sentiments():
    sia = SentimentIntensityAnalyzer()

    def compute_sentiment(row):
        start_time = time.perf_counter()
        scores = sia.polarity_scores(row['sentence'])
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        row['positive_score'] = scores['pos']
        row['negative_score'] = scores['neg']
        row['sentiment_time'] = elapsed_time
        return row

    aux = data.apply(compute_sentiment, axis=1)
    data['positive_score'] = aux['positive_score']
    data['negative_score'] = aux['negative_score']
    data['sentiment_time'] = aux['sentiment_time']


def triangular_membership(inference_result):
    if inference_result < 3.3:
        return 'NEGATIVE'
    elif 3.3 <= inference_result < 6.7:
        return 'NEUTRAL'
    elif 6.7 <= inference_result:
        return 'POSITIVE'

def fuzzy():
    min_positive = data['positive_score'].min()  # Mínimo de puntaje positivo
    max_positive = data['positive_score'].max()  # Máximo de puntaje positivo
    min_negative = data['negative_score'].min()  # Mínimo de puntaje negativo
    max_negative = data['negative_score'].max()  # Máximo de puntaje negativo

    # Calcular el valor medio (mid) para positivo y negativo
    mid_positive = (min_positive + max_positive) / 2
    mid_negative = (min_negative + max_negative) / 2

    # Generar variables universales de entrada y salida
    positive = ctrl.Antecedent(np.arange(min_positive, max_positive + 0.1, 0.1), 'positive')
    negative = ctrl.Antecedent(np.arange(min_negative, max_negative + 0.1, 0.1), 'negative')
    opt = ctrl.Consequent(np.arange(0, 10.1, 0.1), 'opt')

    # Generar funciones de membresía para positive
    positive['low'] = fuzz.trimf(positive.universe, [min_positive, min_positive, mid_positive])
    positive['medium'] = fuzz.trimf(positive.universe, [min_positive, mid_positive, max_positive])
    positive['high'] = fuzz.trimf(positive.universe, [mid_positive, max_positive, max_positive])

    # Generar funciones de membresía para negative
    negative['low'] = fuzz.trimf(negative.universe, [min_negative, min_negative, mid_negative])
    negative['medium'] = fuzz.trimf(negative.universe, [min_negative, mid_negative, max_negative])
    negative['high'] = fuzz.trimf(negative.universe, [mid_negative, max_negative, max_negative])

    # Generar funciones de membresía para opt
    opt['negative'] = fuzz.trimf(opt.universe, [0, 0, 5])
    opt['neutral'] = fuzz.trimf(opt.universe, [0, 5, 10])
    opt['positive'] = fuzz.trimf(opt.universe, [5, 10, 10])

    # Reglas
    rules = [
        ctrl.Rule(positive['low'] & negative['low'], opt['neutral']),
        ctrl.Rule(positive['medium'] & negative['low'], opt['positive']),
        ctrl.Rule(positive['high'] & negative['low'], opt['positive']),
        ctrl.Rule(positive['low'] & negative['medium'], opt['negative']),
        ctrl.Rule(positive['medium'] & negative['medium'], opt['neutral']),
        ctrl.Rule(positive['high'] & negative['medium'], opt['positive']),
        ctrl.Rule(positive['low'] & negative['high'], opt['negative']),
        ctrl.Rule(positive['medium'] & negative['high'], opt['negative']),
        ctrl.Rule(positive['high'] & negative['high'], opt['neutral'])
    ]

    sentiment_ctrl = ctrl.ControlSystem(rules)
    sentiment = ctrl.ControlSystemSimulation(sentiment_ctrl)

    sentiment_output = []
    times = []

    for index, row in data.iterrows():
        start_time = time.perf_counter()
        sentiment.input['positive'] = row['positive_score']
        sentiment.input['negative'] = row['negative_score']
        sentiment.compute()
        end_time = time.perf_counter()

        sentiment_output.append(sentiment.output['opt'])
        times.append(end_time - start_time)

    data['inference_time'] = times
    data['inference_result'] = sentiment_output
    data['inference_class'] = data['inference_result'].apply(triangular_membership)

def seconds_to_milliseconds(seconds):
    return str(round(seconds * 1000, 4)) + "ms"

def generate_result():
    result = pd.DataFrame()
    result['Oracion Original'] = data['original_sentence']
    result['Label Original'] = data['sentiment']
    result['Puntaje Positivo'] = data['positive_score']
    result['Puntaje Negativo'] = data['negative_score']
    result['Resultado de Inferencia'] = data['inference_class']
    # result['Puntaje del Sentimiento'] = data['inference_class']
    result['Tiempo de Inferencia'] = data['inference_time'].apply(seconds_to_milliseconds)
    result.to_csv(RESULT_PATH, index=False)
    print(f"\nResultados guardados en {RESULT_PATH}")

def benchmarks():
    data['time'] = data['preprocess_time'] + data['sentiment_time'] + data['inference_time']

    time_avg = data['time'].mean()
    sentiment_time_avg = data['inference_time'].mean()

    positive_score_count = (data['inference_class'] == 'POSITIVE').sum()
    negative_score_count = (data['inference_class'] == 'NEGATIVE').sum()
    neutral_score_count = (data['inference_class'] == 'NEUTRAL').sum()

    print(f"\nTiempo promedio de ejecución: {seconds_to_milliseconds(time_avg)}")
    print(f"Tiempo promedio de inferencia: {seconds_to_milliseconds(sentiment_time_avg)}")

    print(f"\nTotal de tweets positivos: {positive_score_count}")
    print(f"Total de tweets negativos: {negative_score_count}")
    print(f"Total de tweets neutrales: {neutral_score_count}")

if __name__ == "__main__":
    try:
        data = pd.read_csv(DATASET_PATH)
    except Exception as e:
        print(f'No se pudo abrir el archivo en: {DATASET_PATH}\n Error: {e}')
        exit(2)
    preprocess()
    sentiments()
    fuzzy()
    generate_result()
    benchmarks()
    if SAVE_AUX_DATASET:
        data.to_csv('data/aux.csv', index=False)
