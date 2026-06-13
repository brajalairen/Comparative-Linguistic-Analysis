!pip install textstat nltk openpyxl
!pip install NRCLex==3.0.0

import json
import pandas as pd
import textstat
import nltk

nltk.download('punkt_tab')
nltk.download('vader_lexicon')

from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
from collections import Counter
from nrclex import NRCLex

print("Enter ""1"" for human sample analysis")
print("Enter ""2"" for machine generated sample analysis")
choice=int(input("Enter choice: "))

def extract_dataset():
  if choice==1:
    originals = []
    files=input("Enter the JSON filenames separated by commas: ")
    filenames=[f.strip() for f in files.split(",")]
    for filename in filenames:
      with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
      originals.extend(data["original"])
    df = pd.DataFrame({"text": originals})
    return df, "Human"
  elif choice==2:
    filename = input("Enter the JSON filename: ")
    with open(filename, "r", encoding="utf-8") as f: data = json.load(f)
    df = pd.DataFrame({"text": data["sampled"]})
    return df, filename.replace(".json","")
  else:
    print("Invalid choice")
    return None

def char_count(txt_column): return len(str(txt_column))

def word_count(txt_column): return len(str(txt_column).split())

def sentence_count(txt_column): return len(sent_tokenize(str(txt_column)))

def paragraph_count(txt_column):
  paragraphs = str(txt_column).split('\n\n')
  paragraphs = [ p for p in paragraphs if p.strip() ]
  return len(paragraphs)

def average_word_length(txt_column):
  words = str(txt_column).split()
  if len(words) == 0:
    return 0
  total_chars = sum(len(word) for word in words)
  return total_chars / len(words)

def average_sentence_length(txt_column):
  sentences = sentence_count(txt_column)
  if sentences == 0:
    return 0
  return word_count(txt_column) / sentences

def lexical_diversity(txt_column):
  words = str(txt_column).lower().split()
  if len(words) == 0:
    return 0
  return len(set(words)) / len(words)

def hapax_legomena(txt_column):
  words = str(txt_column).lower().split()
  counts = Counter(words)
  return sum(1 for freq in counts.values() if freq == 1 )

def hapax_ratio(txt_column):
  words = str(txt_column).lower().split()
  if len(words) == 0:
    return 0
  counts = Counter(words)
  hapax = sum( 1 for freq in counts.values() if freq == 1 )
  return hapax / len(words)

def sentiment_positive(txt_column):
    return analyzer.polarity_scores(str(txt_column))["pos"]

def sentiment_negative(txt_column):
    return analyzer.polarity_scores(str(txt_column))["neg"]

def sentiment_neutral(txt_column):
    return analyzer.polarity_scores(str(txt_column))["neu"]

def sentiment_compound(txt_column):
    return analyzer.polarity_scores(str(txt_column))["compound"]

def get_emotion_score(txt_column, emotion):
    try:
        emotions = NRCLex(str(txt_column)).affect_frequencies
        return emotions.get(emotion, 0)
    except:
        return 0

df, name = extract_dataset()
txt_column="text"
summary = {
    "Word Count": df[txt_column].apply(word_count).mean(),
    "Character Count": df[txt_column].apply(char_count).mean(),
    "Sentence Count": df[txt_column].apply(sentence_count).mean(),
    "Paragraph Count": df[txt_column].apply(paragraph_count).mean(),
    "Average Word Length": df[txt_column].apply(average_word_length).mean(),
    "Average Sentence Length": df[txt_column].apply(average_sentence_length).mean(),
    "Lexical Diversity": df[txt_column].apply(lexical_diversity).mean(),
    "Hapax Legomena": df[txt_column].apply(hapax_legomena).mean(),
    "Hapax Ratio": df[txt_column].apply(hapax_ratio).mean(),
    "Flesch Reading Ease": df[txt_column].apply(textstat.flesch_reading_ease).mean(),
    "Flesch-Kincaid Grade": df[txt_column].apply(textstat.flesch_kincaid_grade).mean(),
    "Gunning Fog": df[txt_column].apply(textstat.gunning_fog).mean(),
    "SMOG Index": df[txt_column].apply(textstat.smog_index).mean(),
    "Sentiment Positive": df["text"].apply(sentiment_positive).mean(),
    "Sentiment Negative": df["text"].apply(sentiment_negative).mean(),
    "Sentiment Neutral": df["text"].apply(sentiment_neutral).mean(),
    "Sentiment Compound": df["text"].apply(sentiment_compound).mean(),
    "NRCLex Anger": df["text"].apply(lambda x: get_emotion_score(x, "anger")).mean(),
    "NRCLex Fear": df["text"].apply(lambda x: get_emotion_score(x, "fear")).mean(),
    "NRCLex Anticipation": df["text"].apply(lambda x: get_emotion_score(x, "anticipation")).mean(),
    "NRCLex Trust": df["text"].apply(lambda x: get_emotion_score(x, "trust")).mean(),
    "NRCLex Surprise": df["text"].apply(lambda x: get_emotion_score(x, "surprise")).mean(),
    "NRCLex Sadness": df["text"].apply(lambda x: get_emotion_score(x, "sadness")).mean(),
    "NRCLex Joy": df["text"].apply(lambda x: get_emotion_score(x, "joy")).mean(),
    "NRCLex Disgust": df["text"].apply(lambda x: get_emotion_score(x, "disgust")).mean(),
    "NRCLex Positive": df["text"].apply(lambda x: get_emotion_score(x, "positive")).mean(),
    "NRCLex Negative": df["text"].apply(lambda x: get_emotion_score(x, "negative")).mean()
}

result=pd.DataFrame(list(summary.items()), columns=["Linguistic Features(avg)", "Values"])
output_file = f"{name}_OutputTask1.xlsx"
result.to_excel(output_file,index=False)
print(result)
print(f"Saved successfully as {output_file}")
