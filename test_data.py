import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import fitz
import re
import os
import sys
import io
from nltk import WhitespaceTokenizer
import spacy
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline,BartTokenizer, BartForConditionalGeneration
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
from collections import Counter
from nltk.util import bigrams
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity


try:
    nlp_spacy = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp_spacy = spacy.load("en_core_web_sm")

# """### Data collection and Preparation"""

def extraction_from_pdf(path):
  doc=fitz.open(path)
  content= ""
  for page in doc:
    blocks= page.get_text("blocks")
    for block in blocks:
      if block[6]==0:
        content += block[4] +"\n"
    # content.strip()
  content=content.strip()
  return content


def text_preprocessing(content):
  cleaned_content=content.lower()
  text = re.sub(r"[^\w\s'-]|(?<!\w)'|'(?!\w)|(?<!\w)-|-(?!\w)", "", cleaned_content) # this removes the special characters  and retain Hyphenated terms and alphnumeric
  text = re.sub(r"Fig(?:ure)?\.?\s?\d+[A-Za-z]?", "", text, flags=re.IGNORECASE) # removes the figure references (e.g "FIG8")
  text = re.sub(r"US\s+Patent.*?\d{4}", "", text, flags=re.IGNORECASE) # Removes the Us Patent Headers and date related text
  text = re.sub(r"sheet\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE) #remove the sheet number reference
  text = re.sub(r"^\s*[a-zA-Z]{1,2}\s*$", "", text, flags=re.MULTILINE) # remove the stand alone letters
  text = re.sub(r"[_\-\s]{4,}", " ", text) # normalizes the undersores and excessive white spaces
  text=re.sub(r"\u200b","", text) #remove the izero-width soaces
  text = re.sub(r"\b[a-zA-Z]{1,2}\d+[a-zA-Z]*\b", "", text)  # Removes OCR-misread words like "i1fox"
  text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
  return text


# """STOP--WORDS REMOVAL"""

# RETAINING DOMAIN SPECIFIC WORDS
def domain_specific_words(text):
  wt=WhitespaceTokenizer()
  tokenized_text = wt.tokenize(text)
  doc= nlp_spacy(" ".join(tokenized_text))
  doc_spacy=nlp_spacy(doc)
  spacy_ents={ent.text for ent in doc_spacy.ents}
  model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
  tokenizer1 = AutoTokenizer.from_pretrained(model_name, cache_dir="./models")
  model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir="./models")
  ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer1)
  hr_results=ner_pipeline(text)
  hr_entities={entity['word'] for entity in hr_results}
  combined_entities=spacy_ents.union(hr_entities)
  stop_words = set(stopwords.words('english'))
  filter_tokens=[word for word in tokenized_text if word not in stop_words or word in combined_entities]
  return filter_tokens

# COUNTING OF TOKENIZED_WORDS
def word_count(filter_tokens):
  word_freq=Counter(filter_tokens)
  most_common_words=word_freq.most_common(10)
  return most_common_words , word_freq
# print(most_common_words)

# """EDA"""

# Perfoming a word count EDA
def word_cloud(filter_tokens):
  text=" ".join(filter_tokens)
  wordcloud=WordCloud(width=800,height=400,background_color="white").generate(text)
  return wordcloud

def word_freq_plot(most_common_words):
  data=pd.DataFrame(most_common_words,columns=["Word","Frequency"])
  fig, ax=plt.subplots(figsize=(8,4))
  sns.barplot(data=data,x="Word",y="Frequency",ax=ax)
  ax.set_title("Most_frequent_words")
  plt.xticks(rotation=45)
  plt.tight_layout()
  return fig

# WORD CO-OCCURENCES
def word_cooccurance(filter_tokens):
  bigram_list =list(bigrams(filter_tokens))
  bigram_count=Counter(bigram_list)
  return bigram_count

def bigram_freq_plot(bigram_count):
  data2=pd.DataFrame(bigram_count.most_common(10),columns=["Bigram","Frequency"])
  data2["Bigram"]=data2["Bigram"].apply(lambda x: " ".join(x))
  fig,ax=plt.subplots(figsize=(8,4))
  sns.barplot(data=data2,x="Bigram",y="Frequency",ax=ax)
  ax.set_title("Most_frequent_bigrams")
  plt.xticks(rotation=45)
  plt.tight_layout()
  return fig






# """### SIMILARITY ANALYSIS"""

def similarity_analysis(filter_tokens,filter_tokens2):
  doc1=" ".join(filter_tokens)
  doc2=" ".join(filter_tokens2)
  vectorizer=TfidfVectorizer()
  tfidf_matrix=vectorizer.fit_transform([doc1,doc2])
  words_similarity = (round(np.array(cosine_similarity(tfidf_matrix[0],tfidf_matrix[1])).item(),5)*100)
  return words_similarity

# Jaccard similarity
def jaccard_similarity(tokens1,tokens2):
  set1=set(tokens1)
  set2=set(tokens2)
  intersection=len(set1.intersection(set2))
  union=len(set1.union(set2))
  shared_words_similarity=f"{(intersection/union if union!= 0 else 0 )*100:.4f}%"
  return shared_words_similarity


# !pip install sentence-transformers


def bert_similarity(filter_tokens,filter_tokens2):
  model=SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")
  doc1=" ".join(filter_tokens)
  doc2=" ".join(filter_tokens2)
  embedding1=model.encode(doc1,convert_to_tensor=True)
  embedding2=model.encode(doc2,convert_to_tensor=True)
  context_similarity=f"{util.cos_sim(embedding1,embedding2).detach().item()*100:.3f}%"
  return context_similarity


def chunk_text(text, max_tokens=512):
    """Splits a document into smaller chunks based on token length."""
    tokenizer = AutoTokenizer.from_pretrained("t5-small",cache_dir="./models")
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]  # Tokenize entire text
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]  # Split into chunks

    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]  # Convert back to text

def summarize_large_text(text):
    """Summarizes large text by summarizing chunks, then re-summarizing the combined summary."""
    chunks = chunk_text(text)
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir="./models")
    model = BartForConditionalGeneration.from_pretrained(model_name, cache_dir="./models")
    summarizer = pipeline("summarization", model=model,tokenizer=tokenizer)

    # Summarize each chunk
    chunk_summaries = [summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]['summary_text'] for chunk in chunks]

    # Combine summaries
    combined_summary = " ".join(chunk_summaries)

    # Final summary
    final_summary = summarizer(combined_summary, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

    return final_summary


st.title("""PATENT SIMILARITY ANALYSER""")
uploaded_files=st.file_uploader("Upload your pdf file *here*", type=["pdf"], accept_multiple_files=True)
pdf_texts_cleaned={}
pdf_tokens_filtered={}

if uploaded_files:
  for uploaded_file in uploaded_files:
    temp_path=f"temp_{uploaded_file.name}"

    with open(temp_path,"wb") as f:
      f.write(uploaded_file.getbuffer())

    raw_text=extraction_from_pdf(temp_path)
    preprocessed_text=text_preprocessing(raw_text)
    filter_tokens=domain_specific_words(preprocessed_text)

    pdf_texts_cleaned[uploaded_file.name] = preprocessed_text
    pdf_tokens_filtered[uploaded_file.name]= filter_tokens

    os.remove(temp_path)

  pdf_names=list(pdf_texts_cleaned.keys())
  selected_pdf1=st.selectbox("Select first PDF to compare",pdf_names)
  selected_pdf2=st.selectbox("Select second PDF to compare",pdf_names,index=1 if len(pdf_names)>1 else 0)

  if selected_pdf1 and selected_pdf2:
    text1=pdf_texts_cleaned[selected_pdf1]
    text2=pdf_texts_cleaned[selected_pdf2]

    st.write(f"**Comparison between:** '{selected_pdf1}' **and** '{selected_pdf2}'")

    st.write("### Preview Extracted Texts")
    st.text_area(f"Text from {selected_pdf1}", text1, height=200)
    st.text_area(f"Text from {selected_pdf2}", text2, height=200)

  st.header("Exploratory Data Analysis (EDA) per document")

  for filename,tokens in pdf_tokens_filtered.items():
    st.subheader(f"EDA for: {filename}")

    most_common_words, word_freq = word_count(tokens)

    with st.expander("Top 10 Most Common Words"):
      st.dataframe(pd.DataFrame(most_common_words, columns=["Word", "Frequency"]))

    with st.expander("Word Cloud"):
      fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
      text = " ".join(tokens)
      wordcloud_img = WordCloud(width=800, height=400, background_color="white").generate(text)
      ax_wc.imshow(wordcloud_img, interpolation="bilinear")
      ax_wc.axis("off")
      st.pyplot(fig_wc)

    with st.expander("Word Frequency Plot"):
      fig_freq, ax_freq = plt.subplots(figsize=(8, 4))
      data = pd.DataFrame(most_common_words, columns=["Word", "Frequency"])
      sns.barplot(data=data, x="Word", y="Frequency", ax=ax_freq)
      ax_freq.set_title("Most Frequent Words")
      plt.xticks(rotation=45)
      st.pyplot(fig_freq)

  st.write("""# Similarity analysis""")
  with st.expander("Bigram Analysis"):
    bigram_count = word_cooccurance(tokens)
    st.dataframe(pd.DataFrame(bigram_count.most_common(10), columns=["Bigram", "Frequency"]))

    fig_bigram, ax_bigram = plt.subplots(figsize=(8, 4))
    data2 = pd.DataFrame(bigram_count.most_common(10), columns=["Bigram", "Frequency"])
    data2["Bigram"] = data2["Bigram"].apply(lambda x: " ".join(x))
    sns.barplot(data=data2, x="Bigram", y="Frequency", ax=ax_bigram)
    ax_bigram.set_title("Most Frequent Bigrams")
    plt.xticks(rotation=45)
    st.pyplot(fig_bigram)

  # st.markdown("---")

  # st.write("# Similarity Analysis")

  if selected_pdf1 and selected_pdf2:

    tokens1=pdf_tokens_filtered[selected_pdf1]
    tokens2=pdf_tokens_filtered[selected_pdf2]

    # Calculate all similarities
    tfidf_score = similarity_analysis(tokens1, tokens2)
    jaccard_score = jaccard_similarity(tokens1, tokens2)
    bert_score = bert_similarity(tokens1, tokens2)


    # Display in columns
    col1, col2, col3 = st.columns(3)

    with col1:
      st.metric(label="Word Similarity", value=f"{tfidf_score:.2f}%")

    with col2:
      st.metric(label="Shared Word Similarity", value=jaccard_score)

    with col3:
      st.metric(label="Contextual Similarity", value=bert_score)


    st.subheader("Summaries")
    summary1=summarize_large_text( text1 )
    summary2=summarize_large_text( text2)

    st.header("Summarized Texts")

    st.write(f"**Summary of {selected_pdf1}**")
    st.text_area(f"Summary of {selected_pdf1}",summary1,height=200,key="summary_1")

    st.write(f"**Summary of {selected_pdf2}**")
    st.text_area(f"Summary of {selected_pdf2}", summary2, height=200, key="summary_2")

    summary1_bytes=summary1.encode("utf-8")
    summary2_bytes=summary2.encode("utf-8")

    download_file_1=io.StringIO(summary1)
    download_file_2=io.StringIO(summary2)

    st.download_button(
      label="Download Summary 1",
      data=summary1_bytes,
      file_name=f"{selected_pdf1}_summary.txt",
      mime="text/plain"
    )

    st.download_button(
      label="Download Summary 2",
      data=summary2_bytes,
      file_name=f"{selected_pdf2}_summary.txt",
      mime="text/plain"
    )
