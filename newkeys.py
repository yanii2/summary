import os
import fitz  # PyMuPDF
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import nltk
nltk.download('punkt')

# Path to the folder containing PDF files
pdf_folder = 'uploads'

# Function to extract text from PDF files
def extract_text_from_pdf(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text

# Function to preprocess text (tokenization, stop word removal, lemmatization)
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenization and convert to lowercase
    tokens = [token for token in tokens if token.isalnum()]  # Remove non-alphanumeric tokens
    tokens = [token for token in tokens if token not in stopwords.words('english')]  # Remove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatization
    return tokens

# Function to generate keywords from a list of tokens
def generate_keywords(tokens, top_n=5):
    counter = Counter(tokens)
    return [keyword for keyword, _ in counter.most_common(top_n)]

# Iterate over PDF files in the folder
keywords_per_file = {}
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, filename)
        text = extract_text_from_pdf(file_path)
        tokens = preprocess_text(text)
        keywords = generate_keywords(tokens)
        keywords_per_file[filename] = keywords

# Print keywords for each PDF file
for filename, keywords in keywords_per_file.items():
    print(f"Keywords for {filename}: {keywords}")