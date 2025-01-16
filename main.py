import csv
import nltk
import math
import spacy
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
from sri_encodings import gap_encode, compress_gamma_binary, decompress_gamma_binary, gap_decode
from nltk.stem import PorterStemmer
from nltk import pos_tag


def term_frequency_natural(term, doc_id, processed_documents):
    terms = processed_documents[doc_id]
    counter = Counter(terms)
    return counter[term] if term in counter else 0


def term_frequency_log(term, doc_id, processed_documents):
    count = term_frequency_natural(term, doc_id, processed_documents)
    return 1 + math.log(count) if count > 0 else 0


def term_frequency_augmented(term, doc_id, processed_documents):
    terms = processed_documents[doc_id]
    count = term_frequency_natural(term, doc_id, processed_documents)
    return 0.5 + 0.5 * count / max(term_frequency_natural(t, doc_id, processed_documents) for t in terms)


def idf_normal(_documents, _documents_count):
    return 1


def idf_log(documents, documents_count):
    return math.log(len(documents) / documents_count) if documents_count > 0 else 0


def preprocess_document_racinisation(doc: str) -> list[str]:
    # Initialize the Porter Stemmer
    stemmer = PorterStemmer()
    
    # Convert the document to lowercase
    doc = doc.lower()
    
    # Tokenize the document into words
    tokens = word_tokenize(doc)
    
    # Perform stemming (racination)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return stemmed_tokens


def preprocess_document_lemmatization(doc: str):
    tokens = word_tokenize(doc.lower())
    stop_words = set(stopwords.words("english"))
    tokens = [
        word
        for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]

    doc_spacy = nlp(" ".join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc_spacy]

    return lemmatized_tokens


# def preprocess_document_etiquettage(doc: str) -> list[tuple]:
   
#     # Tokenize the document into words
#     tokens = word_tokenize(doc)
    
#     # Perform POS tagging
#     pos_tags = pos_tag(tokens)
#     return pos_tags


class Config:
    term_frequency_function = term_frequency_log
    idf_function = idf_log
    preprocess_function = preprocess_document_lemmatization


def select_config():
    print("Select a configuration:")
    
    print("Preprocessing function:")
    print("1. Lemmatization")
    print("2. Racination")
    # print("3. Etiquettage")
    choice = input("Enter the number of the configuration: ")
    if choice == "1":
        Config.preprocess_function = preprocess_document_lemmatization
    elif choice == "2":
        Config.preprocess_function = preprocess_document_racinisation
    # elif choice == "3":
    #     Config.preprocess_function = preprocess_document_etiquettage
    
    print("Term frequency function:")
    print("1. Natural")
    print("2. Logarithmic")
    print("3. Augmented")
    choice = input("Enter the number of the configuration: ")
    if choice == "1":
        Config.term_frequency_function = term_frequency_natural
    elif choice == "2":
        Config.term_frequency_function = term_frequency_log
    elif choice == "3":
        Config.term_frequency_function = term_frequency_augmented

    print("IDF function:")
    print("1. Normal")
    print("2. Logarithmic")
    choice = input("Enter the number of the configuration: ")
    if choice == "1":
        Config.idf_function = idf_normal
    elif choice == "2":
        Config.idf_function = idf_log


def setup_nltk() -> spacy.language.Language:
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    nltk.download('averaged_perceptron_tagger_eng')
    nlp = spacy.load("en_core_web_sm")
    return nlp


def load_movies_csv() -> list[dict]:
    movies = []
    with open("movies_metadata.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            movies.append(row)
    
    # Process remaining rows
    parsed_movies = []
    COUNT_LIMIT = 200
    LIMIT = True
    i = 0
    for row in movies[1:]:
        if LIMIT and i > COUNT_LIMIT:
            break
        
        parsed_movies.append({
            'belongs_to_collection': row[1], 
            'genres': row[3],
            'original_title': row[8], 
            'overview': row[9],
        })
        i += 1
    return parsed_movies


def create_movie_document(movie: dict) -> str:
    doc = ''
    doc += f"Title: {movie['original_title']}\n"
    
    if len(movie['overview']) > 0:
        doc += f"Overview: {movie['overview']}\n"
    
    genres_list = eval(movie['genres'])
    genres = ','.join([genre['name'] for genre in genres_list])
    if len(genres) > 0:
        doc += f"Genres: {genres}\n"
    return doc


def convert_to_documents(movies: list[dict]) -> list[str]:
    documents = []
    for movie in movies:
        documents.append(create_movie_document(movie))
    return documents


def process_documents(documents: list[str]) -> list[str]:
    processed_documents = []
    for document in documents:
        processed_documents.append(Config.preprocess_function(document))
    return processed_documents


def build_inverted_index(processed_documents):
    inverted_index = defaultdict(list)

    for idx, terms in enumerate(processed_documents):
        for term in set(terms):
            inverted_index[term].append(idx)

    compressed_index = {}
    for term, doc_ids in inverted_index.items():
        gap_encoded = gap_encode(doc_ids)
        compressed_index[term] = compress_gamma_binary(gap_encoded)

    return compressed_index


def retrieve_postings_list(compressed_index, term):
    compressed_postings = compressed_index.get(term)
    if not compressed_postings:
        return []
    delta_decoded = decompress_gamma_binary(compressed_postings)
    return gap_decode(delta_decoded)


def compute_tfidf(processed_documents, compressed_index):
    doc_count = len(processed_documents)
    tf = defaultdict(lambda: defaultdict(int))
    idf = defaultdict(float)
    tfidf = defaultdict(lambda: defaultdict(float))

    for term in compressed_index.keys():
        doc_ids = retrieve_postings_list(compressed_index, term)

        idf[term] = Config.idf_function(processed_documents, len(doc_ids))

        for doc_id in doc_ids:
            tf_val = Config.term_frequency_function(term, doc_id, processed_documents)
 
            tfidf[term][doc_id] = tf_val * idf[term]

    return tfidf


def compute_query_vector(query, tfidf, compressed_index):
    query_terms = Config.preprocess_function(query)
    query_vector = defaultdict(float)

    for term in query_terms:
        if term in compressed_index:
            doc_ids = retrieve_postings_list(compressed_index, term)
            for doc_id in doc_ids:
                query_vector[term] += tfidf[term][doc_id]

    return query_vector


def compute_cosine_similarity(query_vector, tfidf, doc_id):
    dot_product = 0
    query_magnitude = 0
    doc_magnitude = 0

    for term, query_value in query_vector.items():
        dot_product += query_value * tfidf[term][doc_id]

    for term, query_value in query_vector.items():
        query_magnitude += query_value**2
    query_magnitude = math.sqrt(query_magnitude)

    for term in tfidf:
        doc_magnitude += tfidf[term][doc_id] ** 2
    doc_magnitude = math.sqrt(doc_magnitude)

    if query_magnitude == 0 or doc_magnitude == 0:
        return 0

    return dot_product / (query_magnitude * doc_magnitude)


def search(query, processed_documents, tfidf, inverted_index):
    query_vector = compute_query_vector(query, tfidf, inverted_index)
    similarities = {}
    for doc_id in range(len(processed_documents)):
        similarities[doc_id] = compute_cosine_similarity(query_vector, tfidf, doc_id)

    sorted_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:5]


def search_print_result(query, processed_documents, tfidf, inverted_index):
    print(f'Query: {query}')
    results = search(query, processed_documents, tfidf, inverted_index)
    for doc_id, score in results:
        print(f"Document {doc_id}, Movie name: {movies[doc_id]['original_title']}: Score = {score}")


select_config()
nlp = setup_nltk()
movies = load_movies_csv()
documents = convert_to_documents(movies)
processed_documents = process_documents(documents)

print(processed_documents[0])
print(f'{len(processed_documents)} documents')

inverted_index = build_inverted_index(processed_documents)
print(f'{len(inverted_index)} tokens')

tfidf = compute_tfidf(processed_documents, inverted_index)
# print(tfidf)

queries = ["Dracula Dead Vampire", "crime and drama"]
for query in queries:
    search_print_result(query, processed_documents, tfidf, inverted_index)

while True:
    query = input("Enter a query: ")
    search_print_result(query, processed_documents, tfidf, inverted_index)
