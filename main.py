import csv
import nltk
import math
import spacy
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
from sri_encodings import gap_encode, compress_gamma_binary, decompress_gamma_binary, gap_decode


def setup_nltk() -> spacy.language.Language:
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")
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
    COUNT_LIMIT = 100
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


def preprocess_document(doc: str):
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


def process_documents(documents: list[str]) -> list[str]:
    processed_documents = []
    for document in documents:
        processed_documents.append(preprocess_document(document))
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
        idf[term] = math.log(doc_count / (len(doc_ids))) if len(doc_ids) > 0 else 0

        for doc_id in doc_ids:
            terms = processed_documents[doc_id]
            term_count = Counter(terms)
            tf[term][doc_id] = (
                1 + math.log(term_count[term]) if term in term_count else 0
            )
            tfidf[term][doc_id] = tf[term][doc_id] * idf[term]

    return tfidf


def compute_query_vector(query, tfidf, compressed_index):
    query_terms = preprocess_document(query)
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
