from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os

vectorizer = None
resource_tfidf_matrix = None
lemmatizer = None
NLTK_RESOURCES_DOWNLOADED_FLAG = '.nltk_resources_downloaded'

def lemmatize_text(text):
    global lemmatizer
    if lemmatizer is None:
        return text
    if not isinstance(text, str):
        return ""

    words = word_tokenize(text.lower())
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def initialize_nltk_components():
    global lemmatizer

    if not os.path.exists(NLTK_RESOURCES_DOWNLOADED_FLAG):
        print("NLTK resources flag not found. Checking and downloading if necessary...")
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('corpora/omw-1.4')
            print("NLTK resources seem to be present (found by nltk.data.find).")
            with open(NLTK_RESOURCES_DOWNLOADED_FLAG, 'w') as f:
                f.write('NLTK resources checked/downloaded.')

        except LookupError:
            print("Downloading NLTK resources (punkt, wordnet, omw-1.4)...")
            try:
                nltk.download('punkt', quiet=False)
                nltk.download('wordnet', quiet=False)
                nltk.download('omw-1.4', quiet=False)
                print("NLTK resources downloaded successfully.")
                with open(NLTK_RESOURCES_DOWNLOADED_FLAG, 'w') as f:
                    f.write('NLTK resources downloaded.')
            except Exception as download_error:
                print(f"Error downloading NLTK resources: {download_error}")
                print("Please try downloading them manually and then restart the application.")
                print("import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')")
                exit()
    else:
        print("NLTK resources previously downloaded (flag file found).")

    try:
        lemmatizer = WordNetLemmatizer()
        print("NLTK Lemmatizer initialized.")
    except Exception as e:
        print(f"Error initializing NLTK Lemmatizer even after resource check: {e}")
        print("This might indicate an issue with the NLTK installation or downloaded data.")
        exit()


def initialize_tfidf_model(resources_list):
    global vectorizer, resource_tfidf_matrix

    if not resources_list:
        print("Warning: No resources provided to initialize TF-IDF model.")
        return False

    print("Preparing corpus for TF-IDF model (with lemmatization)...")
    corpus = []
    for resource in resources_list:
        name_text = str(resource.get('name', ''))
        desc_text = str(resource.get('description', ''))
        tags_list = resource.get('tags', [])
        if not isinstance(tags_list, list): tags_list = []
        tags_text = ' '.join(str(tag) for tag in tags_list)
        type_text = str(resource.get('type', ''))
        
        combined_text = f"{name_text} {desc_text} {tags_text} {type_text}".strip()
        
        lemmatized_combined_text = lemmatize_text(combined_text)
        corpus.append(lemmatized_combined_text)

    if not corpus:
        print("Error: Corpus is empty after preparing texts. Cannot initialize TF-IDF.")
        return False

    vectorizer = TfidfVectorizer(stop_words='english', min_df=2, ngram_range=(1,2))
    
    try:
        resource_tfidf_matrix = vectorizer.fit_transform(corpus)
        print(f"TF-IDF model initialized. Vocabulary size: {len(vectorizer.get_feature_names_out())}")
        return True
    except ValueError as e:
        print(f"Error initializing TF-IDF: {e}")
        return False


def get_recommendations_by_text_similarity(query_text, all_resources, top_n=5):
    global vectorizer, resource_tfidf_matrix

    if not query_text.strip() or not all_resources or vectorizer is None or resource_tfidf_matrix is None:
        print("Warning: Query is empty, resources are missing, or TF-IDF model not initialized.")
        return []

    lemmatized_query_text = lemmatize_text(query_text)

    query_tfidf_vector = vectorizer.transform([lemmatized_query_text])
    cosine_similarities = cosine_similarity(query_tfidf_vector, resource_tfidf_matrix).flatten()
    
    scored_indices = list(enumerate(cosine_similarities))
    sorted_scored_indices = sorted(scored_indices, key=lambda x: x[1], reverse=True)

    recommendations = []
    for index, score in sorted_scored_indices[:top_n]:
        if score > 0.01: 
            resource = all_resources[index].copy()
            resource['relevance_score'] = score
            recommendations.append(resource)
            
    return recommendations

# if __name__ == "__main__":
#     initialize_nltk_components()
#     CSV_FILEPATH = 'github_data.csv'
#     resources_db = load_resources_from_csv(CSV_FILEPATH)

#     if not resources_db:
#         print("Failed to load resources. Exiting.")
#         exit()
    
#     if not initialize_tfidf_model(resources_db):
#         print("Exiting due to TF-IDF initialization failure.")
#         exit()

#     print("\nWelcome to the Smart Python Resource Recommender (with Lemmatization)!")
#     print("Describe what you're looking for (e.g., 'pandas tutorial for beginners', 'web development with flask', 'advanced numpy techniques') or type 'exit' to quit.")
    
#     while True:
#         user_query = input("\nSearch: ")
#         if user_query.lower() == 'exit':
#             break
#         if not user_query.strip():
#             print("Please enter a search query.")
#             continue

#         recommendations = get_recommendations_by_text_similarity(
#             user_query,
#             resources_db,
#             top_n=5
#         )

#         if recommendations:
#             print(f"\n--- Top {len(recommendations)} recommendations for '{user_query}' ---")
#             for i, rec in enumerate(recommendations):
#                 print(f"\n{i+1}. {rec['name']} (Score: {rec['relevance_score']:.4f})")
#                 print(f"   Type: {rec.get('type', 'N/A')}")
#                 print(f"   URL: {rec.get('url', 'N/A')}")
#                 print(f"   Tags: {rec.get('tags', [])[:7]}")
#             print("-" * 40)
#         else:
#             print("Sorry, no relevant recommendations found for your query.")
            
#     print("Thank you for using the recommender!")