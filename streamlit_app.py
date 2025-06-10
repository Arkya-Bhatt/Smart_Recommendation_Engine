import streamlit as st
from resource_loader import load_resources_from_csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import json
import os

profiles_file = 'user_profiles.json'
global_nltk_lemmatizer = None

def load_user_profiles():
    if os.path.exists(profiles_file):
        try:
            with open(profiles_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.warning("Could not decode user profiles file. Starting with empty profiles.")
            return {}
    return {}

def save_user_profiles(profiles_data):
    try:
        with open(profiles_file, 'w') as f:
            json.dump(profiles_data, f, indent=4)
    except IOError:
        st.error("Could not save user profiles. Please check file permissions.")

def initialize_nltk():
    global global_nltk_lemmatizer
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        st.info("Downloading NLTK resources (punkt, wordnet, omw-1.4, punkt_tab). This may take a moment...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            st.success("NLTK resources downloaded.")
        except Exception as e:
            st.error(f"Failed to download NLTK resources: {e}")
            st.stop()
            
    global_nltk_lemmatizer = WordNetLemmatizer()
    print("NLTK Lemmatizer initialized for Recommendation Engine.")

def lemmatize_text_streamlit(text):
    global global_nltk_lemmatizer
    if global_nltk_lemmatizer is None:
        st.error("Lemmatizer not initialized. Please ensure NLTK setup is complete.")
        initialize_nltk()
        if global_nltk_lemmatizer is None:
             return text

    if not isinstance(text, str): return ""
    words = word_tokenize(text.lower())
    lemmatized_words = [global_nltk_lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

@st.cache_resource
def vectorizer_and_matrix(resources_list):
    if not resources_list:
        return None, None

    corpus = []
    for resource in resources_list:
        name_text = str(resource.get('name', ''))
        desc_text = str(resource.get('description', ''))
        tags_list = resource.get('tags', [])
        if not isinstance(tags_list, list): tags_list = []
        tags_text = ' '.join(str(tag) for tag in tags_list)
        type_text = str(resource.get('type', ''))
        
        combined_text = f"{name_text} {desc_text} {tags_text} {type_text}".strip()
        lemmatized_combined_text = lemmatize_text_streamlit(combined_text)
        corpus.append(lemmatized_combined_text)

    if not corpus:
        return None, None
        
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2, ngram_range=(1,2))
    try:
        resource_tfidf_matrix = vectorizer.fit_transform(corpus)
        print(f"TF-IDF model initialized for Recommendation Engine. Vocab: {len(vectorizer.get_feature_names_out())}")
        return vectorizer, resource_tfidf_matrix
    except ValueError as e:
        st.error(f"Error initializing TF-IDF: {e}")
        return None, None

@st.cache_data
def load_all_resources(csv_filepath):
    print(f"Loading resources from {csv_filepath} for Streamlit app...") # For server console
    return load_resources_from_csv(csv_filepath)

def get_recommendations_streamlit(query_text, all_resources, vectorizer, resource_matrix, top_n=5):
    if not query_text.strip() or not all_resources or vectorizer is None or resource_matrix is None:
        return []

    lemmatized_query_text = lemmatize_text_streamlit(query_text) # Use Streamlit version
    query_tfidf_vector = vectorizer.transform([lemmatized_query_text])
    cosine_similarities = cosine_similarity(query_tfidf_vector, resource_matrix).flatten()
    
    scored_indices = list(enumerate(cosine_similarities))
    sorted_scored_indices = sorted(scored_indices, key=lambda x: x[1], reverse=True)

    recommendations = []
    for index, score in sorted_scored_indices[:top_n]:
        if score > 0.01: 
            resource = all_resources[index].copy()
            resource['relevance_score'] = score
            recommendations.append(resource)
    return recommendations

def main():
    st.set_page_config(page_title="Smart Python Resource Recommendation Engine", layout="wide", initial_sidebar_state="expanded")
    st.title("🐍 Smart Python Resource Recommendation Engine")
    st.markdown("Discover tools, articles, and exercises tailored to your Python learning journey!")
    st.markdown("---")
    
    initialize_nltk()
    all_user_profiles = load_user_profiles()
    
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'current_user_profile' not in st.session_state:
        st.session_state.current_user_profile = None
    if 'user_query_main' not in st.session_state:
        st.session_state.user_query_main = ""
    if 'engine_ready_message_shown' not in st.session_state:
        st.session_state.engine_ready_message_shown = False

    if st.session_state.username is None:
        with st.container():
            st.subheader("👋 Welcome! Get Started")
            login_cols = st.columns([2,1])
            with login_cols[0]:
                username_input = st.text_input("Enter your Username:", key="username_login_main_area", placeholder="e.g., python_learner")
            with login_cols[1]:
                # Add some space above the button to align better if text input is taller
                st.write("") # Creates a little vertical space
                if st.button("Login / Register", key="login_button_main_area", type="primary", use_container_width=True):
                    if username_input.strip():
                        st.session_state.username = username_input.strip()
                        if st.session_state.username in all_user_profiles:
                            st.session_state.current_user_profile = all_user_profiles[st.session_state.username]
                        else:
                            st.session_state.current_user_profile = {"search_history": []}
                            all_user_profiles[st.session_state.username] = st.session_state.current_user_profile
                            save_user_profiles(all_user_profiles)
                        st.session_state.user_query_main = ""
                        st.session_state.engine_ready_message_shown = False # Reset for new login
                        st.rerun()
                    else:
                        st.warning("Please enter a username.")
    
    else:
        # --- User is "Logged In" ---
        # Sidebar for User Info and Logout
        st.sidebar.success(f"👤 Logged in as: **{st.session_state.username}**")
        if st.sidebar.button("Logout", use_container_width=True):
            st.session_state.username = None
            st.session_state.current_user_profile = None
            st.session_state.user_query_main = ""
            st.session_state.engine_ready_message_shown = False
            st.rerun()
        st.sidebar.markdown("---")
    
        st.subheader("Welcome! Please enter your username to continue.")
        username_input = st.text_input("Username:", key="username_login_main")
        if st.button("Login / Register", key="login_button_main"):
            if username_input.strip():
                st.session_state.username = username_input.strip()
                if st.session_state.username in all_user_profiles:
                    st.session_state.current_user_profile = all_user_profiles[st.session_state.username]
                else:
                    st.session_state.current_user_profile = {"search_history": []}
                    all_user_profiles[st.session_state.username] = st.session_state.current_user_profile
                    save_user_profiles(all_user_profiles)
                st.rerun()
            else:
                st.warning("Please enter a username.")
        
        with st.expander("🕰️ Your Recent Searches", expanded=True): # Start expanded
            if st.session_state.current_user_profile and st.session_state.current_user_profile.get("search_history"):
                history = st.session_state.current_user_profile["search_history"]
                recent_searches_to_show = list(reversed(history[-5:]))

                if recent_searches_to_show:
                    # Using more flexible columns based on number of items
                    num_cols = min(len(recent_searches_to_show), 5) # Max 5 columns
                    cols = st.columns(num_cols)
                    for i, past_query in enumerate(recent_searches_to_show):
                        with cols[i % num_cols]: # Distribute among columns
                            if st.button(f"📜 {past_query}", key=f"dashboard_history_query_{i}", help=f"Search for: {past_query}", use_container_width=True):
                                st.session_state.user_query_main = past_query
                                st.rerun()
                else:
                    st.caption("No search history yet. Start searching below!")
            else:
                st.caption("No search history yet. Start searching below!")
        
        st.markdown("---")
        
    # else:
    #     st.sidebar.success(f"Logged in as: {st.session_state.username}")
    #     if st.sidebar.button("Logout"):
    #         st.session_state.username = None
    #         st.session_state.current_user_profile = None
    #         st.session_state.user_query_main = ""
    #         if 'engine_ready_message_shown' in st.session_state:
    #             del st.session_state.engine_ready_message_shown
    #         st.rerun()
        
    #     st.markdown("---")
    #     st.subheader("Your Recent Searches")
    #     if st.session_state.current_user_profile and st.session_state.current_user_profile.get("search_history"):
    #         history = st.session_state.current_user_profile["search_history"]
            
    #         recent_searches_to_show = list(reversed(history[-5:]))

    #         if recent_searches_to_show:
    #             cols = st.columns(len(recent_searches_to_show))
    #             for i, past_query in enumerate(recent_searches_to_show):
    #                 with cols[i]:
    #                     if st.button(past_query, key=f"dashboard_history_query_{i}", help=f"Search for: {past_query}", use_container_width=True):
    #                         st.session_state.user_query_main = past_query
    #                         st.rerun()
    #         else:
    #             st.caption("No search history yet. Start searching!")
    #     else:
    #         st.caption("No search history yet. Start searching!")
    #     st.markdown("---")

    #     if st.session_state.current_user_profile and st.session_state.current_user_profile.get("search_history"):
    #         with st.sidebar.expander("Your Search History", expanded=False):
    #             for i, past_query in enumerate(reversed(st.session_state.current_user_profile["search_history"][-10:])):
    #                 if st.button(past_query, key=f"past_query_{i}", help="Click to search this again"):
    #                     st.session_state.user_query_main = past_query
    #                     st.rerun()

    csv_path = 'github_data.csv'
    
    with st.spinner("Loading resources & recommendation engine... This might take a moment on first load."):
        resources_db = load_all_resources(csv_path)
    
        if not resources_db:
            st.error(f"Failed to load resources from {csv_path}.")
            st.stop()

        vectorizer, resource_tfidf_matrix = vectorizer_and_matrix(resources_db)

        if vectorizer is None or resource_tfidf_matrix is None:
            st.error("Failed to initialize the recommendation engine.")
            st.stop()
    
    if not st.session_state.engine_ready_message_shown:
        st.success("🔍 Recommendation engine is ready! Let's find some resources.")
        st.session_state.engine_ready_message_shown = True
    
    # if 'engine_ready_message_shown' not in st.session_state:
    #         st.success("Recommendation engine ready!")
    #         st.session_state.engine_ready_message_shown = True
    
    # st.markdown("---")
    
    # if 'user_query_main' not in st.session_state:
    #         st.session_state.user_query_main = ""

    st.subheader("🚀 Find Your Next Python Resource")
    user_query = st.text_input("What are you looking for today?",
                               value=st.session_state.user_query_main,
                               key="main_search_input", 
                               placeholder="e.g., pandas tutorial for beginners, web development with flask")

    if user_query != st.session_state.user_query_main and "main_search_input" in st.session_state:
            st.session_state.user_query_main = user_query
    
    if user_query:
        if st.session_state.current_user_profile:
                if not st.session_state.current_user_profile["search_history"] or \
                   st.session_state.current_user_profile["search_history"][-1] != user_query:
                    st.session_state.current_user_profile["search_history"].append(user_query)
                    all_user_profiles[st.session_state.username] = st.session_state.current_user_profile
                    save_user_profiles(all_user_profiles)
        
        with st.spinner(f"Searching for '{user_query}'..."):
            recommendations = get_recommendations_streamlit(
                user_query,
                resources_db,
                vectorizer,
                resource_tfidf_matrix,
                top_n=5
            )

        if recommendations:
            st.markdown("---")
            st.subheader(f"🌟 Top Matches for '{user_query}'")
            
            with st.expander("📊 View Relevance Score Chart", expanded=False):
                    chart_data_list = [{'Resource': rec['name'], 'Score': rec['relevance_score']} for rec in recommendations]
                    if chart_data_list:
                        chart_df = pd.DataFrame(chart_data_list).set_index('Resource')
                        st.bar_chart(chart_df['Score'], height=max(200, len(chart_df) * 40))
            
            for i, rec in enumerate(recommendations):
                with st.container():
                    st.markdown(f"### {i+1}. {rec['name']}")
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**Type:** {rec.get('type', 'N/A')}")
                        description_value = rec.get('description', 'N/A')
                        description_display = str(description_value) if description_value is not None else 'N/A'
                        st.markdown(f"**Description:** {description_display[:200]}...")
                        if rec.get('tags'):
                            tags_html = " ".join([f"<span style='background-color: #e0e0e0; color: #333; border-radius: 5px; padding: 2px 6px; font-size: 0.8em; margin-right: 4px;'>{tag}</span>" 
                                                      for tag in rec.get('tags', [])[:5]])
                            st.markdown(f"**Tags:** {tags_html}", unsafe_allow_html=True)
                    with col2:
                        st.link_button("🔗 Visit Resource", 
                                           rec.get('url', '#'), 
                                           help=rec.get('url', 'No URL provided'), 
                                           use_container_width=True)
                    st.markdown("---")
        else:
            st.info(f"Sorry, no relevant recommendations found for '{user_query}'. Try a different query.")
    
    if "main_search_input" in st.session_state and st.session_state.main_search_input == "" and st.session_state.user_query_main != "":
        if st.session_state.user_query_main not in [None, ""]:
            pass
    elif st.session_state.user_query_main != user_query:
        pass
    else:
        st.session_state.user_query_main = ""

if __name__ == '__main__':
    main()