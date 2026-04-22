import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ---------------------------
# OpenAI client (STREAMLIT CLOUD SAFE)
# ---------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------------------
# Load embedding model
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-mpnet-base-v2')

model = load_model()

# ---------------------------
# Load data + embeddings
# ---------------------------
@st.cache_data
def load_data(_model):
    df = pd.read_excel("clubs.xlsx")
    df.columns = df.columns.str.strip()  # removes hidden spaces from column names
    df = df.dropna(subset=["Club Name", "Club Description"])
    sentences = (df['Club Name'] + ". " + df['Club Description']).tolist()
    embeddings = _model.encode(sentences, show_progress_bar=False)
    df['embedding'] = embeddings.tolist()
    return df, np.array(embeddings)

df, embeddings = load_data(model)

# ---------------------------
# Search function
# ---------------------------
def search_clubs(query, top_k=10):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    return df.iloc[top_indices][['Club Name', 'Club Description']]

# ---------------------------
# LLM recommendation function
# ---------------------------
def get_recommendations(query, experience, commitment):
    candidates = search_clubs(query)
    prompt = f"""
A student is looking for clubs.

Interests: {query}
Experience level: {experience}
Available time: {commitment} hours per week

Here are some possible clubs:
{candidates.to_string(index=False)}

Pick the best 5 clubs and explain why each is a good fit.
Keep it concise and friendly.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎓 Club Recommender")
st.write("Find clubs that match your interests!")

interests = st.multiselect(
    "Select your interests:",
    [
        "AI / Machine Learning",
        "Programming",
        "Robotics",
        "Business",
        "Finance",
        "Art",
        "Music",
        "Sports",
        "Volunteering",
        "Social / Networking"
    ]
)

extra = st.text_input("Anything else you're looking for? (optional)")

query = ", ".join(interests)
if extra:
    query += ". " + extra

if st.button("Find Clubs"):
    if not interests and not extra:
        st.warning("Please select at least one interest or enter a preference.")
    else:
        with st.spinner("Finding the best clubs for you..."):
            recommendations = get_recommendations(query, experience, commitment)
        st.subheader("Recommended Clubs")
        st.write(recommendations)
