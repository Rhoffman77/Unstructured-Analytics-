import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from pydantic import BaseModel
from typing import List
import json

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="ND Club Finder",
    page_icon="☘️",
    layout="centered"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

.stApp {
    background: linear-gradient(160deg, #0C2340 0%, #0a1e38 50%, #071529 100%);
    min-height: 100vh;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #C99700;
    text-align: center;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 1.1rem;
    color: #AE9142;
    text-align: center;
    font-weight: 300;
    margin-bottom: 2.5rem;
    letter-spacing: 0.5px;
}

.section-label {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #C99700;
    margin-bottom: 0.5rem;
}

.card {
    background: rgba(12, 35, 64, 0.6);
    border: 1px solid rgba(174, 145, 66, 0.3);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

.club-rank {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: rgba(201, 151, 0, 0.25);
    line-height: 1;
}

.club-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: #f0ece0;
    margin-bottom: 0.3rem;
}

.fit-badge {
    display: inline-block;
    background: linear-gradient(90deg, #00843D22, #00843D44);
    color: #4cce7f;
    border: 1px solid #00843D;
    border-radius: 20px;
    padding: 0.15rem 0.8rem;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    margin-bottom: 0.8rem;
}

.club-summary {
    color: #AE9142;
    font-size: 0.9rem;
    line-height: 1.6;
    margin-bottom: 0.6rem;
    font-style: italic;
}

.club-why {
    color: #d4dce8;
    font-size: 0.95rem;
    line-height: 1.6;
}

.why-label {
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #C99700;
    font-weight: 600;
    margin-bottom: 0.3rem;
}

.divider {
    border: none;
    border-top: 1px solid rgba(174, 145, 66, 0.2);
    margin: 1.5rem 0;
}

.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(12, 35, 64, 0.8) !important;
    border: 1px solid rgba(174, 145, 66, 0.3) !important;
    border-radius: 10px !important;
    color: #f0ece0 !important;
}

/* Multiselect selected tags — gold on navy */
span[data-baseweb="tag"] {
    background-color: rgba(201, 151, 0, 0.2) !important;
    border: 1px solid rgba(201, 151, 0, 0.6) !important;
    border-radius: 6px !important;
    color: #C99700 !important;
}

/* The × close button inside each tag */
span[data-baseweb="tag"] span[role="presentation"] {
    color: #C99700 !important;
}
span[data-baseweb="tag"] span[role="presentation"]:hover {
    color: #f0ece0 !important;
    background-color: rgba(201, 151, 0, 0.35) !important;
}

.stTextInput > div > div > input {
    background: rgba(12, 35, 64, 0.8) !important;
    border: 1px solid rgba(174, 145, 66, 0.3) !important;
    border-radius: 10px !important;
    color: #f0ece0 !important;
}

/* Kill red focus rings everywhere — replace with gold */
*:focus,
*:focus-visible {
    outline: none !important;
    box-shadow: none !important;
}

div[data-baseweb="select"]:focus-within,
div[data-baseweb="input"]:focus-within {
    border-color: rgba(201, 151, 0, 0.7) !important;
    box-shadow: 0 0 0 2px rgba(201, 151, 0, 0.2) !important;
}

div[data-baseweb="select"] > div:focus-within {
    border-color: rgba(201, 151, 0, 0.7) !important;
    box-shadow: 0 0 0 2px rgba(201, 151, 0, 0.2) !important;
}

.stTextInput > div:focus-within {
    border-color: rgba(201, 151, 0, 0.7) !important;
    box-shadow: 0 0 0 2px rgba(201, 151, 0, 0.2) !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #AE9142, #C99700) !important;
    color: #0C2340 !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    margin-top: 1rem !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(201, 151, 0, 0.35) !important;
}

label[data-testid="stWidgetLabel"] p {
    color: #AE9142 !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.85rem !important;
}

.stSpinner > div {
    border-top-color: #C99700 !important;
}

.card:hover {
    border-color: rgba(0, 132, 61, 0.5) !important;
    transition: border-color 0.2s ease;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Pydantic models
# ---------------------------
class ClubRecommendation(BaseModel):
    rank: int
    club_name: str
    fit_percentage: int
    summary: str
    why: str

class RecommendationResponse(BaseModel):
    recommendations: List[ClubRecommendation]

# ---------------------------
# OpenAI client
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
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["Club Name", "Club Description"])
    sentences = (df['Club Name'] + ". " + df['Club Description']).tolist()
    embeddings = _model.encode(sentences, show_progress_bar=False)
    df['embedding'] = embeddings.tolist()
    return df, np.array(embeddings)

df, embeddings = load_data(model)

# ---------------------------
# Search function
# ---------------------------
def search_clubs(query, top_k=10, exclude_grad=False):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k * 2]  # grab extra to filter from
    results = df.iloc[top_indices][['Club Name', 'Club Description']].copy()

    if exclude_grad:
        grad_keywords = ['graduate', 'masters', 'MBA', 'PhD', 'doctoral', 'law school', 'grad student']
        pattern = '|'.join(grad_keywords)
        results = results[~results['Club Name'].str.contains(pattern, case=False, na=False)]
        results = results[~results['Club Description'].str.contains(pattern, case=False, na=False)]

    return results.head(top_k)

# ---------------------------
# LLM recommendation function
# ---------------------------
def get_recommendations(query, dorm, class_year, majors):
    is_undergrad = class_year in ["2026", "2027", "2028", "2029"]
    candidates = search_clubs(query, exclude_grad=is_undergrad)

    major_str = ", ".join(majors) if majors else "Undecided"

    level_instruction = ""
    if is_undergrad:
        level_instruction = "IMPORTANT: This is an undergraduate student. Do NOT recommend any graduate, masters, MBA, PhD, or law school clubs."

    prompt = f"""
You are recommending clubs to a Notre Dame student. Address them directly using "you" and "your" throughout.

Their profile:
- Interests: {query}
- Dorm: {dorm}
- Class Year: {class_year}
- Major(s): {major_str}

{level_instruction}

Here are some possible clubs:
{candidates.to_string(index=False)}

Pick the best 5 clubs. For each, provide:
- rank (1-5)
- club_name (exact name from the list)
- fit_percentage (0-100, how well it matches this student)
- summary (1-2 sentence description of what the club does)
- why (1-2 sentences on why it fits YOU specifically — use "you" and "your", referencing their major(s), dorm, class year, or interests directly)

Respond ONLY with a valid JSON object in this exact format, no markdown or extra text:
{{
  "recommendations": [
    {{
      "rank": 1,
      "club_name": "...",
      "fit_percentage": 92,
      "summary": "...",
      "why": "..."
    }}
  ]
}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)
    return RecommendationResponse(**data)

# ---------------------------
# UI
# ---------------------------
st.markdown('<div class="hero-title">☘️ ND Club Finder</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Discover your place at Notre Dame</div>', unsafe_allow_html=True)

# Sorted lists
DORMS = sorted([
    "Off Campus", "Alumni", "Badin", "Baumer", "Breen-Phillips", "Carroll",
    "Cavanaugh", "Dillon", "Duncan", "Farley", "Fisher", "Flaherty",
    "Graham", "Howard", "Johnson Family", "Keenan", "Keough", "Knott",
    "Lewis", "Lyons", "McGlinn", "Morrissey", "O'Neill Family", "Pangborn",
    "Pasquerilla East", "Pasquerilla West", "Ryan", "Siegfried", "Sorin",
    "Stanford", "St. Edward's", "Walsh", "Welsh Family", "Zahm"
])

MAJORS = sorted([
    "Undecided", "Accountancy", "Aerospace Engineering", "Africana Studies",
    "American Studies", "Anthropology",
    "Applied & Computational Mathematics and Statistics (ACMS)",
    "Arabic Studies", "Architecture", "Art History", "Biochemistry",
    "Biological Sciences", "Business Analytics", "Chemical Engineering",
    "Chemistry", "Chinese", "Civil Engineering", "Classics",
    "Computer Science", "Economics", "Electrical Engineering",
    "Engineering (General)", "English", "Environmental Engineering",
    "Environmental Sciences", "Film, Television, and Theatre", "Finance",
    "French", "German", "History", "International Economics",
    "Italian Studies", "Management Consulting", "Marketing", "Mathematics",
    "Mechanical Engineering", "Medieval Studies", "Music",
    "Neuroscience and Behavior", "Philosophy", "Physics",
    "Political Science", "Psychology", "Romance Languages and Literatures",
    "Russian", "Sociology", "Spanish", "Theology", "Theology and Philosophy"
])

INTERESTS = sorted([
    "Arts & Creative (Music, Theatre, Design)", "Business & Finance",
    "Club Sports", "Consulting", "Cultural & Identity Groups",
    "Data Science / AI", "Entrepreneurship / Startups", "Faith & Service",
    "Fitness & Wellness", "Leadership & Networking", "Media & Communications",
    "Outdoors & Adventure", "Politics & International Relations", "Pre-Law / Government",
    "Pre-Med / Health", "Psychology & Social Sciences", "Science & Research",
    "Social / Fun", "Sports & Athletics", "Technology & Engineering",
    "Volunteering / Community Service"
])

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-label">Class Year</div>', unsafe_allow_html=True)
        class_year = st.selectbox("Class Year", [
            "2026", "2027", "2028", "2029", "Graduate / Masters"
        ], label_visibility="collapsed")

    with col2:
        st.markdown('<div class="section-label">Dorm</div>', unsafe_allow_html=True)
        dorm = st.selectbox("Dorm", DORMS, label_visibility="collapsed")

    st.markdown('<div class="section-label">Major(s)</div>', unsafe_allow_html=True)
    majors = st.multiselect("Major(s)", MAJORS, label_visibility="collapsed")

    st.markdown('<div class="section-label">Interests</div>', unsafe_allow_html=True)
    interests = st.multiselect("Interests", INTERESTS, label_visibility="collapsed")

    st.markdown('<div class="section-label">Anything else?</div>', unsafe_allow_html=True)
    extra = st.text_input("Extra", placeholder="e.g. I want to meet people outside my major...", label_visibility="collapsed")

query = ", ".join(interests)
if extra:
    query += ". " + extra

if st.button("Find My Clubs →"):
    if not interests and not extra:
        st.warning("Please select at least one interest or add a note above.")
    else:
        with st.spinner("Finding your perfect clubs..."):
            result = get_recommendations(query, dorm, class_year, majors)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown('<div class="hero-subtitle" style="margin-bottom:1.5rem">Your Top Matches</div>', unsafe_allow_html=True)

        for club in result.recommendations:
            st.markdown(f"""
            <div class="card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <div class="club-rank">#{club.rank}</div>
                        <div class="club-name">{club.club_name}</div>
                    </div>
                    <div class="fit-badge">☘️ {club.fit_percentage}% fit</div>
                </div>
                <div class="club-summary">{club.summary}</div>
                <div class="why-label">Why it's right for you</div>
                <div class="club-why">{club.why}</div>
            </div>
            """, unsafe_allow_html=True)
