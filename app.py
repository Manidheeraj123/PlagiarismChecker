from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# ---------- Ensure NLTK Data ----------
def safe_nltk_download(resource):
    try:
        nltk.data.find(resource)    
    except LookupError:
        nltk.download(resource.split('/')[-1])

safe_nltk_download('tokenizers/punkt')
safe_nltk_download('tokenizers/punkt_tab')
safe_nltk_download('corpora/wordnet')

# Initialize Flask app
app = Flask(__name__)

# Load SBERT model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------- Utility Functions ----------
def cosine_similarity_sbert(text1, text2):
    """Cosine similarity using Sentence-BERT embeddings."""
    embeddings = model.encode([text1, text2])
    cos_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
    cos_sim = (cos_sim + 1) / 2  # Scale [-1,1] → [0,1]
    return cos_sim

def cosine_similarity_tfidf(text1, text2):
    """Cosine similarity using TF-IDF vectors."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    # cosine similarity = dot(A,B) / (||A||*||B||)
    A = tfidf_matrix[0].toarray()[0]
    B = tfidf_matrix[1].toarray()[0]
    dot = (A * B).sum()
    norm_A = (A**2).sum() ** 0.5
    norm_B = (B**2).sum() ** 0.5
    return dot / (norm_A * norm_B) if norm_A and norm_B else 0.0

def combined_score(text1, text2, w_tfidf=0.6, w_sbert=0.4):
    """Weighted average of TF-IDF and SBERT similarities."""
    tfidf_sim = cosine_similarity_tfidf(text1, text2)
    sbert_sim = cosine_similarity_sbert(text1, text2)
    final = w_tfidf * tfidf_sim + w_sbert * sbert_sim
    return final * 100, tfidf_sim * 100, sbert_sim * 100

# ---------- Flask Routes ----------
@app.route("/")
def index():
    """Render upload form."""
    return render_template("index.html")

@app.route("/compare", methods=["POST"])
def compare():
    """Handle plagiarism checking."""
    file1 = request.files["file1"]
    file2 = request.files["file2"]

    text1 = file1.read().decode("utf-8", errors="ignore")
    text2 = file2.read().decode("utf-8", errors="ignore")

    final_score, tfidf_score, sbert_score = combined_score(text1, text2)

    return render_template(
        "result.html",
        final_score=round(final_score, 2),
        tfidf_score=round(tfidf_score, 2),
        sbert_score=round(sbert_score, 2),
    )

if __name__ == "__main__":
    app.run(debug=True)
