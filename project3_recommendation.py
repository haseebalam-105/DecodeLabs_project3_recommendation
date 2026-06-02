# ============================================================
# PROJECT 3: AI Recommendation Logic
# DecodeLabs Industrial Training Kit | Batch 2026
# ============================================================
# System     : Tech Stack Recommender (Content-Based Filtering)
# Algorithm  : TF-IDF Weighting + Cosine Similarity
# Pipeline   : Ingestion → Scoring → Sorting → Filtering (Top-N)
# ============================================================

import math
from collections import Counter

# ─────────────────────────────────────────────────────────────
# DATASET: Job Roles as "Items" (raw_skills.csv equivalent)
# Each role has a list of associated skills/tags.
# ─────────────────────────────────────────────────────────────
JOB_ROLES = {
    "Data Scientist": [
        "python", "sql", "machine learning", "data analysis",
        "statistics", "pandas", "numpy", "tensorflow", "visualization"
    ],
    "Machine Learning Engineer": [
        "python", "tensorflow", "pytorch", "deep learning",
        "algorithms", "model deployment", "docker", "machine learning"
    ],
    "Backend Developer": [
        "python", "java", "sql", "apis", "docker",
        "databases", "rest", "microservices", "node"
    ],
    "Frontend Developer": [
        "javascript", "react", "css", "html", "typescript",
        "ui", "ux", "figma", "responsive design"
    ],
    "DevOps Engineer": [
        "aws", "docker", "kubernetes", "ci/cd", "linux",
        "automation", "terraform", "cloud", "monitoring"
    ],
    "Cloud Architect": [
        "aws", "cloud", "kubernetes", "terraform",
        "networking", "security", "azure", "gcp", "automation"
    ],
    "Data Engineer": [
        "python", "sql", "spark", "hadoop", "pipelines",
        "etl", "databases", "kafka", "cloud", "aws"
    ],
    "Cybersecurity Analyst": [
        "networking", "security", "linux", "penetration testing",
        "firewalls", "cryptography", "siem", "incident response"
    ],
    "Full Stack Developer": [
        "python", "javascript", "react", "sql", "apis",
        "html", "css", "docker", "databases", "rest"
    ],
    "AI Research Scientist": [
        "python", "deep learning", "pytorch", "tensorflow",
        "mathematics", "statistics", "algorithms", "research", "nlp"
    ],
    "Systems Administrator": [
        "linux", "networking", "automation", "bash",
        "monitoring", "security", "virtualization", "servers"
    ],
    "Mobile Developer": [
        "swift", "kotlin", "react", "javascript", "ui",
        "ux", "apis", "mobile", "android", "ios"
    ],
}


# ─────────────────────────────────────────────────────────────
# PHASE 1: VECTOR MAPPING — Build the Shared Vocabulary Space
# ─────────────────────────────────────────────────────────────

def build_vocabulary(job_roles: dict) -> list:
    """Build the unique vocabulary across all job role skill sets."""
    vocab = set()
    for skills in job_roles.values():
        vocab.update(skills)
    return sorted(vocab)


# ─────────────────────────────────────────────────────────────
# PHASE 2: TF-IDF WEIGHTING
# Solves the "binary overlap limitation" — penalizes generic
# terms and rewards rare, specific ones.
#
# TF  = count(term in doc) / total_terms_in_doc
# IDF = log(total_documents / documents_containing_term)
# TF-IDF = TF × IDF
# ─────────────────────────────────────────────────────────────

def compute_tfidf(skills: list, all_roles: dict) -> dict:
    """Compute TF-IDF vector for a given skill list."""
    total_docs = len(all_roles)
    tf = Counter(skills)
    tfidf = {}

    for term, count in tf.items():
        # Term Frequency
        term_freq = count / len(skills)
        # Document Frequency: how many roles contain this term
        doc_freq = sum(1 for s in all_roles.values() if term in s)
        # Inverse Document Frequency (log dampens high-freq words)
        inv_doc_freq = math.log(total_docs / doc_freq) if doc_freq else 0.0
        tfidf[term] = term_freq * inv_doc_freq

    return tfidf


# ─────────────────────────────────────────────────────────────
# PHASE 3: COSINE SIMILARITY — The Industry Standard
# Measures the angle between two vectors, invariant to magnitude.
# Score 1 = perfectly aligned | Score 0 = no overlap
# cos(θ) = (A · B) / (||A|| × ||B||)
# ─────────────────────────────────────────────────────────────

def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Compute cosine similarity between two TF-IDF weight dictionaries."""
    # Dot product (shared terms only)
    dot_product = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in vec_b)

    # Magnitudes
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0   # Cold Start: user vector is all zeros → return 0

    return dot_product / (mag_a * mag_b)


# ─────────────────────────────────────────────────────────────
# PHASE 4: THE 4-STEP RANKING PIPELINE
# Step 1 — Ingestion : Capture user state (skills)
# Step 2 — Scoring   : Cosine similarity vs every job role
# Step 3 — Sorting   : Descending by score
# Step 4 — Filtering : Return Top-N (prevent choice overload)
# ─────────────────────────────────────────────────────────────

def recommend(user_skills: list, job_roles: dict, top_n: int = 3) -> list:
    """Run the full 4-step recommendation pipeline."""

    if not user_skills:
        return []   # Cold Start guard

    # Step 1: Ingestion — Build user TF-IDF vector
    user_vector = compute_tfidf(user_skills, job_roles)

    scores = []

    # Step 2: Scoring — Score every item in the dataset
    for role_name, role_skills in job_roles.items():
        role_vector = compute_tfidf(role_skills, job_roles)
        score = cosine_similarity(user_vector, role_vector)
        scores.append((role_name, score))

    # Step 3: Sorting — Highest score first
    scores.sort(key=lambda x: x[1], reverse=True)

    # Step 4: Filtering — Truncate to Top-N list
    return scores[:top_n]


# ─────────────────────────────────────────────────────────────
# MAIN: Interactive Tech Stack Recommender
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DecodeLabs | Project 3: AI Recommendation Logic")
    print("  Tech Stack Recommender — Content-Based Filtering")
    print("=" * 60)
    print("\nAvailable skills (examples):")
    vocab = build_vocabulary(JOB_ROLES)
    # Print a nice sample of the vocabulary
    sample = vocab[:30]
    print("  " + " | ".join(sample) + " | ...")

    # ── STEP 1: INGESTION — Get at least 3 user inputs ──────
    print("\n📥 Enter at least 3 skills you know (comma-separated).")
    print('   Example: python, sql, machine learning\n')
    raw = input("Your skills: ")
    user_skills = [s.strip().lower() for s in raw.split(",") if s.strip()]

    if len(user_skills) < 3:
        print("\n⚠️  Please enter at least 3 skills for accurate matching.")
        return

    print(f"\n✅ Skills captured: {user_skills}")

    # ── STEPS 2-4: SCORING → SORTING → FILTERING ────────────
    top_n = 3
    results = recommend(user_skills, JOB_ROLES, top_n=top_n)

    # ── OUTPUT ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🏆 Top {top_n} Recommended Career Paths for You:")
    print(f"{'='*60}")

    for rank, (role, score) in enumerate(results, 1):
        bar_len = int(score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"\n  #{rank}  {role}")
        print(f"       Match Score : {score*100:.1f}%")
        print(f"       [{bar}]")
        print(f"       Skills needed: {', '.join(JOB_ROLES[role][:5])}...")

    print(f"\n{'='*60}")
    print("💡 Tip: Add more specific skills to improve match accuracy.")
    print("   Generic terms (python, sql) score lower due to TF-IDF.")
    print("=" * 60)


if __name__ == "__main__":
    main()
