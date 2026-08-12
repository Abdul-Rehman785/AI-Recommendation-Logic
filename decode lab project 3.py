import math
from collections import Counter

DATASET = {
    "Data Scientist": ["python", "sql", "machine learning", "statistics"],
    "DevOps Engineer": ["aws", "docker", "kubernetes", "automation"],
    "Backend Developer": ["java", "python", "sql", "apis"],
    "Cloud Architect": ["aws", "cloud computing", "automation", "networking"],
    "Frontend Developer": ["javascript", "css", "html", "react"],
}

def get_user_input(min_skills=3):
    print(f"Enter at least {min_skills} skills (comma-separated):")
    raw = input("> ")
    skills = [s.strip().lower() for s in raw.split(',') if s.strip()]

    if len(skills) < min_skills:
        raise ValueError(f"Need at least {min_skills} skills, got {len(skills)}.")

    return skills

def compute_tf(skill_list):
    total_terms = len(skill_list)
    counts = Counter(skill_list)
    return {term: count / total_terms for term, count in counts.items()}

def compute_idf(all_documents):
    total_docs = len(all_documents)
    idf = {}

    vocabulary = set(term for doc in all_documents for term in doc)

    for term in vocabulary:
        doc_count = sum(1 for doc in all_documents if term in doc)
        idf[term] = math.log(total_docs / doc_count) if doc_count else 0.0

    return idf

def compute_tfidf_vector(skill_list, idf_scores):
    tf = compute_tf(skill_list)
    return {term: tf_val * idf_scores.get(term, 0.0) for term, tf_val in tf.items()}

def cosine_similarity(vec_a, vec_b):
    common_terms = set(vec_a.keys()) & set(vec_b.keys())
    dot_product = sum(vec_a[t] * vec_b[t] for t in common_terms)

    norm_a = math.sqrt(sum(val ** 2 for val in vec_a.values()))
    norm_b = math.sqrt(sum(val ** 2 for val in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)

def recommend(user_skills, dataset, top_n=3):
    all_documents = list(dataset.values())
    idf_scores = compute_idf(all_documents)

    user_vector = compute_tfidf_vector(user_skills, idf_scores)

    scores = []
    for role, skills in dataset.items():
        role_vector = compute_tfidf_vector(skills, idf_scores)
        similarity = cosine_similarity(user_vector, role_vector)
        scores.append((role, similarity))

    scores.sort(key=lambda pair: pair[1], reverse=True)

    return scores[:top_n]

def main():
    user_skills = get_user_input(min_skills=3)

    results = recommend(user_skills, DATASET, top_n=3)

    print("\nTop matching career paths:")
    for rank, (role, score) in enumerate(results, start=1):
        print(f"{rank}. {role}  (match score: {score:.2f})")

if __name__ == "__main__":
    main()