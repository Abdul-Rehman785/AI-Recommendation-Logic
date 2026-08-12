# Tech Stack Recommender

A simple content-based recommendation engine that maps a user's raw skills to the most relevant career paths using **TF-IDF weighting** and **Cosine Similarity**. Built as Project 3 (AI Recommendation Logic) of the DecodeLabs Industrial Training Kit.

## Overview

Instead of relying on collaborative filtering (which needs historical user data), this project uses **content-based filtering** — it matches a user's stated skills directly against the intrinsic attributes (required skills) of each job role. This means it works immediately, with no "cold start" problem for new roles or new users.

The engine follows a standard **Input → Process → Output** pipeline:

1. **Ingestion** — Capture at least 3 skills from the user.
2. **Scoring** — Convert skills and job-role skill sets into TF-IDF weighted vectors, then compute cosine similarity between the user vector and each role vector.
3. **Sorting** — Rank roles by similarity score, highest first.
4. **Filtering** — Return only the Top-N (default: 3) matches.

## How It Works

### 1. TF-IDF (Term Frequency–Inverse Document Frequency)
- **Term Frequency (TF)** measures how often a skill appears relative to the total skills listed.
- **Inverse Document Frequency (IDF)** down-weights skills that appear across many roles (e.g., "python") and up-weights rarer, more distinctive skills (e.g., "kubernetes").
- Combining TF and IDF ensures common/generic skills don't dominate the similarity score.

### 2. Cosine Similarity
Instead of raw overlap counts, the engine measures the **angle** between the user's skill vector and each role's skill vector. This makes the comparison insensitive to the size of the skill list and focuses purely on how closely the *direction* of interests aligns.

```
cos(θ) = (A · B) / (‖A‖ ‖B‖)
```

A score close to `1` means a strong match; a score close to `0` means little to no overlap.

## Dataset

Roles and their associated skills are currently defined inline in `DATASET`:

| Role | Skills |
|---|---|
| Data Scientist | python, sql, machine learning, statistics |
| DevOps Engineer | aws, docker, kubernetes, automation |
| Backend Developer | java, python, sql, apis |
| Cloud Architect | aws, cloud computing, automation, networking |
| Frontend Developer | javascript, css, html, react |

You can easily swap this for a `raw_skills.csv` file or expand it with more roles/skills.

## Requirements

- Python 3.7+
- No external dependencies (uses only the standard library: `math`, `collections`)

## Usage

Clone the repo and run the script:

```bash
python tech_stack_recommender.py
```

You'll be prompted to enter at least 3 comma-separated skills:

```
Enter at least 3 skills (comma-separated):
> python, aws, automation
```

Example output:

```
Top matching career paths:
1. DevOps Engineer  (match score: 0.71)
2. Cloud Architect  (match score: 0.54)
3. Data Scientist  (match score: 0.22)
```

## Project Structure

```
.
├── tech_stack_recommender.py   # Main script
└── README.md                   # This file
```

## Key Functions

| Function | Purpose |
|---|---|
| `get_user_input()` | Collects and validates user skill input |
| `compute_tf()` | Computes term frequency for a skill list |
| `compute_idf()` | Computes inverse document frequency across all roles |
| `compute_tfidf_vector()` | Combines TF and IDF into a weighted vector |
| `cosine_similarity()` | Measures similarity between two TF-IDF vectors |
| `recommend()` | Runs the full scoring, sorting, and filtering pipeline |

