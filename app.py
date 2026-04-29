from flask import Flask, render_template, request
import pandas as pd
import os
import re
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

if not os.path.exists("uploads"):
    os.makedirs("uploads")

app.config["UPLOAD_FOLDER"] = "uploads"

data = pd.read_csv("jobs.csv")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9+#. ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_raw_text(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"

    return text


def get_skill_bank():
    skill_bank = set()

    for row in data["skills"]:
        skills = str(row).split(",")

        for skill in skills:
            skill = skill.strip().lower()

            if len(skill) > 2:
                skill_bank.add(skill)

    return skill_bank


def get_detected_skills(raw_text):
    text = raw_text.lower()
    detected = []

    skill_bank = get_skill_bank()

    # First try to find skills section only
    section_match = re.search(
        r"(technical skills|key skills|skills)(.*?)(languages|education|experience|projects|certifications|objective|course)",
        text,
        re.DOTALL
    )

    if section_match:
        section = section_match.group(2)

        # remove bracket details like (basic, loops, functions...)
        section = re.sub(r"\(.*?\)", "", section)

        # make separators clean
        section = section.replace("•", ",")
        section = section.replace("·", ",")
        section = section.replace("\n", ",")
        section = section.replace(".", "")
        section = re.sub(r"\betc\b", "", section)
        section = re.sub(r"\be\s*t\s*c\b", "", section)

        parts = section.split(",")

        for part in parts:
            skill = part.strip().lower()
            skill = re.sub(r"[^a-zA-Z0-9+#. ]", " ", skill)
            skill = re.sub(r"\s+", " ", skill).strip()

            if len(skill) <= 2:
                continue

            # avoid theory lines
            if len(skill.split()) > 4:
                continue

            if skill not in detected:
                detected.append(skill)

    # If skills section is not clean/found, detect only known job skills from jobs.csv
    if not detected:
        clean_resume = clean_text(raw_text)

        for skill in sorted(skill_bank):
            pattern = r"\b" + re.escape(skill) + r"\b"

            if re.search(pattern, clean_resume):
                detected.append(skill)

    return detected


def smart_match(resume_text):
    results = []

    job_texts = data["skills"].astype(str).str.lower().tolist()
    corpus = [resume_text] + job_texts

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    resume_vector = tfidf_matrix[0]
    job_vectors = tfidf_matrix[1:]

    scores = cosine_similarity(resume_vector, job_vectors)[0]

    for i, score in enumerate(scores):
        job_title = data.iloc[i]["job_title"]
        salary = data.iloc[i]["salary"]

        job_skills = [
            skill.strip().lower()
            for skill in str(data.iloc[i]["skills"]).split(",")
        ]

        matched_skills = []
        missing_skills = []

        for skill in job_skills:
            pattern = r"\b" + re.escape(skill) + r"\b"

            if re.search(pattern, resume_text):
                matched_skills.append(skill)
            else:
                missing_skills.append(skill)

        match_score = len(matched_skills)
        total = len(job_skills)

        summary = "This job role requires relevant technical knowledge, problem-solving skills, and practical experience."

        if "summary" in data.columns:
            summary = data.iloc[i]["summary"]

        if score > 0 or match_score > 0:
            results.append({
                "title": job_title,
                "salary": salary,
                "match": match_score,
                "total": total,
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "summary": summary
            })

    results = sorted(results, key=lambda x: (x["match"], x["salary"]), reverse=True)

    return results[:5]


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["resume"]

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        raw_text = extract_raw_text(file_path)
        resume_text = clean_text(raw_text)

        skills = get_detected_skills(raw_text)
        results = smart_match(resume_text)

        best_job = results[0] if results else None

        return render_template(
            "result.html",
            results=results,
            best_job=best_job,
            skills=skills
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
