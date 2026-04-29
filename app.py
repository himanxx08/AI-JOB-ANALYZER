from flask import Flask, render_template, request
import os
import re
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

data = pd.read_csv("jobs.csv")


def extract_raw_text(file_path):
    text = ""

    reader = PdfReader(file_path)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9+#. ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_only_skills(raw_text):
    text = raw_text.replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    skill_heading_pattern = re.compile(
        r"^(skills|technical skills|key skills|core skills|professional skills|computer skills|it skills|tools|technologies)\s*:?\s*(.*)$",
        re.IGNORECASE
    )

    stop_heading_pattern = re.compile(
        r"^(summary|profile|objective|career objective|education|qualification|academic|experience|work experience|employment|projects|project|certifications|certificate|achievements|languages|hobbies|personal details|declaration|contact|address|role)\s*:?",
        re.IGNORECASE
    )

    skills_text = ""

    for i, line in enumerate(lines):
        match = skill_heading_pattern.match(line)

        if match:
            # same line me skills ho to lo
            same_line_skills = match.group(2).strip()
            if same_line_skills:
                skills_text += same_line_skills + " "

            # next lines lo jab tak next section heading na aa jaye
            for next_line in lines[i + 1:]:
                if stop_heading_pattern.match(next_line):
                    break

                skills_text += next_line + " "

            break

    if not skills_text.strip():
        return []

    # brackets ke andar explanation remove
    skills_text = re.sub(r"\(.*?\)", "", skills_text)
    skills_text = re.sub(r"\[.*?\]", "", skills_text)

    # separators normalize
    for sep in ["•", "●", "▪", "·", "|", "/", ";", "\n", "\t"]:
        skills_text = skills_text.replace(sep, ",")

    # agar bullet nahi hai aur comma bhi nahi hai, multiple spaces ko comma mat banao
    raw_skills = skills_text.split(",")

    final_skills = []

    for skill in raw_skills:
        skill = skill.strip()

        skill = re.sub(r"^[\-–—]+", "", skill)
        skill = re.sub(r"[^a-zA-Z0-9+#. ]", "", skill)
        skill = re.sub(r"\s+", " ", skill).strip()

        if not skill:
            continue

        # long sentence remove
        if len(skill.split()) > 5:
            continue

        # garbage words remove
        if skill.lower() in [
            "skills", "technical skills", "key skills", "tools",
            "technologies", "etc", "and", "or"
        ]:
            continue

        skill = skill.title()

        if skill not in final_skills:
            final_skills.append(skill)

    return final_skills


def smart_match(resume_text):
    results = []

    job_texts = data["skills"].astype(str).str.lower().tolist()
    corpus = [resume_text] + job_texts

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]

    for i, score in enumerate(scores):
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

        if score > 0 or matched_skills:
            results.append({
                "title": data.iloc[i]["job_title"],
                "salary": data.iloc[i]["salary"],
                "match": len(matched_skills),
                "total": len(job_skills),
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "summary": data.iloc[i]["summary"] if "summary" in data.columns else ""
            })

    results = sorted(results, key=lambda x: x["match"], reverse=True)

    return results[:5]


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["resume"]

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        raw_text = extract_raw_text(file_path)
        resume_text = clean_text(raw_text)

        skills = extract_only_skills(raw_text)
        results = smart_match(resume_text)

        best_job = results[0] if results else None

        return render_template(
            "result.html",
            skills=skills,
            results=results,
            best_job=best_job
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
