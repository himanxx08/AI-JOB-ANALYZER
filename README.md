# 🤖 AI Job Analyzer

AI Job Analyzer is a web-based application developed by Himanshu Sharma that analyzes a user's resume and recommends suitable job roles based on detected skills using NLP techniques.

---

## 🚀 Features

- 📄 Upload Resume (PDF)
- 🧠 Automatic Skill Detection
- 💼 Job Recommendation System
- 🎯 Match Score Calculation
- ❌ Missing Skills Identification
- 📊 Clean UI Dashboard
- 🌐 Deployable on Cloud (Render)

---

## 🛠️ Technologies Used

- Python (Flask)
- Pandas
- Scikit-learn (TF-IDF, Cosine Similarity)
- PyPDF2 (PDF Reader)
- HTML, CSS, Bootstrap

---

## ⚙️ How It Works

1. User uploads a resume (PDF format)
2. System extracts text from resume
3. Skills are detected automatically
4. Resume is compared with job dataset
5. Best matching jobs are displayed with:
   - Match Score
   - Matched Skills
   - Missing Skills

---

## 📦 Installation

```bash
pip install -r requirements.txt
