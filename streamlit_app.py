import re
import warnings

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------
# Self-contained skill vocabulary (no external taxonomy files, so the app is portable)
# ----------------------------------------------------------------------------------
SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "scala", "go", "r", "sql",
    "html", "css", "bash", "machine learning", "deep learning", "data analysis",
    "data analytics", "data science", "nlp", "natural language processing",
    "computer vision", "statistics", "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "spark", "hadoop", "etl", "aws", "azure", "google cloud", "docker",
    "kubernetes", "jenkins", "ci/cd", "linux", "git", "devops", "react", "angular", "vue",
    "node.js", "django", "flask", "rest api", "microservices", "postgresql", "mysql",
    "mongodb", "redis", "neo4j", "power bi", "tableau", "excel", "sas", "agile", "scrum",
    "project management", "stakeholder management", "jira", "communication",
    "leadership", "teamwork", "problem solving", "financial analysis", "accounting",
    "budgeting", "auditing", "taxation", "recruitment", "onboarding", "payroll",
    "employee relations", "performance management", "salesforce", "crm", "seo",
    "digital marketing", "adobe photoshop", "figma", "ux design", "autocad", "six sigma",
]

# ----------------------------------------------------------------------------------
# Personalised learning recommendations: a missing skill maps to a named course and link
# ----------------------------------------------------------------------------------
LEARNING_RESOURCES = {
    "python": "Python for Everybody (Coursera): https://www.coursera.org/specializations/python",
    "java": "Java Programming (Coursera, Duke): https://www.coursera.org/specializations/java-programming",
    "javascript": "JavaScript curriculum (freeCodeCamp): https://www.freecodecamp.org/learn",
    "typescript": "TypeScript Handbook (official): https://www.typescriptlang.org/docs/handbook/intro.html",
    "c++": "C++ tutorials (LearnCpp): https://www.learncpp.com",
    "c#": "C# guide (Microsoft Learn): https://learn.microsoft.com/dotnet/csharp/",
    "r": "Data Science with R (Coursera, JHU): https://www.coursera.org/specializations/jhu-data-science",
    "sql": "SQL for Data Science (Coursera): https://www.coursera.org/learn/sql-for-data-science",
    "html": "Responsive Web Design (freeCodeCamp): https://www.freecodecamp.org/learn",
    "css": "Responsive Web Design (freeCodeCamp): https://www.freecodecamp.org/learn",
    "machine learning": "Machine Learning Specialisation (Coursera, DeepLearning.AI): https://www.coursera.org/specializations/machine-learning-introduction",
    "deep learning": "Deep Learning Specialisation (Coursera, DeepLearning.AI): https://www.coursera.org/specializations/deep-learning",
    "data analysis": "Data Analysis with Python (freeCodeCamp): https://www.freecodecamp.org/learn/data-analysis-with-python/",
    "data analytics": "Google Data Analytics Certificate (Coursera): https://www.coursera.org/professional-certificates/google-data-analytics",
    "data science": "IBM Data Science Certificate (Coursera): https://www.coursera.org/professional-certificates/ibm-data-science",
    "nlp": "NLP Specialisation (Coursera, DeepLearning.AI): https://www.coursera.org/specializations/natural-language-processing",
    "natural language processing": "NLP Specialisation (Coursera, DeepLearning.AI): https://www.coursera.org/specializations/natural-language-processing",
    "computer vision": "OpenCV courses (official): https://opencv.org/courses/",
    "statistics": "Introduction to Statistics (Coursera, Stanford): https://www.coursera.org/learn/stanford-statistics",
    "tensorflow": "TensorFlow tutorials (official): https://www.tensorflow.org/tutorials",
    "pytorch": "PyTorch tutorials (official): https://pytorch.org/tutorials/",
    "keras": "Keras developer guides (official): https://keras.io/guides/",
    "scikit-learn": "scikit-learn tutorials (official): https://scikit-learn.org/stable/tutorial/index.html",
    "pandas": "pandas getting started (official): https://pandas.pydata.org/docs/getting_started/index.html",
    "numpy": "NumPy quickstart (official): https://numpy.org/doc/stable/user/quickstart.html",
    "spark": "Apache Spark documentation (official): https://spark.apache.org/docs/latest/",
    "hadoop": "Hadoop documentation (official): https://hadoop.apache.org/docs/stable/",
    "etl": "ETL and Data Pipelines (Coursera, IBM): https://www.coursera.org/learn/etl-and-data-pipelines-shell-airflow-kafka",
    "aws": "AWS Cloud Practitioner (AWS Skill Builder): https://explore.skillbuilder.aws/",
    "azure": "Azure Fundamentals (Microsoft Learn): https://learn.microsoft.com/training/paths/azure-fundamentals/",
    "google cloud": "Google Cloud training (official): https://cloud.google.com/learn/training",
    "docker": "Docker getting started (official): https://docs.docker.com/get-started/",
    "kubernetes": "Kubernetes Basics (official): https://kubernetes.io/docs/tutorials/kubernetes-basics/",
    "jenkins": "Jenkins documentation (official): https://www.jenkins.io/doc/",
    "ci/cd": "CI/CD with GitHub Actions (GitHub docs): https://docs.github.com/actions",
    "linux": "The Linux Commands Handbook (freeCodeCamp): https://www.freecodecamp.org/news/the-linux-commands-handbook/",
    "git": "Pro Git book (official, free): https://git-scm.com/book",
    "devops": "DevOps Culture and Mindset (Coursera): https://www.coursera.org/learn/devops-culture-and-mindset",
    "react": "React documentation (official): https://react.dev/learn",
    "angular": "Angular tutorials (official): https://angular.dev/tutorials",
    "vue": "Vue.js guide (official): https://vuejs.org/guide/introduction.html",
    "node.js": "Node.js learn (official): https://nodejs.org/en/learn",
    "django": "Django tutorial (official): https://docs.djangoproject.com/en/stable/intro/tutorial01/",
    "flask": "Flask tutorial (official): https://flask.palletsprojects.com/tutorial/",
    "rest api": "REST API design (Microsoft guidelines): https://github.com/microsoft/api-guidelines",
    "microservices": "Microservices guide (martinfowler.com): https://martinfowler.com/microservices/",
    "postgresql": "PostgreSQL tutorial (official): https://www.postgresql.org/docs/current/tutorial.html",
    "mysql": "MySQL tutorial (official): https://dev.mysql.com/doc/mysql-tutorial-excerpt/en/",
    "mongodb": "MongoDB University (official): https://learn.mongodb.com/",
    "redis": "Redis University (official): https://university.redis.com/",
    "neo4j": "Neo4j GraphAcademy (official): https://graphacademy.neo4j.com/",
    "power bi": "Power BI learning path (Microsoft Learn): https://learn.microsoft.com/training/powerplatform/power-bi",
    "tableau": "Tableau free training (official): https://www.tableau.com/learn/training",
    "excel": "Excel training (Microsoft): https://support.microsoft.com/excel",
    "sas": "SAS training (official): https://www.sas.com/en_us/training.html",
    "agile": "Agile coach resources (Atlassian): https://www.atlassian.com/agile",
    "scrum": "The Scrum Guide (official): https://scrumguides.org/",
    "project management": "Google Project Management Certificate (Coursera): https://www.coursera.org/professional-certificates/google-project-management",
    "stakeholder management": "Google Project Management Certificate (Coursera): https://www.coursera.org/professional-certificates/google-project-management",
    "jira": "Jira guides (Atlassian): https://www.atlassian.com/software/jira/guides",
    "communication": "Improving Communication Skills (Coursera, Wharton): https://www.coursera.org/learn/wharton-communication-skills",
    "leadership": "Leading Teams (Coursera, Michigan): https://www.coursera.org/specializations/leading-teams",
    "teamwork": "Teamwork Skills (Coursera, Colorado): https://www.coursera.org/learn/teamwork-skills-effective-communication",
    "problem solving": "Creative Problem Solving (Coursera, Minnesota): https://www.coursera.org/learn/creative-problem-solving",
    "financial analysis": "Financial analysis resources (Corporate Finance Institute): https://corporatefinanceinstitute.com/",
    "accounting": "Introduction to Financial Accounting (Coursera, Wharton): https://www.coursera.org/learn/wharton-accounting",
    "budgeting": "Finance for non-financial professionals (Coursera): https://www.coursera.org/learn/finance-for-non-finance-managers",
    "auditing": "Auditing courses (Coursera): https://www.coursera.org/search?query=auditing",
    "taxation": "US Federal Taxation (Coursera, Illinois): https://www.coursera.org/specializations/united-states-federal-taxation",
    "recruitment": "Recruiting, Hiring and Onboarding (Coursera, Minnesota): https://www.coursera.org/learn/recruiting-hiring-onboarding-employees",
    "onboarding": "Recruiting, Hiring and Onboarding (Coursera, Minnesota): https://www.coursera.org/learn/recruiting-hiring-onboarding-employees",
    "payroll": "Payroll and tax fundamentals (Coursera): https://www.coursera.org/search?query=payroll",
    "employee relations": "Human Resource Management (Coursera, Minnesota): https://www.coursera.org/specializations/human-resource-management",
    "performance management": "Managing Employee Performance (Coursera, Minnesota): https://www.coursera.org/learn/managing-employee-performance",
    "salesforce": "Salesforce Trailhead (official): https://trailhead.salesforce.com/",
    "crm": "Salesforce Trailhead (official): https://trailhead.salesforce.com/",
    "seo": "SEO Specialisation (Coursera, UC Davis): https://www.coursera.org/specializations/seo",
    "digital marketing": "Google Digital Marketing Certificate (Coursera): https://www.coursera.org/professional-certificates/google-digital-marketing-ecommerce",
    "adobe photoshop": "Photoshop tutorials (Adobe, official): https://helpx.adobe.com/photoshop/tutorials.html",
    "figma": "Figma learn (official): https://help.figma.com/hc/en-us/categories/360002051613",
    "ux design": "Google UX Design Certificate (Coursera): https://www.coursera.org/professional-certificates/google-ux-design",
    "autocad": "AutoCAD tutorials (Autodesk, official): https://www.autodesk.com/learn",
    "six sigma": "Six Sigma Specialisation (Coursera): https://www.coursera.org/specializations/six-sigma-green-belt",
}

def recommend(skill):
    return LEARNING_RESOURCES.get(
        skill,
        "a structured online course (Coursera, edX or LinkedIn Learning) plus the official documentation",
    )

# ----------------------------------------------------------------------------------
# Text cleaning and rule-based extraction of known skills, experience and qualifications
# ----------------------------------------------------------------------------------
EMAIL_RE = re.compile(r"\S+@\S+")
URL_RE = re.compile(r"http\S+|www\.\S+")
PHONE_RE = re.compile(r"\+?\d[\d\-\(\)\s]{7,}\d")

def clean(text):
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    text = re.sub(r"\.(?!\w)", " ", text)   # sentence-ending periods become spaces; node.js is preserved
    return re.sub(r"\s+", " ", text).strip()

_skill_pats = sorted(SKILLS, key=len, reverse=True)

def extract(text):
    low = clean(text)
    found, occ = set(), set()
    for sk in _skill_pats:
        m = re.search(r"(?<![a-z0-9+#.])" + re.escape(sk) + r"(?![a-z0-9+#.])", low)
        if m:
            span = range(m.start(), m.end())
            if not any(p in occ for p in span):
                found.add(sk)
                occ.update(span)
    return sorted(found)

YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:years|yrs|year)\b(?!\s+(?:degree|programme|old))", re.I)
RANGE_RE = re.compile(r"(19|20)\d{2}\s*(?:-|to)\s*((?:19|20)\d{2}|present|current)", re.I)
QUAL_LEVELS = [
    (("phd", "doctorate", "doctoral"), 1.0, "Doctorate"),
    (("master", "msc", "mba", "postgraduate"), 0.8, "Master's"),
    (("bachelor", "bsc", "b.tech", "undergraduate"), 0.6, "Bachelor's"),
    (("diploma", "associate"), 0.4, "Diploma"),
    (("high school", "secondary"), 0.2, "High school"),
]
CERT_CUES = ("certified", "certification", "certificate", "pmp", "cpa", "cfa", "scrum master", "comptia")

def exp_years(text):
    low = text.lower()
    cands = [int(m) for m in YEARS_RE.findall(low)]
    for m in RANGE_RE.finditer(low):
        s = int(m.group(0)[:4])
        t = m.group(2)
        cands.append(min((2025 if not t[:2].isdigit() else int(t[:4])) - s, 40))
    return min(max(cands) if cands else 0, 40)

def qual_score(text):
    low = text.lower()
    for cues, sc, lbl in QUAL_LEVELS:
        if any(c in low for c in cues):
            return sc, lbl
    return 0.3, "Unspecified"

def cert_count(text):
    low = text.lower()
    return sum(1 for c in CERT_CUES if c in low)

# ----------------------------------------------------------------------------------
# Sentence-BERT embeddings (the semantic core of the system), loaded once and cached
# ----------------------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TRANSFERABLE_THRESHOLD = 0.45   # a requirement is transferable if a held skill is at least this close

@st.cache_resource(show_spinner=False)
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource(show_spinner=False)
def skill_matrix():
    model = load_model()
    vecs = model.encode(SKILLS, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")

SKILL_INDEX = {s: i for i, s in enumerate(SKILLS)}

def embed_text(text):
    model = load_model()
    v = model.encode([str(text)], normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, dtype="float32")[0]

def semantic_doc_match(resume_text, jd_text):
    # cosine similarity between the whole resume and the whole job description
    rv = embed_text(clean(resume_text))
    jv = embed_text(clean(jd_text))
    return float(np.dot(rv, jv))

def classify_requirements(cand_skills, req_skills, smat):
    # split each required skill into matched (held), transferable (semantically close), or missing
    cand_set = set(cand_skills)
    cand_in = [s for s in cand_skills if s in SKILL_INDEX]
    cand_vecs = np.array([smat[SKILL_INDEX[s]] for s in cand_in]) if cand_in else None
    matched, transferable, missing = [], [], []
    for r in req_skills:
        if r in cand_set:
            matched.append(r)
        elif cand_in and r in SKILL_INDEX:
            sims = cand_vecs @ smat[SKILL_INDEX[r]]
            j = int(np.argmax(sims))
            best = float(sims[j])
            if best >= TRANSFERABLE_THRESHOLD:
                transferable.append((r, cand_in[j], best))
            else:
                missing.append(r)
        else:
            missing.append(r)
    return matched, transferable, missing

# ----------------------------------------------------------------------------------
# Career Readiness Score: weighted blend of skill, experience, qualification, certification
# ----------------------------------------------------------------------------------
WEIGHTS = {"skill": 0.45, "exp": 0.25, "qual": 0.20, "cert": 0.10}
EXP_TGT, CERT_TGT = 8, 3

def readiness_components(sa, yrs, qs, cc):
    return {
        "Skills": 100 * WEIGHTS["skill"] * sa,
        "Experience": 100 * WEIGHTS["exp"] * min(yrs / EXP_TGT, 1),
        "Qualification": 100 * WEIGHTS["qual"] * qs,
        "Certifications": 100 * WEIGHTS["cert"] * min(cc / CERT_TGT, 1),
    }

def skill_alignment(matched, transferable, req_skills):
    # transferable skills count as half credit, which is where the embeddings feed the score
    if not req_skills:
        return 0.0
    return (len(matched) + 0.5 * len(transferable)) / len(req_skills)

# ----------------------------------------------------------------------------------
# User interface
# ----------------------------------------------------------------------------------
st.set_page_config(page_title="Resume Intelligence System", layout="wide")
st.title("AI-Powered Resume Intelligence System")
st.caption("MSc Data Analytics Dissertation")

with st.expander("How the matching works"):
    st.write(
        "Skills are read from the text, then compared by meaning using Sentence-BERT "
        "(all-MiniLM-L6-v2) sentence embeddings rather than exact keywords. A required skill the "
        "candidate does not hold is flagged as transferable when a skill they do hold is "
        "semantically close to it (cosine similarity at or above {:.2f}); otherwise it is a genuine "
        "gap. The readiness score combines skill alignment, experience, qualification and "
        "certifications.".format(TRANSFERABLE_THRESHOLD)
    )

# Load the model up front so the first interaction is responsive and any failure is explicit.
try:
    _ = skill_matrix()
    _model_ready = True
except Exception as exc:
    _model_ready = False
    st.error(
        "The Sentence-BERT model could not be loaded in this environment "
        f"({type(exc).__name__}). The app needs the 'sentence-transformers' package, which is "
        "listed in requirements.txt; on first launch the model is downloaded once and then cached."
    )

mode = st.radio(
    "Mode",
    ["Analyse a candidate resume", "Match resume to a job description"],
    horizontal=True,
)

if mode == "Analyse a candidate resume":
    resume_text = st.text_area("Paste the candidate resume here", height=280)
    jd_text = st.text_area("Paste the target job description (for gap analysis)", height=150)

    if st.button("Analyse") and _model_ready and resume_text.strip():
        smat = skill_matrix()
        cand_skills = extract(resume_text)
        req_skills = extract(jd_text) if jd_text.strip() else []

        matched, transferable, missing = classify_requirements(cand_skills, req_skills, smat)
        sa = skill_alignment(matched, transferable, req_skills)
        yrs = exp_years(resume_text)
        qs, ql = qual_score(resume_text)
        cc = cert_count(resume_text)
        comps = readiness_components(sa, yrs, qs, cc)
        score = sum(comps.values())
        sem = semantic_doc_match(resume_text, jd_text) if jd_text.strip() else None

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Career Readiness", f"{score:.1f}/100")
        c2.metric("Semantic match", f"{sem * 100:.0f}%" if sem is not None else "Add a JD")
        c3.metric("Skill alignment", f"{sa * 100:.0f}%" if req_skills else "Add a JD")
        c4.metric("Years experience", str(yrs))
        st.progress(min(int(score), 100))

        left, right = st.columns(2)
        with left:
            st.subheader("Readiness breakdown")
            st.bar_chart(pd.DataFrame({"Points": comps}))
        with right:
            st.subheader("Skill coverage")
            coverage = pd.DataFrame({
                "Category": ["Matched", "Transferable", "Missing"],
                "Skills": [len(matched), len(transferable), len(missing)],
            })
            coverage_chart = (
                alt.Chart(coverage)
                .mark_bar()
                .encode(
                    x=alt.X("Category:N", sort=["Matched", "Transferable", "Missing"], title=None),
                    y=alt.Y("Skills:Q", title="Skills"),
                    color=alt.Color(
                        "Category:N",
                        scale=alt.Scale(
                            domain=["Matched", "Transferable", "Missing"],
                            range=["#2e7d32", "#ef6c00", "#c62828"],
                        ),
                        legend=None,
                    ),
                    tooltip=["Category", "Skills"],
                )
            )
            st.altair_chart(coverage_chart, use_container_width=True)

        st.subheader("Extracted skills")
        st.write(", ".join(cand_skills) if cand_skills else "None detected")

        if req_skills:
            st.subheader("Matched requirements")
            st.write(", ".join(matched) if matched else "None held directly")

            st.subheader("Transferable skills")
            if transferable:
                st.write("Requirements the candidate does not hold, but has a related skill for:")
                for req, held, sim in sorted(transferable, key=lambda x: -x[2]):
                    st.write(f"- {req}: already has **{held}** (similarity {sim:.2f})")
            else:
                st.write("No transferable skills detected for the missing requirements.")

            st.subheader("Missing skills and recommended learning")
            if missing:
                for sk in missing:
                    st.markdown(f"- **{sk}** -> {recommend(sk)}")
            else:
                st.write("No genuine gaps - the candidate covers the requirements.")
        else:
            st.info("Paste a job description above to see matched, transferable and missing skills with recommendations.")

        st.subheader("Profile")
        st.write(f"Qualification: **{ql}**  |  Certifications detected: **{cc}**")

elif mode == "Match resume to a job description":
    jd_text2 = st.text_area("Paste the job description", height=200)
    resumes_input = st.text_area("Paste multiple resumes, separated by '---'", height=320)

    if st.button("Rank candidates") and _model_ready and jd_text2.strip() and resumes_input.strip():
        smat = skill_matrix()
        jd_skills = extract(jd_text2)
        jd_vec = embed_text(clean(jd_text2))

        rows = []
        for i, rt in enumerate(resumes_input.split("---")):
            rt = rt.strip()
            if not rt:
                continue
            cs = extract(rt)
            matched, transferable, missing = classify_requirements(cs, jd_skills, smat)
            sa = skill_alignment(matched, transferable, jd_skills)
            yrs = exp_years(rt)
            qs, ql = qual_score(rt)
            cc = cert_count(rt)
            score = sum(readiness_components(sa, yrs, qs, cc).values())
            sem = float(np.dot(embed_text(clean(rt)), jd_vec))
            rows.append({
                "Candidate": f"#{i + 1}",
                "Readiness": round(score, 1),
                "Semantic match %": round(sem * 100, 0),
                "Matched": len(matched),
                "Transferable": len(transferable),
                "Missing": len(missing),
                "Exp yrs": yrs,
                "Qualification": ql,
            })

        if rows:
            df = pd.DataFrame(rows).sort_values("Readiness", ascending=False).reset_index(drop=True)
            st.subheader("Ranked candidates")
            st.dataframe(df, use_container_width=True)
            st.subheader("Readiness by candidate")
            st.bar_chart(df.set_index("Candidate")[["Readiness"]])
            st.subheader("Semantic match to the job description")
            st.bar_chart(df.set_index("Candidate")[["Semantic match %"]])
