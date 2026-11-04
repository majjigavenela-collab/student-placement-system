import sqlite3
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash


PROJECT_FOLDER = Path(__file__).resolve().parent
TEMPLATE_FOLDER = PROJECT_FOLDER / "templates"
DATABASE = PROJECT_FOLDER / "database" / "database.db"
MODEL_FILE = PROJECT_FOLDER / "model" / "placement_model.pkl"

DATABASE.parent.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(TEMPLATE_FOLDER))
app.secret_key = "student-placement-project-key"

# Global model cache to avoid re-reading disk on every request
_cached_model = None


def get_model():
    """Load and return the trained ML pipeline from disk."""
    global _cached_model
    if _cached_model is None:
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_FILE}")
        _cached_model = joblib.load(MODEL_FILE)
    return _cached_model


@contextmanager
def get_db_connection():
    """Create and return an SQLite database connection context that commits and closes cleanly."""
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def create_tables():
    """Initialize database tables and apply backward-compatible schema migrations."""
    with sqlite3.connect(DATABASE) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS student_profiles (
                user_id INTEGER PRIMARY KEY,
                cgpa REAL,
                internships INTEGER,
                projects INTEGER,
                certifications INTEGER,
                aptitude_score INTEGER,
                soft_skills_score INTEGER,
                major_projects INTEGER,
                workshops_certifications INTEGER,
                mini_projects INTEGER,
                skills INTEGER,
                communication_skill_rating REAL,
                internship TEXT,
                hackathon TEXT,
                twelfth_percentage REAL,
                tenth_percentage REAL,
                backlogs INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # Add any missing profile columns without modifying or wiping existing data
        existing_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(student_profiles)")
        }
        new_columns = {
            "major_projects": "INTEGER",
            "workshops_certifications": "INTEGER",
            "mini_projects": "INTEGER",
            "skills": "INTEGER",
            "communication_skill_rating": "REAL",
            "internship": "TEXT",
            "hackathon": "TEXT",
            "twelfth_percentage": "REAL",
            "tenth_percentage": "REAL",
            "backlogs": "INTEGER",
        }
        for column, column_type in new_columns.items():
            if column not in existing_columns:
                connection.execute(
                    f"ALTER TABLE student_profiles ADD COLUMN {column} {column_type}"
                )

        # Prediction history table for logging every student prediction
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                prediction_result TEXT NOT NULL,
                probability REAL NOT NULL,
                cgpa REAL,
                major_projects INTEGER,
                workshops_certifications INTEGER,
                mini_projects INTEGER,
                skills INTEGER,
                communication_skill_rating REAL,
                internship TEXT,
                hackathon TEXT,
                twelfth_percentage REAL,
                tenth_percentage REAL,
                backlogs INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # Student technical skills table for skill gap analysis and persistence
        connection.execute("""
            CREATE TABLE IF NOT EXISTS student_skills (
                user_id INTEGER PRIMARY KEY,
                python_score INTEGER DEFAULT 60,
                sql_score INTEGER DEFAULT 50,
                html_css_score INTEGER DEFAULT 70,
                git_score INTEGER DEFAULT 40,
                dsa_score INTEGER DEFAULT 45,
                api_score INTEGER DEFAULT 35,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)


create_tables()


def login_required(view_func):
    """Decorator to require login for student-only pages."""
    @wraps(view_func)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.")
            return redirect(f"/login?next={request.path}")
        return view_func(*args, **kwargs)
    return decorated_function


# Profile required fields and labels
PROFILE_FIELDS = [
    ("cgpa", "CGPA"),
    ("major_projects", "Major Projects"),
    ("workshops_certifications", "Workshops / Certifications"),
    ("mini_projects", "Mini Projects"),
    ("skills", "Skills"),
    ("communication_skill_rating", "Communication Skill Rating"),
    ("internship", "Internship"),
    ("hackathon", "Hackathon"),
    ("twelfth_percentage", "12th Percentage"),
    ("tenth_percentage", "10th Percentage"),
    ("backlogs", "Backlogs"),
]


def get_profile_completion(profile_dict):
    """Calculate profile completion percentage and identify missing required fields."""
    if not profile_dict:
        return {
            "percent": 0,
            "is_complete": False,
            "missing_fields": [label for _, label in PROFILE_FIELDS],
            "filled_count": 0,
            "total_count": len(PROFILE_FIELDS),
        }
    missing = []
    filled = 0
    for key, label in PROFILE_FIELDS:
        val = profile_dict.get(key)
        if val is None or str(val).strip() == "":
            missing.append(label)
        else:
            filled += 1
    percent = round((filled / len(PROFILE_FIELDS)) * 100)
    return {
        "percent": percent,
        "is_complete": (filled == len(PROFILE_FIELDS)),
        "missing_fields": missing,
        "filled_count": filled,
        "total_count": len(PROFILE_FIELDS),
    }


# Technical skill definitions matching skill gap analysis
SKILL_SPECS = [
    {
        "id": "python_score",
        "name": "Python",
        "benchmark": 70,
        "default": 60,
        "recommendation": "Build a small Python application and practice functions, files, and error handling.",
    },
    {
        "id": "sql_score",
        "name": "SQL",
        "benchmark": 70,
        "default": 50,
        "recommendation": "Practice SELECT, JOIN, GROUP BY, and database design with a small project.",
    },
    {
        "id": "html_css_score",
        "name": "HTML/CSS",
        "benchmark": 70,
        "default": 70,
        "recommendation": "Create and style a responsive multi-page website.",
    },
    {
        "id": "git_score",
        "name": "Git/GitHub",
        "benchmark": 70,
        "default": 40,
        "recommendation": "Practice commits, branches, and uploading a project to GitHub.",
    },
    {
        "id": "dsa_score",
        "name": "Data Structures and Algorithms",
        "benchmark": 70,
        "default": 45,
        "recommendation": "Practice arrays, strings, sorting, and problem solving.",
    },
    {
        "id": "api_score",
        "name": "REST APIs",
        "benchmark": 70,
        "default": 35,
        "recommendation": "Build a Flask API and practice sending and receiving JSON.",
    },
]


def get_student_skill_analysis(connection, user_id):
    """Retrieve saved student skills and perform skill gap analysis against target benchmarks."""
    row = connection.execute(
        "SELECT * FROM student_skills WHERE user_id = ?", (user_id,)
    ).fetchone()

    scores = dict(row) if row else {spec["id"]: spec["default"] for spec in SKILL_SPECS}

    analyzed_skills = []
    weak_skills = []
    total_score = 0

    for spec in SKILL_SPECS:
        score = scores.get(spec["id"], spec["default"])
        try:
            score = int(score)
        except (ValueError, TypeError):
            score = spec["default"]

        total_score += score
        is_weak = score < spec["benchmark"]
        item = {
            "id": spec["id"],
            "name": spec["name"],
            "score": score,
            "benchmark": spec["benchmark"],
            "is_weak": is_weak,
            "recommendation": spec["recommendation"],
            "gap": max(0, spec["benchmark"] - score),
        }
        analyzed_skills.append(item)
        if is_weak:
            weak_skills.append(item)

    avg_score = round(total_score / len(SKILL_SPECS), 1) if SKILL_SPECS else 0

    return {
        "skills": analyzed_skills,
        "weak_skills": weak_skills,
        "weak_count": len(weak_skills),
        "total_skills": len(SKILL_SPECS),
        "average_score": avg_score,
        "summary": f"{len(weak_skills)} of {len(SKILL_SPECS)} skills are below the 70% placement benchmark.",
    }


def get_profile_insights(profile):
    """Generate practical readiness recommendations based on saved academic profile metrics."""
    if not profile:
        return []

    insights = []

    # Internships
    if profile.get("internship") == "Yes":
        insights.append({
            "title": "Internship Experience",
            "badge": "Strong",
            "status": "good",
            "message": "Having completed an internship gives you a practical advantage during campus recruitment.",
        })
    else:
        insights.append({
            "title": "Internship Experience",
            "badge": "Action Needed",
            "status": "warn",
            "message": "No industry internship recorded. Seek virtual or summer internships to build industry credibility.",
        })

    # Hackathon
    if profile.get("hackathon") == "Yes":
        insights.append({
            "title": "Hackathon Participation",
            "badge": "Strong",
            "status": "good",
            "message": "Hackathon participation proves problem solving, team collaboration, and rapid development capabilities.",
        })
    else:
        insights.append({
            "title": "Hackathon Participation",
            "badge": "Recommended",
            "status": "warn",
            "message": "Participating in college or online hackathons provides great talking points during interviews.",
        })

    # Backlogs
    backlogs = profile.get("backlogs")
    if backlogs is not None:
        if backlogs == 0:
            insights.append({
                "title": "Academic Standing (Backlogs)",
                "badge": "Clear (0)",
                "status": "good",
                "message": "0 standing backlogs satisfies the zero-arrears criteria enforced by major recruitment drives.",
            })
        else:
            insights.append({
                "title": "Academic Standing (Backlogs)",
                "badge": f"{backlogs} Backlog(s)",
                "status": "warn",
                "message": f"You have {backlogs} standing backlog(s). Many companies require zero backlogs at recruitment time.",
            })

    # Projects
    major = profile.get("major_projects", 0) or 0
    mini = profile.get("mini_projects", 0) or 0
    if major + mini >= 3:
        insights.append({
            "title": "Project Portfolio",
            "badge": f"{major + mini} Projects",
            "status": "good",
            "message": "Solid project portfolio. Keep your code repositories documented and deployment links active.",
        })
    else:
        insights.append({
            "title": "Project Portfolio",
            "badge": "Build More",
            "status": "warn",
            "message": "Build at least 1 end-to-end major project and multiple mini projects to showcase technical skills.",
        })

    # Communication
    comm = profile.get("communication_skill_rating")
    if comm is not None:
        if comm >= 3.5:
            insights.append({
                "title": "Communication Rating",
                "badge": f"{comm} / 5.0",
                "status": "good",
                "message": "Good communication rating for HR discussions and technical explanation rounds.",
            })
        else:
            insights.append({
                "title": "Communication Rating",
                "badge": f"{comm} / 5.0",
                "status": "warn",
                "message": "Practice mock interviews, technical presentations, and group discussions to improve confidence.",
            })

    return insights


def generate_prediction_explanation(values, predicted_status, probability):
    """Generate a clear explanation of placement prediction results highlighting key factors."""
    strengths = []
    improvements = []

    # CGPA evaluation
    cgpa = values["CGPA"]
    if cgpa >= 8.0:
        strengths.append(f"Strong academic performance (CGPA: {cgpa:.2f} / 10).")
    elif cgpa < 7.0:
        improvements.append(f"CGPA is {cgpa:.2f}; many campus recruiters require 7.0 or higher.")

    # Projects
    total_projects = values["Major Projects"] + values["Mini Projects"]
    if total_projects >= 3:
        strengths.append(f"Solid project experience ({values['Major Projects']} major, {values['Mini Projects']} mini projects).")
    elif values["Major Projects"] == 0:
        improvements.append("No major projects listed; complete at least one end-to-end capstone project.")

    # Certifications / Workshops
    workshops = values["Workshops/Certificatios"]
    if workshops >= 2:
        strengths.append(f"Active participation in certifications and workshops ({workshops} completed).")
    elif workshops == 0:
        improvements.append("Earning recognized industry certifications helps validate your technical skills.")

    # Technical Skills
    skills_score = values["Skills"]
    if skills_score >= 7:
        strengths.append(f"High technical skill rating ({skills_score}/10).")
    elif skills_score < 5:
        improvements.append(f"Current technical skill score is {skills_score}/10; prioritize core programming and domain skills.")

    # Communication Skill
    comm = values["Communication Skill Rating"]
    if comm >= 4.0:
        strengths.append(f"High communication rating ({comm:.1f} / 5.0) for technical and HR rounds.")
    elif comm < 3.0:
        improvements.append(f"Communication rating is {comm:.1f} / 5.0; practice mock interviews to build fluency.")

    # Internship
    if values["Internship"] == "Yes":
        strengths.append("Valuable real-world industry experience through internship.")
    else:
        improvements.append("Complete an internship or industry-sponsored project to build hands-on experience.")

    # Hackathon
    if values["Hackathon"] == "Yes":
        strengths.append("Hackathon experience demonstrates collaborative problem solving and agile coding.")
    else:
        improvements.append("Participating in hackathons or coding challenges boosts your practical exposure.")

    # Backlogs
    backlogs = values["backlogs"]
    if backlogs == 0:
        strengths.append("Zero active backlogs ensures eligibility for all standard campus recruitment criteria.")
    else:
        improvements.append(f"Active backlogs ({backlogs}) are a critical filter for eligibility in most companies.")

    if predicted_status == 1:
        summary = (
            f"The model predicts Placed with an estimated {probability}% placement probability. "
            "Your profile shows strong competitive indicators for campus placement selection."
        )
    else:
        risk = round(100.0 - probability, 1)
        summary = (
            f"The model predicts Not Placed with an estimated {probability}% placement probability "
            f"({risk}% risk factor). Addressing the recommended improvement areas below will significantly enhance your placement readiness."
        )

    return {
        "summary": summary,
        "strengths": strengths,
        "improvements": improvements,
    }


# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def home():
    """Landing page."""
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Student registration route."""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("Please fill in all fields.")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.")
            return render_template("register.html")

        try:
            with get_db_connection() as connection:
                connection.execute(
                    """
                    INSERT INTO users (name, email, password_hash)
                    VALUES (?, ?, ?)
                    """,
                    (name, email, generate_password_hash(password)),
                )
            flash("Account created successfully! You can now log in.")
            return redirect("/login")
        except sqlite3.IntegrityError:
            flash("That email address is already registered.")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Student login route."""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        with get_db_connection() as connection:
            user = connection.execute(
                "SELECT id, name, password_hash FROM users WHERE email = ?",
                (email,),
            ).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            next_page = request.args.get("next", "/dashboard")
            if not next_page.startswith("/") or next_page.startswith("//"):
                next_page = "/dashboard"
            return redirect(next_page)

        flash("Invalid email or password.")

    return render_template("login.html")


@app.route("/logout")
def logout():
    """Clear session and log student out."""
    session.clear()
    flash("You have been logged out.")
    return redirect("/login")


@app.route("/dashboard")
@login_required
def dashboard():
    """Student dashboard displaying all 8 key metrics and roadmaps."""
    user_id = session["user_id"]

    with get_db_connection() as connection:
        user_row = connection.execute(
            "SELECT name, email FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        student_name = user_row["name"] if user_row else session.get("user_name", "Student")

        profile_row = connection.execute(
            "SELECT * FROM student_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        profile_data = dict(profile_row) if profile_row else {}

        # Fetch history records
        history_rows = connection.execute(
            """
            SELECT * FROM prediction_history
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,),
        ).fetchall()
        total_predictions = len(history_rows)
        latest_prediction = dict(history_rows[0]) if history_rows else None
        recent_history = [dict(r) for r in history_rows[:5]]

        # Skill gap analysis
        skill_analysis = get_student_skill_analysis(connection, user_id)

    profile_completion = get_profile_completion(profile_data)

    return render_template(
        "dashboard.html",
        student_name=student_name,
        profile_data=profile_data,
        profile_completion=profile_completion,
        total_predictions=total_predictions,
        latest_prediction=latest_prediction,
        recent_history=recent_history,
        skill_analysis=skill_analysis,
    )


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """Student profile creation and update route."""
    user_id = session["user_id"]
    form_data = request.form if request.method == "POST" else {}

    with get_db_connection() as connection:
        if request.method == "POST":
            try:
                cgpa = float(request.form["cgpa"])
                major_projects = int(request.form["major_projects"])
                workshops_certifications = int(request.form["workshops_certifications"])
                mini_projects = int(request.form["mini_projects"])
                skills = int(request.form["skills"])
                communication_skill_rating = float(request.form["communication_skill_rating"])
                internship = request.form["internship"].strip()
                hackathon = request.form["hackathon"].strip()
                twelfth_percentage = float(request.form["twelfth_percentage"])
                tenth_percentage = float(request.form["tenth_percentage"])
                backlogs = int(request.form["backlogs"])

                # Validation bounds
                if not (0.0 <= cgpa <= 10.0):
                    flash("CGPA must be between 0.0 and 10.0.")
                    raise ValueError
                if min(major_projects, workshops_certifications, mini_projects, backlogs) < 0:
                    flash("Project counts, certifications, and backlogs cannot be negative.")
                    raise ValueError
                if not (0 <= skills <= 10):
                    flash("Skills score must be between 0 and 10.")
                    raise ValueError
                if not (0.0 <= communication_skill_rating <= 5.0):
                    flash("Communication skill rating must be between 0.0 and 5.0.")
                    raise ValueError
                if not (0.0 <= twelfth_percentage <= 100.0) or not (0.0 <= tenth_percentage <= 100.0):
                    flash("Percentages must be between 0.0 and 100.0.")
                    raise ValueError
                if internship not in {"Yes", "No"} or hackathon not in {"Yes", "No"}:
                    flash("Internship and Hackathon selections must be 'Yes' or 'No'.")
                    raise ValueError

                connection.execute(
                    """
                    INSERT OR REPLACE INTO student_profiles
                    (user_id, cgpa, major_projects, workshops_certifications,
                     mini_projects, skills, communication_skill_rating, internship,
                     hackathon, twelfth_percentage, tenth_percentage, backlogs)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        cgpa,
                        major_projects,
                        workshops_certifications,
                        mini_projects,
                        skills,
                        communication_skill_rating,
                        internship,
                        hackathon,
                        twelfth_percentage,
                        tenth_percentage,
                        backlogs,
                    ),
                )
                flash("Profile saved successfully!")
                form_data = {}  # Clear form_data so values load cleanly from saved row

            except (KeyError, ValueError):
                if not get_flashed_messages_waiting():
                    flash("Check the form: all fields are required and must be within the valid ranges.")
            except sqlite3.Error:
                app.logger.exception("Could not save student profile")
                flash("Database error: the profile could not be saved.")

        row = connection.execute(
            "SELECT * FROM student_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()

    return render_template(
        "profile.html",
        profile=dict(row) if row else {},
        form_data=form_data,
    )


def get_flashed_messages_waiting():
    """Helper to check if any flash message is already queued."""
    return "_flashes" in session and len(session["_flashes"]) > 0


@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():
    """ML placement prediction route connected to student's saved profile."""
    user_id = session["user_id"]
    result = None
    probability = None
    confidence = None
    explanation = None

    with get_db_connection() as connection:
        profile_row = connection.execute(
            "SELECT * FROM student_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        profile_data = dict(profile_row) if profile_row else {}

    profile_status = get_profile_completion(profile_data)
    has_saved_profile = profile_status["is_complete"]

    if request.method == "POST":
        try:
            model = get_model()

            values = {
                "CGPA": float(request.form["CGPA"]),
                "Major Projects": int(request.form["Major_Projects"]),
                "Workshops/Certificatios": int(request.form["Workshops_Certifications"]),
                "Mini Projects": int(request.form["Mini_Projects"]),
                "Skills": int(request.form["Skills"]),
                "Communication Skill Rating": float(request.form["Communication_Skill_Rating"]),
                "Internship": str(request.form["Internship"]).strip(),
                "Hackathon": str(request.form["Hackathon"]).strip(),
                "12th Percentage": float(request.form["12th_Percentage"]),
                "10th Percentage": float(request.form["10th_Percentage"]),
                "backlogs": int(request.form["backlogs"]),
            }

            # Range validation
            if not (0.0 <= values["CGPA"] <= 10.0):
                raise ValueError("CGPA must be between 0 and 10.")
            if not (0.0 <= values["Communication Skill Rating"] <= 5.0):
                raise ValueError("Communication rating must be between 0 and 5.")
            if not (0 <= values["Skills"] <= 10):
                raise ValueError("Skills must be between 0 and 10.")
            if values["Internship"] not in {"Yes", "No"} or values["Hackathon"] not in {"Yes", "No"}:
                raise ValueError("Internship and Hackathon must be Yes or No.")

            # Create DataFrame matching trained model pipeline features
            input_data = pd.DataFrame([values])
            predicted_status = int(model.predict(input_data)[0])

            # Class probability
            classes = list(model.classes_)
            placed_index = classes.index(1) if 1 in classes else 1
            prob_placed = float(model.predict_proba(input_data)[0][placed_index])
            probability = round(prob_placed * 100.0, 1)

            # Placed vs Not Placed outcome
            result = "Placed" if predicted_status == 1 else "Not Placed"
            confidence = probability if predicted_status == 1 else round(100.0 - probability, 1)

            # Generate structured explanation
            explanation = generate_prediction_explanation(values, predicted_status, probability)

            # Save prediction record to SQLite prediction_history
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            with get_db_connection() as connection:
                connection.execute(
                    """
                    INSERT INTO prediction_history (
                        user_id, prediction_result, probability, cgpa,
                        major_projects, workshops_certifications, mini_projects,
                        skills, communication_skill_rating, internship,
                        hackathon, twelfth_percentage, tenth_percentage,
                        backlogs, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        result,
                        probability,
                        values["CGPA"],
                        values["Major Projects"],
                        values["Workshops/Certificatios"],
                        values["Mini Projects"],
                        values["Skills"],
                        values["Communication Skill Rating"],
                        values["Internship"],
                        values["Hackathon"],
                        values["12th Percentage"],
                        values["10th Percentage"],
                        values["backlogs"],
                        now_str,
                    ),
                )

            flash("Placement prediction calculated and saved to your history.")

            return render_template(
                "prediction.html",
                result=result,
                probability=probability,
                confidence=confidence,
                explanation=explanation,
                form_data=request.form,
                profile_status=profile_status,
                has_saved_profile=has_saved_profile,
            )

        except (KeyError, ValueError, FileNotFoundError) as error:
            app.logger.exception("Prediction failed")
            flash(f"Prediction error: {str(error)}. Please check entered values.")
            return render_template(
                "prediction.html",
                form_data=request.form,
                profile_status=profile_status,
                has_saved_profile=has_saved_profile,
            )

    # GET request: prefill form with saved student profile data
    form_data = {}
    if profile_data:
        form_data = {
            "CGPA": profile_data.get("cgpa", ""),
            "Major_Projects": profile_data.get("major_projects", ""),
            "Workshops_Certifications": profile_data.get("workshops_certifications", ""),
            "Mini_Projects": profile_data.get("mini_projects", ""),
            "Skills": profile_data.get("skills", ""),
            "Communication_Skill_Rating": profile_data.get("communication_skill_rating", ""),
            "Internship": profile_data.get("internship", "Yes"),
            "Hackathon": profile_data.get("hackathon", "Yes"),
            "12th_Percentage": profile_data.get("twelfth_percentage", ""),
            "10th_Percentage": profile_data.get("tenth_percentage", ""),
            "backlogs": profile_data.get("backlogs", "0"),
        }

    return render_template(
        "prediction.html",
        form_data=form_data,
        profile_status=profile_status,
        has_saved_profile=has_saved_profile,
    )


@app.route("/history")
@login_required
def history():
    """Prediction history page showing only the logged-in student's past predictions."""
    user_id = session["user_id"]
    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM prediction_history
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,),
        ).fetchall()
        history_records = [dict(row) for row in rows]

    return render_template("history.html", history_records=history_records)


@app.route("/skill-gap", methods=["GET", "POST"])
@login_required
def skill_gap():
    """Skill gap analysis and recommendation engine with database persistence."""
    user_id = session["user_id"]

    with get_db_connection() as connection:
        if request.method == "POST":
            try:
                p_score = int(request.form.get("python_score", 60))
                s_score = int(request.form.get("sql_score", 50))
                h_score = int(request.form.get("html_css_score", 70))
                g_score = int(request.form.get("git_score", 40))
                d_score = int(request.form.get("dsa_score", 45))
                a_score = int(request.form.get("api_score", 35))

                for val in (p_score, s_score, h_score, g_score, d_score, a_score):
                    if not (0 <= val <= 100):
                        raise ValueError("Skill scores must be between 0 and 100.")

                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                connection.execute(
                    """
                    INSERT OR REPLACE INTO student_skills
                    (user_id, python_score, sql_score, html_css_score, git_score, dsa_score, api_score, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, p_score, s_score, h_score, g_score, d_score, a_score, now_str),
                )
                flash("Skill scores saved successfully and synced with your dashboard!")
            except ValueError as e:
                flash(f"Invalid input: {e}")

        # Fetch saved student skills
        skill_row = connection.execute(
            "SELECT * FROM student_skills WHERE user_id = ?", (user_id,)
        ).fetchone()
        skill_scores = dict(skill_row) if skill_row else {spec["id"]: spec["default"] for spec in SKILL_SPECS}

        # Analyze skills
        server_analysis = get_student_skill_analysis(connection, user_id)

        # Fetch profile for academic placement insights
        profile_row = connection.execute(
            "SELECT * FROM student_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        profile_insights = get_profile_insights(dict(profile_row) if profile_row else {})

    return render_template(
        "skill_gap.html",
        skill_scores=skill_scores,
        server_analysis=server_analysis,
        profile_insights=profile_insights,
    )


import os

if __name__ == "__main__":
    print(app.url_map)
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5001)),
        debug=False
    )