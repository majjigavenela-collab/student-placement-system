# Student Placement Prediction System

A complete Flask web application that enables students to manage academic profiles, predict campus placement outcomes using a trained scikit-learn machine learning pipeline, track prediction history, and identify technical skill gaps with personalized improvement recommendations.

## Features

1. **Student Profile Management**
   - Save and update comprehensive academic records (CGPA, 10th %, 12th %, backlogs), project experience (major & mini), certifications, and competencies (skills, communication, internship, hackathons).
   - Stored securely in local SQLite database with input range validations.
   - Profile completion tracking with missing-field alerts.

2. **ML Placement Prediction**
   - Connects directly to the student's saved profile data.
   - Evaluates placement outcome using the trained Random Forest classification pipeline (`model/placement_model.pkl`).
   - Displays prediction status (**Placed** / **Not Placed**), placement probability %, and model confidence %.
   - Generates detailed explanations highlighting profile strengths and actionable areas for improvement.
   - Handles missing profile information gracefully with clear warnings and prompts.

3. **Prediction History**
   - Logs every prediction with the student's ID, result, probability, profile snapshot, and timestamp into `prediction_history` table.
   - Dedicated history page displaying all previous predictions.
   - Secure student isolation: each student can view only their own records.

4. **Skill Gap Analysis**
   - Benchmarks student competencies in key tech stacks (Python, SQL, HTML/CSS, Git/GitHub, Data Structures & Algorithms, REST APIs) against the 70% placement benchmark.
   - Identifies weak skills and provides practical learning recommendations.
   - Integrates academic profile placement factors (internships, hackathons, backlogs, projects, communication).
   - Persists student skill ratings to SQLite (`student_skills`) and automatically syncs with the dashboard.
   - Supports both instant interactive browser assessment and saved assessments.

5. **Student Dashboard**
   - Displays student name and greeting.
   - Visual profile completion percentage and missing fields summary.
   - Latest placement prediction status and date.
   - Prediction probability and model confidence score.
   - Total number of predictions recorded.
   - Recent prediction history table.
   - Technical skill gap summary and average readiness score.
   - Prioritized list of recommended skills to improve.
   - Quick navigation tools.

6. **Complete User Flow**
   - Login → Dashboard → Profile → Save Profile → Placement Prediction → Prediction Result → Prediction History → Skill Gap Analysis.

## Run on Windows

Open a terminal in the project folder (the folder containing `app.py`) and run:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Or using Command Prompt:

```cmd
.venv\Scripts\activate.bat
python app.py
```

Then open `http://127.0.0.1:5001` in your web browser.

## Retraining the Model

To retrain and compare machine learning models on the placement dataset:

```powershell
python training\train_model.py
```

The script evaluates Logistic Regression, Decision Tree, and Random Forest classifier candidates on a 20% stratified test split, printing accuracy, precision, recall, and F1 score, and automatically saves the best pipeline to `model/placement_model.pkl`.
