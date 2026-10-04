import os
import json
from datetime import datetime
from functools import wraps

import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.middleware.proxy_fix import ProxyFix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_model.joblib")
SYMPTOMS_PATH = os.path.join(BASE_DIR, "models", "symptoms.pkl")
META_PATH = os.path.join(BASE_DIR, "models", "model_meta.json")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "swasthsathi-college-demo-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "instance", "swasthsathi.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

MODEL = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
SYMPTOMS = joblib.load(SYMPTOMS_PATH) if os.path.exists(SYMPTOMS_PATH) else []
META = {}
if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        META = json.load(f)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(160), unique=True, nullable=False)
    name = db.Column(db.String(100), default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Assessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    symptoms = db.Column(db.Text, nullable=False)
    profile = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(200), nullable=False)
    confidence = db.Column(db.Float, default=0)
    top_predictions = db.Column(db.Text, default="[]")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

def current_user():
    uid = session.get("user_id")
    return User.query.get(uid) if uid else None

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please login first.", "warning")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("admin"):
            return redirect(url_for("admin_login"))
        return fn(*args, **kwargs)
    return wrapper

def model_predict(selected_names):
    if MODEL is None:
        raise RuntimeError("ML model not found. Run train_model.py first.")
    row = {symptom: (1 if symptom in selected_names else 0) for symptom in SYMPTOMS}
    frame = pd.DataFrame([row], columns=SYMPTOMS)
    probabilities = MODEL.predict_proba(frame)[0]
    classes = MODEL.classes_
    order = probabilities.argsort()[::-1][:3]
    top = [
        {"disease": str(classes[i]), "score": round(float(probabilities[i]) * 100, 1)}
        for i in order
    ]
    # Feature importance is a model-level signal, not a causal explanation.
    importances = getattr(MODEL, "feature_importances_", None)
    signals = []
    if importances is not None:
        for symptom in selected_names:
            idx = SYMPTOMS.index(symptom)
            signals.append({"symptom": symptom, "importance": round(float(importances[idx]) * 100, 2)})
        signals.sort(key=lambda x: x["importance"], reverse=True)
    return top, signals[:6]

def specialist_for(disease):
    name = disease.lower()
    if any(x in name for x in ["heart", "cardiac"]):
        return "General Physician / Cardiologist as advised"
    if any(x in name for x in ["asthma", "pneumonia", "tuberculosis", "bronchial"]):
        return "General Physician / Pulmonologist as advised"
    if any(x in name for x in ["liver", "hepatitis", "cholestasis"]):
        return "General Physician / Gastroenterologist as advised"
    if any(x in name for x in ["kidney", "urinary"]):
        return "General Physician / Urologist as advised"
    if any(x in name for x in ["diabetes", "thyroid", "hypoglycemia", "hyperthyroidism"]):
        return "General Physician / Endocrinologist as advised"
    if any(x in name for x in ["skin", "acne", "psoriasis", "impetigo", "fungal"]):
        return "General Physician / Dermatologist as advised"
    return "General Physician"

@app.context_processor
def inject_globals():
    return {"app_name": "SwasthSathi", "year": datetime.now().year, "current_user": current_user()}

@app.route("/")
def home():
    return render_template("landing.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        name = request.form.get("name", "").strip()
        if not email:
            flash("Please enter an email.", "danger")
            return redirect(url_for("login"))
        # College-demo login: no real SMTP credential is stored in the project.
        otp = os.getenv("DEMO_OTP", "123456")
        session["pending_email"] = email
        session["pending_otp"] = otp
        session["pending_name"] = name
        flash("Demo OTP generated. Use 123456 unless DEMO_OTP is changed.", "info")
        return redirect(url_for("verify"))
    return render_template("login.html")

@app.route("/verify", methods=["GET", "POST"])
def verify():
    if request.method == "POST":
        if request.form.get("otp", "") == session.get("pending_otp"):
            email = session.get("pending_email")
            name = session.get("pending_name", "")
            user = User.query.filter_by(email=email).first()
            if not user:
                user = User(email=email, name=name)
                db.session.add(user)
                db.session.commit()
            elif name and not user.name:
                user.name = name
                db.session.commit()
            session.clear()
            session["user_id"] = user.id
            return redirect(url_for("dashboard"))
        flash("Invalid OTP. For demo use 123456.", "danger")
    return render_template("verify.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    recent = Assessment.query.filter_by(user_id=user.id).order_by(Assessment.created_at.desc()).limit(5).all()
    total = Assessment.query.filter_by(user_id=user.id).count()
    return render_template("dashboard.html", recent=recent, total=total, meta=META)

@app.route("/assessment")
@login_required
def assessment():
    return render_template("assessment.html", symptoms=SYMPTOMS)

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        selected = [s for s in SYMPTOMS if request.form.get(s) == "Yes"]
        if not selected:
            flash("Please select at least one symptom.", "warning")
            return redirect(url_for("assessment"))
        top, signals = model_predict(selected)
        primary = top[0]
        profile = {
            "Age": request.form.get("Age", "Not provided"),
            "Gender": request.form.get("Gender", "Not provided"),
        }
        assessment = Assessment(
            user_id=current_user().id,
            symptoms=json.dumps(selected),
            profile=json.dumps(profile),
            prediction=primary["disease"],
            confidence=primary["score"],
            top_predictions=json.dumps(top),
        )
        db.session.add(assessment)
        db.session.commit()
        session["last_signals"] = signals
        return redirect(url_for("result", assessment_id=assessment.id))
    except Exception as e:
        flash(f"Prediction could not be completed: {e}", "danger")
        return redirect(url_for("assessment"))

@app.route("/result/<int:assessment_id>")
@login_required
def result(assessment_id):
    a = Assessment.query.get_or_404(assessment_id)
    if a.user_id != current_user().id:
        return redirect(url_for("dashboard"))
    symptoms = json.loads(a.symptoms)
    profile = json.loads(a.profile)
    top = json.loads(a.top_predictions or "[]")
    _, signals = model_predict(symptoms)
    tips = [
        "This output is for educational screening only and is not a medical diagnosis.",
        "Persistent, severe, or worsening symptoms should be discussed with a qualified healthcare professional.",
        "Do not start, stop, or change medication based only on this result.",
    ]
    return render_template(
        "result.html", assessment=a, symptoms=symptoms, profile=profile,
        top=top, signals=signals, tips=tips,
        specialist=specialist_for(a.prediction), model_meta=META
    )

@app.route("/history")
@login_required
def history():
    records = Assessment.query.filter_by(user_id=current_user().id).order_by(Assessment.created_at.desc()).all()
    return render_template("history.html", records=records)

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = current_user()
    if request.method == "POST":
        user.name = request.form.get("name", "").strip()
        db.session.commit()
        flash("Profile updated.", "success")
        return redirect(url_for("profile"))
    return render_template("profile.html", user=user)

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if request.form.get("password", "") == os.getenv("ADMIN_PASSWORD", "admin123"):
            session["admin"] = True
            return redirect(url_for("admin"))
        flash("Invalid admin password.", "danger")
    return render_template("admin_login.html")

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("home"))

@app.route("/admin")
@admin_required
def admin():
    users = User.query.count()
    assessments = Assessment.query.count()
    predictions = db.session.query(Assessment.prediction, db.func.count(Assessment.id)).group_by(Assessment.prediction).order_by(db.func.count(Assessment.id).desc()).limit(8).all()
    return render_template("admin.html", users=users, assessments=assessments, predictions=predictions, model_meta=META)

@app.route("/api/stats")
@admin_required
def stats():
    rows = db.session.query(Assessment.prediction, db.func.count(Assessment.id)).group_by(Assessment.prediction).order_by(db.func.count(Assessment.id).desc()).limit(10).all()
    return jsonify({"labels": [r[0] for r in rows], "values": [r[1] for r in rows]})

@app.route("/about")
def about():
    return render_template("about.html")

with app.app_context():
    os.makedirs(os.path.join(BASE_DIR, "instance"), exist_ok=True)
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)
