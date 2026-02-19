from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ------------------ Flask Config ------------------
app = Flask(__name__, static_folder='static')
app.secret_key = "dyuiknbvcxswe678ijc6i"

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------ SQLite DB ------------------
DB_PATH = 'database.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            password TEXT NOT NULL
        )
    ''')

    # Patients table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            address TEXT NOT NULL,
            phone TEXT NOT NULL,
            image_path TEXT,
            result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

init_db()

# ------------------ Load ML Model ------------------
MODEL_PATH = "best_model.h5"
model = load_model(MODEL_PATH)
IMG_SIZE = 224

# ------------------ Routes ------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            flash("Login successful!", "success")
            return redirect(url_for("prediction"))
        else:
            flash("Invalid email or password.", "danger")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password)

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (name, email, phone, password) VALUES (?, ?, ?, ?)",
                (name, email, phone, hashed_password)
            )
            conn.commit()
            conn.close()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.Error as e:
            flash(f"Registration failed: {e}", "danger")
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("index"))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

# ------------------ Prediction Route ------------------
@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if "user_id" not in session:
        flash("Please log in to access prediction.", "warning")
        return redirect(url_for("login"))

    if request.method == "POST":
        # Get patient details
        patient_name = request.form.get("patient_name")
        age = request.form.get("age")
        gender = request.form.get("gender")
        address = request.form.get("address")
        phone = request.form.get("phone")
        image = request.files.get("image")

        # Basic validation
        if not all([patient_name, age, gender, address, phone, image]):
            flash("Please fill all patient details and upload an image.", "danger")
            return redirect(request.url)

        if image.filename == '':
            flash("No image selected.", "danger")
            return redirect(request.url)

        # Save image
        filename = secure_filename(image.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(file_path)

        relative_image_path = os.path.relpath(file_path, start='static').replace("\\", "/")

        # Preprocess image
        img = load_img(file_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0][0]

        if prediction > 0.5:
            result = "Affected (Dementia)"
            color = "red"
        else:
            result = "Normal"
            color = "green"

        # Save patient record
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO patients (user_id, name, age, gender, address, phone, image_path, result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session["user_id"], patient_name, age, gender, address, phone, relative_image_path, result))
        conn.commit()
        conn.close()

        return render_template(
            "prediction.html",
            image_path=relative_image_path,
            result=result,
            color=color
        )

    return render_template("prediction.html")

# ------------------ Run App ------------------
if __name__ == "__main__":
    app.run(debug=True)
