from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pandas as pd
import joblib
import os
import sqlite3
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, 'oncogen_care.db')

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
import os
app.secret_key = os.environ.get("SECRET_KEY", "local-dev-only-change-me")

model = joblib.load(os.path.join(BASE_DIR, 'cancer_model.pkl'))
model_columns = joblib.load(os.path.join(BASE_DIR, 'model_columns.pkl'))
target_encoder = joblib.load(os.path.join(BASE_DIR, 'target_encoder.pkl'))


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            gender TEXT,
            age INTEGER,
            age_group TEXT,
            phone TEXT,
            country TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )
    conn.commit()
    conn.close()


init_db()


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('home'))
        return view_func(*args, **kwargs)
    return wrapped_view


@app.route('/')
def home():
    if session.get('user_id'):
        return redirect(url_for('dashboard'))
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        if session.get('user_id'):
            return redirect(url_for('dashboard'))
        return render_template('signup.html')

    try:
        data = request.get_json(silent=True) or request.form.to_dict()

        full_name = (data.get('fullName') or data.get('full_name') or '').strip()
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''
        gender = (data.get('gender') or '').strip()
        age = data.get('age')
        age_group = (data.get('ageGroup') or data.get('age_group') or '').strip()
        phone = (data.get('phone') or '').strip()
        country = (data.get('country') or '').strip()

        if not full_name:
            return jsonify({'success': False, 'error': 'Full name is required'}), 400
        if not email:
            return jsonify({'success': False, 'error': 'Email is required'}), 400
        if not password or len(password) < 8:
            return jsonify({'success': False, 'error': 'Password must be at least 8 characters'}), 400

        try:
            age_int = int(age)
            if age_int < 0 or age_int > 120:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': 'Age must be between 0 and 120'}), 400

        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        existing = conn.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
        if existing:
            conn.close()
            return jsonify({'success': False, 'error': 'Email already registered'}), 409

        conn.execute(
            '''
            INSERT INTO users (full_name, email, password, gender, age, age_group, phone, country)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (full_name, email, hashed_password, gender, age_int, age_group, phone, country)
        )
        conn.commit()
        conn.close()

        return jsonify({'success': True, 'message': 'Account created successfully'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json(silent=True) or request.form.to_dict()

        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''

        if not email or not password:
            return jsonify({'success': False, 'error': 'Email and password are required'}), 400

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['user_name'] = user['full_name']
            session['user_email'] = user['email']
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'redirect': url_for('dashboard')
            })

        return jsonify({'success': False, 'error': 'Invalid email or password'}), 401

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('index.html', user_name=session.get('user_name'))


@app.route('/blog')
@login_required
def blog():
    return render_template('Blog.html', user_name=session.get('user_name'))


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'No input data received'
            }), 400

        input_df = pd.DataFrame([data])
        input_df = input_df.fillna('Unknown')
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_df)[0]
        predicted_label = target_encoder.inverse_transform([prediction])[0]

        print("INPUT DATA:", data)
        print("PREDICTION:", predicted_label)

        return jsonify({
            'success': True,
            'prediction': predicted_label
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=False)