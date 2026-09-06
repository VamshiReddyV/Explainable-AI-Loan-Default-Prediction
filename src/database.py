"""
SQLite database for storing loan predictions, user accounts, admin profiles,
and performance tracking. Handles role-based authentication, user data isolation,
time-filtered performance analytics, and explainability persistence.
"""
import os
import sqlite3
import json
import uuid
import random
import hashlib
import hmac
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from werkzeug.security import generate_password_hash, check_password_hash


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'instance', 'predictions.db')


def get_db():
    """Get a database connection with Row factory."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Create tables if they don't exist, apply migrations, and seed initial users and predictions.
    """
    conn = get_db()
    
    # 1. Users Table (Role-based authentication)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'admin')),
            phone TEXT DEFAULT '',
            avatar_url TEXT DEFAULT '',
            age INTEGER DEFAULT 32,
            gender TEXT DEFAULT 'Male',
            marital_status TEXT DEFAULT 'Married',
            number_of_dependents INTEGER DEFAULT 2,
            employment_type TEXT DEFAULT 'Salaried',
            employment_experience INTEGER DEFAULT 4,
            monthly_income REAL DEFAULT 48000.0,
            annual_income REAL DEFAULT 576000.0,
            credit_score INTEGER DEFAULT 580,
            existing_loans INTEGER DEFAULT 2,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 2. Predictions Table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            applicant_name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            marital_status TEXT,
            number_of_dependents INTEGER DEFAULT 0,
            employment_type TEXT,
            employment_experience INTEGER,
            monthly_income REAL,
            annual_income REAL,
            credit_score INTEGER,
            loan_amount REAL,
            loan_purpose TEXT,
            loan_term INTEGER,
            interest_rate REAL,
            existing_loans INTEGER DEFAULT 0,
            prediction TEXT NOT NULL,
            default_probability REAL NOT NULL,
            risk_level TEXT,
            shap_explanation TEXT,
            lime_explanation TEXT,
            actual_outcome TEXT DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Migrations for existing predictions table
    columns = [c[1] for c in conn.execute('PRAGMA table_info(predictions)').fetchall()]
    if 'actual_outcome' not in columns:
        conn.execute('ALTER TABLE predictions ADD COLUMN actual_outcome TEXT DEFAULT NULL')
    if 'user_id' not in columns:
        conn.execute('ALTER TABLE predictions ADD COLUMN user_id INTEGER DEFAULT 1')
    if 'loan_type' not in columns:
        conn.execute('ALTER TABLE predictions ADD COLUMN loan_type TEXT DEFAULT "Personal"')

    # Migrations for users table (extended financial profile & verification status)
    u_cols = [c[1] for c in conn.execute('PRAGMA table_info(users)').fetchall()]
    for col_name, col_def in [
        ('monthly_expenses', 'REAL DEFAULT 0'),
        ('existing_emi', 'REAL DEFAULT 0'),
        ('assets_value', 'REAL DEFAULT 0'),
        ('liabilities_value', 'REAL DEFAULT 0'),
        ('address', 'TEXT DEFAULT ""'),
        ('pan_number', 'TEXT DEFAULT ""'),
        ('phone_verified', 'INTEGER DEFAULT 0'),
        ('is_active', 'INTEGER DEFAULT 1'),
    ]:
        if col_name not in u_cols:
            conn.execute(f'ALTER TABLE users ADD COLUMN {col_name} {col_def}')

    # Ensure existing demo accounts and admins have phone_verified active
    conn.execute('''
        UPDATE users SET phone_verified = 1, is_active = 1
        WHERE role = 'admin' OR LOWER(email) IN ('rahul@example.com', 'priya@example.com')
    ''')

    # 4. Loan Applications Table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS loan_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            application_id TEXT NOT NULL UNIQUE,
            user_id INTEGER NOT NULL,
            loan_type TEXT NOT NULL,
            loan_amount REAL NOT NULL,
            loan_term INTEGER NOT NULL,
            loan_purpose TEXT,
            applicant_data TEXT NOT NULL,
            status TEXT DEFAULT 'Submitted',
            risk_assessment_id INTEGER DEFAULT NULL,
            notes TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # 5. Loan Documents Table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS loan_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            application_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            document_type TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER DEFAULT 0,
            status TEXT DEFAULT 'Uploaded',
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (application_id) REFERENCES loan_applications(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # 6. User Settings Table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id INTEGER PRIMARY KEY,
            notify_assessment INTEGER DEFAULT 1,
            notify_risk_shift INTEGER DEFAULT 1,
            notify_monthly_digest INTEGER DEFAULT 1,
            notify_sms_auth INTEGER DEFAULT 1,
            notify_market_rates INTEGER DEFAULT 0,
            enable_2fa INTEGER DEFAULT 0,
            session_timeout INTEGER DEFAULT 1,
            export_shap_table INTEGER DEFAULT 1,
            export_lime_table INTEGER DEFAULT 1,
            anonymized_telemetry INTEGER DEFAULT 1,
            currency_format TEXT DEFAULT 'INR',
            date_format TEXT DEFAULT 'DD-MM-YYYY',
            high_contrast_charts INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # 7. Phone OTP Verifications Table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS otp_verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT NULL,
            phone TEXT NOT NULL,
            otp_hash TEXT NOT NULL,
            attempts INTEGER DEFAULT 0,
            is_verified INTEGER DEFAULT 0,
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # 3. Admin Profile Table (legacy compatibility & admin settings)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS admin_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            full_name TEXT NOT NULL,
            display_name TEXT NOT NULL,
            email TEXT NOT NULL,
            role TEXT NOT NULL,
            department TEXT DEFAULT 'Credit Risk & Analytics',
            phone TEXT DEFAULT '+91 98765 43210',
            avatar_url TEXT DEFAULT '',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Seed default accounts if empty
    user_count = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    if user_count == 0:
        seed_users_and_accounts(conn)
    
    admin_count = conn.execute('SELECT COUNT(*) FROM admin_profile').fetchone()[0]
    if admin_count == 0:
        conn.execute('''
            INSERT INTO admin_profile (username, full_name, display_name, email, role, department, phone)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            'admin',
            'Vamshi Reddy',
            'Vamshi',
            'admin@loanpredict.ai',
            'Chief Risk Officer & Lead Auditor',
            'Credit Risk & Quantitative Analytics',
            '+91 98765 43210'
        ))
    
    conn.commit()

    # Check if predictions need seeding or date updates
    count = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    if count == 0:
        seed_comprehensive_predictions(conn)
    else:
        # Align seed dates to active year
        align_seed_dates(conn)

    conn.close()


def seed_users_and_accounts(conn):
    """Seed default User and Admin accounts with hashed passwords."""
    # User 1: Rahul Sharma (Primary demo user matching user dashboard spec)
    conn.execute('''
        INSERT INTO users (
            name, email, password_hash, role, phone, age, gender, marital_status,
            number_of_dependents, employment_type, employment_experience,
            monthly_income, annual_income, credit_score, existing_loans,
            phone_verified, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1)
    ''', (
        'Rahul Sharma',
        'rahul@example.com',
        generate_password_hash('password123'),
        'user',
        '+91 98123 45678',
        32, 'Male', 'Married', 2,
        'Salaried', 4, 48000.0, 576000.0, 580, 2
    ))

    # User 2: Priya Patel
    conn.execute('''
        INSERT INTO users (
            name, email, password_hash, role, phone, age, gender, marital_status,
            number_of_dependents, employment_type, employment_experience,
            monthly_income, annual_income, credit_score, existing_loans,
            phone_verified, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1)
    ''', (
        'Priya Patel',
        'priya@example.com',
        generate_password_hash('password123'),
        'user',
        '+91 98234 56789',
        28, 'Female', 'Single', 0,
        'Salaried', 5, 65000.0, 780000.0, 745, 0
    ))

    # Admin: Vamshi Reddy
    conn.execute('''
        INSERT INTO users (
            name, email, password_hash, role, phone, age, gender, marital_status,
            number_of_dependents, employment_type, employment_experience,
            monthly_income, annual_income, credit_score, existing_loans,
            phone_verified, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1)
    ''', (
        'Vamshi Reddy',
        'admin@loanpredict.ai',
        generate_password_hash('admin123'),
        'admin',
        '+91 98765 43210',
        36, 'Male', 'Married', 1,
        'Salaried', 10, 120000.0, 1440000.0, 820, 0
    ))
    conn.commit()


# =========================================================
# Authentication, Registration & Phone OTP Verification
# =========================================================

def normalize_phone(phone):
    """Normalize phone number to standard 10-digit Indian numeric string."""
    if not phone:
        return ""
    digits = "".join(ch for ch in str(phone) if ch.isdigit())
    if len(digits) > 10 and digits.startswith('91'):
        digits = digits[2:]
    return digits[-10:] if len(digits) >= 10 else digits


def get_user_by_phone(phone):
    """Retrieve user dictionary by normalized phone number."""
    norm = normalize_phone(phone)
    if not norm:
        return None
    conn = get_db()
    rows = conn.execute('SELECT * FROM users').fetchall()
    conn.close()
    for r in rows:
        if normalize_phone(r['phone']) == norm:
            return dict(r)
    return None


def get_user_by_email_or_phone(identifier):
    """Retrieve user dictionary by either email address or phone number."""
    if not identifier:
        return None
    ident = str(identifier).strip()
    if '@' in ident:
        return get_user_by_email(ident)
    return get_user_by_phone(ident)


def create_registered_user(name, email, phone, password, age=25, gender='Male', marital_status='Single', employment_type='Salaried'):
    """Create a new pending user account awaiting phone OTP verification."""
    clean_name = str(name).strip()
    clean_email = str(email).strip().lower()
    clean_phone = normalize_phone(phone)

    if not clean_email or '@' not in clean_email:
        return None, "A valid email address is required."

    if len(clean_phone) != 10:
        return None, "Please provide a valid 10-digit mobile number."

    existing_by_email = get_user_by_email(clean_email)
    if existing_by_email:
        return None, "An account with this email address already exists. Please log in."

    existing_by_phone = get_user_by_phone(clean_phone)
    if existing_by_phone:
        return None, "This mobile number is already registered. Please log in."

    conn = get_db()
    cursor = conn.execute('''
        INSERT INTO users (
            name, email, phone, password_hash, role, age, gender, marital_status,
            employment_type, phone_verified, is_active, credit_score, monthly_income, annual_income
        ) VALUES (?, ?, ?, ?, 'user', ?, ?, ?, ?, 0, 1, 650, 50000.0, 600000.0)
    ''', (
        clean_name,
        clean_email,
        clean_phone,
        generate_password_hash(password),
        int(age or 25),
        gender,
        marital_status,
        employment_type
    ))
    conn.commit()
    user_id = cursor.lastrowid
    row = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None, None


def activate_user_phone(identifier):
    """Mark user account as phone verified and active by user_id or phone."""
    conn = get_db()
    if isinstance(identifier, int):
        conn.execute('''
            UPDATE users
            SET phone_verified = 1, is_active = 1, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (identifier,))
    else:
        norm = normalize_phone(str(identifier))
        conn.execute('''
            UPDATE users
            SET phone_verified = 1, is_active = 1, updated_at = CURRENT_TIMESTAMP
            WHERE phone = ? OR phone LIKE ?
        ''', (norm, f'%{norm}'))
    conn.commit()
    conn.close()


def create_otp_verification(user_id, phone, otp_input, expires_minutes=5):
    """Generate and store hashed OTP record for phone verification."""
    norm_phone = normalize_phone(phone)
    otp_str = str(otp_input).strip()
    if len(otp_str) == 64 and all(c in '0123456789abcdefABCDEF' for c in otp_str):
        otp_hash = otp_str.lower()
    else:
        otp_hash = hashlib.sha256(otp_str.encode('utf-8')).hexdigest()

    now_dt = datetime.now()
    now_str = now_dt.strftime('%Y-%m-%d %H:%M:%S')
    expires_at = (now_dt + timedelta(minutes=expires_minutes)).strftime('%Y-%m-%d %H:%M:%S')

    conn = get_db()
    cursor = conn.execute('''
        INSERT INTO otp_verifications (user_id, phone, otp_hash, expires_at, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, norm_phone, otp_hash, expires_at, now_str))
    conn.commit()
    rec_id = cursor.lastrowid
    conn.close()
    return rec_id


def verify_otp_code(phone, otp_input):
    """
    Validate OTP code against latest active record.
    Returns: (success_bool, message, error_code)
    """
    if not phone or not otp_input:
        return False, "Phone number and 6-digit OTP code are required.", "missing_data"

    norm_phone = normalize_phone(phone)
    code_str = str(otp_input).strip()
    if len(code_str) == 64 and all(c in '0123456789abcdefABCDEF' for c in code_str):
        expected_hash = code_str.lower()
    else:
        expected_hash = hashlib.sha256(code_str.encode('utf-8')).hexdigest()

    conn = get_db()
    row = conn.execute('''
        SELECT * FROM otp_verifications
        WHERE phone = ? AND is_verified = 0
        ORDER BY id DESC LIMIT 1
    ''', (norm_phone,)).fetchone()

    if not row:
        conn.close()
        return False, "No active verification request found. Please request a new OTP.", "no_record"

    rec = dict(row)
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Check attempt limit (max 5)
    if rec['attempts'] >= 5:
        conn.close()
        return False, "Maximum verification attempts exceeded. Please request a new OTP.", "max_attempts"

    # Check expiration (5 minutes)
    if rec['expires_at'] < now_str:
        conn.close()
        return False, "OTP has expired. Please request a new verification code.", "expired"

    if hmac.compare_digest(rec['otp_hash'].lower(), expected_hash.lower()):
        conn.execute('UPDATE otp_verifications SET is_verified = 1 WHERE id = ?', (rec['id'],))
        conn.commit()
        conn.close()
        return True, "Phone number verified successfully.", None
    else:
        new_attempts = rec['attempts'] + 1
        conn.execute('UPDATE otp_verifications SET attempts = ? WHERE id = ?', (new_attempts, rec['id']))
        conn.commit()
        conn.close()
        remaining = 5 - new_attempts
        if remaining <= 0:
            return False, "Maximum verification attempts exceeded. Please request a new OTP.", "max_attempts"
        return False, f"Incorrect verification code. {remaining} attempt(s) remaining.", "invalid_otp"


def can_resend_otp(phone, cooldown_seconds=30):
    """
    Check if a resend request is permitted under rate limiting and cooldown rules.
    Returns: (can_resend_bool, remaining_cooldown_seconds)
    """
    norm_phone = normalize_phone(phone)
    conn = get_db()
    row = conn.execute('''
        SELECT created_at FROM otp_verifications
        WHERE phone = ?
        ORDER BY id DESC LIMIT 1
    ''', (norm_phone,)).fetchone()
    conn.close()

    if not row:
        return True, 0

    try:
        created_dt = datetime.strptime(row['created_at'], '%Y-%m-%d %H:%M:%S')
        elapsed = (datetime.now() - created_dt).total_seconds()
        if elapsed < cooldown_seconds:
            return False, int(cooldown_seconds - elapsed)
    except Exception:
        pass
    return True, 0


def authenticate_user(identifier, password, expected_role=None):
    """
    Authenticate a user by email OR phone number and password with optional role verification.
    Enforces that regular users must have a verified phone number before logging in.
    Returns: (user_dict, error_message, error_code)
    """
    if not identifier or not password:
        return None, "Please provide both email/phone number and password.", "missing_credentials"

    user = get_user_by_email_or_phone(identifier)
    if not user:
        return None, "Invalid email/phone number or password.", "invalid_credentials"

    if not check_password_hash(user['password_hash'], password):
        return None, "Invalid email/phone number or password.", "invalid_credentials"

    if expected_role and user['role'] != expected_role:
        if expected_role == 'admin':
            return None, "Access denied: This account does not possess administrator privileges.", "access_denied"
        elif expected_role == 'user' and user['role'] == 'admin':
            return user, None, None

    # Check phone verification for normal borrowers/users
    if user['role'] == 'user' and user.get('phone_verified', 0) == 0:
        return user, "Your phone number is not yet verified. Please enter the OTP to activate your account.", "unverified_phone"

    return user, None, None


def get_user_by_id(user_id):
    """Retrieve user dictionary by ID."""
    conn = get_db()
    row = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_email(email):
    """Retrieve user dictionary by email."""
    conn = get_db()
    row = conn.execute('SELECT * FROM users WHERE LOWER(email) = LOWER(?)', (email.strip(),)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_user_profile(user_id, data):
    """Update general profile fields for a user."""
    conn = get_db()
    conn.execute('''
        UPDATE users
        SET name = ?, phone = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (
        data.get('name', '').strip(),
        data.get('phone', '').strip(),
        user_id
    ))
    conn.commit()
    conn.close()
    return get_user_by_id(user_id)


def update_user_financial_profile(user_id, data):
    """Update complete financial parameters for a user."""
    conn = get_db()
    conn.execute('''
        UPDATE users
        SET age = ?, gender = ?, marital_status = ?, number_of_dependents = ?,
            employment_type = ?, employment_experience = ?, monthly_income = ?,
            annual_income = ?, credit_score = ?, existing_loans = ?,
            monthly_expenses = ?, existing_emi = ?, assets_value = ?, liabilities_value = ?,
            address = ?, pan_number = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (
        int(data.get('age') or 30),
        data.get('gender') or 'Male',
        data.get('marital_status') or 'Single',
        int(data.get('number_of_dependents') or 0),
        data.get('employment_type') or 'Salaried',
        int(data.get('employment_experience') or 0),
        float(data.get('monthly_income') or 0),
        float(data.get('annual_income') or (float(data.get('monthly_income') or 0) * 12)),
        int(data.get('credit_score') or 650),
        int(data.get('existing_loans') or 0),
        float(data.get('monthly_expenses') or 0),
        float(data.get('existing_emi') or 0),
        float(data.get('assets_value') or 0),
        float(data.get('liabilities_value') or 0),
        str(data.get('address') or '').strip(),
        str(data.get('pan_number') or '').strip().upper(),
        user_id
    ))
    conn.commit()
    conn.close()
    return get_user_by_id(user_id)


def update_user_password(user_id, old_password, new_password):
    """Update user password with verification."""
    user = get_user_by_id(user_id)
    if not user:
        return False, "User not found."
    if not check_password_hash(user['password_hash'], old_password):
        return False, "Current password does not match."
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters."

    new_hash = generate_password_hash(new_password)
    conn = get_db()
    conn.execute('UPDATE users SET password_hash = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?', (new_hash, user_id))
    conn.commit()
    conn.close()
    return True, "Password updated successfully."


def get_all_users():
    """Retrieve all users for admin management."""
    conn = get_db()
    rows = conn.execute('''
        SELECT u.id, u.name, u.email, u.role, u.phone, u.employment_type,
               u.credit_score, u.created_at,
               COUNT(p.id) as prediction_count
        FROM users u
        LEFT JOIN predictions p ON u.id = p.user_id
        GROUP BY u.id
        ORDER BY u.id ASC
    ''').fetchall()
    conn.close()
    return [dict(r) for r in rows]


# =========================================================
# Admin Profile & Settings (Legacy & compatibility)
# =========================================================

def get_admin_profile():
    """Retrieve administrator profile from database."""
    conn = get_db()
    row = conn.execute('SELECT * FROM admin_profile ORDER BY id ASC LIMIT 1').fetchone()
    conn.close()
    if row:
        return dict(row)
    return {
        'id': 1,
        'username': 'admin',
        'full_name': 'Vamshi Reddy',
        'display_name': 'Vamshi',
        'email': 'admin@loanpredict.ai',
        'role': 'Chief Risk Officer & Lead Auditor',
        'department': 'Credit Risk & Quantitative Analytics',
        'phone': '+91 98765 43210'
    }


def update_admin_profile(data):
    """Update administrator profile fields."""
    conn = get_db()
    conn.execute('''
        UPDATE admin_profile
        SET full_name = ?, display_name = ?, email = ?, role = ?, department = ?, phone = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (
        data.get('full_name', 'Vamshi Reddy'),
        data.get('display_name', 'Vamshi'),
        data.get('email', 'admin@loanpredict.ai'),
        data.get('role', 'Chief Risk Officer & Lead Auditor'),
        data.get('department', 'Credit Risk & Quantitative Analytics'),
        data.get('phone', '+91 98765 43210'),
        data.get('id', 1)
    ))
    conn.commit()
    conn.close()
    return get_admin_profile()


# =========================================================
# Seed Data with User Associations
# =========================================================

def seed_comprehensive_predictions(conn):
    """
    Seed realistic loan application records with temporal spread spanning
    This Month, Last 3 Months, and Year to Date.
    Rahul Sharma (user_id=1) has a rich prediction history matching user dashboard specs.
    """
    now = datetime.now()
    
    def dt_str(days_ago, hour=10, minute=15):
        d = now - timedelta(days=days_ago)
        return d.replace(hour=hour, minute=minute, second=0).strftime('%Y-%m-%d %H:%M:%S')

    records = [
        # --- User 1: Rahul Sharma's Historical Predictions (Ordered latest to earliest) ---
        {
            'user_id': 1,
            'applicant_name': 'Rahul Sharma',
            'age': 32, 'gender': 'Male', 'marital_status': 'Married', 'number_of_dependents': 2,
            'employment_type': 'Salaried', 'employment_experience': 4,
            'monthly_income': 48000, 'annual_income': 576000, 'credit_score': 580,
            'loan_amount': 250000, 'loan_purpose': 'Personal', 'loan_term': 24,
            'interest_rate': 12.5, 'existing_loans': 2, 'prediction': 'Non-Default',
            'default_probability': 0.1683, 'risk_level': 'Low Risk',
            'actual_outcome': None,
            'created_at': dt_str(0, 11, 20)  # Today
        },
        {
            'user_id': 1,
            'applicant_name': 'Rahul Sharma',
            'age': 32, 'gender': 'Male', 'marital_status': 'Married', 'number_of_dependents': 2,
            'employment_type': 'Salaried', 'employment_experience': 4,
            'monthly_income': 48000, 'annual_income': 576000, 'credit_score': 580,
            'loan_amount': 300000, 'loan_purpose': 'Personal', 'loan_term': 36,
            'interest_rate': 13.0, 'existing_loans': 2, 'prediction': 'Non-Default',
            'default_probability': 0.2842, 'risk_level': 'Medium Risk',
            'actual_outcome': None,
            'created_at': dt_str(16, 14, 10) # 21 Aug
        },
        {
            'user_id': 1,
            'applicant_name': 'Rahul Sharma',
            'age': 32, 'gender': 'Male', 'marital_status': 'Married', 'number_of_dependents': 2,
            'employment_type': 'Salaried', 'employment_experience': 4,
            'monthly_income': 45000, 'annual_income': 540000, 'credit_score': 560,
            'loan_amount': 400000, 'loan_purpose': 'Business', 'loan_term': 48,
            'interest_rate': 14.5, 'existing_loans': 3, 'prediction': 'Default',
            'default_probability': 0.6120, 'risk_level': 'High Risk',
            'actual_outcome': 'Default',
            'created_at': dt_str(58, 10, 30) # 10 Jul
        },
        {
            'user_id': 1,
            'applicant_name': 'Rahul Sharma',
            'age': 32, 'gender': 'Male', 'marital_status': 'Married', 'number_of_dependents': 2,
            'employment_type': 'Salaried', 'employment_experience': 3,
            'monthly_income': 45000, 'annual_income': 540000, 'credit_score': 570,
            'loan_amount': 200000, 'loan_purpose': 'Personal', 'loan_term': 24,
            'interest_rate': 12.0, 'existing_loans': 2, 'prediction': 'Non-Default',
            'default_probability': 0.2215, 'risk_level': 'Low Risk',
            'actual_outcome': 'Non-Default',
            'created_at': dt_str(83, 16, 45) # 15 Jun
        },
        {
            'user_id': 1,
            'applicant_name': 'Rahul Sharma',
            'age': 31, 'gender': 'Male', 'marital_status': 'Married', 'number_of_dependents': 2,
            'employment_type': 'Salaried', 'employment_experience': 3,
            'monthly_income': 42000, 'annual_income': 504000, 'credit_score': 550,
            'loan_amount': 350000, 'loan_purpose': 'Personal', 'loan_term': 36,
            'interest_rate': 13.5, 'existing_loans': 2, 'prediction': 'Non-Default',
            'default_probability': 0.4833, 'risk_level': 'Medium Risk',
            'actual_outcome': 'Non-Default',
            'created_at': dt_str(124, 9, 15) # 05 May
        },
        {
            'user_id': 1,
            'applicant_name': 'Rahul Sharma',
            'age': 31, 'gender': 'Male', 'marital_status': 'Married', 'number_of_dependents': 2,
            'employment_type': 'Salaried', 'employment_experience': 3,
            'monthly_income': 42000, 'annual_income': 504000, 'credit_score': 550,
            'loan_amount': 250000, 'loan_purpose': 'Personal', 'loan_term': 24,
            'interest_rate': 13.0, 'existing_loans': 2, 'prediction': 'Non-Default',
            'default_probability': 0.3650, 'risk_level': 'Medium Risk',
            'actual_outcome': 'Non-Default',
            'created_at': dt_str(145, 11, 0)
        },
        # --- User 2: Priya Patel's Predictions ---
        {
            'user_id': 2,
            'applicant_name': 'Priya Patel',
            'age': 28, 'gender': 'Female', 'marital_status': 'Single', 'number_of_dependents': 0,
            'employment_type': 'Salaried', 'employment_experience': 5,
            'monthly_income': 65000, 'annual_income': 780000, 'credit_score': 745,
            'loan_amount': 180000, 'loan_purpose': 'Education', 'loan_term': 36,
            'interest_rate': 8.5, 'existing_loans': 0, 'prediction': 'Non-Default',
            'default_probability': 0.1420, 'risk_level': 'Low Risk',
            'actual_outcome': None,
            'created_at': dt_str(2, 11, 45)
        },
        {
            'user_id': 2,
            'applicant_name': 'Priya Patel',
            'age': 28, 'gender': 'Female', 'marital_status': 'Single', 'number_of_dependents': 0,
            'employment_type': 'Salaried', 'employment_experience': 5,
            'monthly_income': 65000, 'annual_income': 780000, 'credit_score': 745,
            'loan_amount': 220000, 'loan_purpose': 'Personal', 'loan_term': 24,
            'interest_rate': 9.0, 'existing_loans': 0, 'prediction': 'Non-Default',
            'default_probability': 0.1836, 'risk_level': 'Low Risk',
            'actual_outcome': 'Non-Default',
            'created_at': dt_str(45, 13, 20)
        },
        # --- Additional historical records for platform portfolio analytics ---
        {
            'user_id': 1,
            'applicant_name': 'Amit Kumar',
            'age': 39, 'gender': 'Male', 'marital_status': 'Married', 'number_of_dependents': 3,
            'employment_type': 'Self-Employed', 'employment_experience': 7,
            'monthly_income': 52000, 'annual_income': 624000, 'credit_score': 610,
            'loan_amount': 320000, 'loan_purpose': 'Business', 'loan_term': 48,
            'interest_rate': 13.0, 'existing_loans': 2, 'prediction': 'Default',
            'default_probability': 0.6521, 'risk_level': 'High Risk',
            'actual_outcome': None,
            'created_at': dt_str(4, 14, 20)
        },
        {
            'user_id': 2,
            'applicant_name': 'Sneha Reddy',
            'age': 31, 'gender': 'Female', 'marital_status': 'Single', 'number_of_dependents': 1,
            'employment_type': 'Salaried', 'employment_experience': 6,
            'monthly_income': 78000, 'annual_income': 936000, 'credit_score': 790,
            'loan_amount': 400000, 'loan_purpose': 'Home', 'loan_term': 60,
            'interest_rate': 7.8, 'existing_loans': 1, 'prediction': 'Non-Default',
            'default_probability': 0.1275, 'risk_level': 'Low Risk',
            'actual_outcome': None,
            'created_at': dt_str(5, 16, 10)
        },
        {
            'user_id': 1,
            'applicant_name': 'Vikram Singh',
            'age': 45, 'gender': 'Male', 'marital_status': 'Married', 'number_of_dependents': 2,
            'employment_type': 'Freelancer', 'employment_experience': 3,
            'monthly_income': 38000, 'annual_income': 456000, 'credit_score': 540,
            'loan_amount': 200000, 'loan_purpose': 'Vehicle', 'loan_term': 24,
            'interest_rate': 14.5, 'existing_loans': 3, 'prediction': 'Default',
            'default_probability': 0.8162, 'risk_level': 'High Risk',
            'actual_outcome': 'Default',
            'created_at': dt_str(35, 9, 20)
        },
        {
            'user_id': 2,
            'applicant_name': 'Ananya Deshmukh',
            'age': 29, 'gender': 'Female', 'marital_status': 'Single', 'number_of_dependents': 0,
            'employment_type': 'Salaried', 'employment_experience': 4,
            'monthly_income': 72000, 'annual_income': 864000, 'credit_score': 760,
            'loan_amount': 280000, 'loan_purpose': 'Home', 'loan_term': 36,
            'interest_rate': 8.2, 'existing_loans': 0, 'prediction': 'Non-Default',
            'default_probability': 0.1420, 'risk_level': 'Low Risk',
            'actual_outcome': 'Non-Default',
            'created_at': dt_str(42, 13, 15)
        },
        {
            'user_id': 1,
            'applicant_name': 'Rajesh Gupta',
            'age': 41, 'gender': 'Male', 'marital_status': 'Married', 'number_of_dependents': 2,
            'employment_type': 'Salaried', 'employment_experience': 8,
            'monthly_income': 61000, 'annual_income': 732000, 'credit_score': 710,
            'loan_amount': 300000, 'loan_purpose': 'Personal', 'loan_term': 36,
            'interest_rate': 10.2, 'existing_loans': 1, 'prediction': 'Non-Default',
            'default_probability': 0.2250, 'risk_level': 'Low Risk',
            'actual_outcome': 'Non-Default',
            'created_at': dt_str(50, 11, 0)
        },
        {
            'user_id': 1,
            'applicant_name': 'Kavita Verma',
            'age': 36, 'gender': 'Female', 'marital_status': 'Divorced', 'number_of_dependents': 1,
            'employment_type': 'Self-Employed', 'employment_experience': 5,
            'monthly_income': 43000, 'annual_income': 516000, 'credit_score': 595,
            'loan_amount': 220000, 'loan_purpose': 'Business', 'loan_term': 24,
            'interest_rate': 13.8, 'existing_loans': 2, 'prediction': 'Default',
            'default_probability': 0.6890, 'risk_level': 'High Risk',
            'actual_outcome': 'Default',
            'created_at': dt_str(65, 15, 30)
        },
        {
            'user_id': 1,
            'applicant_name': 'Meera Joshi',
            'age': 48, 'gender': 'Female', 'marital_status': 'Married', 'number_of_dependents': 3,
            'employment_type': 'Unemployed', 'employment_experience': 1,
            'monthly_income': 22000, 'annual_income': 264000, 'credit_score': 510,
            'loan_amount': 150000, 'loan_purpose': 'Personal', 'loan_term': 18,
            'interest_rate': 16.0, 'existing_loans': 3, 'prediction': 'Default',
            'default_probability': 0.8870, 'risk_level': 'High Risk',
            'actual_outcome': 'Default',
            'created_at': dt_str(85, 16, 0)
        }
    ]

    for r in records:
        conn.execute('''
            INSERT INTO predictions (
                user_id, applicant_name, age, gender, marital_status, number_of_dependents,
                employment_type, employment_experience, monthly_income, annual_income,
                credit_score, loan_amount, loan_purpose, loan_term, interest_rate,
                existing_loans, prediction, default_probability, risk_level,
                shap_explanation, lime_explanation, actual_outcome, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            r.get('user_id', 1),
            r['applicant_name'], r['age'], r['gender'], r['marital_status'], r['number_of_dependents'],
            r['employment_type'], r['employment_experience'], r['monthly_income'], r['annual_income'],
            r['credit_score'], r['loan_amount'], r['loan_purpose'], r['loan_term'], r['interest_rate'],
            r['existing_loans'], r['prediction'], r['default_probability'], r['risk_level'],
            json.dumps([]), json.dumps([]), r['actual_outcome'], r['created_at']
        ))
    conn.commit()


def align_seed_dates(conn):
    """Ensure existing seed dates match current active year."""
    now = datetime.now()
    rows = conn.execute('SELECT id, created_at FROM predictions').fetchall()
    for row in rows:
        old_date_str = row['created_at']
        try:
            old_dt = datetime.strptime(old_date_str, '%Y-%m-%d %H:%M:%S')
            if old_dt.year != now.year:
                new_dt = old_dt.replace(year=now.year)
                if new_dt > now:
                    new_dt = new_dt.replace(month=max(1, now.month - 1))
                conn.execute('UPDATE predictions SET created_at = ? WHERE id = ?', (
                    new_dt.strftime('%Y-%m-%d %H:%M:%S'), row['id']
                ))
        except Exception:
            pass
    conn.commit()


# =========================================================
# User-Isolated Dashboard & Query Logic
# =========================================================

def get_user_dashboard_stats(user_id):
    """
    Calculate comprehensive, real-time statistics for the authenticated USER dashboard.
    Returns:
      - Latest prediction, probability, risk category, credit score, loan amount
      - Percentage change from previous prediction
      - Loan progress tracker state
      - Risk overview distribution (% Low, % Med, % High)
      - Probability trend timeline for line chart
      - Recent predictions list
      - Data-driven key insights and personalized recommendations
    """
    conn = get_db()
    user = get_user_by_id(user_id)
    
    rows = conn.execute(
        'SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC', (user_id,)
    ).fetchall()
    conn.close()

    predictions = []
    for row in rows:
        r = dict(row)
        try:
            r['shap_explanation'] = json.loads(r.get('shap_explanation', '[]'))
        except Exception:
            r['shap_explanation'] = []
        try:
            r['lime_explanation'] = json.loads(r.get('lime_explanation', '[]'))
        except Exception:
            r['lime_explanation'] = []
        
        created_str = r.get('created_at', '')
        try:
            dt = datetime.strptime(created_str, '%Y-%m-%d %H:%M:%S')
            r['formatted_date'] = dt.strftime('%d %b %Y')
            r['chart_date'] = dt.strftime('%b %d')
            r['month_label'] = dt.strftime('%b')
        except Exception:
            r['formatted_date'] = created_str
            r['chart_date'] = created_str
            r['month_label'] = 'Recent'
        predictions.append(r)

    # Empty State Handling
    if not predictions:
        credit = user.get('credit_score', 650) if user else 650
        credit_rating = "Excellent" if credit >= 750 else ("Good" if credit >= 700 else ("Fair" if credit >= 600 else "Needs Work"))
        return {
            'has_predictions': False,
            'user': user,
            'credit_score': credit,
            'credit_rating': credit_rating,
            'latest_prediction': None,
            'probability_display': '0.00%',
            'probability_val': 0.0,
            'prob_change_text': 'No prior predictions',
            'prob_change_dir': 'neutral',
            'risk_category': 'PENDING ASSESSMENT',
            'risk_badge_class': 'badge-neutral',
            'loan_amount_display': '₹0',
            'loan_progress_step': 1,
            'risk_overview': {'low': 0.0, 'medium': 0.0, 'high': 0.0, 'low_count': 0, 'med_count': 0, 'high_count': 0, 'total': 0},
            'probability_trend': {'labels': [], 'probabilities': []},
            'recent_predictions': [],
            'key_insights': [
                {'icon': 'fa-circle-info', 'title': 'Get Started with AI Underwriting', 'text': 'Submit your financial details to receive an instant machine-learning loan default assessment and XAI explanation.'}
            ],
            'recommendations': [
                'Complete your first loan assessment using the "Assess My Loan" wizard to generate custom AI recommendations.'
            ]
        }

    # 1. Latest & Previous Predictions
    latest = predictions[0]
    prev = predictions[1] if len(predictions) > 1 else None

    latest_prob = round(latest['default_probability'] * 100, 2)
    latest_result = latest['prediction'].upper()
    latest_risk = latest.get('risk_level', 'Low Risk').upper()
    
    credit_score = latest['credit_score'] or (user.get('credit_score') if user else 580)
    credit_rating = "Excellent" if credit_score >= 750 else ("Good" if credit_score >= 700 else ("Fair" if credit_score >= 580 else "Poor"))
    loan_amount = latest['loan_amount'] or 0

    # Probability Change vs Previous
    if prev:
        prev_prob = round(prev['default_probability'] * 100, 2)
        diff = round(latest_prob - prev_prob, 2)
        if diff < 0:
            prob_change_text = f"↓ {abs(diff)}% from last prediction"
            prob_change_dir = "down" # good (risk decreased)
        elif diff > 0:
            prob_change_text = f"↑ {abs(diff)}% from last prediction"
            prob_change_dir = "up" # risk increased
        else:
            prob_change_text = "Unchanged from last assessment"
            prob_change_dir = "neutral"
    else:
        prob_change_text = "Initial baseline assessment"
        prob_change_dir = "neutral"

    # Risk Category description & badge
    if latest_prob >= 70:
        risk_category = "HIGH RISK"
        risk_subtext = "Significant risk of default detected"
        risk_badge_class = "badge-danger"
    elif latest_prob >= 40:
        risk_category = "MEDIUM RISK"
        risk_subtext = "Moderate repayment risk detected"
        risk_badge_class = "badge-warning"
    else:
        risk_category = "LOW RISK"
        risk_subtext = "You are in a safe zone"
        risk_badge_class = "badge-success"

    # 2. Risk Overview Donut Distribution
    total_preds = len(predictions)
    low_count = sum(1 for p in predictions if (p['default_probability'] * 100) < 40)
    med_count = sum(1 for p in predictions if 40 <= (p['default_probability'] * 100) < 70)
    high_count = sum(1 for p in predictions if (p['default_probability'] * 100) >= 70)

    low_pct = round((low_count / total_preds) * 100, 1)
    med_pct = round((med_count / total_preds) * 100, 1)
    high_pct = round((high_count / total_preds) * 100, 1)

    # 3. Probability Trend (Chronological order)
    trend_preds = list(reversed(predictions[:8]))
    trend_labels = [p['chart_date'] for p in trend_preds]
    trend_probs = [round(p['default_probability'] * 100, 2) for p in trend_preds]

    # 4. Key Insights (Data-driven from user history & feature values)
    insights = []
    if prev and diff < 0:
        insights.append({
            'icon': 'fa-arrow-trend-down',
            'color': 'text-green',
            'title': 'Default Risk Reduction',
            'text': f"Your default probability decreased by {abs(diff)}% compared to your previous assessment."
        })
    elif prev and diff > 0:
        insights.append({
            'icon': 'fa-arrow-trend-up',
            'color': 'text-red',
            'title': 'Risk Escalation Notice',
            'text': f"Your default risk increased by {abs(diff)}% due to requested loan terms and active exposure."
        })

    if latest.get('employment_type') == 'Salaried' and latest.get('monthly_income', 0) >= 45000:
        insights.append({
            'icon': 'fa-briefcase',
            'color': 'text-blue',
            'title': 'Positive Income Stability',
            'text': f"Your salaried status and monthly earnings (₹{latest.get('monthly_income', 0):,.0f}) provide a positive protective buffer."
        })

    if credit_score < 650:
        insights.append({
            'icon': 'fa-gauge-high',
            'color': 'text-orange',
            'title': 'Credit Score Opportunity',
            'text': f"Your current credit score of {credit_score} ({credit_rating}) represents the single largest factor elevating your loan cost."
        })
    else:
        insights.append({
            'icon': 'fa-shield-check',
            'color': 'text-green',
            'title': 'Solid Credit Profile',
            'text': f"Your credit score of {credit_score} ({credit_rating}) strengthens your underwriting credibility."
        })

    insights.append({
        'icon': 'fa-award',
        'color': 'text-purple',
        'title': 'Risk Category Status',
        'text': f"You are currently positioned in the {risk_category} tier ({latest_subtext if 'latest_subtext' in locals() else risk_subtext})."
    })

    # 5. Personalized Recommendations based on model attributes
    recommendations = []
    if credit_score < 650:
        recommendations.append("Focus on improving your credit score above 680 by reducing credit utilization and ensuring zero late payments.")
    if latest.get('existing_loans', 0) >= 2:
        recommendations.append("Consolidate your active outstanding loans to lower your monthly debt-service-to-income obligations.")
    if latest.get('loan_amount', 0) > (latest.get('monthly_income', 1) * 6):
        recommendations.append(f"Consider adjusting your loan amount to under ₹{int(latest.get('monthly_income', 1) * 5):,} to improve debt burden ratios.")
    if not recommendations:
        recommendations.append("Maintain your consistent repayment history and stable income to unlock prime tier interest rate discounts.")

    # 6. Loan Progress Step (1: Details, 2: Predicted, 3: Explanation, 4: Recommended)
    progress_step = 4 if len(predictions) >= 1 else 1

    return {
        'has_predictions': True,
        'user': user,
        'latest_prediction': latest,
        'latest_result': latest_result,
        'probability_display': f"{latest_prob:.2f}%",
        'probability_val': latest_prob,
        'prob_change_text': prob_change_text,
        'prob_change_dir': prob_change_dir,
        'risk_category': risk_category,
        'risk_subtext': risk_subtext,
        'risk_badge_class': risk_badge_class,
        'credit_score': credit_score,
        'credit_rating': credit_rating,
        'loan_amount_display': f"₹{loan_amount:,.0f}",
        'loan_progress_step': progress_step,
        'risk_overview': {
            'low': low_pct,
            'medium': med_pct,
            'high': high_pct,
            'low_count': low_count,
            'med_count': med_count,
            'high_count': high_count,
            'total': total_preds
        },
        'probability_trend': {
            'labels': trend_labels,
            'probabilities': trend_probs
        },
        'recent_predictions': predictions[:5],
        'key_insights': insights,
        'recommendations': recommendations
    }


def get_user_predictions(user_id, filter_status=None, search_query=None):
    """
    Fetch all predictions strictly belonging to the specified user_id.
    Guarantees user data isolation.
    """
    conn = get_db()
    query = 'SELECT * FROM predictions WHERE user_id = ?'
    params = [user_id]

    if filter_status and filter_status != 'All':
        query += ' AND prediction = ?'
        params.append(filter_status)

    if search_query:
        query += ' AND (loan_purpose LIKE ? OR CAST(loan_amount AS TEXT) LIKE ?)'
        params.extend([f'%{search_query}%', f'%{search_query}%'])

    query += ' ORDER BY id DESC'
    rows = conn.execute(query, tuple(params)).fetchall()
    conn.close()

    results = []
    for row in rows:
        r = dict(row)
        try:
            r['shap_explanation'] = json.loads(r.get('shap_explanation', '[]'))
        except Exception:
            r['shap_explanation'] = []
        try:
            r['lime_explanation'] = json.loads(r.get('lime_explanation', '[]'))
        except Exception:
            r['lime_explanation'] = []
            
        created_str = r.get('created_at', '')
        try:
            dt = datetime.strptime(created_str, '%Y-%m-%d %H:%M:%S')
            r['formatted_date'] = dt.strftime('%d %b %Y')
        except Exception:
            r['formatted_date'] = created_str

        results.append(r)
    return results


def get_user_prediction_by_id(prediction_id, user_id):
    """
    Fetch a prediction by ID strictly verifying user ownership.
    Returns None if the prediction does not belong to the user.
    """
    conn = get_db()
    row = conn.execute('SELECT * FROM predictions WHERE id = ? AND user_id = ?', (prediction_id, user_id)).fetchone()
    conn.close()
    
    if row:
        r = dict(row)
        try:
            r['shap_explanation'] = json.loads(r.get('shap_explanation', '[]'))
        except Exception:
            r['shap_explanation'] = []
        try:
            r['lime_explanation'] = json.loads(r.get('lime_explanation', '[]'))
        except Exception:
            r['lime_explanation'] = []
        return r
    return None


def get_user_risk_analytics(user_id, period='all'):
    """
    Calculate dedicated personal risk analytics and correlations for a user.
    """
    conn = get_db()
    
    query = 'SELECT * FROM predictions WHERE user_id = ?'
    params = [user_id]
    
    now = datetime.now()
    if period == 'last_30_days':
        start_str = (now - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        query += ' AND created_at >= ?'
        params.append(start_str)
    elif period == 'last_6_months':
        start_str = (now - relativedelta(months=6)).strftime('%Y-%m-%d %H:%M:%S')
        query += ' AND created_at >= ?'
        params.append(start_str)
    elif period == 'year_to_date':
        start_str = datetime(now.year, 1, 1).strftime('%Y-%m-%d %H:%M:%S')
        query += ' AND created_at >= ?'
        params.append(start_str)

    query += ' ORDER BY id ASC'
    rows = conn.execute(query, tuple(params)).fetchall()
    conn.close()

    predictions = [dict(r) for r in rows]
    total = len(predictions)

    if not predictions:
        return {
            'total_assessments': 0,
            'avg_default_probability': 0.0,
            'risk_tier': 'N/A',
            'trend_labels': [],
            'trend_probs': [],
            'credit_scores': [],
            'loan_vs_prob': [],
            'income_vs_prob': [],
            'insights': ['No assessment history found for this period.']
        }

    probs = [p['default_probability'] * 100 for p in predictions]
    avg_prob = round(sum(probs) / total, 2)
    
    trend_labels = []
    for p in predictions:
        try:
            dt = datetime.strptime(p['created_at'], '%Y-%m-%d %H:%M:%S')
            trend_labels.append(dt.strftime('%b %d'))
        except Exception:
            trend_labels.append(p['created_at'][:10])

    credit_scores = [p.get('credit_score', 600) for p in predictions]
    loan_vs_prob = [{'x': round(p.get('loan_amount', 0) / 1000, 1), 'y': round(p['default_probability'] * 100, 1)} for p in predictions]
    income_vs_prob = [{'x': round(p.get('monthly_income', 0) / 1000, 1), 'y': round(p['default_probability'] * 100, 1)} for p in predictions]

    insights = []
    if len(probs) >= 2:
        if probs[-1] < probs[0]:
            insights.append(f"Your default probability has improved from {probs[0]:.1f}% to {probs[-1]:.1f}% over the tracked period.")
        else:
            insights.append(f"Your recent loan inquiries show an elevated risk level ({probs[-1]:.1f}%) compared to baseline ({probs[0]:.1f}%).")
    insights.append(f"Your personal average default risk across all evaluated applications is {avg_prob:.1f}%.")

    return {
        'total_assessments': total,
        'avg_default_probability': avg_prob,
        'trend_labels': trend_labels,
        'trend_probs': [round(x, 2) for x in probs],
        'credit_scores': credit_scores,
        'loan_vs_prob': loan_vs_prob,
        'income_vs_prob': income_vs_prob,
        'insights': insights
    }


# =========================================================
# Prediction Saving & Platform-wide Queries
# =========================================================

def save_prediction(data):
    """Save a new prediction record associated with a user_id."""
    conn = get_db()
    cursor = conn.execute('''
        INSERT INTO predictions (
            user_id, applicant_name, age, gender, marital_status, number_of_dependents,
            employment_type, employment_experience, monthly_income, annual_income,
            credit_score, loan_amount, loan_purpose, loan_term, interest_rate,
            existing_loans, prediction, default_probability, risk_level,
            shap_explanation, lime_explanation, actual_outcome, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data.get('user_id', 1),
        data.get('applicant_name', 'Unknown'),
        data.get('age'),
        data.get('gender'),
        data.get('marital_status'),
        data.get('number_of_dependents', 0),
        data.get('employment_type'),
        data.get('employment_experience'),
        data.get('monthly_income'),
        data.get('annual_income'),
        data.get('credit_score'),
        data.get('loan_amount'),
        data.get('loan_purpose'),
        data.get('loan_term'),
        data.get('interest_rate'),
        data.get('existing_loans', 0),
        data.get('prediction', 'Unknown'),
        data.get('default_probability', 0.0),
        data.get('risk_level', 'Unknown'),
        json.dumps(data.get('shap_explanation', [])),
        json.dumps(data.get('lime_explanation', [])),
        data.get('actual_outcome', None),
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_recent_predictions(limit=5):
    """Fetch the most recent predictions across the platform (for admin dashboard)."""
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM predictions ORDER BY id DESC LIMIT ?', (limit,)
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        r = dict(row)
        try:
            r['shap_explanation'] = json.loads(r.get('shap_explanation', '[]'))
        except Exception:
            r['shap_explanation'] = []
        try:
            r['lime_explanation'] = json.loads(r.get('lime_explanation', '[]'))
        except Exception:
            r['lime_explanation'] = []
        
        created_str = r.get('created_at', '')
        try:
            dt = datetime.strptime(created_str, '%Y-%m-%d %H:%M:%S')
            r['formatted_date'] = dt.strftime('%d %b %Y %I:%M %p')
        except Exception:
            r['formatted_date'] = created_str

        results.append(r)
    return results


def get_all_predictions(filter_status=None, search_query=None):
    """Fetch all predictions across the entire platform (for admin auditing)."""
    conn = get_db()
    query = 'SELECT * FROM predictions WHERE 1=1'
    params = []

    if filter_status and filter_status != 'All':
        query += ' AND prediction = ?'
        params.append(filter_status)

    if search_query:
        query += ' AND (applicant_name LIKE ? OR loan_purpose LIKE ?)'
        params.extend([f'%{search_query}%', f'%{search_query}%'])

    query += ' ORDER BY id DESC'
    rows = conn.execute(query, tuple(params)).fetchall()
    conn.close()

    results = []
    for row in rows:
        r = dict(row)
        try:
            r['shap_explanation'] = json.loads(r.get('shap_explanation', '[]'))
        except Exception:
            r['shap_explanation'] = []
        try:
            r['lime_explanation'] = json.loads(r.get('lime_explanation', '[]'))
        except Exception:
            r['lime_explanation'] = []
            
        created_str = r.get('created_at', '')
        try:
            dt = datetime.strptime(created_str, '%Y-%m-%d %H:%M:%S')
            r['formatted_date'] = dt.strftime('%b %d, %Y')
        except Exception:
            r['formatted_date'] = created_str

        results.append(r)
    return results


def get_prediction_by_id(prediction_id):
    """Fetch a single prediction by its ID (admin global query)."""
    conn = get_db()
    row = conn.execute('SELECT * FROM predictions WHERE id = ?', (prediction_id,)).fetchone()
    conn.close()
    
    if row:
        r = dict(row)
        try:
            r['shap_explanation'] = json.loads(r.get('shap_explanation', '[]'))
        except Exception:
            r['shap_explanation'] = []
        try:
            r['lime_explanation'] = json.loads(r.get('lime_explanation', '[]'))
        except Exception:
            r['lime_explanation'] = []
        return r
    return None


def get_dashboard_stats():
    """Calculate aggregate platform metrics for the executive admin dashboard."""
    conn = get_db()
    total = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    approved = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Non-Default'").fetchone()[0]
    defaulted = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Default'").fetchone()[0]
    avg_prob_row = conn.execute("SELECT AVG(default_probability) FROM predictions").fetchone()[0]
    avg_prob = round((avg_prob_row or 0) * 100, 2)
    user_count = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'user'").fetchone()[0]
    conn.close()

    default_rate = round((defaulted / total * 100), 2) if total > 0 else 0.0

    return {
        'total_predictions': f"{total:,}",
        'raw_total': total,
        'approved_loans': f"{approved:,}",
        'raw_approved': approved,
        'defaulted_loans': f"{defaulted:,}",
        'raw_defaulted': defaulted,
        'default_rate': default_rate,
        'avg_default_probability': avg_prob,
        'total_users': user_count
    }


def get_risk_distribution():
    """Get count of applications in each default risk bucket."""
    dist = {'0-20%': 0, '20-40%': 0, '40-60%': 0, '60-80%': 0, '80-100%': 0}
    conn = get_db()
    rows = conn.execute('SELECT default_probability FROM predictions').fetchall()
    conn.close()

    for row in rows:
        prob = (row[0] or 0) * 100
        if prob < 20: dist['0-20%'] += 1
        elif prob < 40: dist['20-40%'] += 1
        elif prob < 60: dist['40-60%'] += 1
        elif prob < 80: dist['60-80%'] += 1
        else: dist['80-100%'] += 1

    return dist


def get_model_performance_by_period(period='this_month'):
    """
    Calculate actual empirical model performance for predictions within the selected timeframe.
    """
    now = datetime.now()
    
    if period == 'this_month':
        start_dt = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        label = now.strftime('%B %Y')
    elif period == 'last_3_months':
        start_dt = (now - relativedelta(months=3)).replace(hour=0, minute=0, second=0, microsecond=0)
        label = f"{(now - relativedelta(months=3)).strftime('%b')} - {now.strftime('%b %Y')}"
    elif period == 'year_to_date':
        start_dt = datetime(now.year, 1, 1, 0, 0, 0)
        label = f"Jan 1, {now.year} - Present"
    else:
        start_dt = datetime(2000, 1, 1)
        label = "All Time"
        
    start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S')

    conn = get_db()
    total = conn.execute('SELECT COUNT(*) FROM predictions WHERE created_at >= ?', (start_str,)).fetchone()[0]
    defaulted = conn.execute("SELECT COUNT(*) FROM predictions WHERE created_at >= ? AND prediction = 'Default'", (start_str,)).fetchone()[0]
    approved = conn.execute("SELECT COUNT(*) FROM predictions WHERE created_at >= ? AND prediction = 'Non-Default'", (start_str,)).fetchone()[0]

    evaluated_rows = conn.execute(
        "SELECT prediction, actual_outcome FROM predictions WHERE created_at >= ? AND actual_outcome IS NOT NULL AND actual_outcome != ''",
        (start_str,)
    ).fetchall()
    
    evaluated_count = len(evaluated_rows)
    correct_count = sum(1 for r in evaluated_rows if r['prediction'] == r['actual_outcome'])
    incorrect_count = evaluated_count - correct_count

    if evaluated_count > 0:
        accuracy = round((correct_count / evaluated_count) * 100, 2)
        accuracy_status = 'available'
        accuracy_display = f"{accuracy}%"
        accuracy_message = f"Based on {evaluated_count} observed loan outcomes ({correct_count} correct, {incorrect_count} incorrect)."
    else:
        accuracy = None
        accuracy_status = 'unavailable'
        accuracy_display = "Unavailable"
        accuracy_message = "Accuracy unavailable – actual outcomes not yet recorded for recent loans."

    avg_prob_row = conn.execute('SELECT AVG(default_probability) FROM predictions WHERE created_at >= ?', (start_str,)).fetchone()[0]
    avg_default_probability = round((avg_prob_row or 0) * 100, 2)

    timeline_rows = conn.execute('''
        SELECT DATE(created_at) as pred_date,
               COUNT(*) as daily_total,
               SUM(CASE WHEN prediction = 'Default' THEN 1 ELSE 0 END) as daily_default,
               SUM(CASE WHEN actual_outcome IS NOT NULL AND actual_outcome != '' AND prediction = actual_outcome THEN 1 ELSE 0 END) as daily_correct,
               SUM(CASE WHEN actual_outcome IS NOT NULL AND actual_outcome != '' THEN 1 ELSE 0 END) as daily_evaluated
        FROM predictions
        WHERE created_at >= ?
        GROUP BY DATE(created_at)
        ORDER BY pred_date ASC
    ''', (start_str,)).fetchall()

    timeline_labels = []
    timeline_accuracy = []
    timeline_volume = []
    
    for r in timeline_rows:
        date_obj = datetime.strptime(r['pred_date'], '%Y-%m-%d')
        timeline_labels.append(date_obj.strftime('%b %d'))
        timeline_volume.append(r['daily_total'])
        if r['daily_evaluated'] > 0:
            timeline_accuracy.append(round((r['daily_correct'] / r['daily_evaluated']) * 100, 1))
        else:
            timeline_accuracy.append(None)

    conn.close()

    return {
        'period': period,
        'label': label,
        'start_date': start_str,
        'total_predictions': total,
        'approved_predictions': approved,
        'default_predictions': defaulted,
        'evaluated_predictions': evaluated_count,
        'correct_predictions': correct_count,
        'incorrect_predictions': incorrect_count,
        'accuracy': accuracy,
        'accuracy_status': accuracy_status,
        'accuracy_display': accuracy_display,
        'accuracy_message': accuracy_message,
        'benchmark_validation_accuracy': 89.67,
        'avg_default_probability': avg_default_probability,
        'timeline': {
            'labels': timeline_labels,
            'accuracy': timeline_accuracy,
            'volume': timeline_volume
        }
    }


def get_risk_analytics(period='all'):
    """
    Calculate comprehensive, data-driven platform risk metrics,
    distributions, and feature correlations for administrator review.
    """
    conn = get_db()
    
    now = datetime.now()
    if period == 'this_month':
        start_str = now.replace(day=1, hour=0, minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
        query_suffix = f"WHERE created_at >= '{start_str}'"
    elif period == 'last_3_months':
        start_str = (now - relativedelta(months=3)).strftime('%Y-%m-%d %H:%M:%S')
        query_suffix = f"WHERE created_at >= '{start_str}'"
    elif period == 'year_to_date':
        start_str = datetime(now.year, 1, 1).strftime('%Y-%m-%d %H:%M:%S')
        query_suffix = f"WHERE created_at >= '{start_str}'"
    else:
        query_suffix = ""

    total = conn.execute(f'SELECT COUNT(*) FROM predictions {query_suffix}').fetchone()[0]
    default_count = conn.execute(f"SELECT COUNT(*) FROM predictions {query_suffix} {'AND' if query_suffix else 'WHERE'} prediction = 'Default'").fetchone()[0]
    non_default_count = conn.execute(f"SELECT COUNT(*) FROM predictions {query_suffix} {'AND' if query_suffix else 'WHERE'} prediction = 'Non-Default'").fetchone()[0]
    
    high_risk_count = conn.execute(f"SELECT COUNT(*) FROM predictions {query_suffix} {'AND' if query_suffix else 'WHERE'} default_probability >= 0.70").fetchone()[0]
    med_risk_count = conn.execute(f"SELECT COUNT(*) FROM predictions {query_suffix} {'AND' if query_suffix else 'WHERE'} default_probability >= 0.40 AND default_probability < 0.70").fetchone()[0]
    low_risk_count = conn.execute(f"SELECT COUNT(*) FROM predictions {query_suffix} {'AND' if query_suffix else 'WHERE'} default_probability < 0.40").fetchone()[0]
    
    avg_prob_row = conn.execute(f"SELECT AVG(default_probability) FROM predictions {query_suffix}").fetchone()[0]
    avg_default_probability = round((avg_prob_row or 0) * 100, 2)
    default_rate = round((default_count / total * 100), 2) if total > 0 else 0.0

    highest_risk_rows = conn.execute(
        f"SELECT id, applicant_name, credit_score, loan_amount, default_probability, risk_level FROM predictions {query_suffix} ORDER BY default_probability DESC LIMIT 5"
    ).fetchall()
    highest_risk = [dict(r) for r in highest_risk_rows]

    lowest_risk_rows = conn.execute(
        f"SELECT id, applicant_name, credit_score, loan_amount, default_probability, risk_level FROM predictions {query_suffix} ORDER BY default_probability ASC LIMIT 5"
    ).fetchall()
    lowest_risk = [dict(r) for r in lowest_risk_rows]

    dist = {'0-20%': 0, '20-40%': 0, '40-60%': 0, '60-80%': 0, '80-100%': 0}
    prob_rows = conn.execute(f"SELECT default_probability FROM predictions {query_suffix}").fetchall()
    for row in prob_rows:
        prob = (row[0] or 0) * 100
        if prob < 20: dist['0-20%'] += 1
        elif prob < 40: dist['20-40%'] += 1
        elif prob < 60: dist['40-60%'] += 1
        elif prob < 80: dist['60-80%'] += 1
        else: dist['80-100%'] += 1

    corr_rows = conn.execute(
        f"SELECT credit_score, loan_amount, monthly_income, default_probability FROM predictions {query_suffix}"
    ).fetchall()
    
    scatter_credit = [{'x': r['credit_score'], 'y': round(r['default_probability'] * 100, 1)} for r in corr_rows]
    scatter_loan = [{'x': round(r['loan_amount'] / 1000, 1), 'y': round(r['default_probability'] * 100, 1)} for r in corr_rows]
    scatter_income = [{'x': round(r['monthly_income'] / 1000, 1), 'y': round(r['default_probability'] * 100, 1)} for r in corr_rows]

    monthly_rows = conn.execute(f'''
        SELECT strftime('%Y-%m', created_at) as yr_mo,
               COUNT(*) as total,
               SUM(CASE WHEN prediction = 'Default' THEN 1 ELSE 0 END) as defaults,
               AVG(default_probability) as avg_prob
        FROM predictions
        {query_suffix}
        GROUP BY yr_mo
        ORDER BY yr_mo ASC
    ''').fetchall()
    
    monthly_labels = []
    monthly_defaults = []
    monthly_totals = []
    for r in monthly_rows:
        if r['yr_mo']:
            try:
                dt = datetime.strptime(r['yr_mo'], '%Y-%m')
                monthly_labels.append(dt.strftime('%b %Y'))
            except Exception:
                monthly_labels.append(r['yr_mo'])
            monthly_defaults.append(r['defaults'])
            monthly_totals.append(r['total'])

    insights = []
    if corr_rows:
        scores = [r['credit_score'] for r in corr_rows if r['credit_score'] is not None]
        probs = [r['default_probability'] for r in corr_rows if r['default_probability'] is not None]
        if len(scores) > 1:
            import numpy as np
            corr_coef = np.corrcoef(scores, probs)[0, 1] if len(scores) > 1 else 0
            if corr_coef < -0.3:
                insights.append({
                    'title': 'Credit Score Inversely Associated with Default Risk',
                    'detail': f'Statistical analysis reveals a negative correlation (r = {corr_coef:.2f}) between credit score and default probability. Applicants with lower credit scores (under 620) exhibit substantially higher predicted default incidence.',
                    'type': 'warning'
                })
            
        high_loan_defaults = sum(1 for r in corr_rows if r['loan_amount'] > 250000 and r['default_probability'] >= 0.50)
        total_high_loan = sum(1 for r in corr_rows if r['loan_amount'] > 250000)
        if total_high_loan > 0:
            rate = round((high_loan_defaults / total_high_loan) * 100, 1)
            insights.append({
                'title': 'Elevated Risk Concentration in High-Value Loans',
                'detail': f'{rate}% of loan applications exceeding ₹2,50,000 are classified as elevated risk. Increased principal exposure amplifies repayment burden relative to disposable income.',
                'type': 'info'
            })

    insights.append({
        'title': 'Statistical Correlation vs Causation Notice',
        'detail': 'Observed associations highlight statistical patterns identified by the ensemble model. They represent predictive correlations across applicant features rather than direct causative mechanisms.',
        'type': 'neutral'
    })

    conn.close()

    return {
        'total_applicants': total,
        'default_count': default_count,
        'non_default_count': non_default_count,
        'default_rate': default_rate,
        'high_risk_count': high_risk_count,
        'med_risk_count': med_risk_count,
        'low_risk_count': low_risk_count,
        'avg_default_probability': avg_default_probability,
        'highest_risk': highest_risk,
        'lowest_risk': lowest_risk,
        'risk_distribution': dist,
        'scatter_credit': scatter_credit,
        'scatter_loan': scatter_loan,
        'scatter_income': scatter_income,
        'monthly_trend': {
            'labels': monthly_labels,
            'defaults': monthly_defaults,
            'totals': monthly_totals
        },
        'insights': insights
    }


# =========================================================
# Loan Applications & Document Management
# =========================================================

def create_loan_application(user_id, loan_type, loan_amount, loan_term, loan_purpose, applicant_data, risk_assessment_id=None):
    """
    Create a new structured loan application record with a unique tracking ID (e.g., LP-2026-XXXX).
    """
    conn = get_db()
    while True:
        app_code = f"LP-2026-{random.randint(1000, 9999)}"
        exists = conn.execute('SELECT 1 FROM loan_applications WHERE application_id = ?', (app_code,)).fetchone()
        if not exists:
            break

    cursor = conn.execute('''
        INSERT INTO loan_applications (
            application_id, user_id, loan_type, loan_amount, loan_term,
            loan_purpose, applicant_data, status, risk_assessment_id, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'Submitted', ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    ''', (
        app_code,
        user_id,
        loan_type,
        float(loan_amount),
        int(loan_term),
        loan_purpose,
        json.dumps(applicant_data) if isinstance(applicant_data, dict) else str(applicant_data),
        risk_assessment_id
    ))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id, app_code


def get_user_applications(user_id):
    """Fetch all loan applications submitted by user_id."""
    conn = get_db()
    rows = conn.execute('''
        SELECT * FROM loan_applications WHERE user_id = ? ORDER BY id DESC
    ''', (user_id,)).fetchall()
    conn.close()
    apps = []
    for r in rows:
        d = dict(r)
        d['application_code'] = d.get('application_id', '')
        try:
            d['applicant_data'] = json.loads(d.get('applicant_data', '{}'))
        except Exception:
            d['applicant_data'] = {}
        apps.append(d)
    return apps


def get_user_application_by_id(app_id, user_id):
    """Strictly user-isolated retrieval of an application record."""
    conn = get_db()
    row = conn.execute('''
        SELECT * FROM loan_applications WHERE id = ? AND user_id = ?
    ''', (app_id, user_id)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d['application_code'] = d.get('application_id', '')
    try:
        d['applicant_data'] = json.loads(d.get('applicant_data', '{}'))
    except Exception:
        d['applicant_data'] = {}
    return d


def update_application_status(app_id, user_id, new_status, notes=None):
    """Update status of a loan application (e.g. Withdrawn by user or admin updated)."""
    conn = get_db()
    if notes:
        conn.execute('''
            UPDATE loan_applications SET status = ?, notes = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
        ''', (new_status, notes, app_id, user_id))
    else:
        conn.execute('''
            UPDATE loan_applications SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
        ''', (new_status, app_id, user_id))
    conn.commit()
    conn.close()


def add_loan_document(application_id, user_id, document_type, original_filename, stored_filename, file_path, file_size):
    """Store an uploaded verification document metadata record."""
    conn = get_db()
    cursor = conn.execute('''
        INSERT INTO loan_documents (
            application_id, user_id, document_type, original_filename, stored_filename, file_path, file_size, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'Uploaded')
    ''', (application_id, user_id, document_type, original_filename, stored_filename, file_path, file_size))
    conn.commit()
    doc_id = cursor.lastrowid
    conn.close()
    return doc_id


def get_application_documents(application_id, user_id):
    """Retrieve all uploaded documents for a specific application."""
    conn = get_db()
    rows = conn.execute('''
        SELECT * FROM loan_documents WHERE application_id = ? AND user_id = ? ORDER BY id ASC
    ''', (application_id, user_id)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_document_by_id(doc_id, user_id):
    """Fetch single document metadata with strict user authorization check."""
    conn = get_db()
    row = conn.execute('''
        SELECT * FROM loan_documents WHERE id = ? AND user_id = ?
    ''', (doc_id, user_id)).fetchone()
    conn.close()
    return dict(row) if row else None


# =========================================================
# User Settings Management
# =========================================================

def get_user_settings(user_id):
    """Fetch settings for a user, creating defaults if not yet present."""
    conn = get_db()
    row = conn.execute('SELECT * FROM user_settings WHERE user_id = ?', (user_id,)).fetchone()
    if not row:
        conn.execute('''
            INSERT OR IGNORE INTO user_settings (user_id) VALUES (?)
        ''', (user_id,))
        conn.commit()
        row = conn.execute('SELECT * FROM user_settings WHERE user_id = ?', (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else {
        'user_id': user_id,
        'notify_assessment': 1,
        'notify_risk_shift': 1,
        'notify_monthly_digest': 1,
        'notify_sms_auth': 1,
        'notify_market_rates': 0,
        'enable_2fa': 0,
        'session_timeout': 1,
        'export_shap_table': 1,
        'export_lime_table': 1,
        'anonymized_telemetry': 1,
        'currency_format': 'INR',
        'date_format': 'DD-MM-YYYY',
        'high_contrast_charts': 0
    }


def update_user_settings(user_id, form_data):
    """Update settings for a user from form data."""
    conn = get_db()
    notify_assessment = 1 if form_data.get('notify_assessment') else 0
    notify_risk_shift = 1 if form_data.get('notify_risk_shift') else 0
    notify_monthly_digest = 1 if form_data.get('notify_monthly_digest') else 0
    notify_sms_auth = 1 if form_data.get('notify_sms_auth') else 0
    notify_market_rates = 1 if form_data.get('notify_market_rates') else 0
    enable_2fa = 1 if form_data.get('enable_2fa') else 0
    session_timeout = 1 if form_data.get('session_timeout') else 0
    export_shap_table = 1 if form_data.get('export_shap_table') else 0
    export_lime_table = 1 if form_data.get('export_lime_table') else 0
    anonymized_telemetry = 1 if form_data.get('anonymized_telemetry') else 0
    currency_format = form_data.get('currency_format', 'INR')
    date_format = form_data.get('date_format', 'DD-MM-YYYY')
    high_contrast_charts = 1 if form_data.get('high_contrast_charts') else 0

    conn.execute('''
        INSERT INTO user_settings (
            user_id, notify_assessment, notify_risk_shift, notify_monthly_digest,
            notify_sms_auth, notify_market_rates, enable_2fa, session_timeout,
            export_shap_table, export_lime_table, anonymized_telemetry,
            currency_format, date_format, high_contrast_charts, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(user_id) DO UPDATE SET
            notify_assessment = excluded.notify_assessment,
            notify_risk_shift = excluded.notify_risk_shift,
            notify_monthly_digest = excluded.notify_monthly_digest,
            notify_sms_auth = excluded.notify_sms_auth,
            notify_market_rates = excluded.notify_market_rates,
            enable_2fa = excluded.enable_2fa,
            session_timeout = excluded.session_timeout,
            export_shap_table = excluded.export_shap_table,
            export_lime_table = excluded.export_lime_table,
            anonymized_telemetry = excluded.anonymized_telemetry,
            currency_format = excluded.currency_format,
            date_format = excluded.date_format,
            high_contrast_charts = excluded.high_contrast_charts,
            updated_at = CURRENT_TIMESTAMP
    ''', (
        user_id, notify_assessment, notify_risk_shift, notify_monthly_digest,
        notify_sms_auth, notify_market_rates, enable_2fa, session_timeout,
        export_shap_table, export_lime_table, anonymized_telemetry,
        currency_format, date_format, high_contrast_charts
    ))
    conn.commit()
    conn.close()
    return get_user_settings(user_id)


# =========================================================
# Institutional Loan Recommendations
# =========================================================

def get_financial_institutions(loan_type=None, credit_score=None, monthly_income=None):
    """
    Return curated, real-world financial institutions with transparent eligibility criteria.
    Scores suitability based on applicant's actual credit profile.
    """
    institutions = [
        {
            'name': 'State Bank of India (SBI)',
            'code': 'sbi',
            'type': 'Public Sector Bank',
            'supported_loans': ['Personal', 'Home', 'Land', 'Education', 'Vehicle', 'Agricultural', 'Business'],
            'min_credit_score': 650,
            'preferred_credit_score': 720,
            'min_income': 25000,
            'interest_range': '8.40% – 11.15%',
            'processing_fee': '0.50% – 1.0%',
            'max_loan_limit': 'Up to ₹20,00,000 (Personal) / ₹1,50,00,000 (Home)',
            'key_features': 'Lowest repo-rate linked APR, zero prepayment penalty for floating loans, special concessions for women & agriculture.',
            'eligibility_note': 'Requires minimum 1 year employment stability or 2 years IT returns for self-employed.'
        },
        {
            'name': 'HDFC Bank',
            'code': 'hdfc',
            'type': 'Private Commercial Bank',
            'supported_loans': ['Personal', 'Home', 'Business', 'Vehicle', 'Education'],
            'min_credit_score': 700,
            'preferred_credit_score': 750,
            'min_income': 35000,
            'interest_range': '9.50% – 12.75%',
            'processing_fee': 'Up to ₹4,999',
            'max_loan_limit': 'Up to ₹40,00,000 (Personal)',
            'key_features': 'Instant 10-second digital disbursement for pre-approved salary holders, transparent digital loan journey.',
            'eligibility_note': 'Prefers prime score borrowers (720+) with salaried corporate profiles.'
        },
        {
            'name': 'ICICI Bank',
            'code': 'icici',
            'type': 'Private Commercial Bank',
            'supported_loans': ['Personal', 'Home', 'Land', 'Vehicle', 'Business'],
            'min_credit_score': 680,
            'preferred_credit_score': 740,
            'min_income': 30000,
            'interest_range': '9.80% – 13.50%',
            'processing_fee': '1.0% – 2.0%',
            'max_loan_limit': 'Up to ₹50,00,000 (Personal)',
            'key_features': 'Flexible repayment tenure up to 72 months, minimal physical paperwork, competitive car & home loans.',
            'eligibility_note': 'Considers income stability and debt-to-income under 45%.'
        },
        {
            'name': 'Axis Bank',
            'code': 'axis',
            'type': 'Private Commercial Bank',
            'supported_loans': ['Personal', 'Home', 'Education', 'Vehicle', 'Business'],
            'min_credit_score': 675,
            'preferred_credit_score': 730,
            'min_income': 28000,
            'interest_range': '9.90% – 13.99%',
            'processing_fee': '1.0% – 1.5%',
            'max_loan_limit': 'Up to ₹40,00,000',
            'key_features': 'Express 24-hour turnaround, education loan coverage up to 100% of overseas tuition.',
            'eligibility_note': 'Open to both salaried and professional self-employed applicants.'
        },
        {
            'name': 'Bank of Baroda',
            'code': 'bob',
            'type': 'Public Sector Bank',
            'supported_loans': ['Personal', 'Home', 'Land', 'Agricultural', 'Vehicle', 'Education'],
            'min_credit_score': 650,
            'preferred_credit_score': 710,
            'min_income': 22000,
            'interest_range': '8.60% – 11.50%',
            'processing_fee': 'Nil to 0.75%',
            'max_loan_limit': 'Up to ₹25,00,000',
            'key_features': 'Concessional rates for government/PSU employees and agricultural plot developments.',
            'eligibility_note': 'Accepts agricultural and rural land documentation.'
        },
        {
            'name': 'Tata Capital',
            'code': 'tata',
            'type': 'Non-Banking Financial Company (NBFC)',
            'supported_loans': ['Personal', 'Business', 'Vehicle', 'Home'],
            'min_credit_score': 640,
            'preferred_credit_score': 700,
            'min_income': 25000,
            'interest_range': '10.50% – 15.00%',
            'processing_fee': '1.5% – 2.5%',
            'max_loan_limit': 'Up to ₹35,00,000',
            'key_features': 'Flexible underwriting criteria, accommodates diverse income streams and small business cash flows.',
            'eligibility_note': 'Suitable for applicants recovering credit scores or with non-standard salary credit.'
        },
        {
            'name': 'Bajaj Finserv',
            'code': 'bajaj',
            'type': 'Non-Banking Financial Company (NBFC)',
            'supported_loans': ['Personal', 'Business', 'Home'],
            'min_credit_score': 650,
            'preferred_credit_score': 720,
            'min_income': 30000,
            'interest_range': '10.00% – 16.00%',
            'processing_fee': 'Up to 3.0%',
            'max_loan_limit': 'Up to ₹40,00,000',
            'key_features': 'Flexi-hybrid loan facility (pay interest only on drawn amount, lower initial EMIs).',
            'eligibility_note': 'Fast approvals; flexible EMI structure beneficial for short-term liquidity needs.'
        }
    ]

    results = []
    user_score = int(credit_score or 700)
    user_inc = float(monthly_income or 50000)

    for inst in institutions:
        matches_loan = (not loan_type) or (loan_type in inst['supported_loans'])
        meets_score = user_score >= inst['min_credit_score']
        meets_income = user_inc >= inst['min_income']

        if meets_score and meets_income and user_score >= inst['preferred_credit_score']:
            suitability = 'High Suitability'
            suit_class = 'badge-success'
            suit_reason = f"Your credit score ({user_score}) meets their prime benchmark ({inst['preferred_credit_score']}+) for preferential rates."
        elif meets_score and meets_income:
            suitability = 'Moderate Suitability'
            suit_class = 'badge-warning'
            suit_reason = "Your profile satisfies all minimum thresholds. Standard documentation will be required."
        elif not meets_score:
            suitability = 'Credit Review Required'
            suit_class = 'badge-danger'
            suit_reason = f"Minimum credit score benchmark is {inst['min_credit_score']}. Consider improving your credit profile before applying."
        else:
            suitability = 'Income Verification Required'
            suit_class = 'badge-warning'
            suit_reason = f"Minimum monthly income requirement is ₹{inst['min_income']:,}."

        inst_copy = dict(inst)
        inst_copy['suitability'] = suitability
        inst_copy['suit_class'] = suit_class
        inst_copy['suit_reason'] = suit_reason
        inst_copy['is_recommended'] = matches_loan and meets_score and meets_income

        if matches_loan:
            results.append(inst_copy)

    priority = {'High Suitability': 0, 'Moderate Suitability': 1, 'Income Verification Required': 2, 'Credit Review Required': 3}
    results.sort(key=lambda x: priority.get(x['suitability'], 9))
    return results
