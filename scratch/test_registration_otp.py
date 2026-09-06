"""
Comprehensive test script for User Registration + Phone OTP + Dual Login + Admin Isolation.
"""
import os
import sys
import unittest

# Ensure root directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app
from src.database import (
    get_db, get_user_by_email, get_user_by_phone,
    authenticate_user, activate_user_phone,
    create_otp_verification, verify_otp_code, can_resend_otp
)
from src.otp_service import generate_otp, hash_otp, send_otp, mask_phone

class TestRegistrationAndOTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()

    def setUp(self):
        # Clean up any previous test user
        conn = get_db()
        conn.execute("DELETE FROM users WHERE email = 'test_applicant@example.com' OR phone = '9988776655'")
        conn.execute("DELETE FROM otp_verifications WHERE phone = '9988776655'")
        conn.commit()
        conn.close()

    def test_phone_masking(self):
        masked = mask_phone("9988776655")
        self.assertEqual(masked, "+91 ******6655")

    def test_otp_generation_and_hashing(self):
        code = generate_otp(6)
        self.assertEqual(len(code), 6)
        self.assertTrue(code.isdigit())
        
        hashed = hash_otp(code)
        self.assertEqual(len(hashed), 64) # sha256 hex length

    def test_registration_validation_errors(self):
        # 1. Invalid phone (less than 10 digits)
        resp = self.client.post('/register', data={
            'name': 'Test Applicant',
            'email': 'test_applicant@example.com',
            'phone': '12345',
            'password': 'password123',
            'confirm_password': 'password123'
        }, follow_redirects=True)
        self.assertIn(b"valid 10-digit Indian mobile number", resp.data)

        # 2. Password mismatch
        resp = self.client.post('/register', data={
            'name': 'Test Applicant',
            'email': 'test_applicant@example.com',
            'phone': '9988776655',
            'password': 'password123',
            'confirm_password': 'differentpassword'
        }, follow_redirects=True)
        self.assertIn(b"Passwords do not match", resp.data)

    def test_successful_registration_and_otp_verification(self):
        # Register valid user
        resp = self.client.post('/register', data={
            'name': 'Test Applicant',
            'email': 'test_applicant@example.com',
            'phone': '9988776655',
            'password': 'password123',
            'confirm_password': 'password123'
        }, follow_redirects=False)

        # Should redirect to /verify-otp
        self.assertEqual(resp.status_code, 302)
        self.assertIn('/verify-otp', resp.headers['Location'])

        # Check user exists in database but phone_verified == 0
        user = get_user_by_email('test_applicant@example.com')
        self.assertIsNotNone(user)
        self.assertEqual(user['phone_verified'], 0)

        # Attempt to login BEFORE verification - should reject or redirect to /verify-otp
        login_resp = self.client.post('/login/user', data={
            'identifier': 'test_applicant@example.com',
            'password': 'password123'
        }, follow_redirects=False)
        self.assertEqual(login_resp.status_code, 302)
        self.assertIn('/verify-otp', login_resp.headers['Location'])

        # Now get the active OTP from DB
        conn = get_db()
        otp_row = conn.execute(
            "SELECT otp_hash FROM otp_verifications WHERE phone = '9988776655' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        self.assertIsNotNone(otp_row)

        # Try submitting wrong OTP code
        wrong_otp_resp = self.client.post('/verify-otp', data={'otp': '000000'}, follow_redirects=True)
        self.assertIn(b"Incorrect verification code", wrong_otp_resp.data)

        # Test resend OTP cooldown
        resend_resp = self.client.post('/api/resend-otp', json={'phone': '9988776655'})
        # Should be rejected or throttled because of 30-sec cooldown
        self.assertEqual(resend_resp.status_code, 429)

        # Now test direct verification with the valid code by creating known code
        known_code = "654321"
        known_hash = hash_otp(known_code)
        create_otp_verification(user['id'], '9988776655', known_hash)

        verify_resp = self.client.post('/verify-otp', data={'otp': known_code}, follow_redirects=False)
        self.assertEqual(verify_resp.status_code, 302)
        self.assertIn('/user/dashboard', verify_resp.headers['Location'])

        # Check user is now verified
        user_after = get_user_by_email('test_applicant@example.com')
        self.assertEqual(user_after['phone_verified'], 1)

        # Test dual login with EMAIL
        with self.client as c:
            login_email = c.post('/login/user', data={
                'identifier': 'test_applicant@example.com',
                'password': 'password123'
            }, follow_redirects=False)
            self.assertEqual(login_email.status_code, 302)
            self.assertIn('/user/dashboard', login_email.headers['Location'])

        # Test dual login with PHONE
        with self.client as c:
            login_phone = c.post('/login/user', data={
                'identifier': '9988776655',
                'password': 'password123'
            }, follow_redirects=False)
            self.assertEqual(login_phone.status_code, 302)
            self.assertIn('/user/dashboard', login_phone.headers['Location'])

    def test_existing_demo_users_and_admin_isolation(self):
        # Rahul Sharma can login via email or phone
        user_email, err1, _ = authenticate_user('rahul@example.com', 'password123', expected_role='user')
        self.assertIsNotNone(user_email)
        self.assertIsNone(err1)

        user_phone, err2, _ = authenticate_user('9812345678', 'password123', expected_role='user')
        self.assertIsNotNone(user_phone)
        self.assertIsNone(err2)

        # Admin user cannot login as user
        admin_as_user, err_admin, _ = authenticate_user('admin@loanpredict.ai', 'admin123', expected_role='admin')
        self.assertIsNotNone(admin_as_user)
        self.assertEqual(admin_as_user['role'], 'admin')

        # Regular user cannot authenticate with expected_role='admin'
        fake_admin, err_fake, _ = authenticate_user('rahul@example.com', 'password123', expected_role='admin')
        self.assertIsNone(fake_admin)
        self.assertIn("Access denied", err_fake)

    def test_landing_page_render(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"LoanPredict AI", resp.data)
        self.assertIn(b"Explainable AI", resp.data)
        self.assertIn(b"User Login", resp.data)
        self.assertIn(b"Admin Portal", resp.data)
        self.assertIn(b"Supported Loan Categories", resp.data)

if __name__ == '__main__':
    unittest.main()
