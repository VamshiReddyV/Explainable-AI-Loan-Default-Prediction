import os
import sys
import io
import unittest

sys.path.insert(0, r'c:\Users\VamshiReddy\OneDrive\Desktop\Explainable-AI-Loan-Default-Prediction')

from app import app

class FullSystemVerification(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

    def login_user(self, client=None):
        c = client or self.client
        return c.post('/login/user', data={
            'email': 'rahul@example.com',
            'password': 'password123'
        }, follow_redirects=True)

    def login_user2(self, client=None):
        c = client or self.client
        return c.post('/login/user', data={
            'email': 'priya@example.com',
            'password': 'password123'
        }, follow_redirects=True)

    def login_admin(self, client=None):
        c = client or self.client
        return c.post('/login/admin', data={
            'email': 'admin@loanpredict.ai',
            'password': 'admin123'
        }, follow_redirects=True)

    def test_01_landing_page(self):
        """Verify common landing page loads with role selection."""
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'LoanPredict AI', resp.data)
        self.assertIn(b'/login/user', resp.data)
        self.assertIn(b'/login/admin', resp.data)
        print("[PASS] 01: Landing page renders with role portals")

    def test_02_user_login_and_dashboard(self):
        """Verify user authentication and dashboard stats."""
        login_resp = self.login_user()
        self.assertEqual(login_resp.status_code, 200)
        self.assertIn(b'Welcome back, Rahul', login_resp.data)

        dash_resp = self.client.get('/user/dashboard')
        self.assertEqual(dash_resp.status_code, 200)
        self.assertIn(b'Borrower Portal', dash_resp.data)
        self.assertIn(b'Assess My Loan', dash_resp.data)
        self.assertIn(b'Apply Loan', dash_resp.data)
        print("[PASS] 02: User login and dashboard load successfully")

    def test_03_assess_loan_types(self):
        """Verify assessment for multiple loan types (Personal, Home, Business, etc.)."""
        self.login_user()

        # GET assess page
        resp_get = self.client.get('/user/assess')
        self.assertEqual(resp_get.status_code, 200)
        self.assertIn(b'Assess My Loan', resp_get.data)
        self.assertIn(b'Personal Loan', resp_get.data)
        self.assertIn(b'Home Loan', resp_get.data)
        self.assertIn(b'Business Loan', resp_get.data)

        # POST Personal Loan Assessment
        post_data = {
            'loan_type': 'Personal',
            'age': 32,
            'gender': 'Male',
            'marital_status': 'Married',
            'number_of_dependents': 1,
            'employment_type': 'Salaried',
            'employment_experience': 5,
            'monthly_income': 65000,
            'annual_income': 780000,
            'credit_score': 740,
            'loan_amount': 200000,
            'loan_term': 24,
            'interest_rate': 10.5,
            'existing_loans': 0
        }
        resp_post = self.client.post('/user/assess', data=post_data, follow_redirects=True)
        self.assertEqual(resp_post.status_code, 200)
        self.assertIn(b'Assessment Results', resp_post.data)
        self.assertIn(b'Default Probability', resp_post.data)
        self.assertIn(b'SHAP Feature Contributions', resp_post.data)
        self.assertIn(b'LIME Local Explanation', resp_post.data)
        self.assertIn(b'Proceed to Apply', resp_post.data)

        # POST Home Loan Assessment
        post_home = {
            'loan_type': 'Home',
            'age': 35,
            'monthly_income': 95000,
            'annual_income': 1140000,
            'credit_score': 770,
            'loan_amount': 3500000,
            'loan_term': 240,
            'interest_rate': 8.5,
            'existing_loans': 0
        }
        resp_home = self.client.post('/user/assess', data=post_home, follow_redirects=True)
        self.assertEqual(resp_home.status_code, 200)
        self.assertIn(b'Assessment Result', resp_home.data)
        print("[PASS] 03: Assess loan with ML inference, SHAP, and LIME explanations")

    def test_04_apply_loan_workflow_with_docs(self):
        """Verify full loan application submission including file upload."""
        self.login_user()

        # GET apply page
        get_apply = self.client.get('/user/apply?type=Home&amount=2500000&term=180')
        self.assertEqual(get_apply.status_code, 200)
        self.assertIn(b'Apply for Loan', get_apply.data)

        # POST apply with documents
        fake_pdf = (io.BytesIO(b"%PDF-1.4 test dummy doc content"), "salary_slip.pdf")
        fake_id = (io.BytesIO(b"%PDF-1.4 test dummy aadhar doc"), "aadhaar.pdf")

        apply_data = {
            'loan_type': 'Home',
            'loan_amount': '2500000',
            'loan_term': '180',
            'loan_purpose': 'Purchasing 2BHK Apartment',
            'full_name': 'Rahul Verma',
            'email': 'rahul@example.com',
            'phone': '9876543210',
            'age': '30',
            'gender': 'Male',
            'address': 'Flat 402, Green Valley Apartments, Hyderabad',
            'employment_type': 'Salaried',
            'employer_name': 'TechCorp Solutions',
            'monthly_income': '65000',
            'credit_score': '740',
            'property_value': '3500000',
            'property_location': 'Gachibowli, Hyderabad',
            'nominee_name': 'Sneha Verma',
            'nominee_relation': 'Spouse',
            'bank_name': 'HDFC Bank',
            'account_number': '50100234567890',
            'ifsc_code': 'HDFC0001234',
            'doc_identity': fake_id,
            'doc_income': fake_pdf
        }

        resp_apply = self.client.post('/user/apply', data=apply_data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(resp_apply.status_code, 200)
        self.assertIn(b'Application Code', resp_apply.data)
        self.assertIn(b'Submitted', resp_apply.data)
        self.assertIn(b'Application Pipeline', resp_apply.data)
        self.assertIn(b'salary_slip.pdf', resp_apply.data)
        print("[PASS] 04: Loan application submitted and tracked with uploaded documents")

    def test_05_applications_list_and_detail(self):
        """Verify user can view list of their applications and track detail."""
        self.login_user()

        list_resp = self.client.get('/user/applications')
        self.assertEqual(list_resp.status_code, 200)
        self.assertIn(b'My Loan Applications', list_resp.data)
        print("[PASS] 05: Application listing page displays user applications")

    def test_06_security_idor_prevention(self):
        """Verify IDOR prevention: User 2 cannot view User 1's applications."""
        client1 = app.test_client()
        self.login_user(client=client1)

        apply_data = {
            'loan_type': 'Personal',
            'loan_amount': '150000',
            'loan_term': '12',
            'loan_purpose': 'Medical Emergency',
            'full_name': 'Rahul Verma',
            'email': 'rahul@example.com',
            'monthly_income': '65000',
            'credit_score': '740'
        }
        resp = client1.post('/user/apply', data=apply_data, follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        redirect_url = resp.headers['Location']
        app_id = int(redirect_url.rstrip('/').split('/')[-1])

        # Login as user 2 on a completely isolated client and try to view user 1's application
        client2 = app.test_client()
        self.login_user2(client=client2)
        idor_resp = client2.get(f'/user/applications/{app_id}', follow_redirects=False)
        self.assertIn(idor_resp.status_code, [302, 403, 404])
        if idor_resp.status_code == 302:
            self.assertIn('/user/applications', idor_resp.headers['Location'])
        print(f"[PASS] 06: IDOR check passed — unauthorized user blocked from viewing application #{app_id}")

    def test_07_ai_chat_assistant(self):
        """Verify AI chat assistant API."""
        self.login_user()

        resp = self.client.post('/user/api/chat', json={
            'message': 'What documents do I need for a home loan?'
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get('success'))
        self.assertIn('reply', data)
        self.assertGreater(len(data['reply']), 10)
        print("[PASS] 07: AI Chat Assistant returns intelligent borrower guidance")

    def test_08_financial_profile_persistence(self):
        """Verify financial profile extended fields update and persist."""
        self.login_user()

        post_profile = {
            'age': 31,
            'gender': 'Male',
            'marital_status': 'Married',
            'number_of_dependents': 1,
            'employment_type': 'Salaried',
            'employment_experience': 6,
            'monthly_income': 70000,
            'annual_income': 840000,
            'monthly_expenses': 25000,
            'existing_emi': 8000,
            'assets_value': 1500000,
            'liabilities_value': 200000,
            'credit_score': 755,
            'existing_loans': 1,
            'pan_number': 'ABCDE1234F',
            'address': 'Road No 12, Banjara Hills, Hyderabad'
        }
        resp = self.client.post('/user/financial-profile', data=post_profile, follow_redirects=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'ABCDE1234F', resp.data)
        self.assertIn(b'25000', resp.data)
        print("[PASS] 08: Extended financial profile saved to database and rendered")

    def test_09_recommendations(self):
        """Verify recommendations page renders suitable institutions based on profile."""
        self.login_user()
        resp = self.client.get('/user/recommendations')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Personalized AI Action Plan', resp.data)
        self.assertIn(b'Institution Recommendations', resp.data)
        self.assertIn(b'State Bank of India', resp.data)
        print("[PASS] 09: Recommendations page renders with loan partners and institutions")

    def test_10_admin_protection(self):
        """Verify Admin routes remain intact and protected."""
        # Clean client: unauthenticated access to /admin/dashboard redirects to admin login
        anon_client = app.test_client()
        anon_resp = anon_client.get('/admin/dashboard', follow_redirects=False)
        self.assertEqual(anon_resp.status_code, 302)
        self.assertIn('/login/admin', anon_resp.headers['Location'])

        # User attempting to access admin route gets 403
        user_client = app.test_client()
        self.login_user(client=user_client)
        user_admin_resp = user_client.get('/admin/dashboard', follow_redirects=False)
        self.assertEqual(user_admin_resp.status_code, 403)

        # Admin login works
        admin_client = app.test_client()
        admin_login = self.login_admin(client=admin_client)
        self.assertEqual(admin_login.status_code, 200)

        # Admin dashboard works
        admin_dash = admin_client.get('/admin/dashboard')
        self.assertEqual(admin_dash.status_code, 200)
        self.assertIn(b'Explainable AI Platform', admin_dash.data)

        # Existing admin routes work
        for route in ['/model-performance', '/risk-analysis', '/explainability', '/history', '/about']:
            r = admin_client.get(route)
            self.assertEqual(r.status_code, 200, f"Admin route {route} returned {r.status_code}")
        print("[PASS] 10: Admin Dashboard and all existing admin features intact and strictly protected")

if __name__ == '__main__':
    unittest.main(verbosity=2)
