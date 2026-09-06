"""
AI Loan Assistant service layer for LoanPredict AI borrower portal.
Provides comprehensive answers regarding loan eligibility, document requirements,
loan types, application tracking, SHAP/LIME explainability, and financial profile optimization.
Supports external LLM providers via environment variables with a rich fallback intelligence engine.
"""
import os
import re

def get_ai_chat_response(user_message, user_context=None):
    """
    Generate an intelligent, context-aware answer for the borrower.
    user_context can include:
      - name
      - credit_score
      - monthly_income
      - active_applications (list of application summaries)
      - latest_risk_level
    """
    if not user_message or not user_message.strip():
        return "Please type a question regarding your loan assessment, eligibility, or application process."

    clean_msg = user_message.strip()
    msg_lower = clean_msg.lower()

    # Context variables
    user_name = user_context.get('name', 'Applicant') if user_context else 'Applicant'
    credit_score = user_context.get('credit_score', 700) if user_context else 700
    monthly_income = user_context.get('monthly_income', 50000) if user_context else 50000
    applications = user_context.get('active_applications', []) if user_context else []
    latest_risk = user_context.get('latest_risk_level', 'Low Risk') if user_context else 'Low Risk'

    # Check for external API (OpenAI / Anthropic / Gemini) if configured
    openai_key = os.environ.get('OPENAI_API_KEY')
    if openai_key:
        try:
            import urllib.request
            import json
            system_prompt = f"""You are LoanPredict AI Assistant, an expert loan underwriting and explainable AI advisor for borrower {user_name}.
User's financial profile: Credit Score: {credit_score}, Monthly Income: ₹{monthly_income:,.0f}, Latest Risk: {latest_risk}.
Never expose any other user's confidential data. Answer concisely, professionally, with structured bullet points."""
            req = urllib.request.Request(
                'https://api.openai.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {openai_key}', 'Content-Type': 'application/json'},
                data=json.dumps({
                    'model': 'gpt-3.5-turbo',
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': clean_msg}
                    ],
                    'max_tokens': 350,
                    'temperature': 0.3
                }).encode('utf-8')
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                return res_data['choices'][0]['message']['content']
        except Exception:
            pass # Fall through to comprehensive internal engine

    # --- Comprehensive Domain Knowledge Engine ---

    # 1. Application Status & Tracking
    if any(k in msg_lower for k in ['my application', 'status', 'tracking', 'track', 'application id']):
        if applications:
            latest_app = applications[0]
            status_text = f"<strong>Your Latest Application:</strong><br><br>"
            status_text += f"• <strong>Application ID:</strong> {latest_app.get('application_id', 'N/A')}<br>"
            status_text += f"• <strong>Loan Type:</strong> {latest_app.get('loan_type', 'Personal')} Loan<br>"
            status_text += f"• <strong>Requested Amount:</strong> ₹{latest_app.get('loan_amount', 0):,.0f}<br>"
            status_text += f"• <strong>Current Status:</strong> <span class='badge-success'>{latest_app.get('status', 'Submitted')}</span><br>"
            status_text += f"• <strong>Submitted Date:</strong> {latest_app.get('created_at', 'Recently')}<br><br>"
            status_text += "You can track stage-by-stage progress under <strong>Apply Loan → View Applications</strong>."
            return status_text
        else:
            return "You currently have no active loan applications submitted. You can apply anytime by navigating to <strong>Apply Loan</strong> in the main menu."

    # 2. Document Requirements
    if any(k in msg_lower for k in ['document', 'documents', 'proof', 'paperwork', 'what do i need to submit']):
        if 'home' in msg_lower:
            return """<strong>Required Documents for Home Loan:</strong><br><br>
1. <strong>Identity & Address:</strong> PAN Card, Aadhaar Card, Passport / Voter ID.<br>
2. <strong>Income Proof:</strong> Last 3 months' salary slips, Form 16, or 2 years' ITR.<br>
3. <strong>Bank Statements:</strong> 6 months' primary bank account statements.<br>
4. <strong>Property Documents:</strong> Agreement of Sale / Allotment Letter, Title Deed, Approved Building Plan, and Property Tax Receipts."""
        elif 'land' in msg_lower or 'plot' in msg_lower:
            return """<strong>Required Documents for Land / Plot Loan:</strong><br><br>
1. <strong>Identity Proof:</strong> PAN Card, Aadhaar Card.<br>
2. <strong>Income Verification:</strong> 3 months' salary slips or 2 years' business ITR.<br>
3. <strong>Land Title Documents:</strong> Sale Deed, Patta / Khata Certificate, Non-encumbrance certificate (13-30 years), and Land Survey sketch."""
        elif 'business' in msg_lower:
            return """<strong>Required Documents for Business Loan:</strong><br><br>
1. <strong>Business Identity:</strong> GST Registration, Certificate of Incorporation / Partnership Deed, MSME / Udyam registration.<br>
2. <strong>Financial Statements:</strong> Audited P&L and Balance Sheet for past 2 years, 2 years' Business ITR.<br>
3. <strong>Bank Records:</strong> 12 months' business current account statements.<br>
4. <strong>Promoter KYC:</strong> PAN and Aadhaar of Directors / Partners."""
        elif 'education' in msg_lower or 'student' in msg_lower:
            return """<strong>Required Documents for Education Loan:</strong><br><br>
1. <strong>Student Records:</strong> Admission Letter, detailed college fee structure, 10th / 12th / Degree marks sheets.<br>
2. <strong>Co-Applicant KYC:</strong> Parent / Guardian PAN and Aadhaar Card.<br>
3. <strong>Co-Applicant Income:</strong> 3 months' salary slips, Form 16, 6 months' bank statements."""
        elif 'vehicle' in msg_lower or 'car' in msg_lower or 'auto' in msg_lower:
            return """<strong>Required Documents for Vehicle Loan:</strong><br><br>
1. <strong>KYC:</strong> PAN Card, Aadhaar Card, Valid Driving License.<br>
2. <strong>Income Proof:</strong> Latest 3 months' salary slips or 1 year ITR.<br>
3. <strong>Vehicle Details:</strong> Official dealer proforma invoice or on-road price quotation."""
        elif 'agri' in msg_lower or 'farm' in msg_lower or 'kisan' in msg_lower:
            return """<strong>Required Documents for Agricultural Loan:</strong><br><br>
1. <strong>Applicant KYC:</strong> Aadhaar Card, PAN Card.<br>
2. <strong>Land Ownership:</strong> Pattadar Passbook / 7/12 extract / RTC (Record of Rights).<br>
3. <strong>Agricultural Proof:</strong> Crop cultivation certificate or Kisan Credit Card (KCC) passbook."""
        else:
            return """<strong>General Document Requirements by Category:</strong><br><br>
• <strong>Identity & Address:</strong> Aadhaar Card, PAN Card.<br>
• <strong>Income Proof:</strong> Latest 3 months' salary slips, Form 16, or 2 years' ITR.<br>
• <strong>Banking:</strong> 3–6 months' bank statements showing regular salary/revenue.<br>
• <strong>Collateral / Specifics:</strong> Title deeds for Home/Land, dealer quotations for Vehicle, admission letters for Education.<br><br>
You can upload PDF, JPG, or PNG files directly during the <strong>Apply Loan</strong> flow."""

    # 3. SHAP vs LIME Explainability
    if ('shap' in msg_lower and 'lime' in msg_lower) or 'difference between shap' in msg_lower:
        return """<strong>SHAP vs LIME Comparison:</strong><br><br>
• <strong>SHAP (SHapley Additive exPlanations):</strong> Uses cooperative game theory to measure the exact mathematical contribution of each feature towards deviating from the 35% portfolio baseline. It guarantees global consistency.<br><br>
• <strong>LIME (Local Interpretable Model-agnostic Explanations):</strong> Builds a local linear surrogate model around your specific loan numbers by generating slight perturbations to identify the immediate threshold boundary rules.<br><br>
<em>In LoanPredict AI, both frameworks typically achieve 85%–90% directional agreement on primary risk drivers.</em>"""

    if 'shap' in msg_lower:
        return """<strong>Understanding SHAP in LoanPredict AI:</strong><br><br>
SHAP quantifies exactly why the AI assigned your default probability score.<br>
• <strong>Red / Positive Values (+):</strong> Factors that increased your default risk (e.g., high loan-to-income multiple or existing loan count).<br>
• <strong>Blue / Negative Values (-):</strong> Protective factors that lowered your default risk (e.g., strong 700+ credit score, salaried stability).<br><br>
You can inspect interactive waterfalls under <strong>EXPLAINABILITY → SHAP Explanation</strong>."""

    if 'lime' in msg_lower:
        return """<strong>Understanding LIME in LoanPredict AI:</strong><br><br>
LIME tests what would happen if your parameters were slightly altered. It answers: <em>'What local condition most quickly changes this decision from Default to Approved?'</em><br>
For example, LIME might indicate that keeping your existing loans to 1 or fewer has the strongest local protective weight."""

    # 4. How to improve credit / Lower Default Risk
    if any(k in msg_lower for k in ['lower risk', 'improve', 'reduce risk', 'eligible', 'eligibility', 'chances']):
        return f"""<strong>Personalized Guidance to Maximize Approval (Profile: {credit_score} Score):</strong><br><br>
1. <strong>Optimize Loan Tenure:</strong> Selecting a 36–48 month tenure spreads repayments, keeping your monthly EMI under 35% of monthly net income.<br>
2. <strong>Limit Requested Principal:</strong> Ensure the requested loan amount is within 4–5x your verified monthly salary (₹{monthly_income:,.0f}).<br>
3. <strong>Credit Card Utilization:</strong> Keep credit card balances below 30% of authorized limits.<br>
4. <strong>Consolidate Debts:</strong> If you have 2+ active loans, combining them into one reduces the model's active liability penalty.<br>
5. <strong>Test with Simulator:</strong> Use our <strong>Recommendations → What-If Simulator</strong> to preview real-time impact before applying!"""

    # 5. Loan Types
    if any(k in msg_lower for k in ['loan type', 'types of loan', 'what loans']):
        return """<strong>Available Loan Products on LoanPredict AI:</strong><br><br>
1. <strong>Personal Loan:</strong> Unsecured financing for emergencies, debt consolidation, or lifestyle needs.<br>
2. <strong>Home Loan:</strong> High-value financing for purchasing apartments, villas, or independent houses.<br>
3. <strong>Land Loan:</strong> Tailored for purchasing residential or commercial plots/land.<br>
4. <strong>Business Loan:</strong> Working capital and term expansion for MSMEs, traders, and enterprises.<br>
5. <strong>Education Loan:</strong> Covers tuition fees and living expenses for higher studies.<br>
6. <strong>Vehicle Loan:</strong> Attractive financing for two-wheelers, cars, and commercial transport.<br>
7. <strong>Agricultural Loan:</strong> Support for farming activities, irrigation, and crop cultivation.<br><br>
You can assess or apply for any of these directly via the <strong>Assess My Loan</strong> or <strong>Apply Loan</strong> menu."""

    # 6. Default Fallback
    return f"""Hello {user_name}! I can assist you with:<br><br>
• <strong>Document Checklist:</strong> Ask <em>'What documents do I need for a home/business loan?'</em><br>
• <strong>Application Tracking:</strong> Ask <em>'What is my application status?'</em><br>
• <strong>Explainable AI:</strong> Ask <em>'Explain SHAP vs LIME'</em><br>
• <strong>Risk Optimization:</strong> Ask <em>'How can I lower my default probability?'</em><br>
• <strong>Institutional Options:</strong> Check the <strong>Recommendations</strong> tab for matching banks.<br><br>
What would you like to explore?"""
