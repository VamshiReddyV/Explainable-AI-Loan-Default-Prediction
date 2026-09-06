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
