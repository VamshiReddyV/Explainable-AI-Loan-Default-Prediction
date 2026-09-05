document.addEventListener('DOMContentLoaded', function () {

    // =========================================================
    // 1. Dashboard Charts (Chart.js)
    // =========================================================

    const dashData = window.dashboardData || {
        approved: 812,
        defaulted: 436,
        total: 1248,
        defaultRate: 34.94,
        riskDistribution: { '0-20%': 320, '20-40%': 280, '40-60%': 250, '60-80%': 210, '80-100%': 188 }
    };

    // 1.1 Prediction Overview (Doughnut)
    const ctxOverview = document.getElementById('predictionOverviewChart');
    if (ctxOverview) {
        const approved = dashData.approved || 812;
        const defaulted = dashData.defaulted || 436;
        const total = dashData.total || (approved + defaulted);
        new Chart(ctxOverview, {
            type: 'doughnut',
            data: {
                labels: ['Non-Default (Approved)', 'Default'],
                datasets: [{
                    data: total > 0 ? [approved, defaulted] : [1, 0],
                    backgroundColor: ['#25C266', '#F85A5A'],
                    borderWidth: 0,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '72%',
                plugins: { legend: { display: false } }
            }
        });
    }

    // 1.2 Default Risk Distribution (Bar Chart)
    const ctxRisk = document.getElementById('riskDistributionChart');
    if (ctxRisk) {
        const rd = dashData.riskDistribution || {};
        new Chart(ctxRisk, {
            type: 'bar',
            data: {
                labels: ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'],
                datasets: [{
                    label: 'Number of Applicants',
                    data: [
                        rd['0-20%'] || 320,
                        rd['20-40%'] || 280,
                        rd['40-60%'] || 250,
                        rd['60-80%'] || 210,
                        rd['80-100%'] || 188
                    ],
                    backgroundColor: ['#25C266', '#86D779', '#FFC107', '#FF8A65', '#F85A5A'],
                    borderRadius: 4,
                    barPercentage: 0.55
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 500,
                        grid: { color: '#F1F5F9' },
                        border: { display: false },
                        title: { display: true, text: 'Number of Applicants' }
                    },
                    x: {
                        grid: { display: false },
                        border: { display: false },
                        title: { display: true, text: 'Default Probability (%)' }
                    }
                }
            }
        });
    }

    // 1.3 Model Performance (Line Chart)
    const ctxPerformance = document.getElementById('modelPerformanceChart');
    if (ctxPerformance) {
        new Chart(ctxPerformance, {
            type: 'line',
            data: {
                labels: ['May 5', '', '', 'May 12', '', '', 'May 19', '', '', 'May 26', '', '', 'Jun 2'],
                datasets: [{
                    label: 'Accuracy (%)',
                    data: [82, 80, 85, 83, 86, 84, 87, 85, 88, 86, 89, 87, 89.67],
                    borderColor: '#0A58CA',
                    backgroundColor: 'rgba(10, 88, 202, 0.08)',
                    borderWidth: 2.2,
                    pointBackgroundColor: '#0A58CA',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 3.5,
                    pointHoverRadius: 6,
                    tension: 0.35,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom',
                        labels: {
                            usePointStyle: true,
                            boxWidth: 8,
                            generateLabels: function () {
                                return [{
                                    text: 'Current Accuracy: 89.67%',
                                    fillStyle: '#0A58CA',
                                    strokeStyle: 'transparent',
                                    pointStyle: 'circle'
                                }];
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { stepSize: 25 },
                        grid: { color: '#F1F5F9' },
                        border: { display: false },
                        title: { display: true, text: 'Accuracy (%)' }
                    },
                    x: {
                        grid: { display: false },
                        border: { display: false }
                    }
                }
            }
        });
    }

    // =========================================================
    // 2. Interactive Tab Switching (SHAP vs LIME)
    // =========================================================

    const tabButtons = document.querySelectorAll('.tabs-container .tab-btn');
    tabButtons.forEach(btn => {
        btn.addEventListener('click', function () {
            const targetTab = this.getAttribute('data-tab');
            const parentCard = this.closest('.card') || document;
            
            // Toggle active button
            parentCard.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');

            // Toggle corresponding pane
            const shapPane = parentCard.querySelector('#shapTab');
            const limePane = parentCard.querySelector('#limeTab');

            if (targetTab === 'lime') {
                if (shapPane) shapPane.classList.remove('active');
                if (limePane) limePane.classList.add('active');
            } else {
                if (limePane) limePane.classList.remove('active');
                if (shapPane) shapPane.classList.add('active');
            }
        });
    });

    // Hash navigation for #lime anchor
    if (window.location.hash === '#lime') {
        const limeTabBtn = document.querySelector('.tab-btn[data-tab="lime"]');
        if (limeTabBtn) limeTabBtn.click();
    }

    // =========================================================
    // 3. Multi-Step Wizard Navigation (New Loan Prediction)
    // =========================================================

    let currentStep = 1;
    const totalSteps = 5;
    const stepperContainer = document.getElementById('formStepper');
    const btnNext = document.getElementById('btnNextStep');
    const btnPrev = document.getElementById('btnPrevStep');
    const btnSubmit = document.getElementById('btnSubmitPrediction');
    const loanForm = document.getElementById('loanForm');

    function updateWizardUI() {
        if (!stepperContainer) return;

        // 1. Update Step Indicator Circles and connecting lines
        const steps = stepperContainer.querySelectorAll('.step');
        steps.forEach(stepEl => {
            const stepNum = parseInt(stepEl.getAttribute('data-step'));
            stepEl.classList.remove('active', 'completed');
            if (stepNum === currentStep) {
                stepEl.classList.add('active');
            } else if (stepNum < currentStep) {
                stepEl.classList.add('completed');
            }
        });

        for (let i = 1; i < totalSteps; i++) {
            const line = stepperContainer.querySelector(`.step-line[data-line="${i}"]`);
            if (line) {
                if (i < currentStep) {
                    line.classList.add('active');
                } else {
                    line.classList.remove('active');
                }
            }
        }

        // 2. Show/hide wizard panels
        for (let i = 1; i <= totalSteps; i++) {
            const panel = document.getElementById(`panel-step-${i}`);
            if (panel) {
                if (i === currentStep) {
                    panel.classList.add('active');
                } else {
                    panel.classList.remove('active');
                }
            }
        }

        // 3. Update Action Buttons
        if (btnPrev) {
            btnPrev.style.display = currentStep > 1 ? 'inline-flex' : 'none';
        }

        if (currentStep === totalSteps) {
            if (btnNext) btnNext.style.display = 'none';
            if (btnSubmit) btnSubmit.style.display = 'inline-flex';
            populateReviewSummary();
        } else {
            if (btnNext) btnNext.style.display = 'inline-flex';
            if (btnSubmit) btnSubmit.style.display = 'none';
        }
    }

    function validateCurrentStep() {
        const currentPanel = document.getElementById(`panel-step-${currentStep}`);
        if (!currentPanel) return true;

        const inputs = currentPanel.querySelectorAll('input[required], select[required]');
        for (let input of inputs) {
            if (!input.value || input.value.trim() === '') {
                input.focus();
                input.reportValidity();
                return false;
            }
        }
        return true;
    }

    function populateReviewSummary() {
        const reviewContainer = document.getElementById('reviewSummary');
        if (!reviewContainer) return;

        const fullName = document.getElementById('fullName')?.value || 'Not provided';
        const age = document.getElementById('age')?.value || '30';
        const gender = document.getElementById('gender')?.value || 'Male';
        const maritalStatus = document.getElementById('maritalStatus')?.value || 'Single';
        const dependents = document.getElementById('dependents')?.value || '0';
        const employmentType = document.getElementById('employmentType')?.value || 'Salaried';
        const experience = document.getElementById('experience')?.value || '0';
        const monthlyIncome = parseFloat(document.getElementById('monthlyIncome')?.value || 0);
        const annualIncome = (monthlyIncome * 12).toLocaleString('en-IN');
        const loanAmount = parseFloat(document.getElementById('loanAmount')?.value || 0).toLocaleString('en-IN');
        const loanPurpose = document.getElementById('loanPurpose')?.value || 'Personal';
        const loanTerm = document.getElementById('loanTerm')?.value || '12';
        const interestRate = document.getElementById('interestRate')?.value || '10.0';
        const existingLoans = document.getElementById('existingLoans')?.value || '0';
        const creditScore = document.getElementById('creditScore')?.value || '650';

        reviewContainer.innerHTML = `
            <div class="review-item"><span class="label">Applicant Name</span><span class="val">${fullName}</span></div>
            <div class="review-item"><span class="label">Age / Gender</span><span class="val">${age} yrs, ${gender} (${maritalStatus})</span></div>
            <div class="review-item"><span class="label">Dependents</span><span class="val">${dependents}</span></div>
            <div class="review-item"><span class="label">Employment</span><span class="val">${employmentType} (${experience} yrs experience)</span></div>
            <div class="review-item"><span class="label">Monthly Income</span><span class="val">₹ ${monthlyIncome.toLocaleString('en-IN')}</span></div>
            <div class="review-item"><span class="label">Annual Income</span><span class="val">₹ ${annualIncome}</span></div>
            <div class="review-item"><span class="label">Credit Score</span><span class="val"><span class="score-pill ${parseInt(creditScore) >= 700 ? 'high' : (parseInt(creditScore) >= 600 ? 'mid' : 'low')}">${creditScore}</span></span></div>
            <div class="review-item"><span class="label">Loan Requested</span><span class="val">₹ ${loanAmount} (${loanPurpose})</span></div>
            <div class="review-item"><span class="label">Term & Rate</span><span class="val">${loanTerm} Months @ ${interestRate}%</span></div>
            <div class="review-item"><span class="label">Existing Loans</span><span class="val">${existingLoans} active facilities</span></div>
        `;
    }

    if (btnNext) {
        btnNext.addEventListener('click', function () {
            if (validateCurrentStep()) {
                if (currentStep < totalSteps) {
                    currentStep++;
                    updateWizardUI();
                }
            }
        });
    }

    if (btnPrev) {
        btnPrev.addEventListener('click', function () {
            if (currentStep > 1) {
                currentStep--;
                updateWizardUI();
            }
        });
    }

    // Direct step click on stepper header
    if (stepperContainer) {
        stepperContainer.querySelectorAll('.step').forEach(stepEl => {
            stepEl.addEventListener('click', function () {
                const targetStep = parseInt(this.getAttribute('data-step'));
                if (targetStep < currentStep || validateCurrentStep()) {
                    currentStep = targetStep;
                    updateWizardUI();
                }
            });
        });
    }

    // =========================================================
    // 4. Auto-Calculate Annual Income
    // =========================================================

    const monthlyIncomeInput = document.getElementById('monthlyIncome');
    const annualIncomeInput = document.getElementById('annualIncome');
    if (monthlyIncomeInput && annualIncomeInput) {
        monthlyIncomeInput.addEventListener('input', function () {
            const monthly = parseFloat(this.value) || 0;
            annualIncomeInput.value = (monthly * 12).toLocaleString('en-IN');
        });
    }

    // =========================================================
    // 5. Prediction Form AJAX Submission
    // =========================================================

    if (loanForm) {
        loanForm.addEventListener('submit', async function (e) {
            e.preventDefault();

            if (!validateCurrentStep()) return;

            const submitBtn = btnSubmit || loanForm.querySelector('button[type="submit"]');
            const originalText = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Assessing Risk...';
            submitBtn.disabled = true;

            const formData = {
                full_name: document.getElementById('fullName')?.value || 'Unknown',
                age: parseInt(document.getElementById('age')?.value || 30),
                gender: document.getElementById('gender')?.value || 'Male',
                marital_status: document.getElementById('maritalStatus')?.value || 'Single',
                number_of_dependents: parseInt(document.getElementById('dependents')?.value || 0),
                employment_type: document.getElementById('employmentType')?.value || 'Salaried',
                employment_experience: parseInt(document.getElementById('experience')?.value || 0),
                monthly_income: parseFloat(document.getElementById('monthlyIncome')?.value || 0),
                credit_score: parseInt(document.getElementById('creditScore')?.value || 650),
                loan_amount: parseFloat(document.getElementById('loanAmount')?.value || 0),
                loan_purpose: document.getElementById('loanPurpose')?.value || 'Personal',
                loan_term: parseInt(document.getElementById('loanTerm')?.value || 12),
                interest_rate: parseFloat(document.getElementById('interestRate')?.value || 10.0),
                existing_loans: parseInt(document.getElementById('existingLoans')?.value || 0)
            };

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });

                const result = await response.json();

                if (result.success) {
                    showResultModal(result);
                } else {
                    alert('Prediction error: ' + (result.error || 'Unable to assess risk'));
                }
            } catch (error) {
                alert('Connection error: ' + error.message);
            } finally {
                submitBtn.innerHTML = originalText;
                submitBtn.disabled = false;
            }
        });
    }

    // =========================================================
    // 6. Interactive Prediction Result Modal
    // =========================================================

    function showResultModal(result) {
        const existing = document.getElementById('resultModal');
        if (existing) existing.remove();

        const isDefault = result.prediction === 'Default';
        const color = isDefault ? '#F85A5A' : '#25C266';
        const icon = isDefault ? 'fa-triangle-exclamation' : 'fa-circle-check';

        const modal = document.createElement('div');
        modal.id = 'resultModal';
        modal.className = 'result-modal-overlay';
        modal.innerHTML = `
            <div class="result-modal">
                <div class="result-modal-header" style="background: ${color};">
                    <i class="fa-solid ${icon}" style="font-size: 44px; color: white;"></i>
                    <h2 style="color: white; margin-top: 10px; font-size: 24px;">${result.prediction}</h2>
                    <p style="color: rgba(255,255,255,0.9); font-size: 13px;">AI Underwriting Risk Assessment</p>
                </div>
                <div class="result-modal-body">
                    <div class="result-stat">
                        <span class="label">Default Probability</span>
                        <span class="value" style="color: ${color};">${result.default_probability}%</span>
                    </div>
                    <div class="result-stat">
                        <span class="label">Risk Tier</span>
                        <span class="badge" style="background: ${isDefault ? '#FEEEEE' : '#E8FAED'}; color: ${color}; font-weight: 700;">
                            ${result.risk_level}
                        </span>
                    </div>
                    
                    <h4 style="margin: 20px 0 10px; font-size: 14px; font-weight: 700;">Top Decision Factors (SHAP)</h4>
                    <div class="result-features">
                        ${result.shap_explanation.slice(0, 4).map(item => `
                            <div class="result-feature-row">
                                <span>${item.display_name}</span>
                                <span style="color: ${item.shap_value >= 0 ? '#F85A5A' : '#25C266'}; font-weight: 700;">
                                    ${item.shap_value >= 0 ? '+' : ''}${item.shap_value}
                                </span>
                            </div>
                        `).join('')}
                    </div>
                    
                    <div class="result-modal-actions">
                        <a href="/prediction/${result.prediction_id}" class="btn btn-primary">
                            <i class="fa-solid fa-chart-pie"></i> View Full Audit
                        </a>
                        <a href="/" class="btn btn-secondary">
                            <i class="fa-solid fa-house"></i> Dashboard
                        </a>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        modal.addEventListener('click', function (e) {
            if (e.target === modal) modal.remove();
        });
    }

    // =========================================================
    // 7. Mobile Sidebar Menu Toggle
    // =========================================================

    const mobileToggle = document.getElementById('mobileMenuToggle');
    const sidebar = document.getElementById('sidebar');
    if (mobileToggle && sidebar) {
        mobileToggle.addEventListener('click', function () {
            sidebar.classList.toggle('open');
        });
    }

    // Initialize Wizard UI state on load
    updateWizardUI();
});
