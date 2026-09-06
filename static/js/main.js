document.addEventListener('DOMContentLoaded', function () {

    // =========================================================
    // 1. Admin Profile Dropdown (Top Navigation)
    // =========================================================

    const profileTrigger = document.getElementById('userProfileTrigger');
    const profileDropdown = document.getElementById('profileDropdownMenu');
    if (profileTrigger && profileDropdown) {
        profileTrigger.addEventListener('click', function (e) {
            e.stopPropagation();
            const isOpen = profileDropdown.classList.toggle('show');
            profileTrigger.setAttribute('aria-expanded', isOpen);
        });

        document.addEventListener('click', function (e) {
            if (!profileTrigger.contains(e.target) && !profileDropdown.contains(e.target)) {
                profileDropdown.classList.remove('show');
                profileTrigger.setAttribute('aria-expanded', 'false');
            }
        });

        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') {
                profileDropdown.classList.remove('show');
                profileTrigger.setAttribute('aria-expanded', 'false');
            }
        });
    }

    // =========================================================
    // 2. Executive Dashboard Charts (Overview & Distribution)
    // =========================================================

    const dashData = window.dashboardData || {
        approved: 0,
        defaulted: 0,
        total: 0,
        defaultRate: 0,
        riskDistribution: { '0-20%': 0, '20-40%': 0, '40-60%': 0, '60-80%': 0, '80-100%': 0 },
        performance: null
    };

    // 2.1 Prediction Overview (Doughnut)
    const ctxOverview = document.getElementById('predictionOverviewChart');
    if (ctxOverview) {
        const approved = dashData.approved || 0;
        const defaulted = dashData.defaulted || 0;
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

    // 2.2 Default Risk Distribution (Bar Chart)
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
                        rd['0-20%'] || 0,
                        rd['20-40%'] || 0,
                        rd['40-60%'] || 0,
                        rd['60-80%'] || 0,
                        rd['80-100%'] || 0
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
                        ticks: { stepSize: 1 },
                        grid: { color: '#F1F5F9' },
                        border: { display: false },
                        title: { display: true, text: 'Applicants' }
                    },
                    x: {
                        grid: { display: false },
                        border: { display: false },
                        title: { display: true, text: 'Default Probability Bucket' }
                    }
                }
            }
        });
    }

    // =========================================================
    // 3. Dynamic Model Performance Timeframe (This Month, Last 3 Months, YTD)
    // =========================================================

    let perfChartInstance = null;

    function updateModelPerformanceUI(perf) {
        if (!perf) return;

        const subtext = document.getElementById('perfPeriodSubtext');
        if (subtext && perf.label) subtext.textContent = perf.label;

        const accVal = document.getElementById('perfAccuracyVal');
        if (accVal) {
            accVal.textContent = perf.accuracy_display || 'Unavailable';
            accVal.className = 'p-val ' + (perf.accuracy_status === 'available' ? 'text-primary' : 'text-muted');
        }

        const totCount = document.getElementById('perfTotalCount');
        if (totCount) totCount.textContent = perf.total_predictions || 0;

        const evalCount = document.getElementById('perfEvaluatedCount');
        if (evalCount) evalCount.textContent = perf.evaluated_predictions || 0;

        const corrCount = document.getElementById('perfCorrectCount');
        if (corrCount) corrCount.textContent = perf.correct_predictions || 0;

        const incorrCount = document.getElementById('perfIncorrectCount');
        if (incorrCount) incorrCount.textContent = perf.incorrect_predictions || 0;

        const footnote = document.getElementById('perfFootnoteText');
        if (footnote) footnote.textContent = perf.accuracy_message || '';

        // Update Chart
        const ctxPerformance = document.getElementById('modelPerformanceChart');
        if (!ctxPerformance || !perf.timeline) return;

        const labels = perf.timeline.labels || ['Start'];
        const accuracyData = perf.timeline.accuracy || [];
        const volumeData = perf.timeline.volume || [];

        if (perfChartInstance) {
            perfChartInstance.data.labels = labels;
            perfChartInstance.data.datasets[0].data = accuracyData;
            perfChartInstance.data.datasets[1].data = volumeData;
            perfChartInstance.update();
        } else {
            perfChartInstance = new Chart(ctxPerformance, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Empirical Accuracy (%)',
                            data: accuracyData,
                            borderColor: '#0A58CA',
                            backgroundColor: 'rgba(10, 88, 202, 0.08)',
                            borderWidth: 2.2,
                            pointBackgroundColor: '#0A58CA',
                            pointBorderColor: '#fff',
                            pointBorderWidth: 2,
                            pointRadius: 4,
                            pointHoverRadius: 6,
                            tension: 0.25,
                            fill: true,
                            yAxisID: 'y'
                        },
                        {
                            label: 'Prediction Volume',
                            data: volumeData,
                            borderColor: '#94A3B8',
                            backgroundColor: 'rgba(148, 163, 184, 0.15)',
                            borderWidth: 1.5,
                            borderDash: [4, 4],
                            pointBackgroundColor: '#94A3B8',
                            pointRadius: 3,
                            tension: 0.2,
                            fill: false,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'bottom',
                            labels: { boxWidth: 12, usePointStyle: true }
                        },
                        tooltip: {
                            callbacks: {
                                label: function (context) {
                                    if (context.datasetIndex === 0) {
                                        return context.raw !== null ? `Accuracy: ${context.raw}%` : 'Accuracy: Outcomes Pending';
                                    }
                                    return `Volume: ${context.raw} applications`;
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
                            title: { display: true, text: 'Accuracy (%)' }
                        },
                        y1: {
                            beginAtZero: true,
                            position: 'right',
                            grid: { display: false },
                            ticks: { stepSize: 1 },
                            title: { display: true, text: 'Volume' }
                        },
                        x: {
                            grid: { display: false }
                        }
                    }
                }
            });
        }
    }

    // Initial render from server-injected performance
    if (dashData.performance) {
        updateModelPerformanceUI(dashData.performance);
    }

    // Listen to timeframe dropdown changes
    const timeframeSelect = document.getElementById('timeframeSelect');
    if (timeframeSelect) {
        timeframeSelect.addEventListener('change', async function () {
            try {
                const period = this.value;
                const response = await fetch(`/api/model-performance?period=${period}`);
                if (response.ok) {
                    const data = await response.json();
                    updateModelPerformanceUI(data);
                }
            } catch (err) {
                console.error('Error fetching model performance timeframe:', err);
            }
        });
    }

    // =========================================================
    // 4. Dedicated SHAP Waterfall Chart (shap_explanation.html)
    // =========================================================

    const ctxShap = document.getElementById('shapWaterfallChart');
    if (ctxShap && window.shapChartData) {
        const labels = window.shapChartData.labels || [];
        const values = window.shapChartData.values || [];
        const colors = values.map(v => v >= 0 ? '#EF4444' : '#10B981');

        new Chart(ctxShap, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'SHAP Contribution to Default Risk',
                    data: values,
                    backgroundColor: colors,
                    borderRadius: 4,
                    barPercentage: 0.6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                const val = ctx.raw;
                                const dir = val >= 0 ? 'Increases Default Risk' : 'Decreases Default Risk';
                                return `SHAP Value: ${val > 0 ? '+' : ''}${val} (${dir})`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#F1F5F9' },
                        title: { display: true, text: 'Marginal Impact on P(Default) Relative to Baseline' }
                    },
                    y: {
                        grid: { display: false }
                    }
                }
            }
        });
    }

    // =========================================================
    // 5. Dedicated LIME Weights Chart (lime_explanation.html)
    // =========================================================

    const ctxLime = document.getElementById('limeWeightsChart');
    if (ctxLime && window.limeChartData) {
        const labels = window.limeChartData.labels || [];
        const weights = window.limeChartData.weights || [];
        const colors = weights.map(w => w >= 0 ? '#EF4444' : '#10B981');

        new Chart(ctxLime, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'LIME Sparse Surrogate Weight',
                    data: weights,
                    backgroundColor: colors,
                    borderRadius: 4,
                    barPercentage: 0.6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                const val = ctx.raw;
                                const dir = val >= 0 ? 'Pushes toward Default' : 'Pushes toward Non-Default';
                                return `Rule Weight: ${val > 0 ? '+' : ''}${val} (${dir})`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#F1F5F9' },
                        title: { display: true, text: 'Local Surrogate Weight (ω)' }
                    },
                    y: {
                        grid: { display: false }
                    }
                }
            }
        });
    }

    // =========================================================
    // 6. Comprehensive Risk Analysis Charts (performance.html)
    // =========================================================

    const riskData = window.riskAnalyticsData;
    if (riskData) {
        // 6.1 Outcome Doughnut
        const ctxRiskOutcome = document.getElementById('riskOutcomeChart');
        if (ctxRiskOutcome) {
            new Chart(ctxRiskOutcome, {
                type: 'doughnut',
                data: {
                    labels: ['Non-Default', 'Default'],
                    datasets: [{
                        data: [riskData.approvedCount, riskData.defaultCount],
                        backgroundColor: ['#10B981', '#EF4444'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '70%',
                    plugins: { legend: { position: 'bottom' } }
                }
            });
        }

        // 6.2 Risk Buckets
        const ctxRiskBuckets = document.getElementById('riskBucketsChart');
        if (ctxRiskBuckets) {
            const b = riskData.buckets || {};
            new Chart(ctxRiskBuckets, {
                type: 'bar',
                data: {
                    labels: ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'],
                    datasets: [{
                        label: 'Applicants',
                        data: [b['0-20%'] || 0, b['20-40%'] || 0, b['40-60%'] || 0, b['60-80%'] || 0, b['80-100%'] || 0],
                        backgroundColor: ['#10B981', '#34D399', '#FBBF24', '#F97316', '#EF4444'],
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true, ticks: { stepSize: 1 }, grid: { color: '#F1F5F9' } },
                        x: { grid: { display: false } }
                    }
                }
            });
        }

        // 6.3 Risk Trend
        const ctxRiskTrend = document.getElementById('riskTrendChart');
        if (ctxRiskTrend && riskData.trend) {
            new Chart(ctxRiskTrend, {
                type: 'line',
                data: {
                    labels: riskData.trend.labels || [],
                    datasets: [
                        {
                            label: 'Total Applications',
                            data: riskData.trend.totals || [],
                            borderColor: '#3B82F6',
                            backgroundColor: 'rgba(59, 130, 246, 0.08)',
                            fill: true,
                            tension: 0.2
                        },
                        {
                            label: 'Defaults',
                            data: riskData.trend.defaults || [],
                            borderColor: '#EF4444',
                            borderDash: [3, 3],
                            tension: 0.2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { position: 'bottom' } },
                    scales: {
                        y: { beginAtZero: true, ticks: { stepSize: 1 }, grid: { color: '#F1F5F9' } },
                        x: { grid: { display: false } }
                    }
                }
            });
        }

        // 6.4 Credit Score vs Default Probability (Scatter)
        const ctxCredit = document.getElementById('creditVsProbChart');
        if (ctxCredit && riskData.scatterCredit) {
            new Chart(ctxCredit, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Applicant Risk Score',
                        data: riskData.scatterCredit,
                        backgroundColor: '#3B82F6',
                        pointRadius: 5,
                        pointHoverRadius: 7
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: ctx => `Credit: ${ctx.raw.x}, Default Prob: ${ctx.raw.y}%`
                            }
                        }
                    },
                    scales: {
                        x: { title: { display: true, text: 'Credit Score' }, grid: { color: '#F1F5F9' } },
                        y: { beginAtZero: true, max: 100, title: { display: true, text: 'Default Probability (%)' }, grid: { color: '#F1F5F9' } }
                    }
                }
            });
        }

        // 6.5 Loan Amount vs Default Probability
        const ctxLoan = document.getElementById('loanVsProbChart');
        if (ctxLoan && riskData.scatterLoan) {
            new Chart(ctxLoan, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Applicant Exposure',
                        data: riskData.scatterLoan,
                        backgroundColor: '#F97316',
                        pointRadius: 5,
                        pointHoverRadius: 7
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: ctx => `Loan: ₹${ctx.raw.x}k, Default Prob: ${ctx.raw.y}%`
                            }
                        }
                    },
                    scales: {
                        x: { title: { display: true, text: 'Loan Requested (₹ Thousands)' }, grid: { color: '#F1F5F9' } },
                        y: { beginAtZero: true, max: 100, title: { display: true, text: 'Default Probability (%)' }, grid: { color: '#F1F5F9' } }
                    }
                }
            });
        }

        // 6.6 Income vs Default Probability
        const ctxIncome = document.getElementById('incomeVsProbChart');
        if (ctxIncome && riskData.scatterIncome) {
            new Chart(ctxIncome, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Applicant Cashflow',
                        data: riskData.scatterIncome,
                        backgroundColor: '#10B981',
                        pointRadius: 5,
                        pointHoverRadius: 7
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: ctx => `Income: ₹${ctx.raw.x}k/mo, Default Prob: ${ctx.raw.y}%`
                            }
                        }
                    },
                    scales: {
                        x: { title: { display: true, text: 'Monthly Income (₹ Thousands)' }, grid: { color: '#F1F5F9' } },
                        y: { beginAtZero: true, max: 100, title: { display: true, text: 'Default Probability (%)' }, grid: { color: '#F1F5F9' } }
                    }
                }
            });
        }
    }

    // =========================================================
    // 7. Interactive Tab Switching (Dashboard Local XAI)
    // =========================================================

    const tabButtons = document.querySelectorAll('.tabs-container .tab-btn');
    tabButtons.forEach(btn => {
        btn.addEventListener('click', function () {
            const targetTab = this.getAttribute('data-tab');
            const parentCard = this.closest('.card') || document;
            
            parentCard.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');

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

    // =========================================================
    // 8. Multi-Step Wizard Navigation (New Loan Prediction)
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
            if (!input.checkValidity()) {
                input.reportValidity();
                return false;
            }
        }
        return true;
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

    function populateReviewSummary() {
        const reviewGrid = document.getElementById('reviewSummaryGrid');
        if (!reviewGrid) return;

        const summary = [
            { label: 'Applicant Name', val: document.getElementById('fullName')?.value || '—' },
            { label: 'Age / Gender', val: `${document.getElementById('age')?.value || '—'} yrs / ${document.getElementById('gender')?.value || '—'}` },
            { label: 'Employment', val: `${document.getElementById('employmentType')?.value || '—'} (${document.getElementById('experience')?.value || 0} yrs exp)` },
            { label: 'Monthly Income', val: `₹ ${Number(document.getElementById('monthlyIncome')?.value || 0).toLocaleString()}` },
            { label: 'Credit Score', val: document.getElementById('creditScore')?.value || '—' },
            { label: 'Loan Requested', val: `₹ ${Number(document.getElementById('loanAmount')?.value || 0).toLocaleString()}` },
            { label: 'Loan Purpose', val: document.getElementById('loanPurpose')?.value || '—' },
            { label: 'Loan Term', val: `${document.getElementById('loanTerm')?.value || '—'} months` },
            { label: 'Interest Rate', val: `${document.getElementById('interestRate')?.value || '—'} %` },
            { label: 'Existing Active Loans', val: document.getElementById('existingLoans')?.value || '0' }
        ];

        reviewGrid.innerHTML = summary.map(item => `
            <div class="summary-item">
                <span class="summary-label">${item.label}</span>
                <span class="summary-value">${item.val}</span>
            </div>
        `).join('');
    }

    // =========================================================
    // 9. Synchronize Range Sliders & Inputs
    // =========================================================

    function syncSlider(sliderId, inputId) {
        const slider = document.getElementById(sliderId);
        const input = document.getElementById(inputId);
        if (slider && input) {
            slider.addEventListener('input', () => input.value = slider.value);
            input.addEventListener('input', () => slider.value = input.value);
        }
    }

    syncSlider('loanAmountSlider', 'loanAmount');
    syncSlider('creditScoreSlider', 'creditScore');
    syncSlider('interestRateSlider', 'interestRate');

    // =========================================================
    // 10. Form Submission & Real-time Inference API Call
    // =========================================================

    if (loanForm) {
        loanForm.addEventListener('submit', async function (e) {
            e.preventDefault();

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
    // 11. Interactive Prediction Result Modal
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
                        ${(result.shap_explanation || []).slice(0, 4).map(item => `
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
    // 12. Mobile Sidebar Menu Toggle
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
