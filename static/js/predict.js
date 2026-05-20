document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    if (!form) return;

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const btn = document.getElementById('predictBtn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<span class="spinner" style="margin-right:8px;">⏳</span> Processing...';
        btn.disabled = true;

        const formData = new FormData(form);
        const data = {};
        
        for (let [key, value] of formData.entries()) {
            if (key === 'OverTime') {
                data[key] = value;
            } else {
                data[key] = parseFloat(value);
            }
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            const result = await response.json();
            
            if (!response.ok) throw new Error(result.error || 'Failed to predict');
            
            showResult(result, data);
            
        } catch (error) {
            alert('Assessment Error: ' + error.message);
        } finally {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    });
});

function showResult(result, inputData) {
    const card = document.getElementById('resultCard');
    const statusDiv = document.getElementById('resultStatus');
    const probSpan = document.getElementById('riskProb');
    const confSpan = document.getElementById('confidenceLvl');
    const factorsList = document.getElementById('factorsList');
    const actionBox = document.getElementById('hrAction');
    
    // Calculate metrics
    const prob = result.probability;
    const probPercentage = (prob * 100).toFixed(1) + '%';
    probSpan.innerText = probPercentage;
    
    // Confidence logic: distance from threshold (0 to 1 scale roughly)
    const dist = Math.abs(prob - result.threshold);
    let confidence = 'Moderate';
    if (dist > 0.2) confidence = 'High';
    if (dist < 0.05) confidence = 'Low';
    confSpan.innerText = confidence;
    
    // Determine risk status
    let statusText = '';
    statusDiv.className = 'result-status'; // reset
    
    let isRisk = (prob >= result.threshold);
    
    if (isRisk && prob > 0.8) {
        statusText = 'High Attrition Risk';
        statusDiv.classList.add('status-danger');
    } else if (isRisk) {
        statusText = 'Moderate Attrition Risk';
        statusDiv.classList.add('status-warning');
    } else {
        statusText = 'Low Attrition Risk';
        statusDiv.classList.add('status-success');
    }
    
    statusDiv.innerText = statusText;
    
    // Populate risk factors based on inputs
    const factors = [];
    if (inputData.OverTime === 'Yes') factors.push('Frequent overtime requirements');
    if (inputData.JobSatisfaction <= 2) factors.push('Self-reported low job satisfaction');
    if (inputData.WorkLifeBalance <= 2) factors.push('Challenging work-life balance');
    if (inputData.YearsSinceLastPromotion >= 3) factors.push('Long gap since last promotion');
    if (inputData.MonthlyIncome < 4000) factors.push('Below average compensation level');
    if (inputData.EnvironmentSatisfaction <= 2) factors.push('Poor environment satisfaction');
    if (inputData.DistanceFromHome > 15) factors.push('Long commute distance');

    if (factors.length > 0) {
        factorsList.innerHTML = `<ul>${factors.map(f => `<li>${f}</li>`).join('')}</ul>`;
    } else {
        factorsList.innerHTML = `<p class="small-text">No distinct single risk factors identified; risk is based on cumulative profile evaluation.</p>`;
    }
    
    // Suggested Action
    if (isRisk) {
        actionBox.innerHTML = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--primary-blue);flex-shrink:0;"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
            <span>Consider scheduling a retention discussion and reviewing workload distribution or compensation equity for this employee.</span>
        `;
    } else {
        actionBox.innerHTML = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--success);flex-shrink:0;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
            <span>Employee profile indicates stability. Maintain regular check-ins to ensure continued engagement and satisfaction.</span>
        `;
    }
    
    // Reveal
    card.classList.remove('hidden');
    
    // Scroll smoothly
    setTimeout(() => {
        card.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}
