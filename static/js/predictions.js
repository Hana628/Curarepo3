// Handle all prediction form submissions and results display
document.addEventListener('DOMContentLoaded', function() {
    // Blood Pressure Prediction
    setupBloodPressurePrediction();
    
    // Diabetes Prediction
    setupDiabetesPrediction();
    
    // Lifestyle Recommendations
    setupLifestyleRecommendations();
    
    // Anomaly Detection
    setupAnomalyDetection();
    
    // BMI Calculator
    setupBmiCalculator();
});

// Blood Pressure Prediction Functions
function setupBloodPressurePrediction() {
    const bpForm = document.getElementById('bp-prediction-form');
    const bpResult = document.getElementById('bp-result');
    const bpLoading = document.getElementById('bp-loading');
    const bpError = document.getElementById('bp-error');
    const bpErrorMessage = document.getElementById('bp-error-message');
    const bpStartOver = document.getElementById('bp-start-over');
    
    if (!bpForm) return;
    
    bpForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading
        bpForm.classList.add('d-none');
        bpResult.classList.add('d-none');
        bpError.classList.add('d-none');
        bpLoading.classList.remove('d-none');
        
        // Collect form data
        const data = {
            age: document.getElementById('bp-age').value,
            gender: document.getElementById('bp-gender').value,
            weight: document.getElementById('bp-weight').value,
            height: document.getElementById('bp-height').value,
            smoking: document.getElementById('bp-smoking').value,
            alcohol: document.getElementById('bp-alcohol').value,
            exercise: document.getElementById('bp-exercise').value
        };
        
        // Send prediction request
        fetch('/predict/blood_pressure', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading
            bpLoading.classList.add('d-none');
            
            if (data.status === 'success') {
                // Display results
                displayBloodPressureResults(data.prediction);
                bpResult.classList.remove('d-none');
            } else {
                // Show error
                bpErrorMessage.textContent = data.message || 'An error occurred during prediction.';
                bpError.classList.remove('d-none');
                bpForm.classList.remove('d-none');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            bpLoading.classList.add('d-none');
            bpErrorMessage.textContent = 'Network error. Please try again.';
            bpError.classList.remove('d-none');
            bpForm.classList.remove('d-none');
        });
    });
    
    // Reset form on start over
    if (bpStartOver) {
        bpStartOver.addEventListener('click', function() {
            bpResult.classList.add('d-none');
            bpForm.classList.remove('d-none');
        });
    }
}

function displayBloodPressureResults(prediction) {
    // Update systolic/diastolic values
    document.getElementById('bp-systolic').textContent = prediction.systolic_estimate;
    document.getElementById('bp-diastolic').textContent = prediction.diastolic_estimate;
    
    // Update risk category
    const riskCategory = document.getElementById('bp-risk-category');
    riskCategory.textContent = prediction.risk_category;
    
    // Set risk category color
    if (prediction.risk_category === 'Low') {
        riskCategory.className = 'display-6 text-success';
    } else if (prediction.risk_category === 'Moderate') {
        riskCategory.className = 'display-6 text-warning';
    } else {
        riskCategory.className = 'display-6 text-danger';
    }
    
    // Update progress bar
    const riskProgress = document.getElementById('bp-risk-progress');
    const riskPercentage = prediction.risk_score * 100;
    riskProgress.style.width = `${riskPercentage}%`;
    
    // Set progress bar color
    if (prediction.risk_score < 0.2) {
        riskProgress.className = 'progress-bar bg-success';
    } else if (prediction.risk_score < 0.5) {
        riskProgress.className = 'progress-bar bg-warning';
    } else {
        riskProgress.className = 'progress-bar bg-danger';
    }
    
    // Update advice
    document.getElementById('bp-advice').textContent = prediction.advice;
}

// Diabetes Prediction Functions
function setupDiabetesPrediction() {
    const diabetesForm = document.getElementById('diabetes-prediction-form');
    const diabetesResult = document.getElementById('diabetes-result');
    const diabetesLoading = document.getElementById('diabetes-loading');
    const diabetesError = document.getElementById('diabetes-error');
    const diabetesErrorMessage = document.getElementById('diabetes-error-message');
    const diabetesStartOver = document.getElementById('diabetes-start-over');
    
    if (!diabetesForm) return;
    
    diabetesForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading
        diabetesForm.classList.add('d-none');
        diabetesResult.classList.add('d-none');
        diabetesError.classList.add('d-none');
        diabetesLoading.classList.remove('d-none');
        
        // Collect form data
        const data = {
            age: document.getElementById('diabetes-age').value,
            glucose: document.getElementById('diabetes-glucose').value,
            bmi: document.getElementById('diabetes-bmi').value,
            family_history: document.getElementById('diabetes-family').value,
            physical_activity: document.getElementById('diabetes-activity').value
        };
        
        // Send prediction request
        fetch('/predict/diabetes', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading
            diabetesLoading.classList.add('d-none');
            
            if (data.status === 'success') {
                // Display results
                displayDiabetesResults(data.prediction);
                diabetesResult.classList.remove('d-none');
            } else {
                // Show error
                diabetesErrorMessage.textContent = data.message || 'An error occurred during prediction.';
                diabetesError.classList.remove('d-none');
                diabetesForm.classList.remove('d-none');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            diabetesLoading.classList.add('d-none');
            diabetesErrorMessage.textContent = 'Network error. Please try again.';
            diabetesError.classList.remove('d-none');
            diabetesForm.classList.remove('d-none');
        });
    });
    
    // Reset form on start over
    if (diabetesStartOver) {
        diabetesStartOver.addEventListener('click', function() {
            diabetesResult.classList.add('d-none');
            diabetesForm.classList.remove('d-none');
        });
    }
}

function displayDiabetesResults(prediction) {
    // Update risk value
    const riskValue = document.getElementById('diabetes-risk-value');
    const riskPercentage = Math.round(prediction.risk_score * 100);
    riskValue.textContent = `${riskPercentage}%`;
    
    // Create risk gauge chart
    if (window.createDiabetesGauge) {
        window.createDiabetesGauge(prediction.risk_score);
    }
    
    // Update risk category
    const riskCategory = document.getElementById('diabetes-risk-category');
    riskCategory.textContent = prediction.risk_category;
    
    // Set risk category color
    if (prediction.risk_category === 'Low') {
        riskCategory.className = 'display-6 text-success';
    } else if (prediction.risk_category === 'Moderate') {
        riskCategory.className = 'display-6 text-warning';
    } else {
        riskCategory.className = 'display-6 text-danger';
    }
    
    // Update progress bar
    const riskProgress = document.getElementById('diabetes-risk-progress');
    riskProgress.style.width = `${riskPercentage}%`;
    
    // Set progress bar color
    if (prediction.risk_score < 0.2) {
        riskProgress.className = 'progress-bar bg-success';
    } else if (prediction.risk_score < 0.5) {
        riskProgress.className = 'progress-bar bg-warning';
    } else {
        riskProgress.className = 'progress-bar bg-danger';
    }
    
    // Update advice
    document.getElementById('diabetes-advice').textContent = prediction.advice;
    
    // Update recommendations list
    const recommendationsList = document.getElementById('diabetes-recommendations');
    recommendationsList.innerHTML = '';
    
    if (prediction.recommendations && prediction.recommendations.length > 0) {
        prediction.recommendations.forEach(recommendation => {
            const li = document.createElement('li');
            li.className = 'list-group-item bg-transparent';
            li.innerHTML = `<i class="fas fa-check-circle text-info me-2"></i>${recommendation}`;
            recommendationsList.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.className = 'list-group-item bg-transparent';
        li.innerHTML = '<i class="fas fa-info-circle me-2"></i>No specific recommendations available.';
        recommendationsList.appendChild(li);
    }
}

// Lifestyle Recommendations Functions
function setupLifestyleRecommendations() {
    const lifestyleForm = document.getElementById('lifestyle-form');
    const lifestyleResult = document.getElementById('lifestyle-result');
    const lifestyleLoading = document.getElementById('lifestyle-loading');
    const lifestyleError = document.getElementById('lifestyle-error');
    const lifestyleErrorMessage = document.getElementById('lifestyle-error-message');
    const lifestyleStartOver = document.getElementById('lifestyle-start-over');
    
    if (!lifestyleForm) return;
    
    lifestyleForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading
        lifestyleForm.classList.add('d-none');
        lifestyleResult.classList.add('d-none');
        lifestyleError.classList.add('d-none');
        lifestyleLoading.classList.remove('d-none');
        
        // Collect form data
        const data = {
            age: document.getElementById('lifestyle-age').value,
            weight: document.getElementById('lifestyle-weight').value,
            height: document.getElementById('lifestyle-height').value,
            activity_level: document.getElementById('lifestyle-activity').value,
            diet_preference: document.getElementById('lifestyle-diet').value,
            sleep_hours: document.getElementById('lifestyle-sleep').value
        };
        
        // Send recommendation request
        fetch('/recommend/lifestyle', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading
            lifestyleLoading.classList.add('d-none');
            
            if (data.status === 'success') {
                // Display results
                displayLifestyleResults(data.recommendations);
                lifestyleResult.classList.remove('d-none');
            } else {
                // Show error
                lifestyleErrorMessage.textContent = data.message || 'An error occurred while generating recommendations.';
                lifestyleError.classList.remove('d-none');
                lifestyleForm.classList.remove('d-none');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            lifestyleLoading.classList.add('d-none');
            lifestyleErrorMessage.textContent = 'Network error. Please try again.';
            lifestyleError.classList.remove('d-none');
            lifestyleForm.classList.remove('d-none');
        });
    });
    
    // Reset form on start over
    if (lifestyleStartOver) {
        lifestyleStartOver.addEventListener('click', function() {
            lifestyleResult.classList.add('d-none');
            lifestyleForm.classList.remove('d-none');
        });
    }
}

function displayLifestyleResults(recommendations) {
    // Update BMI and weight status
    document.getElementById('lifestyle-bmi').textContent = recommendations.bmi;
    const weightStatus = document.getElementById('lifestyle-weight-status');
    weightStatus.textContent = capitalizeFirstLetter(recommendations.weight_status);
    
    // Set weight status color
    if (recommendations.weight_status === 'normal') {
        weightStatus.className = 'h5 text-success';
    } else if (recommendations.weight_status === 'underweight') {
        weightStatus.className = 'h5 text-warning';
    } else {
        weightStatus.className = 'h5 text-danger';
    }
    
    // Update sleep quality
    const sleepQuality = document.getElementById('lifestyle-sleep-quality');
    sleepQuality.textContent = capitalizeFirstLetter(recommendations.sleep_quality);
    document.getElementById('lifestyle-sleep-hours').textContent = recommendations.sleep_hours;
    
    // Set sleep quality color
    if (recommendations.sleep_quality === 'good') {
        sleepQuality.className = 'display-6 text-success';
    } else if (recommendations.sleep_quality === 'average') {
        sleepQuality.className = 'display-6 text-warning';
    } else {
        sleepQuality.className = 'display-6 text-danger';
    }
    
    // Update recommendation lists
    updateRecommendationList('exercise-recommendations', recommendations.exercise_recommendations);
    updateRecommendationList('diet-recommendations', recommendations.diet_recommendations);
    updateRecommendationList('sleep-recommendations', recommendations.sleep_recommendations);
    updateRecommendationList('weight-recommendations', recommendations.weight_recommendations);
    updateRecommendationList('general-recommendations', recommendations.general_recommendations);
}

function updateRecommendationList(elementId, recommendations) {
    const list = document.getElementById(elementId);
    if (!list) return;
    
    list.innerHTML = '';
    
    if (recommendations && recommendations.length > 0) {
        recommendations.forEach(recommendation => {
            const li = document.createElement('li');
            li.className = 'list-group-item bg-transparent';
            li.innerHTML = `<i class="fas fa-check-circle text-info me-2"></i>${recommendation}`;
            list.appendChild(li);
        });
    } else {
        const li = document.createElement('li');
        li.className = 'list-group-item bg-transparent';
        li.innerHTML = '<i class="fas fa-info-circle me-2"></i>No specific recommendations available.';
        list.appendChild(li);
    }
}

// Anomaly Detection Functions
function setupAnomalyDetection() {
    const anomalyForm = document.getElementById('anomaly-form');
    const anomalyResult = document.getElementById('anomaly-result');
    const anomalyLoading = document.getElementById('anomaly-loading');
    const anomalyError = document.getElementById('anomaly-error');
    const anomalyErrorMessage = document.getElementById('anomaly-error-message');
    const anomalyStartOver = document.getElementById('anomaly-start-over');
    const sampleDataCheck = document.getElementById('sample-data-check');
    const anomalyData = document.getElementById('anomaly-data');
    
    if (!anomalyForm) return;
    
    // Sample data for different metric types
    const sampleData = {
        heart_rate: "72, 75, 70, 73, 74, 71, 95, 73, 72, 71, 75",
        blood_pressure: "120, 122, 118, 121, 119, 150, 122, 120, 121, 119",
        glucose: "85, 87, 86, 84, 88, 120, 86, 85, 87, 84",
        oxygen: "98, 97, 98, 99, 97, 88, 98, 97, 99, 98",
        sleep: "7.5, 7.2, 7.8, 7.4, 7.6, 4.2, 7.5, 7.8, 7.3, 7.6",
        other: "45, 47, 44, 46, 48, 75, 46, 45, 47, 46"
    };
    
    // Set sample data when checkbox is clicked
    if (sampleDataCheck && anomalyData) {
        sampleDataCheck.addEventListener('change', function() {
            if (this.checked) {
                const metricType = document.getElementById('metric-type').value;
                anomalyData.value = sampleData[metricType] || sampleData.other;
            } else {
                anomalyData.value = '';
            }
        });
        
        // Update sample data when metric type changes if checkbox is checked
        document.getElementById('metric-type').addEventListener('change', function() {
            if (sampleDataCheck.checked) {
                anomalyData.value = sampleData[this.value] || sampleData.other;
            }
        });
    }
    
    anomalyForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Get health metrics data from textarea
        let healthMetricsText = anomalyData.value.trim();
        if (!healthMetricsText) {
            // Show error for empty data
            anomalyErrorMessage.textContent = 'Please enter health metrics data or use sample data.';
            anomalyError.classList.remove('d-none');
            return;
        }
        
        // Parse health metrics data (support for comma-separated or line-separated)
        const healthMetrics = healthMetricsText
            .replace(/\n/g, ',')  // Replace newlines with commas
            .split(',')           // Split by comma
            .map(value => value.trim()) // Trim whitespace
            .filter(value => value)  // Remove empty values
            .map(value => parseFloat(value)); // Convert to numbers
        
        // Validate parsed data
        if (healthMetrics.length < 7) {
            anomalyErrorMessage.textContent = 'Please provide at least 7 data points for anomaly detection.';
            anomalyError.classList.remove('d-none');
            return;
        }
        
        if (healthMetrics.some(isNaN)) {
            anomalyErrorMessage.textContent = 'All values must be valid numbers.';
            anomalyError.classList.remove('d-none');
            return;
        }
        
        // Show loading
        anomalyForm.classList.add('d-none');
        anomalyResult.classList.add('d-none');
        anomalyError.classList.add('d-none');
        anomalyLoading.classList.remove('d-none');
        
        // Prepare data for anomaly detection
        const data = {
            health_metrics: healthMetrics,
            metric_type: document.getElementById('metric-type').value
        };
        
        // Send anomaly detection request
        fetch('/detect/anomaly', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading
            anomalyLoading.classList.add('d-none');
            
            if (data.status === 'success') {
                // Display results
                displayAnomalyResults(data.result, healthMetrics);
                anomalyResult.classList.remove('d-none');
            } else {
                // Show error
                anomalyErrorMessage.textContent = data.message || 'An error occurred during anomaly detection.';
                anomalyError.classList.remove('d-none');
                anomalyForm.classList.remove('d-none');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            anomalyLoading.classList.add('d-none');
            anomalyErrorMessage.textContent = 'Network error. Please try again.';
            anomalyError.classList.remove('d-none');
            anomalyForm.classList.remove('d-none');
        });
    });
    
    // Reset form on start over
    if (anomalyStartOver) {
        anomalyStartOver.addEventListener('click', function() {
            anomalyResult.classList.add('d-none');
            anomalyForm.classList.remove('d-none');
        });
    }
}

function displayAnomalyResults(result, healthMetrics) {
    // Update status and message
    const anomalyStatus = document.getElementById('anomaly-status');
    const anomalyMessage = document.getElementById('anomaly-message');
    const anomalyAlert = document.getElementById('anomaly-alert');
    const anomalyDetails = document.getElementById('anomaly-details');
    
    if (result.anomalies_detected) {
        anomalyStatus.textContent = 'Anomalies Detected';
        anomalyMessage.textContent = `Found ${result.anomaly_indices.length} anomalies in your health data. These could indicate unusual health patterns that might need attention.`;
        anomalyAlert.className = 'alert alert-warning mt-4';
        anomalyDetails.classList.remove('d-none');
    } else {
        anomalyStatus.textContent = 'No Anomalies Detected';
        anomalyMessage.textContent = 'Your health data appears to follow normal patterns. No anomalies were detected.';
        anomalyAlert.className = 'alert alert-success mt-4';
        anomalyDetails.classList.add('d-none');
    }
    
    // Create the anomaly chart
    if (window.createAnomalyChart) {
        window.createAnomalyChart(healthMetrics, result.anomaly_indices);
    }
    
    // Populate the anomaly table if there are anomalies
    if (result.anomalies_detected) {
        const tableBody = document.getElementById('anomaly-table-body');
        tableBody.innerHTML = '';
        
        result.anomaly_indices.forEach(index => {
            const row = document.createElement('tr');
            
            // Point number (index + 1 for human-readable)
            const pointCell = document.createElement('td');
            pointCell.textContent = index + 1;
            row.appendChild(pointCell);
            
            // Value at that point
            const valueCell = document.createElement('td');
            valueCell.textContent = healthMetrics[index];
            row.appendChild(valueCell);
            
            // Anomaly score for that point
            const scoreCell = document.createElement('td');
            const scoreIndex = result.anomaly_scores.length > index ? index : 0;
            const score = result.anomaly_scores[scoreIndex];
            scoreCell.textContent = score ? score.toFixed(4) : 'N/A';
            row.appendChild(scoreCell);
            
            tableBody.appendChild(row);
        });
    }
}

// BMI Calculator Functions
function setupBmiCalculator() {
    const calculateBmiBtn = document.getElementById('calculate-bmi-btn');
    const useBmiBtn = document.getElementById('use-bmi-btn');
    const bmiWeight = document.getElementById('bmi-weight');
    const bmiHeight = document.getElementById('bmi-height');
    const bmiResult = document.getElementById('bmi-result');
    const bmiValue = document.getElementById('bmi-value');
    const bmiCategory = document.getElementById('bmi-category');
    
    if (!calculateBmiBtn) return;
    
    calculateBmiBtn.addEventListener('click', function() {
        // Get weight and height
        const weight = parseFloat(bmiWeight.value);
        const height = parseFloat(bmiHeight.value) / 100; // cm to meters
        
        if (isNaN(weight) || isNaN(height) || weight <= 0 || height <= 0) {
            alert('Please enter valid weight and height values.');
            return;
        }
        
        // Calculate BMI
        const bmi = weight / (height * height);
        
        // Display result
        bmiValue.textContent = bmi.toFixed(1);
        
        // Determine category
        let category;
        let categoryColor;
        
        if (bmi < 18.5) {
            category = 'Underweight';
            categoryColor = 'text-warning';
        } else if (bmi < 25) {
            category = 'Normal weight';
            categoryColor = 'text-success';
        } else if (bmi < 30) {
            category = 'Overweight';
            categoryColor = 'text-warning';
        } else {
            category = 'Obese';
            categoryColor = 'text-danger';
        }
        
        bmiCategory.textContent = `Category: ${category}`;
        bmiCategory.className = categoryColor;
        
        // Show result and enable use button
        bmiResult.classList.remove('d-none');
        useBmiBtn.disabled = false;
    });
    
    // Use calculated BMI in diabetes form
    if (useBmiBtn) {
        useBmiBtn.addEventListener('click', function() {
            const diabetesBmi = document.getElementById('diabetes-bmi');
            if (diabetesBmi && bmiValue.textContent) {
                diabetesBmi.value = bmiValue.textContent;
                
                // Close modal
                const bmiModal = bootstrap.Modal.getInstance(document.getElementById('bmiCalculatorModal'));
                if (bmiModal) {
                    bmiModal.hide();
                }
            }
        });
    }
}

// Utility Functions
function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}
