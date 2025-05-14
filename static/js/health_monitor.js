/**
 * Health Monitor Dashboard
 * 
 * This script provides functionality for the health monitoring dashboard,
 * simulating wearable device connections and displaying health metrics.
 */

// Global variables for chart and timers
let healthMetricsChart = null;
let dataRefreshInterval = null;
let alertRefreshInterval = null;

// Initialize the health monitoring dashboard
function initializeHealthMonitoring() {
    console.log("Initializing health monitoring dashboard");
    
    // Clear any existing intervals
    if (dataRefreshInterval) clearInterval(dataRefreshInterval);
    if (alertRefreshInterval) clearInterval(alertRefreshInterval);
    
    // Set up device connection button
    document.getElementById('connect-device').addEventListener('click', function() {
        simulateDeviceConnection();
    });
    
    // Initialize chart
    setupHealthMetricsChart();
    
    // Load initial data
    fetchHealthData();
    fetchHealthAlerts();
    
    // Set up periodic refreshes (every 10 seconds for data, 20 seconds for alerts)
    dataRefreshInterval = setInterval(fetchHealthData, 10000);
    alertRefreshInterval = setInterval(fetchHealthAlerts, 20000);
}

// Simulate connecting to a wearable device
function simulateDeviceConnection() {
    const deviceStatus = document.getElementById('device-status');
    
    // Show connecting status
    deviceStatus.className = 'status-badge warning';
    deviceStatus.textContent = 'Connecting...';
    
    // Simulate connection process with timeout
    setTimeout(function() {
        const connected = Math.random() > 0.1; // 90% chance of successful connection
        
        if (connected) {
            deviceStatus.className = 'status-badge healthy';
            deviceStatus.textContent = 'Connected';
            fetchHealthData(); // Immediately fetch new data
        } else {
            deviceStatus.className = 'status-badge danger';
            deviceStatus.textContent = 'Connection Failed';
        }
    }, 2000);
}

// Set up the health metrics chart
function setupHealthMetricsChart() {
    const ctx = document.getElementById('healthMetricsChart').getContext('2d');
    
    // Create initial empty datasets
    healthMetricsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Heart Rate',
                    data: [],
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0,
                    pointHoverRadius: 4
                },
                {
                    label: 'Blood Oxygen',
                    data: [],
                    borderColor: '#17a2b8',
                    backgroundColor: 'rgba(23, 162, 184, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0,
                    pointHoverRadius: 4
                },
                {
                    label: 'Glucose',
                    data: [],
                    borderColor: '#ffc107',
                    backgroundColor: 'rgba(255, 193, 7, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointRadius: 0,
                    pointHoverRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(200, 200, 200, 0.1)'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        boxWidth: 12,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.7)',
                    padding: 10,
                    cornerRadius: 6,
                    intersect: false,
                    mode: 'index'
                }
            },
            interaction: {
                intersect: false,
                mode: 'nearest'
            }
        }
    });
}

// Fetch health data from the API
function fetchHealthData() {
    fetch('/api/health-data')
        .then(response => response.json())
        .then(data => {
            updateHealthMetrics(data);
        })
        .catch(error => {
            console.error('Error fetching health data:', error);
            // If API fails, use simulated data
            const simulatedData = generateSimulatedHealthData();
            updateHealthMetrics(simulatedData);
        });
}

// Fetch health alerts from the API
function fetchHealthAlerts() {
    fetch('/api/health-alerts')
        .then(response => response.json())
        .then(data => {
            updateHealthAlerts(data);
        })
        .catch(error => {
            console.error('Error fetching health alerts:', error);
            // If API fails, use simulated alerts
            const simulatedAlerts = generateSimulatedHealthAlerts();
            updateHealthAlerts(simulatedAlerts);
        });
}

// Generate simulated health data if the API fails
function generateSimulatedHealthData() {
    // Current time for the x-axis
    const now = new Date();
    const times = [];
    const heartRateData = [];
    const oxygenData = [];
    const glucoseData = [];
    
    // Generate data for the last 24 hours
    for (let i = 0; i < 24; i++) {
        const time = new Date(now);
        time.setHours(now.getHours() - 23 + i);
        times.push(time.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}));
        
        // Heart rate between 60-100, with some variation
        const baseHeartRate = 75;
        const heartRate = baseHeartRate + Math.sin(i/2) * 15 + (Math.random() * 10 - 5);
        heartRateData.push(Math.round(heartRate));
        
        // Oxygen levels between 95-100%
        const baseOxygen = 97;
        const oxygen = baseOxygen + Math.sin(i/3) * 1.5 + (Math.random() * 1 - 0.5);
        oxygenData.push(Math.min(100, Math.round(oxygen * 10) / 10));
        
        // Glucose levels between 80-120 mg/dL
        const baseGlucose = 100;
        const glucose = baseGlucose + Math.sin(i/4) * 15 + (Math.random() * 10 - 5);
        glucoseData.push(Math.round(glucose));
    }
    
    return {
        times: times,
        heart_rate: {
            values: heartRateData,
            current: heartRateData[heartRateData.length - 1],
            min: Math.min(...heartRateData),
            max: Math.max(...heartRateData),
            status: getHeartRateStatus(heartRateData[heartRateData.length - 1])
        },
        blood_pressure: {
            systolic: Math.floor(Math.random() * 30) + 110,
            diastolic: Math.floor(Math.random() * 20) + 70,
            status: 'Normal'
        },
        oxygen: {
            value: oxygenData[oxygenData.length - 1],
            status: getOxygenStatus(oxygenData[oxygenData.length - 1])
        },
        glucose: {
            value: glucoseData[glucoseData.length - 1],
            status: getGlucoseStatus(glucoseData[glucoseData.length - 1])
        },
        activity: {
            steps: Math.floor(Math.random() * 5000) + 3000,
            active_minutes: Math.floor(Math.random() * 60) + 30,
            goal_percentage: Math.floor(Math.random() * 80) + 20
        }
    };
}

// Generate simulated health alerts if the API fails
function generateSimulatedHealthAlerts() {
    const alertTypes = ['info', 'warning', 'danger', 'success'];
    const alerts = [];
    
    // Generate random number of alerts (0-5)
    const numAlerts = Math.floor(Math.random() * 5);
    
    for (let i = 0; i < numAlerts; i++) {
        const alertType = alertTypes[Math.floor(Math.random() * alertTypes.length)];
        let message = '';
        let icon = '';
        
        switch (alertType) {
            case 'info':
                message = 'Your average heart rate has increased by 10% from last week.';
                icon = 'fa-heartbeat';
                break;
            case 'warning':
                message = 'Your blood glucose level is trending higher than normal this week.';
                icon = 'fa-tachometer-alt';
                break;
            case 'danger':
                message = 'Elevated blood pressure detected. Consider scheduling a check-up.';
                icon = 'fa-exclamation-triangle';
                break;
            case 'success':
                message = 'You met your step goal for 5 consecutive days! Great job!';
                icon = 'fa-award';
                break;
        }
        
        alerts.push({
            type: alertType,
            message: message,
            icon: icon,
            time: getRandomTimeAgo()
        });
    }
    
    return {
        count: alerts.length,
        alerts: alerts
    };
}

// Update health metrics with new data
function updateHealthMetrics(data) {
    // Update chart
    if (healthMetricsChart) {
        healthMetricsChart.data.labels = data.times;
        healthMetricsChart.data.datasets[0].data = data.heart_rate.values;
        healthMetricsChart.data.datasets[1].data = data.oxygen ? data.times.map(() => data.oxygen.value) : [];
        healthMetricsChart.data.datasets[2].data = data.glucose ? data.times.map(() => data.glucose.value) : [];
        healthMetricsChart.update();
    }
    
    // Update heart rate display
    if (data.heart_rate) {
        document.getElementById('heart-rate-value').textContent = `${data.heart_rate.current} bpm`;
        document.getElementById('heart-rate-min').textContent = `Min: ${data.heart_rate.min} bpm`;
        document.getElementById('heart-rate-max').textContent = `Max: ${data.heart_rate.max} bpm`;
        document.getElementById('heart-rate-status').textContent = data.heart_rate.status;
        
        // Update heart rate progress bar
        const heartRateProgress = document.getElementById('heart-rate-progress');
        const heartRatePercentage = Math.min(100, (data.heart_rate.current / 150) * 100);
        heartRateProgress.style.width = `${heartRatePercentage}%`;
        
        // Set progress bar color based on status
        heartRateProgress.className = 'progress-bar';
        if (data.heart_rate.status === 'Normal') {
            heartRateProgress.classList.add('bg-success');
        } else if (data.heart_rate.status === 'Elevated') {
            heartRateProgress.classList.add('bg-warning');
        } else {
            heartRateProgress.classList.add('bg-danger');
        }
    }
    
    // Update blood pressure display
    if (data.blood_pressure) {
        document.getElementById('blood-pressure-value').textContent = `${data.blood_pressure.systolic}/${data.blood_pressure.diastolic} mmHg`;
        document.getElementById('systolic-value').textContent = `${data.blood_pressure.systolic} mmHg`;
        document.getElementById('diastolic-value').textContent = `${data.blood_pressure.diastolic} mmHg`;
        document.getElementById('bp-status').textContent = data.blood_pressure.status;
        
        // Update blood pressure progress bar
        const bpProgress = document.getElementById('bp-progress');
        const bpPercentage = Math.min(100, (data.blood_pressure.systolic / 180) * 100);
        bpProgress.style.width = `${bpPercentage}%`;
        
        // Set progress bar color based on status
        bpProgress.className = 'progress-bar';
        if (data.blood_pressure.status === 'Normal') {
            bpProgress.classList.add('bg-success');
        } else if (data.blood_pressure.status === 'Elevated') {
            bpProgress.classList.add('bg-warning');
        } else {
            bpProgress.classList.add('bg-danger');
        }
    }
    
    // Update glucose display
    if (data.glucose) {
        document.getElementById('glucose-value').textContent = `${data.glucose.value} mg/dL`;
        document.getElementById('glucose-status').textContent = data.glucose.status;
        
        // Update glucose progress bar
        const glucoseProgress = document.getElementById('glucose-progress');
        // For glucose, optimal range is 70-140, scale accordingly
        const glucosePercentage = Math.min(100, (data.glucose.value / 200) * 100);
        glucoseProgress.style.width = `${glucosePercentage}%`;
        
        // Set progress bar color based on status
        glucoseProgress.className = 'progress-bar';
        if (data.glucose.status === 'Normal') {
            glucoseProgress.classList.add('bg-success');
        } else if (data.glucose.status === 'Elevated') {
            glucoseProgress.classList.add('bg-warning');
        } else {
            glucoseProgress.classList.add('bg-danger');
        }
    }
    
    // Update oxygen display
    if (data.oxygen) {
        document.getElementById('oxygen-value').textContent = `${data.oxygen.value}%`;
        document.getElementById('oxygen-status').textContent = data.oxygen.status;
        
        // Update oxygen progress bar
        const oxygenProgress = document.getElementById('oxygen-progress');
        // For oxygen, normal range is 95-100%, scale accordingly
        const oxygenPercentage = Math.min(100, ((data.oxygen.value - 90) / 10) * 100);
        oxygenProgress.style.width = `${oxygenPercentage}%`;
        
        // Set progress bar color based on status
        oxygenProgress.className = 'progress-bar';
        if (data.oxygen.status === 'Normal') {
            oxygenProgress.classList.add('bg-success');
        } else if (data.oxygen.status === 'Low') {
            oxygenProgress.classList.add('bg-warning');
        } else {
            oxygenProgress.classList.add('bg-danger');
        }
    }
    
    // Update activity metrics
    if (data.activity) {
        document.getElementById('steps-value').textContent = data.activity.steps.toLocaleString();
        document.getElementById('active-minutes').textContent = data.activity.active_minutes;
        document.getElementById('activity-completion').textContent = `${data.activity.goal_percentage}%`;
        
        // Update activity progress bar
        const activityProgress = document.getElementById('activity-progress');
        activityProgress.style.width = `${data.activity.goal_percentage}%`;
    }
}

// Update health alerts with new data
function updateHealthAlerts(data) {
    const alertsContainer = document.getElementById('alerts-container');
    const alertsCount = document.getElementById('alerts-count');
    
    // Update alerts count
    alertsCount.textContent = data.count > 0 ? 
        `${data.count} alert${data.count > 1 ? 's' : ''} found` : 
        'No alerts';
    
    // Clear previous alerts
    alertsContainer.innerHTML = '';
    
    if (data.alerts.length === 0) {
        alertsContainer.innerHTML = `
            <div class="text-center py-4">
                <i class="fas fa-check-circle fa-3x text-success mb-3"></i>
                <p>No health alerts at this time. Everything looks good!</p>
            </div>
        `;
    } else {
        // Add each alert to the container
        data.alerts.forEach(alert => {
            const alertElement = document.createElement('div');
            alertElement.className = `alert-item ${alert.type}`;
            alertElement.innerHTML = `
                <div class="d-flex align-items-center">
                    <div class="alert-icon ${alert.type}">
                        <i class="fas ${alert.icon}"></i>
                    </div>
                    <div>
                        <p class="mb-1">${alert.message}</p>
                        <small class="text-muted">${alert.time}</small>
                    </div>
                </div>
            `;
            alertsContainer.appendChild(alertElement);
        });
    }
}

// Helper function to get heart rate status
function getHeartRateStatus(heartRate) {
    if (heartRate < 60) return 'Low';
    if (heartRate > 100) return 'Elevated';
    return 'Normal';
}

// Helper function to get oxygen status
function getOxygenStatus(oxygen) {
    if (oxygen < 95) return 'Low';
    return 'Normal';
}

// Helper function to get glucose status
function getGlucoseStatus(glucose) {
    if (glucose < 70) return 'Low';
    if (glucose > 140) return 'Elevated';
    return 'Normal';
}

// Helper function to generate random time ago string
function getRandomTimeAgo() {
    const units = ['minute', 'hour', 'day'];
    const unit = units[Math.floor(Math.random() * units.length)];
    const value = Math.floor(Math.random() * 5) + 1;
    return `${value} ${unit}${value > 1 ? 's' : ''} ago`;
}

// Initialize when loaded if this section is already visible
document.addEventListener('DOMContentLoaded', function() {
    const healthMonitorSection = document.getElementById('health-monitor-section');
    if (healthMonitorSection && !healthMonitorSection.classList.contains('d-none')) {
        initializeHealthMonitoring();
    }
});