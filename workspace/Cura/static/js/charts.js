// Chart creation and management functions
document.addEventListener('DOMContentLoaded', function() {
    // Store chart instances to allow for updates/destruction
    window.chartInstances = {};
    
    // Function to create or update diabetes risk gauge chart
    window.createDiabetesGauge = function(riskScore) {
        const ctx = document.getElementById('diabetes-risk-gauge');
        if (!ctx) return;
        
        // Destroy existing chart if it exists
        if (window.chartInstances.diabetesGauge) {
            window.chartInstances.diabetesGauge.destroy();
        }
        
        // Risk colors based on score
        let color = '#28a745'; // green for low risk
        if (riskScore > 0.5) {
            color = '#dc3545'; // red for high risk
        } else if (riskScore > 0.2) {
            color = '#ffc107'; // yellow for moderate risk
        }
        
        // Create risk gauge chart
        window.chartInstances.diabetesGauge = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [riskScore * 100, (1 - riskScore) * 100],
                    backgroundColor: [color, 'rgba(0, 0, 0, 0.1)'],
                    borderWidth: 0,
                    cutout: '75%'
                }]
            },
            options: {
                responsive: true,
                circumference: 180,
                rotation: 270,
                plugins: {
                    tooltip: {
                        enabled: false
                    },
                    legend: {
                        display: false
                    }
                },
                layout: {
                    padding: {
                        bottom: 10
                    }
                },
                elements: {
                    arc: {
                        borderRadius: 10
                    }
                }
            }
        });
    };
    
    // Function to create anomaly detection chart
    window.createAnomalyChart = function(data, anomalyIndices) {
        const ctx = document.getElementById('anomaly-chart');
        if (!ctx) return;
        
        // Destroy existing chart if it exists
        if (window.chartInstances.anomalyChart) {
            window.chartInstances.anomalyChart.destroy();
        }
        
        // Create labels (indices)
        const labels = Array.from({ length: data.length }, (_, i) => i + 1);
        
        // Create datasets array for normal and anomaly points
        let normalData = [];
        let anomalyData = [];
        
        // Fill data arrays
        data.forEach((value, index) => {
            if (anomalyIndices.includes(index)) {
                // This is an anomaly point
                normalData.push(null);
                anomalyData.push(value);
            } else {
                // This is a normal point
                normalData.push(value);
                anomalyData.push(null);
            }
        });
        
        // Create the chart
        window.chartInstances.anomalyChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Normal Data',
                        data: normalData,
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderWidth: 2,
                        pointRadius: 4,
                        tension: 0.1
                    },
                    {
                        label: 'Anomalies',
                        data: anomalyData,
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        borderWidth: 2,
                        pointRadius: 6,
                        pointStyle: 'rectRot',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'Value'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Data Point'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    },
                    tooltip: {
                        callbacks: {
                            title: function(tooltipItems) {
                                return `Data Point: ${tooltipItems[0].label}`;
                            }
                        }
                    }
                },
                elements: {
                    line: {
                        fill: false
                    }
                }
            }
        });
    };
});
