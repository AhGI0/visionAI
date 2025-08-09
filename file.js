 let healthData = {
            dates: [],
            heartRate: [],
            systolic: [],
            diastolic: [],
            weight: [],
            steps: [],
            sleep: []
        };

        let anomalyCount = 0;
        let charts = {};

        // TensorFlow model for anomaly detection
        let anomalyModel;

        // Initialize TensorFlow model
        async function initializeAnomalyModel() {
            // Create a simple autoencoder for anomaly detection
            anomalyModel = tf.sequential({
                layers: [
                    tf.layers.dense({ inputShape: [6], units: 4, activation: 'relu' }),
                    tf.layers.dense({ units: 2, activation: 'relu' }),
                    tf.layers.dense({ units: 4, activation: 'relu' }),
                    tf.layers.dense({ units: 6, activation: 'linear' })
                ]
            });

            anomalyModel.compile({
                optimizer: tf.train.adam(0.001),
                loss: 'meanSquaredError'
            });

            console.log('Anomaly detection model initialized');
        }

        // Initialize charts
        function initializeCharts() {
            // Heart Rate Chart
            const heartRateCtx = document.getElementById('heartRateChart').getContext('2d');
            charts.heartRate = new Chart(heartRateCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Heart Rate (BPM)',
                        data: [],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            min: 50,
                            max: 150
                        }
                    }
                }
            });

            // Blood Pressure Chart
            const bpCtx = document.getElementById('bloodPressureChart').getContext('2d');
            charts.bloodPressure = new Chart(bpCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Systolic',
                        data: [],
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        borderWidth: 3,
                        fill: false
                    }, {
                        label: 'Diastolic',
                        data: [],
                        borderColor: '#4ecdc4',
                        backgroundColor: 'rgba(78, 205, 196, 0.1)',
                        borderWidth: 3,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            min: 60,
                            max: 180
                        }
                    }
                }
            });

            // Weight & Steps Chart
            const weightStepsCtx = document.getElementById('weightStepsChart').getContext('2d');
            charts.weightSteps = new Chart(weightStepsCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Weight (kg)',
                        data: [],
                        borderColor: '#764ba2',
                        backgroundColor: 'rgba(118, 75, 162, 0.1)',
                        borderWidth: 3,
                        yAxisID: 'y',
                        fill: true
                    }, {
                        label: 'Steps (thousands)',
                        data: [],
                        borderColor: '#f093fb',
                        backgroundColor: 'rgba(240, 147, 251, 0.1)',
                        borderWidth: 3,
                        yAxisID: 'y1',
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            grid: {
                                drawOnChartArea: false,
                            },
                        }
                    }
                }
            });

            // Sleep Chart
            const sleepCtx = document.getElementById('sleepChart').getContext('2d');
            charts.sleep = new Chart(sleepCtx, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Sleep Hours',
                        data: [],
                        backgroundColor: 'rgba(78, 205, 196, 0.8)',
                        borderColor: '#4ecdc4',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 12
                        }
                    }
                }
            });
        }

        // Add health data
        function addHealthData() {
            const heartRate = parseFloat(document.getElementById('heartRate').value);
            const systolic = parseFloat(document.getElementById('bloodPressureSys').value);
            const diastolic = parseFloat(document.getElementById('bloodPressureDia').value);
            const weight = parseFloat(document.getElementById('weight').value);
            const steps = parseFloat(document.getElementById('steps').value);
            const sleep = parseFloat(document.getElementById('sleep').value);

            if (!heartRate || !systolic || !diastolic || !weight || !steps || !sleep) {
                alert('Please fill in all health metrics!');
                return;
            }

            // Add data to storage
            const now = new Date();
            const dateString = now.toLocaleDateString();
            
            healthData.dates.push(dateString);
            healthData.heartRate.push(heartRate);
            healthData.systolic.push(systolic);
            healthData.diastolic.push(diastolic);
            healthData.weight.push(weight);
            healthData.steps.push(steps);
            healthData.sleep.push(sleep);

            // Keep only last 30 entries
            if (healthData.dates.length > 30) {
                Object.keys(healthData).forEach(key => {
                    healthData[key] = healthData[key].slice(-30);
                });
            }

            // Clear inputs
            document.getElementById('heartRate').value = '';
            document.getElementById('bloodPressureSys').value = '';
            document.getElementById('bloodPressureDia').value = '';
            document.getElementById('weight').value = '';
            document.getElementById('steps').value = '';
            document.getElementById('sleep').value = '';

            // Update charts and run anomaly detection
            updateCharts();
            detectAnomalies();
            updateStatistics();
            generateHealthPredictions();
        }

        // Update all charts
        function updateCharts() {
            // Heart Rate Chart
            charts.heartRate.data.labels = healthData.dates;
            charts.heartRate.data.datasets[0].data = healthData.heartRate;
            charts.heartRate.update();

            // Blood Pressure Chart
            charts.bloodPressure.data.labels = healthData.dates;
            charts.bloodPressure.data.datasets[0].data = healthData.systolic;
            charts.bloodPressure.data.datasets[1].data = healthData.diastolic;
            charts.bloodPressure.update();

            // Weight & Steps Chart
            charts.weightSteps.data.labels = healthData.dates;
            charts.weightSteps.data.datasets[0].data = healthData.weight;
            charts.weightSteps.data.datasets[1].data = healthData.steps.map(s => s / 1000); // Convert to thousands
            charts.weightSteps.update();

            // Sleep Chart
            charts.sleep.data.labels = healthData.dates;
            charts.sleep.data.datasets[0].data = healthData.sleep;
            charts.sleep.update();
        }

        // Detect anomalies using statistical methods and TensorFlow
        async function detectAnomalies() {
            if (healthData.heartRate.length < 5) return;

            const latestData = [
                healthData.heartRate[healthData.heartRate.length - 1],
                healthData.systolic[healthData.systolic.length - 1],
                healthData.diastolic[healthData.diastolic.length - 1],
                healthData.weight[healthData.weight.length - 1],
                healthData.steps[healthData.steps.length - 1] / 10000, // Normalize steps
                healthData.sleep[healthData.sleep.length - 1]
            ];

            // Statistical anomaly detection
            let anomalies = [];

            // Heart rate anomaly (simple threshold)
            const hr = latestData[0];
            if (hr < 50 || hr > 120) {
                anomalies.push(`Unusual heart rate: ${hr} BPM`);
            }

            // Blood pressure anomaly
            const sys = latestData[1];
            const dia = latestData[2];
            if (sys > 140 || dia > 90) {
                anomalies.push(`High blood pressure: ${sys}/${dia} mmHg`);
            }

            // Sleep anomaly
            const sleepHours = latestData[5];
            if (sleepHours < 4 || sleepHours > 12) {
                anomalies.push(`Unusual sleep pattern: ${sleepHours} hours`);
            }

            // If we have enough data, train and use TensorFlow model
            if (healthData.heartRate.length >= 10) {
                await trainAnomalyModel();
                const isAnomaly = await detectTensorFlowAnomaly(latestData);
                if (isAnomaly) {
                    anomalies.push('there is an unusual health pattern');
                }
            }

            // Show anomaly alert
            if (anomalies.length > 0) {
                showAnomalyAlert(anomalies);
                anomalyCount++;
            } else {
                hideAnomalyAlert();
            }
        }

        // Train anomaly detection model
        async function trainAnomalyModel() {
            if (healthData.heartRate.length < 10) return;

            // Prepare training data (normalize)
            const trainingData = [];
            for (let i = 0; i < healthData.heartRate.length; i++) {
                trainingData.push([
                    healthData.heartRate[i] / 100, // Normalize
                    healthData.systolic[i] / 150,
                    healthData.diastolic[i] / 100,
                    healthData.weight[i] / 100,
                    healthData.steps[i] / 10000,
                    healthData.sleep[i] / 10
                ]);
            }

            const xs = tf.tensor2d(trainingData);
            
            // Train autoencoder (unsupervised learning)
            await anomalyModel.fit(xs, xs, {
                epochs: 50,
                batchSize: Math.min(8, trainingData.length),
                verbose: 0
            });

            xs.dispose();
        }

        // Detect anomaly using TensorFlow model
        async function detectTensorFlowAnomaly(data) {
            const normalizedData = [
                data[0] / 100,
                data[1] / 150,
                data[2] / 100,
                data[3] / 100,
                data[4], // Already normalized
                data[5] / 10
            ];

            const input = tf.tensor2d([normalizedData]);
            const prediction = anomalyModel.predict(input);
            const reconstruction = await prediction.data();
            
            // Calculate reconstruction error
            let error = 0;
            for (let i = 0; i < normalizedData.length; i++) {
                error += Math.pow(normalizedData[i] - reconstruction[i], 2);
            }
            
            input.dispose();
            prediction.dispose();
            
            // Threshold for anomaly detection
            return error > 0.1;
        }

        // Show anomaly alert
        function showAnomalyAlert(anomalies) {
            const alertDiv = document.getElementById('anomalyAlert');
            const messageSpan = document.getElementById('anomalyMessage');
            messageSpan.textContent = anomalies.join('; ');
            alertDiv.classList.add('show');
        }

        // Hide anomaly alert
        function hideAnomalyAlert() {
            const alertDiv = document.getElementById('anomalyAlert');
            alertDiv.classList.remove('show');
        }

        // Update statistics
        function updateStatistics() {
            if (healthData.heartRate.length === 0) return;

            // Average heart rate
            const avgHR = (healthData.heartRate.reduce((a, b) => a + b, 0) / healthData.heartRate.length).toFixed(0);
            document.getElementById('avgHeartRate').textContent = avgHR;

            // Average blood pressure
            const avgSys = (healthData.systolic.reduce((a, b) => a + b, 0) / healthData.systolic.length).toFixed(0);
            const avgDia = (healthData.diastolic.reduce((a, b) => a + b, 0) / healthData.diastolic.length).toFixed(0);
            document.getElementById('avgBP').textContent = `${avgSys}/${avgDia}`;

            // Total steps
            const totalSteps = healthData.steps.reduce((a, b) => a + b, 0);
            document.getElementById('totalSteps').textContent = (totalSteps / 1000).toFixed(0) + 'K';

            // Average sleep
            const avgSleep = (healthData.sleep.reduce((a, b) => a + b, 0) / healthData.sleep.length).toFixed(1);
            document.getElementById('avgSleep').textContent = avgSleep + 'h';

            // Anomalies detected
            document.getElementById('anomaliesDetected').textContent = anomalyCount;
        }

        // Generate health predictions
        function generateHealthPredictions() {
            if (healthData.heartRate.length < 5) {
                document.getElementById('healthPrediction').textContent = 'Add more data to see AI predictions...';
                return;
            }

            // Simple trend analysis
            const recentHR = healthData.heartRate.slice(-5);
            const recentWeight = healthData.weight.slice(-5);
            const recentSleep = healthData.sleep.slice(-5);

            let predictions = [];

            // Heart rate trend
            const hrTrend = (recentHR[recentHR.length - 1] - recentHR[0]) / recentHR.length;
            if (hrTrend > 2) {
                predictions.push('Heart rate trending upward - consider rest');
            } else if (hrTrend < -2) {
                predictions.push('Heart rate improving - keep it up!');
            }

            // Weight trend
            const weightTrend = (recentWeight[recentWeight.length - 1] - recentWeight[0]) / recentWeight.length;
            if (Math.abs(weightTrend) > 0.5) {
                predictions.push(weightTrend > 0 ? 'Weight increasing trend detected' : 'Weight decreasing trend detected');
            }

            // Sleep quality
            const avgRecentSleep = recentSleep.reduce((a, b) => a + b, 0) / recentSleep.length;
            if (avgRecentSleep < 6) {
                predictions.push('Consider improving sleep schedule');
            } else if (avgRecentSleep > 8) {
                predictions.push('Excellent sleep patterns detected!');
            }

            if (predictions.length === 0) {
                predictions.push('All metrics appear stable and healthy!');
            }

            document.getElementById('healthPrediction').textContent = predictions.join('. ');
        }

        // Generate sample data for demonstration
        function generateSampleData() {
            // Clear existing data
            healthData = {
                dates: [],
                heartRate: [],
                systolic: [],
                diastolic: [],
                weight: [],
                steps: [],
                sleep: []
            };

            // Generate 14 days of sample data
            const baseDate = new Date();
            baseDate.setDate(baseDate.getDate() - 14);

            for (let i = 0; i < 14; i++) {
                const currentDate = new Date(baseDate);
                currentDate.setDate(currentDate.getDate() + i);
                
                healthData.dates.push(currentDate.toLocaleDateString());
                
                // Generate realistic health data with some variation
                healthData.heartRate.push(65 + Math.random() * 20 + (i > 10 ? Math.random() * 15 : 0)); // Slight increase later
                healthData.systolic.push(110 + Math.random() * 20 + (Math.random() > 0.9 ? 20 : 0)); // Occasional spike
                healthData.diastolic.push(70 + Math.random() * 15);
                healthData.weight.push(70 + Math.sin(i * 0.3) * 2 + Math.random() * 1);
                healthData.steps.push(6000 + Math.random() * 6000);
                healthData.sleep.push(6.5 + Math.random() * 3 + (Math.random() > 0.8 ? -2 : 0)); // Occasional poor sleep
            }

            updateCharts();
            detectAnomalies();
            updateStatistics();
            generateHealthPredictions();
            
            alert('Sample health data generated! The AI will now analyze patterns and detect anomalies.');
        }

        // Initialize everything when page loads
        window.addEventListener('load', async () => {
            await initializeAnomalyModel();
            initializeCharts();
        });