class MusicPopularityApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = new GRUModel();
        this.isTraining = false;
        this.charts = {};
        this.trainingData = null;
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        document.getElementById('csvFile').addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files[0]);
        });

        document.getElementById('trainModel').addEventListener('click', () => {
            this.trainModel();
        });

        document.getElementById('evaluateModel').addEventListener('click', () => {
            this.evaluateModel();
        });

        document.getElementById('saveModel').addEventListener('click', () => {
            this.saveModel();
        });

        document.getElementById('validateData').addEventListener('click', () => {
            this.validateData();
        });

        document.addEventListener('trainingProgress', (e) => {
            this.updateTrainingProgress(e.detail);
        });
    }

    async handleFileUpload(file) {
        if (!file) return;

        try {
            this.showLoading('Loading and processing CSV data...');
            await this.dataLoader.loadCSV(file);
            this.dataLoader.createSlidingWindows();
            
            const isValid = this.dataLoader.validateData();
            if (!isValid) {
                throw new Error('Data validation failed. Check console for details.');
            }
            
            this.trainingData = this.dataLoader.getTrainingData();
            this.hideLoading();
            
            this.updateDataSummary();
            this.showNotification('Data loaded and validated successfully!', 'success');
        } catch (error) {
            this.hideLoading();
            this.showNotification('Error loading file: ' + error.message, 'error');
            console.error('File loading error:', error);
        }
    }

    updateDataSummary() {
        if (!this.trainingData) return;
        
        const summaryElement = document.getElementById('dataSummary');
        const trainSamples = this.trainingData.X_train ? this.trainingData.X_train.shape[0] : 0;
        const testSamples = this.trainingData.X_test ? this.trainingData.X_test.shape[0] : 0;
        
        summaryElement.innerHTML = `
            <div class="summary-grid">
                <div class="summary-item">
                    <h4>Training Samples</h4>
                    <p>${trainSamples}</p>
                </div>
                <div class="summary-item">
                    <h4>Test Samples</h4>
                    <p>${testSamples}</p>
                </div>
                <div class="summary-item">
                    <h4>Input Shape</h4>
                    <p>7 × 30</p>
                </div>
                <div class="summary-item">
                    <h4>Tracks</h4>
                    <p>${this.trainingData.selectedTracks.length}</p>
                </div>
            </div>
        `;
    }

    async trainModel() {
        if (this.isTraining) {
            this.showNotification('Training already in progress', 'warning');
            return;
        }

        try {
            if (!this.trainingData || !this.trainingData.X_train) {
                throw new Error('No training data available. Please load CSV file first.');
            }

            this.isTraining = true;
            this.showLoading('Training model with early stopping...');
            this.initializeTrainingCharts();
            
            document.getElementById('trainModel').disabled = true;
            document.getElementById('trainingProgress').innerHTML = '<span style="color: orange;">Training started...</span>';
            
            await this.model.fit(
                this.trainingData.X_train, 
                this.trainingData.y_train, 
                this.trainingData.X_test, 
                this.trainingData.y_test, 
                100,
                32
            );
            
            this.hideLoading();
            this.showNotification('Model training completed with early stopping!', 'success');
        } catch (error) {
            this.hideLoading();
            this.showNotification('Training error: ' + error.message, 'error');
            console.error('Training error:', error);
        } finally {
            this.isTraining = false;
            document.getElementById('trainModel').disabled = false;
        }
    }

    initializeTrainingCharts() {
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');

        if (this.charts.lossChart) this.charts.lossChart.destroy();
        if (this.charts.accuracyChart) this.charts.accuracyChart.destroy();

        this.charts.lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Training Loss',
                        borderColor: 'rgb(255, 99, 132)',
                        data: [],
                        tension: 0.4
                    },
                    {
                        label: 'Validation Loss',
                        borderColor: 'rgb(54, 162, 235)',
                        data: [],
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'Epoch' }
                    },
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Loss' }
                    }
                }
            }
        });

        this.charts.accuracyChart = new Chart(accuracyCtx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Training Accuracy',
                        borderColor: 'rgb(75, 192, 192)',
                        data: [],
                        tension: 0.4
                    },
                    {
                        label: 'Validation Accuracy',
                        borderColor: 'rgb(153, 102, 255)',
                        data: [],
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'Epoch' }
                    },
                    y: {
                        beginAtZero: true,
                        max: 1,
                        title: { display: true, text: 'Accuracy' }
                    }
                }
            }
        });
    }

    updateTrainingProgress(progress) {
        if (this.charts.lossChart) {
            this.charts.lossChart.data.datasets[0].data.push({x: progress.epoch, y: progress.loss});
            this.charts.lossChart.data.datasets[1].data.push({x: progress.epoch, y: progress.val_loss});
            this.charts.lossChart.update('none');
        }

        if (this.charts.accuracyChart) {
            this.charts.accuracyChart.data.datasets[0].data.push({x: progress.epoch, y: progress.accuracy});
            this.charts.accuracyChart.data.datasets[1].data.push({x: progress.epoch, y: progress.val_accuracy});
            this.charts.accuracyChart.update('none');
        }

        const earlyStoppingInfo = progress.earlyStopping > 0 ? 
            ` | Early stopping patience: ${progress.earlyStopping}` : '';
            
        document.getElementById('trainingProgress').innerHTML = 
            `Epoch: ${progress.epoch} | Loss: ${progress.loss.toFixed(4)} | Acc: ${progress.accuracy.toFixed(4)} | Val Loss: ${progress.val_loss.toFixed(4)} | Val Acc: ${progress.val_accuracy.toFixed(4)}${earlyStoppingInfo}`;
    }

    async evaluateModel() {
        try {
            if (!this.trainingData || !this.trainingData.X_test) {
                throw new Error('No test data available. Please load data and train model first.');
            }

            this.showLoading('Evaluating model with consistent metrics...');
            
            const evaluation = await this.model.evaluate(this.trainingData.X_test, this.trainingData.y_test);
            const predictions = await this.model.predict(this.trainingData.X_test);
            const consistentAccuracy = await this.model.computeConsistentAccuracy(predictions, this.trainingData.y_test);
            const accuracyAnalysis = this.model.computeTrackSpecificAccuracy(
                predictions, this.trainingData.y_test, this.trainingData.trackMetadata
            );

            // NEW: Compute feature importance and breakout detection
            const featureImportance = await this.computeFeatureImportance();
            const breakoutTracks = this.detectBreakoutTracks(predictions, this.trainingData);

            predictions.dispose();

            this.displayEvaluationResults(evaluation, consistentAccuracy, accuracyAnalysis);
            this.createAccuracyRankingChart(accuracyAnalysis.trackAccuracies);
            this.createHitPotentialMeter(accuracyAnalysis.trackAccuracies);
            this.createDayAccuracyChart(accuracyAnalysis.dayAccuracies);
            
            // NEW: Display feature importance and breakout detection
            this.displayFeatureImportance(featureImportance);
            this.displayBreakoutDetection(breakoutTracks);
            
            this.hideLoading();
            this.showNotification('Model evaluation completed! All features updated.', 'success');
        } catch (error) {
            this.hideLoading();
            this.showNotification('Evaluation error: ' + error.message, 'error');
            console.error('Evaluation error:', error);
        }
    }

    // NEW: Compute Feature Importance
    async computeFeatureImportance() {
        try {
            const data = this.trainingData;
            if (!data || !data.X_test) return null;

            // Get baseline accuracy
            const baselineResults = await this.model.evaluate(data.X_test, data.y_test);
            const baselineAccuracy = (await baselineResults[1].data())[0];
            
            baselineResults[0].dispose();
            baselineResults[1].dispose();

            const features = ['Streams', 'Danceability', 'Energy'];
            const importanceScores = [];
            
            // Test importance of each feature by shuffling it
            for (let featureIdx = 0; featureIdx < 3; featureIdx++) {
                const shuffledData = await this.shuffleFeature(data.X_test, featureIdx);
                const shuffledResults = await this.model.evaluate(shuffledData, data.y_test);
                const shuffledAccuracy = (await shuffledResults[1].data())[0];
                
                const importance = baselineAccuracy - shuffledAccuracy;
                importanceScores.push({
                    feature: features[featureIdx],
                    importance: Math.max(importance * 100, 0),
                    description: this.getFeatureDescription(features[featureIdx])
                });
                
                shuffledData.dispose();
                shuffledResults[0].dispose();
                shuffledResults[1].dispose();
            }
            
            return importanceScores.sort((a, b) => b.importance - a.importance);
        } catch (error) {
            console.error('Error computing feature importance:', error);
            return null;
        }
    }

    async shuffleFeature(X, featureIndex) {
        const data = await X.array();
        
        // Shuffle the specified feature across all samples
        for (let sample = 0; sample < data.length; sample++) {
            for (let day = 0; day < data[sample].length; day++) {
                for (let track = 0; track < 10; track++) {
                    const featurePos = track * 3 + featureIndex;
                    const randomSample = Math.floor(Math.random() * data.length);
                    const randomDay = Math.floor(Math.random() * data[randomSample].length);
                    const randomTrack = Math.floor(Math.random() * 10);
                    const randomPos = randomTrack * 3 + featureIndex;
                    
                    const temp = data[sample][day][featurePos];
                    data[sample][day][featurePos] = data[randomSample][randomDay][randomPos];
                    data[randomSample][randomDay][randomPos] = temp;
                }
            }
        }
        
        return tf.tensor3d(data);
    }

    getFeatureDescription(feature) {
        const descriptions = {
            'Streams': 'Historical streaming patterns - most important for trend prediction',
            'Danceability': 'Musical rhythm and dance-friendly characteristics',
            'Energy': 'Intensity and activity level of the track'
        };
        return descriptions[feature] || 'Audio feature characteristic';
    }

    // NEW: Detect Breakout Tracks
    detectBreakoutTracks(predictions, trainingData) {
        try {
            const predData = predictions.arraySync();
            const tracks = Array.from(trainingData.trackMetadata.values());
            
            const breakoutScores = tracks.map((track, trackIndex) => {
                let breakoutScore = 0;
                let confidence = 0;
                let sampleCount = 0;
                
                for (let sampleIdx = 0; sampleIdx < predData.length; sampleIdx++) {
                    const day1Idx = trackIndex * 3;
                    const day2Idx = trackIndex * 3 + 1;
                    const day3Idx = trackIndex * 3 + 2;
                    
                    const day1Prob = predData[sampleIdx][day1Idx];
                    const day2Prob = predData[sampleIdx][day2Idx];
                    const day3Prob = predData[sampleIdx][day3Idx];
                    
                    const trendStrength = (day3Prob - day1Prob) * 2 + (day2Prob - day1Prob);
                    const overallConfidence = (day1Prob + day2Prob + day3Prob) / 3;
                    
                    breakoutScore += trendStrength;
                    confidence += overallConfidence;
                    sampleCount++;
                }
                
                const avgBreakoutScore = breakoutScore / sampleCount;
                const avgConfidence = confidence / sampleCount;
                
                return {
                    trackId: track.id,
                    trackName: track.name,
                    breakoutScore: avgBreakoutScore * 100,
                    confidence: avgConfidence * 100,
                    trend: avgBreakoutScore > 0 ? 'rising' : 'stable',
                    riskLevel: this.calculateRiskLevel(avgBreakoutScore, avgConfidence)
                };
            });
            
            return breakoutScores.sort((a, b) => b.breakoutScore - a.breakoutScore);
        } catch (error) {
            console.error('Error detecting breakout tracks:', error);
            return [];
        }
    }

    calculateRiskLevel(breakoutScore, confidence) {
        if (breakoutScore > 0.1 && confidence > 0.7) return 'low';
        if (breakoutScore > 0.05 && confidence > 0.6) return 'medium';
        if (breakoutScore > 0) return 'high';
        return 'very-high';
    }

    // NEW: Display Feature Importance
    displayFeatureImportance(featureImportance) {
        const featureElement = document.getElementById('featureImportance');
        
        if (!featureImportance || featureImportance.length === 0) {
            featureElement.innerHTML = `
                <h2>🔍 Feature Importance</h2>
                <p>Unable to compute feature importance. Please check if model is trained properly.</p>
            `;
            return;
        }

        let featureHTML = `
            <h2>🔍 Feature Importance</h2>
            <p>How much each feature contributes to popularity predictions:</p>
            <div class="feature-importance-container">
        `;

        featureImportance.forEach(feature => {
            const width = Math.min(feature.importance * 5, 100);
            featureHTML += `
                <div class="feature-item">
                    <div class="feature-header">
                        <span class="feature-name">${feature.feature}</span>
                        <span class="feature-score">${feature.importance.toFixed(1)}%</span>
                    </div>
                    <div class="feature-bar-container">
                        <div class="feature-bar" style="width: ${width}%"></div>
                    </div>
                    <div class="feature-description">${feature.description}</div>
                </div>
            `;
        });

        featureHTML += `</div>`;
        featureElement.innerHTML = featureHTML;
    }

    // NEW: Display Breakout Detection
    displayBreakoutDetection(breakoutTracks) {
        const breakoutElement = document.getElementById('breakoutDetection');
        
        if (!breakoutTracks || breakoutTracks.length === 0) {
            breakoutElement.innerHTML = `
                <h2>🚀 Breakout Detection</h2>
                <p>No breakout patterns detected in current data.</p>
            `;
            return;
        }

        let breakoutHTML = `
            <h2>🚀 Breakout Detection</h2>
            <p>Tracks showing unusual growth patterns:</p>
            <div class="breakout-tracks">
        `;

        breakoutTracks.slice(0, 3).forEach((track, index) => {
            if (track.breakoutScore > 0) {
                breakoutHTML += `
                    <div class="breakout-track-item ${track.riskLevel}-risk">
                        <div class="breakout-rank">${index + 1}</div>
                        <div class="breakout-info">
                            <div class="breakout-track-name">${track.trackName}</div>
                            <div class="breakout-metrics">
                                <span class="breakout-score">Breakout Score: ${track.breakoutScore.toFixed(1)}%</span>
                                <span class="confidence">Confidence: ${track.confidence.toFixed(1)}%</span>
                            </div>
                            <div class="trend-indicator trend-${track.trend}">
                                📈 ${track.trend.toUpperCase()} TREND
                            </div>
                        </div>
                        <div class="risk-level ${track.riskLevel}">
                            ${track.riskLevel.toUpperCase().replace('-', ' ')} RISK
                        </div>
                    </div>
                `;
            }
        });

        if (!breakoutHTML.includes('breakout-track-item')) {
            breakoutHTML += `<p>No strong breakout patterns detected in current evaluation.</p>`;
        }

        breakoutHTML += `</div>`;
        breakoutElement.innerHTML = breakoutHTML;
    }

    displayEvaluationResults(evaluation, consistentAccuracy, accuracyAnalysis) {
        const resultsElement = document.getElementById('evaluationResults');
        
        let trackAccuracyHTML = '';
        const sortedAccuracies = Array.from(accuracyAnalysis.trackAccuracies.entries())
            .sort((a, b) => b[1].accuracy - a[1].accuracy);
        
        sortedAccuracies.forEach(([trackId, data]) => {
            trackAccuracyHTML += `
                <div class="track-accuracy-item">
                    <span class="track-name">${data.trackName}</span>
                    <div class="accuracy-bar-container">
                        <div class="accuracy-bar" style="width: ${data.accuracy}%"></div>
                        <span class="accuracy-text">${data.accuracy.toFixed(1)}%</span>
                    </div>
                </div>
            `;
        });

        resultsElement.innerHTML = `
            <div class="evaluation-summary">
                <h4>Overall Model Performance</h4>
                <p><strong>Loss:</strong> ${evaluation.loss.toFixed(4)}</p>
                <p><strong>Accuracy:</strong> ${(evaluation.accuracy * 100).toFixed(2)}%</p>
                <p><strong>Consistent Accuracy:</strong> ${consistentAccuracy.toFixed(2)}%</p>
            </div>
            <div class="track-accuracies">
                <h4>Track-Specific Accuracy</h4>
                ${trackAccuracyHTML}
            </div>
        `;
    }

    createAccuracyRankingChart(trackAccuracies) {
        const ctx = document.getElementById('accuracyRankingChart').getContext('2d');
        
        const sortedData = Array.from(trackAccuracies.entries())
            .sort((a, b) => a[1].accuracy - b[1].accuracy);
        
        const labels = sortedData.map(([_, data]) => data.trackName);
        const accuracies = sortedData.map(([_, data]) => data.accuracy);

        if (this.charts.accuracyRankingChart) {
            this.charts.accuracyRankingChart.destroy();
        }

        this.charts.accuracyRankingChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Prediction Accuracy (%)',
                    data: accuracies,
                    backgroundColor: 'rgba(54, 162, 235, 0.8)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Track Prediction Accuracy Ranking'
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Accuracy (%)'
                        }
                    }
                }
            }
        });
    }

    createDayAccuracyChart(dayAccuracies) {
        const ctx = document.getElementById('dayAccuracyChart').getContext('2d');
        
        if (this.charts.dayAccuracyChart) {
            this.charts.dayAccuracyChart.destroy();
        }

        this.charts.dayAccuracyChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Day +1', 'Day +2', 'Day +3'],
                datasets: [{
                    label: 'Prediction Accuracy (%)',
                    data: [dayAccuracies.day1, dayAccuracies.day2, dayAccuracies.day3],
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(75, 192, 192, 0.8)',
                        'rgba(153, 102, 255, 0.8)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Prediction Accuracy by Forecast Day'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Accuracy (%)'
                        }
                    }
                }
            }
        });
    }

    createHitPotentialMeter(trackAccuracies) {
        const meterElement = document.getElementById('hitPotentialMeter');
        const sortedTracks = Array.from(trackAccuracies.entries())
            .sort((a, b) => b[1].accuracy - a[1].accuracy)
            .slice(0, 5);

        meterElement.innerHTML = `
            <h4>Hit Potential Meter</h4>
            <div class="hit-tracks">
                ${sortedTracks.map(([trackId, data], index) => `
                    <div class="hit-track-item">
                        <div class="hit-rank">${index + 1}</div>
                        <div class="hit-track-info">
                            <div class="hit-track-name">${data.trackName}</div>
                            <div class="hit-confidence">
                                <div class="confidence-bar" style="width: ${data.accuracy}%"></div>
                                <span>${data.accuracy.toFixed(1)}% confidence</span>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    async validateData() {
        try {
            this.showLoading('Validating data integrity...');
            const isValid = this.dataLoader.validateData();
            this.hideLoading();
            
            if (isValid) {
                this.showNotification('Data validation passed! All checks OK.', 'success');
            } else {
                this.showNotification('Data validation failed. Check console for details.', 'error');
            }
        } catch (error) {
            this.hideLoading();
            this.showNotification('Validation error: ' + error.message, 'error');
        }
    }

    async saveModel() {
        try {
            await this.model.saveModel();
            this.showNotification('Model saved successfully!', 'success');
        } catch (error) {
            this.showNotification('Error saving model: ' + error.message, 'error');
        }
    }

    showLoading(message) {
        document.getElementById('loadingMessage').textContent = message;
        document.getElementById('loadingOverlay').style.display = 'flex';
    }

    hideLoading() {
        document.getElementById('loadingOverlay').style.display = 'none';
    }

    showNotification(message, type) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }

    dispose() {
        this.dataLoader.dispose();
        this.model.dispose();
        
        Object.values(this.charts).forEach(chart => {
            if (chart && chart.destroy) chart.destroy();
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.musicApp = new MusicPopularityApp();
});
