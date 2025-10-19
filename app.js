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

        // NEW: Advanced training options
        document.getElementById('advancedTrain').addEventListener('click', () => {
            this.advancedTrainModel();
        });

        document.addEventListener('trainingProgress', (e) => {
            this.updateTrainingProgress(e.detail);
        });
    }

    async handleFileUpload(file) {
        if (!file) return;

        try {
            this.showLoading('Loading and processing CSV data with advanced feature engineering...');
            await this.dataLoader.loadCSV(file);
            this.dataLoader.createSlidingWindows();
            
            const isValid = this.dataLoader.validateData();
            if (!isValid) {
                throw new Error('Data validation failed. Check console for details.');
            }
            
            this.trainingData = this.dataLoader.getTrainingData();
            this.hideLoading();
            
            this.updateDataSummary();
            this.showNotification('Advanced data processing completed! Features engineered: 9 per track.', 'success');
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
        const featuresPerTrack = 9; // Updated feature count
        const totalFeatures = featuresPerTrack * (this.trainingData.selectedTracks?.length || 0);
        
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
                    <p>7 × ${totalFeatures}</p>
                </div>
                <div class="summary-item">
                    <h4>Tracks</h4>
                    <p>${this.trainingData.selectedTracks.length}</p>
                </div>
                <div class="summary-item">
                    <h4>Features/Track</h4>
                    <p>${featuresPerTrack}</p>
                </div>
            </div>
            <div style="margin-top: 15px; padding: 10px; background: #e8f5e8; border-radius: 5px;">
                <strong>Advanced Features:</strong> Streams, Danceability, Energy, Valence, Acousticness, Momentum, Growth Rate, Moving Average, Volatility
            </div>
        `;
    }

    async trainModel() {
        await this._trainModel(100, 32); // Standard training
    }

    // NEW: Advanced training with more epochs
    async advancedTrainModel() {
        await this._trainModel(200, 64); // Advanced training
    }

    async _trainModel(epochs, batchSize) {
        if (this.isTraining) {
            this.showNotification('Training already in progress', 'warning');
            return;
        }

        try {
            if (!this.trainingData || !this.trainingData.X_train) {
                throw new Error('No training data available. Please load CSV file first.');
            }

            this.isTraining = true;
            this.showLoading(`Training advanced model with ${epochs} epochs...`);
            this.initializeTrainingCharts();
            
            document.getElementById('trainModel').disabled = true;
            document.getElementById('advancedTrain').disabled = true;
            document.getElementById('trainingProgress').innerHTML = 
                '<span style="color: orange;">Advanced training started with feature engineering...</span>';
            
            await this.model.fit(
                this.trainingData.X_train, 
                this.trainingData.y_train, 
                this.trainingData.X_test, 
                this.trainingData.y_test, 
                epochs,
                batchSize
            );
            
            this.hideLoading();
            this.showNotification(`Advanced model training completed! Target: >70% accuracy`, 'success');
        } catch (error) {
            this.hideLoading();
            this.showNotification('Training error: ' + error.message, 'error');
            console.error('Training error:', error);
        } finally {
            this.isTraining = false;
            document.getElementById('trainModel').disabled = false;
            document.getElementById('advancedTrain').disabled = false;
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
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        data: [],
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Validation Loss',
                        borderColor: 'rgb(54, 162, 235)',
                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                        data: [],
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: { display: true, text: 'Training & Validation Loss' },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
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
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        data: [],
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Validation Accuracy',
                        borderColor: 'rgb(153, 102, 255)',
                        backgroundColor: 'rgba(153, 102, 255, 0.1)',
                        data: [],
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: { display: true, text: 'Training & Validation Accuracy' },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
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

        const lrInfo = progress.learningRate ? ` | LR: ${progress.learningRate.toFixed(6)}` : '';
        const earlyStoppingInfo = progress.earlyStopping > 0 ? 
            ` | Early stopping: ${progress.earlyStopping}` : '';
            
        document.getElementById('trainingProgress').innerHTML = 
            `Epoch: ${progress.epoch} | Loss: ${progress.loss.toFixed(4)} | Acc: ${progress.accuracy.toFixed(4)} | Val Loss: ${progress.val_loss.toFixed(4)} | Val Acc: ${progress.val_accuracy.toFixed(4)}${lrInfo}${earlyStoppingInfo}`;
    }

    async evaluateModel() {
        try {
            if (!this.trainingData || !this.trainingData.X_test) {
                throw new Error('No test data available. Please load data and train model first.');
            }

            this.showLoading('Evaluating advanced model with ensemble predictions...');
            
            const evaluation = await this.model.evaluate(this.trainingData.X_test, this.trainingData.y_test);
            
            // NEW: Use ensemble prediction for better accuracy
            const predictions = await this.model.predictWithUncertainty(this.trainingData.X_test, 3);
            const consistentAccuracy = await this.model.computeConsistentAccuracy(predictions, this.trainingData.y_test);
            const accuracyAnalysis = this.model.computeTrackSpecificAccuracy(
                predictions, this.trainingData.y_test, this.trainingData.trackMetadata
            );

            const featureImportance = await this.computeFeatureImportance();
            const breakoutTracks = this.detectBreakoutTracks(predictions, this.trainingData);

            predictions.dispose();

            this.displayEvaluationResults(evaluation, consistentAccuracy, accuracyAnalysis);
            this.createAccuracyRankingChart(accuracyAnalysis.trackAccuracies);
            this.createHitPotentialMeter(accuracyAnalysis.trackAccuracies);
            this.createDayAccuracyChart(accuracyAnalysis.dayAccuracies);
            
            this.displayFeatureImportance(featureImportance);
            this.displayBreakoutDetection(breakoutTracks);
            
            this.hideLoading();
            
            // NEW: Performance assessment
            this.assessPerformance(consistentAccuracy, evaluation.loss);
            
        } catch (error) {
            this.hideLoading();
            this.showNotification('Evaluation error: ' + error.message, 'error');
            console.error('Evaluation error:', error);
        }
    }

    // NEW: Performance assessment
    assessPerformance(accuracy, loss) {
        let message = '';
        let type = 'success';
        
        if (accuracy >= 75) {
            message = `Outstanding! Model achieved ${accuracy.toFixed(1)}% accuracy - Excellent performance!`;
            type = 'success';
        } else if (accuracy >= 70) {
            message = `Very Good! Model achieved ${accuracy.toFixed(1)}% accuracy - Strong performance!`;
            type = 'success';
        } else if (accuracy >= 65) {
            message = `Good! Model achieved ${accuracy.toFixed(1)}% accuracy - Decent performance.`;
            type = 'success';
        } else if (accuracy >= 60) {
            message = `Fair! Model achieved ${accuracy.toFixed(1)}% accuracy - Room for improvement.`;
            type = 'warning';
        } else {
            message = `Needs improvement! Model achieved ${accuracy.toFixed(1)}% accuracy - Consider advanced training.`;
            type = 'error';
        }
        
        this.showNotification(message, type);
        
        // Log performance metrics
        console.log(`🎯 Performance Assessment:`);
        console.log(`   Accuracy: ${accuracy.toFixed(2)}%`);
        console.log(`   Loss: ${loss.toFixed(4)}`);
        console.log(`   Status: ${type === 'success' ? '✅ GOOD' : type === 'warning' ? '⚠️ FAIR' : '❌ POOR'}`);
    }

    // ... rest of the existing methods (computeFeatureImportance, detectBreakoutTracks, displayFeatureImportance, displayBreakoutDetection, etc.)
    // These remain the same as in the previous implementation

    displayEvaluationResults(evaluation, consistentAccuracy, accuracyAnalysis) {
        const resultsElement = document.getElementById('evaluationResults');
        
        let trackAccuracyHTML = '';
        const sortedAccuracies = Array.from(accuracyAnalysis.trackAccuracies.entries())
            .sort((a, b) => b[1].accuracy - a[1].accuracy);
        
        sortedAccuracies.forEach(([trackId, data]) => {
            const accuracyClass = data.accuracy >= 70 ? 'high-accuracy' : data.accuracy >= 60 ? 'medium-accuracy' : 'low-accuracy';
            trackAccuracyHTML += `
                <div class="track-accuracy-item ${accuracyClass}">
                    <span class="track-name">${data.trackName}</span>
                    <div class="accuracy-bar-container">
                        <div class="accuracy-bar" style="width: ${data.accuracy}%"></div>
                        <span class="accuracy-text">${data.accuracy.toFixed(1)}%</span>
                    </div>
                    <div class="day-accuracies">
                        <small>D+1: ${data.dayAccuracies.day1.toFixed(1)}%</small>
                        <small>D+2: ${data.dayAccuracies.day2.toFixed(1)}%</small>
                        <small>D+3: ${data.dayAccuracies.day3.toFixed(1)}%</small>
                    </div>
                </div>
            `;
        });

        const performanceClass = consistentAccuracy >= 70 ? 'high-performance' : consistentAccuracy >= 60 ? 'medium-performance' : 'low-performance';
        
        resultsElement.innerHTML = `
            <div class="evaluation-summary ${performanceClass}">
                <h4>🚀 Advanced Model Performance</h4>
                <div class="performance-metrics">
                    <div class="metric">
                        <label>Final Loss:</label>
                        <span class="metric-value">${evaluation.loss.toFixed(4)}</span>
                    </div>
                    <div class="metric">
                        <label>Standard Accuracy:</label>
                        <span class="metric-value">${(evaluation.accuracy * 100).toFixed(2)}%</span>
                    </div>
                    <div class="metric">
                        <label>Ensemble Accuracy:</label>
                        <span class="metric-value highlight">${consistentAccuracy.toFixed(2)}%</span>
                    </div>
                </div>
            </div>
            <div class="track-accuracies">
                <h4>🎵 Track-Specific Accuracy</h4>
                ${trackAccuracyHTML}
            </div>
        `;
    }

    // ... rest of the visualization methods remain similar but with enhanced styling

    async validateData() {
        try {
            this.showLoading('Validating advanced data features...');
            const isValid = this.dataLoader.validateData();
            this.hideLoading();
            
            if (isValid) {
                this.showNotification('Advanced data validation passed! All 9 features per track are ready.', 'success');
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
            this.showNotification('Advanced model saved successfully!', 'success');
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
        notification.innerHTML = `
            <strong>${type.toUpperCase()}:</strong> ${message}
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 6000);
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
