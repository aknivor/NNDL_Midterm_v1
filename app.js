class MusicPopularityApp {
    constructor() {
        this.dataLoader = new DataLoader();
        this.model = new GRUModel();
        this.isTraining = false;
        this.charts = {};
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // File upload
        document.getElementById('csvFile').addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files[0]);
        });

        // Training controls
        document.getElementById('trainModel').addEventListener('click', () => {
            this.trainModel();
        });

        document.getElementById('evaluateModel').addEventListener('click', () => {
            this.evaluateModel();
        });

        document.getElementById('saveModel').addEventListener('click', () => {
            this.saveModel();
        });

        // Training progress listener
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
            this.hideLoading();
            
            this.updateDataSummary();
            this.showNotification('Data loaded successfully!', 'success');
        } catch (error) {
            this.hideLoading();
            this.showNotification('Error loading file: ' + error.message, 'error');
        }
    }

    updateDataSummary() {
        const data = this.dataLoader.getTrainingData();
        const summaryElement = document.getElementById('dataSummary');
        
        summaryElement.innerHTML = `
            <div class="summary-grid">
                <div class="summary-item">
                    <h4>Training Samples</h4>
                    <p>${data.X_train.shape[0]}</p>
                </div>
                <div class="summary-item">
                    <h4>Test Samples</h4>
                    <p>${data.X_test.shape[0]}</p>
                </div>
                <div class="summary-item">
                    <h4>Input Shape</h4>
                    <p>${data.X_train.shape.slice(1).join(' × ')}</p>
                </div>
                <div class="summary-item">
                    <h4>Tracks</h4>
                    <p>${data.selectedTracks.length}</p>
                </div>
            </div>
        `;
    }

    async trainModel() {
        if (this.isTraining) return;

        try {
            this.isTraining = true;
            const data = this.dataLoader.getTrainingData();
            
            if (!data.X_train) {
                throw new Error('No training data available. Please load CSV file first.');
            }

            this.showLoading('Training model...');
            this.initializeTrainingCharts();
            
            await this.model.fit(data.X_train, data.y_train, data.X_test, data.y_test, 50, 32);
            
            this.hideLoading();
            this.showNotification('Model training completed!', 'success');
        } catch (error) {
            this.hideLoading();
            this.showNotification('Training error: ' + error.message, 'error');
        } finally {
            this.isTraining = false;
        }
    }

    initializeTrainingCharts() {
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');

        // Clean up existing charts
        if (this.charts.lossChart) this.charts.lossChart.destroy();
        if (this.charts.accuracyChart) this.charts.accuracyChart.destroy();

        this.charts.lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Training Loss',
                        borderColor: 'rgb(255, 99, 132)',
                        data: []
                    },
                    {
                        label: 'Validation Loss',
                        borderColor: 'rgb(54, 162, 235)',
                        data: []
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
                        data: []
                    },
                    {
                        label: 'Validation Accuracy',
                        borderColor: 'rgb(153, 102, 255)',
                        data: []
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

        // Update progress text
        document.getElementById('trainingProgress').textContent = 
            `Epoch: ${progress.epoch} | Loss: ${progress.loss.toFixed(4)} | Accuracy: ${progress.accuracy.toFixed(4)}`;
    }

    async evaluateModel() {
        try {
            this.showLoading('Evaluating model...');
            const data = this.dataLoader.getTrainingData();
            
            const evaluation = await this.model.evaluate(data.X_test, data.y_test);
            const predictions = await this.model.predict(data.X_test);
            const trackAccuracies = this.model.computeTrackSpecificAccuracy(
                predictions, data.y_test, data.trackMetadata
            );

            // Clean up tensors
            predictions.dispose();

            this.displayEvaluationResults(evaluation, trackAccuracies);
            this.createAccuracyRankingChart(trackAccuracies);
            this.createHitPotentialMeter(trackAccuracies);
            
            this.hideLoading();
            this.showNotification('Model evaluation completed!', 'success');
        } catch (error) {
            this.hideLoading();
            this.showNotification('Evaluation error: ' + error.message, 'error');
        }
    }

    displayEvaluationResults(evaluation, trackAccuracies) {
        const resultsElement = document.getElementById('evaluationResults');
        
        let trackAccuracyHTML = '';
        const sortedAccuracies = Array.from(trackAccuracies.entries())
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

    createHitPotentialMeter(trackAccuracies) {
        const meterElement = document.getElementById('hitPotentialMeter');
        const sortedTracks = Array.from(trackAccuracies.entries())
            .sort((a, b) => b[1].accuracy - a[1].accuracy)
            .slice(0, 5); // Top 5 tracks

        meterElement.innerHTML = `
            <h4>🎵 Hit Potential Meter</h4>
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
                        <div class="hit-potential ${this.getPotentialClass(data.accuracy)}">
                            ${this.getPotentialText(data.accuracy)}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    getPotentialClass(accuracy) {
        if (accuracy >= 80) return 'high-potential';
        if (accuracy >= 60) return 'medium-potential';
        return 'low-potential';
    }

    getPotentialText(accuracy) {
        if (accuracy >= 80) return 'HIGH POTENTIAL';
        if (accuracy >= 60) return 'MEDIUM POTENTIAL';
        return 'LOW POTENTIAL';
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
        
        // Clean up charts
        Object.values(this.charts).forEach(chart => {
            if (chart && chart.destroy) chart.destroy();
        });
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.musicApp = new MusicPopularityApp();
});
