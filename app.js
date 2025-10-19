class MusicPopularityApp {
    // ... existing code ...

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

            // Simple permutation importance
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
                    importance: Math.max(importance * 100, 0), // Convert to percentage
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
        // Create a copy of the tensor
        const data = await X.array();
        
        // Shuffle the specified feature across all samples, tracks, and days
        for (let sample = 0; sample < data.length; sample++) {
            for (let day = 0; day < data[sample].length; day++) {
                for (let track = 0; track < 10; track++) {
                    const featurePos = track * 3 + featureIndex;
                    // Simple shuffle: swap with random position
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
                    // Look at predictions for days 1, 2, 3 for this track
                    const day1Idx = trackIndex * 3;
                    const day2Idx = trackIndex * 3 + 1;
                    const day3Idx = trackIndex * 3 + 2;
                    
                    const day1Prob = predData[sampleIdx][day1Idx];
                    const day2Prob = predData[sampleIdx][day2Idx];
                    const day3Prob = predData[sampleIdx][day3Idx];
                    
                    // Calculate breakout pattern: increasing probability over time
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
                    breakoutScore: avgBreakoutScore * 100, // Convert to percentage
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
            const width = Math.min(feature.importance * 5, 100); // Scale for visibility
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

        // Show top 3 breakout tracks
        breakoutTracks.slice(0, 3).forEach((track, index) => {
            if (track.breakoutScore > 0) { // Only show tracks with positive breakout potential
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

    // ... rest of existing code ...
}
