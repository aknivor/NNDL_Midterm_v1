class GRUModel {
    constructor(inputShape = [7, 90]) { // Updated for 10 tracks × 9 features
        this.model = null;
        this.inputShape = inputShape;
        this.history = {
            loss: [],
            val_loss: [],
            accuracy: [],
            val_accuracy: []
        };
        this.bestWeights = null;
        this.bestValLoss = Infinity;
        this.learningRate = 0.001;
    }

    buildModel() {
        this.model = tf.sequential({
            layers: [
                // Input layer with batch normalization
                tf.layers.gru({
                    units: 128,
                    returnSequences: true,
                    inputShape: this.inputShape,
                    dropout: 0.3,
                    recurrentDropout: 0.3,
                    kernelRegularizer: tf.regularizers.l2({l2: 0.001}),
                    recurrentRegularizer: tf.regularizers.l2({l2: 0.001}),
                    name: 'gru_1'
                }),
                
                // Batch normalization between GRU layers
                tf.layers.batchNormalization(),
                
                // Second GRU layer
                tf.layers.gru({
                    units: 64,
                    returnSequences: false,
                    dropout: 0.3,
                    recurrentDropout: 0.3,
                    kernelRegularizer: tf.regularizers.l2({l2: 0.001}),
                    recurrentRegularizer: tf.regularizers.l2({l2: 0.001}),
                    name: 'gru_2'
                }),
                
                // Batch normalization
                tf.layers.batchNormalization(),
                
                // First dense layer
                tf.layers.dense({
                    units: 128,
                    activation: 'relu',
                    kernelRegularizer: tf.regularizers.l2({l2: 0.001}),
                    name: 'dense_1'
                }),
                tf.layers.dropout({rate: 0.4}),
                tf.layers.batchNormalization(),
                
                // Second dense layer
                tf.layers.dense({
                    units: 64,
                    activation: 'relu',
                    kernelRegularizer: tf.regularizers.l2({l2: 0.001}),
                    name: 'dense_2'
                }),
                tf.layers.dropout({rate: 0.3}),
                tf.layers.batchNormalization(),
                
                // Third dense layer
                tf.layers.dense({
                    units: 32,
                    activation: 'relu',
                    kernelRegularizer: tf.regularizers.l2({l2: 0.001}),
                    name: 'dense_3'
                }),
                tf.layers.dropout({rate: 0.2}),
                
                // Output layer
                tf.layers.dense({
                    units: 30, // 10 tracks × 3 days
                    activation: 'sigmoid',
                    name: 'output'
                })
            ]
        });

        // NEW: Custom optimizer with learning rate scheduling
        const optimizer = tf.train.adam(this.learningRate);
        
        this.model.compile({
            optimizer: optimizer,
            loss: 'binaryCrossentropy',
            metrics: ['binaryAccuracy', 'precision', 'recall'] // NEW: Additional metrics
        });

        console.log('Advanced model built successfully');
        console.log('Model summary:', this.getModelSummary());
        return this.model;
    }

    async fit(X_train, y_train, X_test, y_test, epochs = 200, batchSize = 32) {
        if (!this.model) {
            this.buildModel();
        }

        this.bestValLoss = Infinity;
        this.bestWeights = null;
        this.history = { loss: [], val_loss: [], accuracy: [], val_accuracy: [] };
        
        const patience = 25;
        let patienceCounter = 0;
        let currentLearningRate = this.learningRate;

        console.log('Starting advanced model training...');
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            // NEW: Learning rate scheduling
            if (epoch > 0 && epoch % 30 === 0) {
                currentLearningRate *= 0.5; // Reduce learning rate every 30 epochs
                const newOptimizer = tf.train.adam(currentLearningRate);
                this.model.compile({
                    optimizer: newOptimizer,
                    loss: 'binaryCrossentropy',
                    metrics: ['binaryAccuracy', 'precision', 'recall']
                });
                console.log(`Reduced learning rate to: ${currentLearningRate}`);
            }

            const history = await this.model.fit(X_train, y_train, {
                epochs: 1,
                batchSize: batchSize,
                validationData: [X_test, y_test],
                verbose: 0
            });

            const loss = history.history.loss[0];
            const accuracy = history.history.binaryAccuracy[0];
            const valLoss = history.history.val_loss[0];
            const valAccuracy = history.history.val_binaryAccuracy[0];

            this.history.loss.push(loss);
            this.history.accuracy.push(accuracy);
            this.history.val_loss.push(valLoss);
            this.history.val_accuracy.push(valAccuracy);

            // NEW: Improved early stopping with momentum
            if (valLoss < this.bestValLoss - 0.001) { // Only update if significant improvement
                this.bestValLoss = valLoss;
                patienceCounter = 0;
                this.bestWeights = await this.model.getWeights();
                console.log(`Epoch ${epoch + 1}: New best validation loss: ${valLoss.toFixed(4)}`);
            } else {
                patienceCounter++;
                if (patienceCounter >= patience) {
                    console.log(`Early stopping triggered at epoch ${epoch + 1}`);
                    await this.model.setWeights(this.bestWeights);
                    break;
                }
            }

            const event = new CustomEvent('trainingProgress', {
                detail: {
                    epoch: epoch + 1,
                    loss: loss,
                    accuracy: accuracy,
                    val_loss: valLoss,
                    val_accuracy: valAccuracy,
                    earlyStopping: patienceCounter,
                    learningRate: currentLearningRate
                }
            });
            document.dispatchEvent(event);

            if ((epoch + 1) % 10 === 0) {
                console.log(`Epoch ${epoch + 1}/${epochs} - LR: ${currentLearningRate.toFixed(6)} - Loss: ${loss.toFixed(4)} - Acc: ${accuracy.toFixed(4)} - Val Loss: ${valLoss.toFixed(4)} - Val Acc: ${valAccuracy.toFixed(4)}`);
            }
        }

        console.log('Training completed');
        return this.history;
    }

    async predict(X) {
        if (!this.model) {
            throw new Error('Model not built or loaded');
        }
        return this.model.predict(X);
    }

    async evaluate(X_test, y_test) {
        if (!this.model) {
            throw new Error('Model not built or loaded');
        }
        
        const results = this.model.evaluate(X_test, y_test);
        const loss = await results[0].data();
        const accuracy = await results[1].data();
        
        results[0].dispose();
        results[1].dispose();
        
        return {
            loss: loss[0],
            accuracy: accuracy[0]
        };
    }

    // NEW: Ensemble prediction for better accuracy
    async predictWithUncertainty(X, numSamples = 5) {
        if (!this.model) {
            throw new Error('Model not built or loaded');
        }
        
        const predictions = [];
        for (let i = 0; i < numSamples; i++) {
            // Enable dropout during inference for uncertainty estimation
            const prediction = this.model.predict(X, {training: true});
            predictions.push(prediction);
        }
        
        // Average the predictions
        let meanPrediction = predictions[0];
        for (let i = 1; i < predictions.length; i++) {
            meanPrediction = meanPrediction.add(predictions[i]);
        }
        meanPrediction = meanPrediction.div(tf.scalar(predictions.length));
        
        // Clean up
        predictions.forEach(p => p.dispose());
        
        return meanPrediction;
    }

    computeTrackSpecificAccuracy(predictions, y_true, trackMetadata) {
        const predData = predictions.arraySync();
        const trueData = y_true.arraySync();
        
        const trackAccuracies = new Map();
        const tracks = Array.from(trackMetadata.keys());
        
        const dayAccuracies = { day1: 0, day2: 0, day3: 0 };
        const dayCounts = { day1: 0, day2: 0, day3: 0 };
        
        tracks.forEach((trackId, trackIndex) => {
            let correct = 0;
            let total = 0;
            const trackDayAccuracies = [0, 0, 0];
            const trackDayCounts = [0, 0, 0];
            
            for (let sampleIdx = 0; sampleIdx < predData.length; sampleIdx++) {
                for (let dayOffset = 0; dayOffset < 3; dayOffset++) {
                    const predIdx = trackIndex * 3 + dayOffset;
                    const prediction = predData[sampleIdx][predIdx] > 0.5 ? 1 : 0;
                    const actual = trueData[sampleIdx][predIdx] > 0.5 ? 1 : 0; // Convert probability to binary
                    
                    if (actual !== undefined) {
                        total++;
                        dayCounts[`day${dayOffset + 1}`]++;
                        
                        if (prediction === actual) {
                            correct++;
                            dayAccuracies[`day${dayOffset + 1}`]++;
                            trackDayAccuracies[dayOffset]++;
                        }
                        trackDayCounts[dayOffset]++;
                    }
                }
            }
            
            const accuracy = total > 0 ? (correct / total) * 100 : 0;
            const day1Acc = trackDayCounts[0] > 0 ? (trackDayAccuracies[0] / trackDayCounts[0]) * 100 : 0;
            const day2Acc = trackDayCounts[1] > 0 ? (trackDayAccuracies[1] / trackDayCounts[1]) * 100 : 0;
            const day3Acc = trackDayCounts[2] > 0 ? (trackDayAccuracies[2] / trackDayCounts[2]) * 100 : 0;
            
            trackAccuracies.set(trackId, {
                accuracy: accuracy,
                trackName: trackMetadata.get(trackId).name || trackId,
                dayAccuracies: {
                    day1: day1Acc,
                    day2: day2Acc,
                    day3: day3Acc
                }
            });
        });
        
        Object.keys(dayAccuracies).forEach(day => {
            dayAccuracies[day] = dayCounts[day] > 0 ? (dayAccuracies[day] / dayCounts[day]) * 100 : 0;
        });
        
        return {
            trackAccuracies,
            dayAccuracies
        };
    }

    async computeConsistentAccuracy(predictions, y_true) {
        const binaryPreds = predictions.greater(0.5).cast('float32');
        const binaryTrue = y_true.greater(0.5).cast('float32');
        const correct = binaryPreds.equal(binaryTrue).sum();
        const total = tf.size(y_true);
        
        const accuracy = (await correct.data())[0] / (await total.data())[0] * 100;
        
        correct.dispose();
        total.dispose();
        binaryPreds.dispose();
        binaryTrue.dispose();
        
        return accuracy;
    }

    getModelSummary() {
        if (!this.model) return 'Model not built';
        
        let summary = 'Advanced Model Architecture:\n';
        let totalParams = 0;
        this.model.layers.forEach((layer, i) => {
            const params = layer.countParams();
            totalParams += params;
            summary += `${i + 1}. ${layer.name} (${layer.getClassName()}) - Params: ${params} - Output: ${JSON.stringify(layer.outputShape)}\n`;
        });
        summary += `Total Parameters: ${totalParams}`;
        return summary;
    }

    async saveModel() {
        if (!this.model) {
            throw new Error('No model to save');
        }
        
        const saveResult = await this.model.save('downloads://music-popularity-advanced-model');
        console.log('Advanced model saved successfully');
        return saveResult;
    }

    async loadModel(modelArtifacts) {
        this.model = await tf.loadLayersModel(modelArtifacts);
        console.log('Advanced model loaded successfully');
        return this.model;
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
        if (this.bestWeights) {
            this.bestWeights.forEach(w => w.dispose());
        }
    }
}
