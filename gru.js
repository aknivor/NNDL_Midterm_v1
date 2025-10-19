class GRUModel {
    constructor(inputShape = [7, 30]) {
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
    }

    buildModel() {
        this.model = tf.sequential({
            layers: [
                tf.layers.gru({
                    units: 32,
                    returnSequences: true,
                    inputShape: this.inputShape,
                    dropout: 0.3,
                    recurrentDropout: 0.3,
                    kernelRegularizer: tf.regularizers.l2({l2: 0.01}),
                    name: 'gru_1'
                }),
                tf.layers.gru({
                    units: 32,
                    dropout: 0.3,
                    recurrentDropout: 0.3,
                    kernelRegularizer: tf.regularizers.l2({l2: 0.01}),
                    name: 'gru_2'
                }),
                tf.layers.dense({
                    units: 64,
                    activation: 'relu',
                    kernelRegularizer: tf.regularizers.l2({l2: 0.01}),
                    name: 'dense_1'
                }),
                tf.layers.dropout({rate: 0.5}),
                tf.layers.dense({
                    units: 30,
                    activation: 'sigmoid',
                    name: 'output'
                })
            ]
        });

        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['binaryAccuracy']
        });

        console.log('Model built successfully');
        return this.model;
    }

    async fit(X_train, y_train, X_test, y_test, epochs = 100, batchSize = 32) {
        if (!this.model) {
            this.buildModel();
        }

        this.bestValLoss = Infinity;
        this.bestWeights = null;
        this.history = { loss: [], val_loss: [], accuracy: [], val_accuracy: [] };
        
        const patience = 15;
        let patienceCounter = 0;

        console.log('Starting model training with early stopping...');
        
        for (let epoch = 0; epoch < epochs; epoch++) {
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

            if (valLoss < this.bestValLoss) {
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
                    earlyStopping: patienceCounter
                }
            });
            document.dispatchEvent(event);

            if ((epoch + 1) % 10 === 0) {
                console.log(`Epoch ${epoch + 1}/${epochs} - loss: ${loss.toFixed(4)} - accuracy: ${accuracy.toFixed(4)} - val_loss: ${valLoss.toFixed(4)} - val_accuracy: ${valAccuracy.toFixed(4)}`);
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
                    const actual = trueData[sampleIdx][predIdx];
                    
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
        const correct = binaryPreds.equal(y_true).sum();
        const total = tf.size(y_true);
        
        const accuracy = (await correct.data())[0] / (await total.data())[0] * 100;
        
        correct.dispose();
        total.dispose();
        binaryPreds.dispose();
        
        return accuracy;
    }

    getModelSummary() {
        if (!this.model) return 'Model not built';
        
        let summary = 'Model Architecture:\n';
        this.model.layers.forEach((layer, i) => {
            summary += `${i + 1}. ${layer.name} (${layer.getClassName()}) - Output: ${JSON.stringify(layer.outputShape)}\n`;
        });
        return summary;
    }

    async saveModel() {
        if (!this.model) {
            throw new Error('No model to save');
        }
        
        const saveResult = await this.model.save('downloads://music-popularity-model');
        console.log('Model saved successfully');
        return saveResult;
    }

    async loadModel(modelArtifacts) {
        this.model = await tf.loadLayersModel(modelArtifacts);
        console.log('Model loaded successfully');
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
