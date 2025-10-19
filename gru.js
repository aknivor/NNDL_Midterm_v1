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
    }

    buildModel() {
        this.model = tf.sequential({
            layers: [
                tf.layers.gru({
                    units: 64,
                    returnSequences: true,
                    inputShape: this.inputShape,
                    dropout: 0.2,
                    recurrentDropout: 0.2
                }),
                tf.layers.gru({
                    units: 64,
                    dropout: 0.2,
                    recurrentDropout: 0.2
                }),
                tf.layers.dense({
                    units: 30,
                    activation: 'sigmoid'
                })
            ]
        });

        this.model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['binaryAccuracy']
        });

        return this.model;
    }

    async fit(X_train, y_train, X_test, y_test, epochs = 100, batchSize = 32) {
        if (!this.model) {
            this.buildModel();
        }

        console.log('Starting model training...');
        
        const history = await this.model.fit(X_train, y_train, {
            epochs: epochs,
            batchSize: batchSize,
            validationData: [X_test, y_test],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    this.history.loss.push(logs.loss);
                    this.history.val_loss.push(logs.val_loss);
                    this.history.accuracy.push(logs.binaryAccuracy);
                    this.history.val_accuracy.push(logs.val_binaryAccuracy);
                    
                    // Dispatch event for UI updates
                    const event = new CustomEvent('trainingProgress', {
                        detail: {
                            epoch: epoch + 1,
                            loss: logs.loss,
                            accuracy: logs.binaryAccuracy,
                            val_loss: logs.val_loss,
                            val_accuracy: logs.val_binaryAccuracy
                        }
                    });
                    document.dispatchEvent(event);
                }
            }
        });

        return history;
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
        
        // Cleanup
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
        
        tracks.forEach((trackId, trackIndex) => {
            let correct = 0;
            let total = 0;
            
            for (let i = 0; i < predData.length; i++) {
                for (let dayOffset = 0; dayOffset < 3; dayOffset++) {
                    const predIdx = trackIndex * 3 + dayOffset;
                    const prediction = predData[i][predIdx] > 0.5 ? 1 : 0;
                    const actual = trueData[i][predIdx];
                    
                    if (actual !== undefined) {
                        total++;
                        if (prediction === actual) {
                            correct++;
                        }
                    }
                }
            }
            
            const accuracy = total > 0 ? (correct / total) * 100 : 0;
            trackAccuracies.set(trackId, {
                accuracy: accuracy,
                trackName: trackMetadata.get(trackId).name || trackId
            });
        });
        
        return trackAccuracies;
    }

    async saveModel() {
        if (!this.model) {
            throw new Error('No model to save');
        }
        
        const saveResult = await this.model.save('downloads://music-popularity-model');
        return saveResult;
    }

    async loadModel(modelArtifacts) {
        this.model = await tf.loadLayersModel(modelArtifacts);
        return this.model;
    }

    getModelSummary() {
        if (!this.model) return '';
        
        let summary = '';
        this.model.layers.forEach(layer => {
            summary += `${layer.name} (${layer.getClassName()}) - Output: ${layer.outputShape}\n`;
        });
        return summary;
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}
