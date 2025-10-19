class DataLoader {
    constructor() {
        this.data = null;
        this.tracks = new Set();
        this.dates = new Set();
        this.X_train = null;
        this.y_train = null;
        this.X_test = null;
        this.y_test = null;
        this.trackMetadata = new Map();
        this.normalizationParams = new Map();
        this.selectedTracks = [];
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    this.parseCSV(e.target.result);
                    resolve(this.data);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = reject;
            reader.readAsText(file);
        });
    }

    parseCSV(csvText) {
        const lines = csvText.split('\n').filter(line => line.trim());
        const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
        
        const dateIdx = headers.findIndex(h => h.toLowerCase().includes('date'));
        const trackIdx = headers.findIndex(h => h.toLowerCase().includes('track'));
        const streamsIdx = headers.findIndex(h => h.toLowerCase().includes('stream'));
        const danceabilityIdx = headers.findIndex(h => h.toLowerCase().includes('danceability'));
        const energyIdx = headers.findIndex(h => h.toLowerCase().includes('energy'));
        const valenceIdx = headers.findIndex(h => h.toLowerCase().includes('valence'));
        const acousticnessIdx = headers.findIndex(h => h.toLowerCase().includes('acousticness'));

        this.data = [];
        for (let i = 1; i < lines.length; i++) {
            const values = this.parseCSVLine(lines[i]);
            if (values.length >= Math.max(dateIdx, trackIdx, streamsIdx, danceabilityIdx, energyIdx)) {
                const entry = {
                    date: values[dateIdx],
                    track_id: values[trackIdx],
                    streams: parseFloat(values[streamsIdx]) || 0,
                    danceability: parseFloat(values[danceabilityIdx]) || 0,
                    energy: parseFloat(values[energyIdx]) || 0,
                    valence: parseFloat(values[valenceIdx]) || 0,
                    acousticness: parseFloat(values[acousticnessIdx]) || 0
                };
                
                if (entry.track_id && entry.date) {
                    this.data.push(entry);
                    this.tracks.add(entry.track_id);
                    this.dates.add(entry.date);
                }
            }
        }

        this.selectTopTracks(10);
        this.engineerFeatures(); // NEW: Feature engineering
        return this.data;
    }

    parseCSVLine(line) {
        const result = [];
        let current = '';
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            if (char === '"') {
                inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
                result.push(current.trim().replace(/"/g, ''));
                current = '';
            } else {
                current += char;
            }
        }
        result.push(current.trim().replace(/"/g, ''));
        return result;
    }

    // NEW: Advanced feature engineering
    engineerFeatures() {
        // Calculate rolling statistics for each track
        this.selectedTracks.forEach(trackId => {
            const trackData = this.data.filter(d => d.track_id === trackId).sort((a, b) => a.date.localeCompare(b.date));
            
            // Add momentum features (rate of change)
            for (let i = 1; i < trackData.length; i++) {
                trackData[i].streams_momentum = trackData[i].streams - trackData[i-1].streams;
                trackData[i].streams_growth_rate = trackData[i-1].streams > 0 ? 
                    (trackData[i].streams - trackData[i-1].streams) / trackData[i-1].streams : 0;
            }
            
            // Add rolling averages
            for (let i = 2; i < trackData.length; i++) {
                trackData[i].streams_ma3 = (trackData[i].streams + trackData[i-1].streams + trackData[i-2].streams) / 3;
                trackData[i].volatility = Math.abs(trackData[i].streams_momentum / (trackData[i].streams_ma3 || 1));
            }
            
            // Fill missing values for first records
            trackData[0].streams_momentum = 0;
            trackData[0].streams_growth_rate = 0;
            trackData[0].streams_ma3 = trackData[0].streams;
            trackData[0].volatility = 0;
            
            if (trackData[1]) {
                trackData[1].streams_ma3 = (trackData[1].streams + trackData[0].streams) / 2;
                trackData[1].volatility = Math.abs(trackData[1].streams_momentum / (trackData[1].streams_ma3 || 1));
            }
        });
    }

    selectTopTracks(n) {
        const trackStreams = new Map();
        
        this.data.forEach(entry => {
            const current = trackStreams.get(entry.track_id) || 0;
            trackStreams.set(entry.track_id, current + entry.streams);
        });

        const sortedTracks = Array.from(trackStreams.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, n)
            .map(entry => entry[0]);

        this.selectedTracks = sortedTracks;
        this.data = this.data.filter(entry => this.selectedTracks.includes(entry.track_id));

        this.selectedTracks.forEach(trackId => {
            const trackData = this.data.find(d => d.track_id === trackId);
            if (trackData) {
                this.trackMetadata.set(trackId, {
                    id: trackId,
                    name: trackId,
                    totalStreams: trackStreams.get(trackId)
                });
            }
        });
    }

    normalizeFeatures() {
        this.normalizationParams = new Map();
        const sortedDates = Array.from(this.dates).sort();
        const splitIndex = Math.floor(sortedDates.length * 0.8);
        const trainingDates = new Set(sortedDates.slice(0, splitIndex));
        
        this.selectedTracks.forEach(trackId => {
            const trackData = this.data.filter(d => d.track_id === trackId && trainingDates.has(d.date));
            
            // NEW: More features to normalize
            const features = ['streams', 'danceability', 'energy', 'valence', 'acousticness', 
                             'streams_momentum', 'streams_growth_rate', 'streams_ma3', 'volatility'];
            const params = {};
            
            features.forEach(feature => {
                const values = trackData.map(d => d[feature]).filter(v => !isNaN(v));
                if (values.length > 0) {
                    const min = Math.min(...values);
                    const max = Math.max(...values);
                    params[feature] = { min, max };
                } else {
                    params[feature] = { min: 0, max: 1 };
                }
            });
            
            this.normalizationParams.set(trackId, params);
        });

        this.data.forEach(entry => {
            const params = this.normalizationParams.get(entry.track_id);
            if (params) {
                // Original features
                entry.streams_normalized = this.minMaxNormalize(entry.streams, params.streams);
                entry.danceability_normalized = this.minMaxNormalize(entry.danceability, params.danceability);
                entry.energy_normalized = this.minMaxNormalize(entry.energy, params.energy);
                entry.valence_normalized = this.minMaxNormalize(entry.valence, params.valence);
                entry.acousticness_normalized = this.minMaxNormalize(entry.acousticness, params.acousticness);
                
                // Engineered features
                entry.streams_momentum_normalized = this.minMaxNormalize(entry.streams_momentum || 0, params.streams_momentum);
                entry.streams_growth_rate_normalized = this.minMaxNormalize(entry.streams_growth_rate || 0, params.streams_growth_rate);
                entry.streams_ma3_normalized = this.minMaxNormalize(entry.streams_ma3 || entry.streams, params.streams_ma3);
                entry.volatility_normalized = this.minMaxNormalize(entry.volatility || 0, params.volatility);
            } else {
                // Fallback normalization
                const fallbackNormalize = (val) => (val - (-100)) / (100 - (-100)); // Rough estimate range
                entry.streams_normalized = fallbackNormalize(entry.streams);
                entry.danceability_normalized = entry.danceability;
                entry.energy_normalized = entry.energy;
                entry.valence_normalized = entry.valence;
                entry.acousticness_normalized = entry.acousticness;
                entry.streams_momentum_normalized = 0.5;
                entry.streams_growth_rate_normalized = 0.5;
                entry.streams_ma3_normalized = fallbackNormalize(entry.streams);
                entry.volatility_normalized = 0.5;
            }
        });
    }

    minMaxNormalize(value, params) {
        if (params && params.max > params.min) {
            return (value - params.min) / (params.max - params.min);
        }
        return 0.5;
    }

    createSlidingWindows() {
        this.normalizeFeatures();
        
        const sortedDates = Array.from(this.dates).sort();
        const samples = [];
        const targets = [];
        const windowSize = 7;

        // NEW: Data augmentation - create multiple variations
        for (let i = windowSize; i < sortedDates.length - 3; i++) {
            const currentDate = sortedDates[i];
            const windowDates = sortedDates.slice(i - windowSize, i);
            
            // Original sample
            const sample = this.createSample(windowDates);
            if (sample) {
                const target = this.createTarget(currentDate);
                if (target && target.length === this.selectedTracks.length * 3) {
                    samples.push(sample);
                    targets.push(target);
                    
                    // NEW: Data augmentation - add slightly noisy versions
                    for (let aug = 0; aug < 2; aug++) {
                        const augmentedSample = this.augmentSample(sample);
                        samples.push(augmentedSample);
                        targets.push(target);
                    }
                }
            }
        }

        this.splitData(samples, targets);
    }

    createSample(windowDates) {
        const sample = [];
        
        for (const date of windowDates) {
            const dayFeatures = [];
            
            for (const trackId of this.selectedTracks) {
                const entry = this.data.find(d => d.date === date && d.track_id === trackId);
                if (entry) {
                    // NEW: More features per track (9 instead of 3)
                    dayFeatures.push(
                        entry.streams_normalized || 0,
                        entry.danceability_normalized || 0,
                        entry.energy_normalized || 0,
                        entry.valence_normalized || 0,
                        entry.acousticness_normalized || 0,
                        entry.streams_momentum_normalized || 0,
                        entry.streams_growth_rate_normalized || 0,
                        entry.streams_ma3_normalized || 0,
                        entry.volatility_normalized || 0
                    );
                } else {
                    // If data missing, use zeros for all 9 features
                    dayFeatures.push(0, 0, 0, 0, 0, 0, 0, 0, 0);
                }
            }
            
            sample.push(dayFeatures);
        }

        return sample.length === windowDates.length ? sample : null;
    }

    // NEW: Data augmentation with noise
    augmentSample(sample) {
        return sample.map(day => 
            day.map(feature => {
                // Add small random noise (5% of value)
                const noise = (Math.random() - 0.5) * 0.1;
                return Math.max(0, Math.min(1, feature + noise));
            })
        );
    }

    createTarget(currentDate) {
        const target = [];
        const sortedDates = Array.from(this.dates).sort();
        const currentDateIndex = sortedDates.indexOf(currentDate);
        
        for (const trackId of this.selectedTracks) {
            const currentEntry = this.data.find(d => d.date === currentDate && d.track_id === trackId);
            if (!currentEntry) return null;

            const currentStreams = currentEntry.streams;
            
            for (let offset = 1; offset <= 3; offset++) {
                const futureDate = sortedDates[currentDateIndex + offset];
                const futureEntry = this.data.find(d => d.date === futureDate && d.track_id === trackId);
                
                if (futureEntry) {
                    // NEW: Use probability instead of binary for smoother learning
                    const increaseRatio = (futureEntry.streams - currentStreams) / (currentStreams || 1);
                    const probability = 1 / (1 + Math.exp(-increaseRatio * 10)); // Sigmoid to get probability
                    target.push(probability);
                } else {
                    return null;
                }
            }
        }

        return target;
    }

    splitData(samples, targets, trainRatio = 0.8) {
        const splitIndex = Math.floor(samples.length * trainRatio);
        
        console.log(`Total samples: ${samples.length}`);
        console.log(`Training samples: ${splitIndex}`);
        console.log(`Test samples: ${samples.length - splitIndex}`);
        console.log(`Features per track: 9, Total features: ${9 * this.selectedTracks.length}`);
        
        this.X_train = tf.tensor3d(samples.slice(0, splitIndex));
        this.y_train = tf.tensor2d(targets.slice(0, splitIndex));
        this.X_test = tf.tensor3d(samples.slice(splitIndex));
        this.y_test = tf.tensor2d(targets.slice(splitIndex));
        
        this.logDataStatistics();
    }

    logDataStatistics() {
        if (this.X_train && this.y_train) {
            const trainMean = this.y_train.mean().dataSync()[0];
            console.log(`Training set - Average target probability: ${trainMean.toFixed(4)}`);
        }
    }

    getTrainingData() {
        return {
            X_train: this.X_train,
            y_train: this.y_train,
            X_test: this.X_test,
            y_test: this.y_test,
            trackMetadata: this.trackMetadata,
            selectedTracks: this.selectedTracks
        };
    }

    validateData() {
        const data = this.getTrainingData();
        
        const hasNaN = (tensor) => {
            if (!tensor) return true;
            const nanCheck = tf.isNaN(tensor).any();
            return nanCheck.dataSync()[0];
        };
        
        console.log("Train has NaN:", hasNaN(data.X_train));
        console.log("Test has NaN:", hasNaN(data.X_test));
        
        return !hasNaN(data.X_train) && !hasNaN(data.X_test);
    }

    dispose() {
        if (this.X_train) this.X_train.dispose();
        if (this.y_train) this.y_train.dispose();
        if (this.X_test) this.X_test.dispose();
        if (this.y_test) this.y_test.dispose();
    }
}
