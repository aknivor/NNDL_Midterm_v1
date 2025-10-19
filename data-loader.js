class DataLoader {
    constructor() {
        this.data = null;
        this.tracks = new Set();
        this.dates = new Set();
        this.processedData = null;
        this.X_train = null;
        this.y_train = null;
        this.X_test = null;
        this.y_test = null;
        this.trackMetadata = new Map();
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
        
        // Find column indices
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

        // Select top 10 tracks by total streams
        this.selectTopTracks(10);
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

    selectTopTracks(n) {
        const trackStreams = new Map();
        
        this.data.forEach(entry => {
            const current = trackStreams.get(entry.track_id) || 0;
            trackStreams.set(entry.track_id, current + entry.streams);
        });

        // Convert to array and sort
        const sortedTracks = Array.from(trackStreams.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, n)
            .map(entry => entry[0]);

        this.selectedTracks = sortedTracks;
        
        // Filter data to only include selected tracks
        this.data = this.data.filter(entry => 
            this.selectedTracks.includes(entry.track_id)
        );

        // Store track metadata
        this.selectedTracks.forEach(trackId => {
            const trackData = this.data.find(d => d.track_id === trackId);
            if (trackData) {
                this.trackMetadata.set(trackId, {
                    id: trackId,
                    name: trackId, // In real implementation, extract track name
                    totalStreams: trackStreams.get(trackId)
                });
            }
        });
    }

    normalizeFeatures() {
        // MinMax normalization per track
        this.selectedTracks.forEach(trackId => {
            const trackData = this.data.filter(d => d.track_id === trackId);
            
            const features = ['streams', 'danceability', 'energy'];
            features.forEach(feature => {
                const values = trackData.map(d => d[feature]).filter(v => !isNaN(v));
                const min = Math.min(...values);
                const max = Math.max(...values);
                
                if (max > min) {
                    trackData.forEach(entry => {
                        entry[`${feature}_normalized`] = (entry[feature] - min) / (max - min);
                    });
                } else {
                    trackData.forEach(entry => {
                        entry[`${feature}_normalized`] = 0.5;
                    });
                }
            });
        });
    }

    createSlidingWindows() {
        this.normalizeFeatures();
        
        const sortedDates = Array.from(this.dates).sort();
        const samples = [];
        const targets = [];
        const windowSize = 7;

        for (let i = windowSize; i < sortedDates.length - 3; i++) {
            const currentDate = sortedDates[i];
            const windowDates = sortedDates.slice(i - windowSize, i);
            
            // Create input sample: [7 days × 30 features]
            const sample = this.createSample(windowDates, currentDate);
            if (sample) {
                const target = this.createTarget(currentDate);
                if (target) {
                    samples.push(sample);
                    targets.push(target);
                }
            }
        }

        this.splitData(samples, targets);
    }

    createSample(windowDates, currentDate) {
        const sample = [];
        
        for (const date of windowDates) {
            const dayFeatures = [];
            
            for (const trackId of this.selectedTracks) {
                const entry = this.data.find(d => d.date === date && d.track_id === trackId);
                if (entry) {
                    dayFeatures.push(
                        entry.streams_normalized || 0,
                        entry.danceability_normalized || 0,
                        entry.energy_normalized || 0
                    );
                } else {
                    // If data missing, use zeros
                    dayFeatures.push(0, 0, 0);
                }
            }
            
            if (dayFeatures.length === this.selectedTracks.length * 3) {
                sample.push(dayFeatures);
            }
        }

        return sample.length === windowDates.length ? sample : null;
    }

    createTarget(currentDate) {
        const target = [];
        const currentDateIndex = Array.from(this.dates).sort().indexOf(currentDate);
        const sortedDates = Array.from(this.dates).sort();
        
        for (const trackId of this.selectedTracks) {
            const currentEntry = this.data.find(d => d.date === currentDate && d.track_id === trackId);
            if (!currentEntry) {
                target.push(0, 0, 0);
                continue;
            }

            const currentStreams = currentEntry.streams;
            
            for (let offset = 1; offset <= 3; offset++) {
                const futureDate = sortedDates[currentDateIndex + offset];
                const futureEntry = this.data.find(d => d.date === futureDate && d.track_id === trackId);
                
                if (futureEntry && futureEntry.streams > currentStreams) {
                    target.push(1);
                } else {
                    target.push(0);
                }
            }
        }

        return target.length === this.selectedTracks.length * 3 ? target : null;
    }

    splitData(samples, targets, trainRatio = 0.8) {
        const splitIndex = Math.floor(samples.length * trainRatio);
        
        this.X_train = tf.tensor3d(samples.slice(0, splitIndex));
        this.y_train = tf.tensor2d(targets.slice(0, splitIndex));
        this.X_test = tf.tensor3d(samples.slice(splitIndex));
        this.y_test = tf.tensor2d(targets.slice(splitIndex));
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

    dispose() {
        if (this.X_train) this.X_train.dispose();
        if (this.y_train) this.y_train.dispose();
        if (this.X_test) this.X_test.dispose();
        if (this.y_test) this.y_test.dispose();
    }
}
