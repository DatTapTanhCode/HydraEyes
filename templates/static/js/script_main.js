// INITIALIZATION AND EVENT HANDLERS
// ================================

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Page loaded, initializing...');
    
    // Set up navigation event listeners
    document.querySelectorAll('.nav-item').forEach(item => {
        console.log('Setting up navigation for:', item.getAttribute('data-page'));
        item.addEventListener('click', function() {
            switchToPage(this.getAttribute('data-page'));
        });
    });
    
    // Enhanced keyboard support
    document.addEventListener('keydown', function(e) {
        const calibrationModalActive = document.getElementById('analogCalibrationModal')?.classList.contains('active');
        const valueModalActive = document.getElementById('valueInputModal')?.classList.contains('active');
        
        if (!calibrationModalActive && !valueModalActive) return;
        
        // ESC key to close modals
        if (e.key === 'Escape') {
            e.preventDefault();
            if (valueModalActive) {
                hideValueInputModal();
            } else if (calibrationModalActive) {
                hideAnalogCalibrationModal();
            }
        }
        
        // Enter key in value input
        if (e.key === 'Enter' && valueModalActive) {
            e.preventDefault();
            submitGaugeValue();
        }
        
        // Ctrl+Z for undo (when calibration modal is active but not value input)
        if (e.key === 'z' && e.ctrlKey && calibrationModalActive && !valueModalActive) {
            e.preventDefault();
            undoLastPoint();
        }
        
        // Delete key for reset (with confirmation)
        if (e.key === 'Delete' && e.ctrlKey && calibrationModalActive && !valueModalActive) {
            e.preventDefault();
            resetAllPoints();
        }
    });
    
    // Demo mode toggle handlers
    const demoModeCheckbox = document.getElementById('demoMode');
    const addressGroup = document.getElementById('addressGroup');
    const cameraAddressInput = document.getElementById('cameraAddress');
    
    const configDemoModeCheckbox = document.getElementById('configDemoMode');
    const configAddressGroup = document.getElementById('configAddressGroup');
    const configCameraAddressInput = document.getElementById('configCameraAddress');
    
    console.log('Demo mode elements:', {
        demoModeCheckbox,
        addressGroup,
        cameraAddressInput
    });
    
    // Add camera form demo mode toggle
    if (demoModeCheckbox && addressGroup) {
        demoModeCheckbox.addEventListener('change', function() {
            console.log('Demo mode checkbox changed:', this.checked);
            if (this.checked) {
                addressGroup.style.opacity = '0.5';
                cameraAddressInput.required = false;
                cameraAddressInput.placeholder = 'Not required in demo mode';
            } else {
                addressGroup.style.opacity = '1';
                cameraAddressInput.required = true;
                cameraAddressInput.placeholder = 'Enter camera IP or URL (e.g., 192.168.1.100 or http://camera-url)';
            }
        });
    }
    
    // Config form demo mode toggle
    if (configDemoModeCheckbox && configAddressGroup) {
        configDemoModeCheckbox.addEventListener('change', function() {
            if (this.checked) {
                configAddressGroup.style.opacity = '0.5';
                configCameraAddressInput.required = false;
                configCameraAddressInput.placeholder = 'Not required in demo mode';
            } else {
                configAddressGroup.style.opacity = '1';
                configCameraAddressInput.required = true;
                configCameraAddressInput.placeholder = 'Enter camera IP or URL (e.g., 192.168.1.100 or http://camera-url)';
            }
        });
    }
    
    // Add event listener for detector type change
    const detectorTypeSelect = document.getElementById('detectorType');
    if (detectorTypeSelect) {
        detectorTypeSelect.addEventListener('change', handleDetectorTypeChange);
    }
    
    // Config form detector type
    const configDetectorTypeSelect = document.getElementById('configDetectorType');
    if (configDetectorTypeSelect) {
        configDetectorTypeSelect.addEventListener('change', function() {
            const detectorType = this.value;
            // Add any config-specific handling here
        });
    }
    
    // Handle Enter key in value input
    const gaugeValueInput = document.getElementById('gaugeValueInput');
    if (gaugeValueInput) {
        gaugeValueInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                submitGaugeValue();
            }
        });
    }

    // Form submission handlers
    // Camera Form
    document.getElementById('cameraForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const address = document.getElementById('cameraAddress').value.trim();
        const type = document.getElementById('gaugeType').value;
        const detectorType = document.getElementById('detectorType').value;
        const location = document.getElementById('cameraLocation').value.trim();
        const threshold = parseFloat(document.getElementById('thresholdValue').value);
        const demoMode = document.getElementById('demoMode').checked;
        
        console.log('Form submission data:', {
            address, type, detectorType, location, threshold, demoMode
        });
        
        if (!type || !detectorType || !location || isNaN(threshold)) {
            alert('Please fill in all required fields (Detector Type, Gauge Type, Location, and Threshold)');
            return;
        }
        
        if (!demoMode && !address) {
            alert('Please provide camera address for detector mode');
            return;
        }
        
        const newCameraId = `CAM-${String(cameras.length + 1).padStart(3, '0')}`;
        
        try {
            const requestData = {
                camera_id: newCameraId,
                address: demoMode ? '' : address,
                demo_mode: demoMode,
                gauge_type: type,
                detector_type: detectorType,
                location: location,
                threshold: threshold
            };
            
            console.log('Sending request data:', requestData);
            
            const response = await apiCall('/api/add-camera', 'POST', requestData);
            console.log('API Response:', response);
            
            const newCamera = {
                id: newCameraId,
                address: demoMode ? '' : address,
                demo_mode: demoMode,
                type: type,
                detector_type: detectorType,
                location: location,
                threshold: threshold
            };
            
            cameras.push(newCamera);
            
            // FIXED: Initialize readings properly based on mode and detector type
            if (demoMode) {
                // Demo mode: generate random reading immediately
                gaugeReadings[newCamera.id] = generateRandomReading(newCamera);
                console.log(`Demo camera ${newCamera.id} initialized with reading: ${gaugeReadings[newCamera.id]}`);
            } else {
                // Real detector mode: initialize with 0, will be updated by backend
                gaugeReadings[newCamera.id] = 0;
                console.log(`Real camera ${newCamera.id} initialized with reading: 0 (awaiting detector)`);
            }
            
            // Initialize chart data for all cameras
            chartData[newCamera.id] = [];
            
            const modeText = demoMode ? '(demo mode)' : `(${detectorType} detector)`;
            addActivityLog(`New ${detectorType} camera ${newCamera.id} added at ${location} (${type}) ${modeText}`, 'success');
            
            renderCameras();
            createChartsGrid();
            hideAddCameraModal();
            
            console.log('Camera added successfully:', newCamera);
            
        } catch (error) {
            console.error('API Error:', error);
            alert('Error adding camera: ' + error.message);
        }
    });

    // Config Form
    document.getElementById('configForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (currentConfigCamera === null) return;
    
        const address = document.getElementById('configCameraAddress').value.trim();
        const type = document.getElementById('configGaugeType').value;
        const detectorType = document.getElementById('configDetectorType').value;
        const location = document.getElementById('configCameraLocation').value.trim();
        const threshold = parseFloat(document.getElementById('configThresholdValue').value);
        const demoMode = document.getElementById('configDemoMode').checked;
        
        if (!type || !detectorType || !location || !threshold) {
            alert('Please fill in all required fields');
            return;
        }
        
        if (!demoMode && !address) {
            alert('Please provide camera address for detector mode');
            return;
        }
        
        const oldCamera = { ...cameras[currentConfigCamera] };
        const cameraId = cameras[currentConfigCamera].id;
        
        try {
            await apiCall('/api/update-camera', 'POST', {
                camera_id: cameraId,
                address: address,
                demo_mode: demoMode,
                gauge_type: type,
                detector_type: detectorType,
                location: location,
                threshold: threshold
            });
            
            cameras[currentConfigCamera].address = demoMode ? '' : address;
            cameras[currentConfigCamera].demo_mode = demoMode;
            cameras[currentConfigCamera].type = type;
            cameras[currentConfigCamera].detector_type = detectorType;
            cameras[currentConfigCamera].location = location;
            cameras[currentConfigCamera].threshold = threshold;
            
            const changes = [];
            if (oldCamera.address !== address) changes.push('address');
            if (oldCamera.demo_mode !== demoMode) changes.push('mode');
            if (oldCamera.type !== type) changes.push('gauge type');
            if (oldCamera.detector_type !== detectorType) changes.push('detector type');
            if (oldCamera.location !== location) changes.push('location');
            if (oldCamera.threshold !== threshold) changes.push('threshold');
            
            if (changes.length > 0) {
                const modeText = demoMode ? 'demo mode' : 'detector mode';
                addActivityLog(`Camera ${cameraId} updated: ${changes.join(', ')} changed (now ${detectorType} in ${modeText})`, 'activity');
            }
            
            renderCameras();
            createChartsGrid();
            hideConfigModal();
            
        } catch (error) {
            alert('Error updating camera: ' + error.message);
        }
    });

    // Team Form
    document.getElementById('teamForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const name = document.getElementById('teamName').value.trim();
        const email = document.getElementById('teamEmail').value.trim();
        const contact = document.getElementById('teamContact').value.trim();
        
        if (!name || !email || !contact) {
            alert('Please fill in all required fields');
            return;
        }
        
        const selectedGauges = [];
        document.querySelectorAll('#gaugeCheckboxes input[type="checkbox"]:checked').forEach(checkbox => {
            selectedGauges.push(checkbox.value);
        });
        
        const newMember = {
            id: `TEAM-${String(teamMembers.length + 1).padStart(3, '0')}`,
            name: name,
            email: email,
            contact: contact,
            assignedGauges: selectedGauges
        };
        
        teamMembers.push(newMember);
        
        const gaugeText = selectedGauges.length > 0 ? 
            ` and assigned to ${selectedGauges.join(', ')}` : '';
        addActivityLog(`New team member ${name} added${gaugeText}`, 'success');
        
        renderTeams();
        hideAddTeamModal();
    });

    // Close modals when clicking overlay
    document.getElementById('cameraModal').addEventListener('click', function(e) {
        if (e.target === this) {
            hideAddCameraModal();
        }
    });

    document.getElementById('configModal').addEventListener('click', function(e) {
        if (e.target === this) {
            hideConfigModal();
        }
    });

    document.getElementById('teamModal').addEventListener('click', function(e) {
        if (e.target === this) {
            hideAddTeamModal();
        }
    });

    // Initialize with some sample data for demonstration
    addActivityLog('System initialized and ready', 'success');
    
    // Initialize empty cameras array and render immediately
    renderCameras();
    renderTeams();
    renderActivityLog();
    renderAlertLog();
    updateDashboardStats();
    createChartsGrid();
    
    // Fetch existing camera configurations after initial render
    fetchCameraConfigs();
    
    // Start live data updates
    const updateInterval = setInterval(() => {
        console.log('Running data updates...', {
            camerasCount: cameras.length,
            demoCameras: cameras.filter(c => c.demo_mode).length,
            onlineCameras: cameras.filter(c => c.address).length,
            chartDataKeys: Object.keys(chartData),
            gaugeReadingsKeys: Object.keys(gaugeReadings)
        });
        
        updateGaugeReadings();
        
        fetchCameraReadings().catch(err => {
            console.log('API fetch failed, using demo mode:', err.message);
        });
    }, 1000);
    
    // Check for alerts every 5 seconds
    setInterval(checkAlerts, 5000);
    
    // AI predictions
    const aiToggle = document.getElementById('ai-predictions-toggle');
    if (aiToggle) {
        aiToggle.addEventListener('change', function() {
            if (this.checked) {
                setInterval(generateAIPredictions, 5000);
                generateAIPredictions();
            }
        });
        
        if (aiToggle.checked) {
            setInterval(generateAIPredictions, 5000);
            setTimeout(generateAIPredictions, 3000);
        }
    }
    
    // Handle window resize for charts with debouncing
    let resizeTimeout;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            console.log('Window resized, recreating charts...');
            
            Object.keys(individualCharts).forEach(cameraId => {
                const canvas = document.getElementById(`chart-${cameraId}`);
                if (canvas) {
                    const container = canvas.parentElement;
                    const containerWidth = container.offsetWidth || 400;
                    
                    canvas.width = containerWidth;
                    canvas.height = 180;
                    canvas.style.width = containerWidth + 'px';
                    canvas.style.height = '180px';
                    
                    if (individualCharts[cameraId]) {
                        individualCharts[cameraId].width = containerWidth;
                        individualCharts[cameraId].height = 180;
                    }
                }
            });
            
            updateAllCharts();
        }, 300);
    });
});
// GLOBAL VARIABLES AND CONFIGURATION
// ================================

// Data storage
let cameras = [];
let teamMembers = [];
let activityLog = [];
let alertLog = [];
let currentConfigCamera = null;
let gaugeReadings = {};
let alertCheckInterval;
let chartData = {};
let individualCharts = {};
let thresholdStates = {};
let detectorEnabled = true;

// Analog calibration variables
let currentAnalogCamera = null;
let analogCalibrationPoints = [];
let calibrationHistory = [];
let pendingCalibrationClick = null;
let calibrationPointsRequired = 4;
let analogStatusInterval = null;

// Gauge type configurations
const gaugeConfigs = {
    'pressure': { unit: 'PSI', min: 0, max: 100, normalMax: 50 },
    'temperature': { unit: '°C', min: 0, max: 200, normalMax: 80 },
    'flow': { unit: 'L/min', min: 0, max: 500, normalMax: 400 },
    'level': { unit: '%', min: 0, max: 100, normalMax: 85 },
    'digital': { unit: '', min: 0, max: 9999, normalMax: 8000 },
    'analog': { unit: 'units', min: 0, max: 100, normalMax: 80 }
};

// ================================
// NAVIGATION AND PAGE MANAGEMENT
// ================================

function switchToPage(page) {
    console.log('Switching to page:', page);
    
    // Remove active class from all nav items
    document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
    // Add active class to clicked item
    const navItem = document.querySelector(`[data-page="${page}"]`);
    if (navItem) {
        navItem.classList.add('active');
    }
    
    // Hide all content sections
    document.querySelectorAll('.content').forEach(content => {
        content.classList.add('hidden');
    });
    
    // Show selected content
    const contentElement = document.getElementById(page + '-content');
    if (contentElement) {
        contentElement.classList.remove('hidden');
        console.log('Switched to page:', page);
    } else {
        console.error('Content element not found for page:', page);
    }
}

function logout() {
    console.log('Logout clicked');
    if (confirm('Are you sure you want to logout?')) {
        window.location.href = '/logout';
    }
}

// ================================
// API FUNCTIONS
// ================================

async function apiCall(endpoint, method = 'GET', data = null) {
    try {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            }
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(endpoint, options);
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'API call failed');
        }
        
        return result;
    } catch (error) {
        console.error('API call error:', error);
        throw error;
    }
}

async function fetchCameraReadings() {
    try {
        const readings = await apiCall('/api/camera-readings');
        
        Object.keys(readings).forEach(cameraId => {
            const camera = cameras.find(c => c.id === cameraId);
            
            // Skip demo mode cameras - don't overwrite their values!
            if (camera && camera.demo_mode) {
                console.log(`Skipping backend reading for demo camera ${cameraId}`);
                return;
            }
            
            const readingData = readings[cameraId];
            const previousValue = gaugeReadings[cameraId];
            
            // Update the reading (this will replace the initial 0)
            gaugeReadings[cameraId] = readingData.value;
            
            console.log(`Backend reading for ${cameraId}: ${readingData.value} (was: ${previousValue})`);
            
            // Add to chart data only if we have a valid reading
            if (readingData.value !== null && readingData.value !== undefined) {
                const now = Date.now();
                if (!chartData[cameraId]) {
                    chartData[cameraId] = [];
                }
                
                chartData[cameraId].push({
                    timestamp: now,
                    value: readingData.value
                });
                
                // Keep only last 30 seconds
                const cutoffTime = now - 30000;
                chartData[cameraId] = chartData[cameraId].filter(point => point.timestamp > cutoffTime);
            }
            
            // Update display for non-demo cameras
            updateCameraDisplay(cameraId, readingData.value);
        });
        
        updateDashboardStats();
        updateAllCharts();
        
    } catch (error) {
        console.error('Error fetching camera readings:', error);
    }
}

async function fetchCameraConfigs() {
    try {
        console.log('Fetching camera configurations...');
        const configs = await apiCall('/api/camera-configs');
        console.log('Received configs:', configs);
        
        cameras = Object.keys(configs).map(cameraId => ({
            id: cameraId,
            address: configs[cameraId].address,
            demo_mode: configs[cameraId].demo_mode,
            type: configs[cameraId].gauge_type,
            detector_type: configs[cameraId].detector_type || 'digital',
            location: configs[cameraId].location,
            threshold: configs[cameraId].threshold
        }));
        
        console.log('Converted cameras array:', cameras);
        
        // FIXED: Initialize readings for existing cameras
        cameras.forEach(camera => {
            if (camera.demo_mode) {
                // Demo cameras get random readings
                gaugeReadings[camera.id] = generateRandomReading(camera);
                console.log(`Existing demo camera ${camera.id} initialized with reading: ${gaugeReadings[camera.id]}`);
            } else {
                // Real detector cameras start with 0
                gaugeReadings[camera.id] = 0;
                console.log(`Existing real camera ${camera.id} initialized with reading: 0`);
            }
            
            // Initialize chart data
            if (!chartData[camera.id]) {
                chartData[camera.id] = [];
            }
        });
        
        renderCameras();
        createChartsGrid();
        
    } catch (error) {
        console.error('Error fetching camera configs:', error);
    }
}

// ================================
// CAMERA MANAGEMENT FUNCTIONS
// ================================

function getTypeIcon(type) {
    const icons = {
        'pressure': '⏲️',
        'temperature': '🌡️',
        'flow': '💧',
        'level': '📏',
        'digital': '🔢',
        'analog': '⏰'
    };
    return icons[type] || '❓';
}

function generateRandomReading(camera) {
    const config = gaugeConfigs[camera.type];
    if (!config) return 0;

    const currentReading = gaugeReadings[camera.id] || (config.max * 0.7);
    const threshold = camera.threshold || config.normalMax;
    
    if (!thresholdStates[camera.id]) {
        thresholdStates[camera.id] = {
            isOverThreshold: false,
            overThresholdStartTime: null,
            maxOverThresholdDuration: 8000,
            trend: 'stable'
        };
    }
    
    const state = thresholdStates[camera.id];
    const now = Date.now();
    
    let newReading;
    
    if (state.isOverThreshold && state.overThresholdStartTime && 
        (now - state.overThresholdStartTime) >= state.maxOverThresholdDuration) {
        
        const targetReading = threshold * 0.95;
        const difference = currentReading - targetReading;
        newReading = currentReading - (difference * 0.2);
        
        if (Math.abs(newReading - targetReading) < threshold * 0.02) {
            newReading = targetReading;
            state.isOverThreshold = false;
            state.overThresholdStartTime = null;
            state.trend = 'stable';
        }
    } else {
        const shouldBreachThreshold = Math.random() < 0.03 && !state.isOverThreshold;
        
        if (shouldBreachThreshold) {
            const overAmount = threshold * (0.15 + Math.random() * 0.1);
            newReading = threshold + overAmount;
            state.isOverThreshold = true;
            state.overThresholdStartTime = now;
            state.trend = 'rising';
        } else {
            let changeAmount;
            
            switch (state.trend) {
                case 'rising':
                    changeAmount = currentReading * (0.005 + Math.random() * 0.01);
                    if (Math.random() < 0.1) state.trend = 'stable';
                    break;
                case 'falling':
                    changeAmount = -currentReading * (0.005 + Math.random() * 0.01);
                    if (Math.random() < 0.1) state.trend = 'stable';
                    break;
                default:
                    changeAmount = currentReading * (Math.random() - 0.5) * 0.02;
                    if (Math.random() < 0.05) {
                        state.trend = Math.random() < 0.5 ? 'rising' : 'falling';
                    }
            }
            
            newReading = currentReading + changeAmount;
        }
    }
    
    newReading = Math.max(config.min, Math.min(config.max, newReading));
    
    if (newReading <= threshold && state.isOverThreshold) {
        state.isOverThreshold = false;
        state.overThresholdStartTime = null;
        state.trend = 'falling';
    }
    
    return Math.round(newReading * 10) / 10;
}

function getReadingStatus(value, camera) {
    const config = gaugeConfigs[camera.type];
    const threshold = camera.threshold || config.normalMax;
    
    if (value > threshold * 1.2) return 'danger';
    if (value > threshold) return 'warning';
    return 'normal';
}

function generateFeedContent(camera) {
    if (!camera.address && !camera.demo_mode) {
        return `
            <div class="gauge-placeholder">
                <div style="font-size: 24px; margin-bottom: 8px;">${getTypeIcon(camera.type)}</div>
                <div>No feed configured</div>
            </div>
        `;
    }
    
    const currentReading = gaugeReadings[camera.id];
    const config = gaugeConfigs[camera.type];
    const status = getReadingStatus(currentReading, camera);
    
    const isAnalog = camera.detector_type === 'analog';
    const needsCalibration = isAnalog && camera.address && !camera.demo_mode;
    
    let content = `
        <div class="live-reading" id="reading-${camera.id}">${currentReading || 0}</div>
        <div class="reading-unit">${config.unit}</div>
        <div class="reading-status status-${status}-reading" id="status-${camera.id}">
            ${status.toUpperCase()}
        </div>
    `;
    
    if (needsCalibration) {
        content += `
            <div style="margin-top: 8px;">
                <button class="calibrate-btn" onclick="showAnalogCalibrationModal('${camera.id}')" title="Calibrate Analog Gauge">
                    📐 CALIBRATE
                </button>
            </div>
        `;
    }
    
    return content;
}

function renderCameras() {
    const container = document.getElementById('cameraFeeds');
    
    container.innerHTML = '';
    
    if (cameras.length === 0) {
        container.innerHTML = `
            <div class="empty-cameras-message">
                <span class="icon">📹</span>
                <div><strong>No cameras configured</strong></div>
                <div>Click "Add Camera" button to add your first camera</div>
            </div>
        `;
    } else {
        const camerasHTML = cameras.map((camera, index) => {
            const typeIcon = getTypeIcon(camera.type);
            const detectorType = camera.detector_type || 'digital';
            const modeStatus = camera.demo_mode ? 'DEMO' : (camera.address ? 'ONLINE' : 'OFFLINE');
            const modeClass = camera.demo_mode ? 'status-demo' : (camera.address ? 'status-online' : 'status-warning');
            
            const detectorIndicator = detectorType === 'analog' ? '⚙️ ANALOG' : '🔢 DIGITAL';
            
            console.log(`Rendering camera ${camera.id} with detector type: ${detectorType}`);
            
            return `
                <div class="camera-feed" data-detector="${detectorType}">
                    <div class="feed-header">
                        <span>${typeIcon} ${camera.id}</span>
                        <div style="display: flex; gap: 8px; align-items: center;">
                            <span class="feed-status status-${detectorType}">${detectorIndicator}</span>
                            <span class="feed-status ${modeClass}">${modeStatus}</span>
                            <button class="remove-btn" onclick="removeCamera(${index})" title="Remove Camera">🗑️</button>
                        </div>
                    </div>
                    <div class="feed-view">
                        <button class="config-btn" onclick="showConfigModal(${index})">⚙️ CONFIG</button>
                        ${detectorType === 'analog' && camera.address && !camera.demo_mode ? 
                            `<button class="calibrate-btn" onclick="showAnalogCalibrationModal('${camera.id}')" title="Calibrate Analog Gauge">📐 CAL</button>` : ''}
                        <div class="gauge-view">
                            ${generateFeedContent(camera)}
                        </div>
                    </div>
                    <div class="feed-info">
                        <div class="info-item">
                            <span>Status:</span>
                            <span class="reading-value">${modeStatus}</span>
                        </div>
                        <div class="info-item">
                            <span>Mode:</span>
                            <span class="reading-value">${camera.demo_mode ? 'DEMO' : 'DETECTOR'}</span>
                        </div>
                        <div class="info-item">
                            <span>Detector:</span>
                            <span class="reading-value">${detectorType.toUpperCase()}</span>
                        </div>
                        <div class="info-item">
                            <span>Type:</span>
                            <span class="reading-value">${camera.type.toUpperCase()}</span>
                        </div>
                        <div class="info-item">
                            <span>Location:</span>
                            <span class="reading-value">${camera.location}</span>
                        </div>
                        <div class="info-item">
                            <span>Threshold:</span>
                            <span class="reading-value">${camera.threshold || 'Not set'}</span>
                        </div>
                        <div class="info-item">
                            <span>Address:</span>
                            <span class="reading-value">${camera.address || 'N/A'}</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        container.innerHTML = camerasHTML;
    }
    
    updateDashboardStats();
}

async function removeCamera(index) {
    if (confirm('Are you sure you want to remove this camera? This will also remove it from all team member assignments.')) {
        const removedCamera = cameras[index];
        
        try {
            await apiCall('/api/remove-camera', 'POST', { camera_id: removedCamera.id });
            
            addActivityLog(`Camera ${removedCamera.id} removed from ${removedCamera.location}`, 'warning');
            
            cameras.splice(index, 1);
            delete gaugeReadings[removedCamera.id];
            delete chartData[removedCamera.id];
            delete thresholdStates[removedCamera.id];
            delete individualCharts[removedCamera.id];
            
            teamMembers.forEach(member => {
                if (member.assignedGauges) {
                    member.assignedGauges = member.assignedGauges.filter(gaugeId => gaugeId !== removedCamera.id);
                }
            });
            
            renderCameras();
            renderTeams();
            updateGaugeCheckboxes();
            createChartsGrid();
            
        } catch (error) {
            alert('Error removing camera: ' + error.message);
        }
    }
}

function updateCameraDisplay(cameraId, newReading) {
    const camera = cameras.find(c => c.id === cameraId);
    if (!camera) return;
    
    const readingElement = document.getElementById(`reading-${cameraId}`);
    const statusElement = document.getElementById(`status-${cameraId}`);
    
    if (readingElement) {
        const formattedReading = typeof newReading === 'number' ? 
            newReading.toFixed(1) : newReading;
        readingElement.textContent = formattedReading;
        
        console.log(`Updated reading for ${cameraId}: ${formattedReading}`);
    } else {
        console.warn(`Reading element not found for camera ${cameraId}`);
    }
    
    if (statusElement) {
        const status = getReadingStatus(newReading, camera);
        statusElement.className = `reading-status status-${status}-reading`;
        statusElement.textContent = status.toUpperCase();
        
        console.log(`Updated status for ${cameraId}: ${status}`);
    } else {
        console.warn(`Status element not found for camera ${cameraId}`);
    }
}

// ================================
// ANALOG CALIBRATION FUNCTIONS
// ================================

function showAnalogCalibrationModal(cameraId) {
    currentAnalogCamera = cameraId;
    analogCalibrationPoints = [];
    calibrationHistory = [];
    
    calibrationPointsRequired = parseInt(document.getElementById('calibrationPointsCount').value) || 4;
    
    const videoFeed = document.getElementById('analogVideoFeed');
    if (videoFeed) {
        videoFeed.src = `/api/video-feed/${cameraId}`;
        videoFeed.onload = function() {
            document.getElementById('videoStatus').textContent = 'Video feed active - Click on gauge numbers';
        };
        videoFeed.onerror = function() {
            document.getElementById('videoStatus').textContent = 'Error loading video feed';
        };
    }
    
    updateCalibrationStatus();
    updateCalibrationProgress();
    updateCalibrationButtons();
    updateCalibrationPoints();
    
    document.getElementById('analogCalibrationModal').classList.add('active');
    document.body.style.overflow = 'hidden';
    
    startAnalogStatusUpdates();
    
    console.log(`Opened calibration modal for camera: ${cameraId}`);
}

function hideAnalogCalibrationModal() {
    if (analogCalibrationPoints.length > 0) {
        const hasMinimum = analogCalibrationPoints.length >= 2;
        const message = hasMinimum 
            ? `You have ${analogCalibrationPoints.length} calibration points. Close without completing calibration?`
            : `You have ${analogCalibrationPoints.length} calibration points (minimum 2 needed). Close anyway?`;
        
        if (!confirm(message)) {
            return;
        }
    }
    
    document.getElementById('analogCalibrationModal').classList.remove('active');
    document.body.style.overflow = 'auto';
    
    const videoFeed = document.getElementById('analogVideoFeed');
    if (videoFeed) {
        videoFeed.src = '';
    }
    
    stopAnalogStatusUpdates();
    
    currentAnalogCamera = null;
    analogCalibrationPoints = [];
    calibrationHistory = [];
    pendingCalibrationClick = null;
    
    console.log('Calibration modal closed');
}

function handleAnalogClick(event) {
    if (!currentAnalogCamera) {
        console.warn('No active camera for calibration');
        return;
    }
    
    if (analogCalibrationPoints.length >= calibrationPointsRequired) {
        alert(`Maximum calibration points (${calibrationPointsRequired}) already reached. Remove some points or increase the limit.`);
        return;
    }
    
    const img = event.target;
    const rect = img.getBoundingClientRect();
    
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    const frameWidth = img.naturalWidth || img.width;
    const frameHeight = img.naturalHeight || img.height;
    
    const scaleX = frameWidth / img.offsetWidth;
    const scaleY = frameHeight / img.offsetHeight;
    
    const actualX = x * scaleX;
    const actualY = y * scaleY;
    
    console.log(`Analog click at: ${actualX}, ${actualY} (frame: ${frameWidth}x${frameHeight})`);
    
    sendAnalogCalibrationClick(actualX, actualY, frameWidth, frameHeight);
    
    img.classList.add('click-feedback');
    setTimeout(() => img.classList.remove('click-feedback'), 300);
}

async function sendAnalogCalibrationClick(x, y, frameWidth, frameHeight) {
    try {
        const response = await apiCall('/api/analog-calibration-click', 'POST', {
            camera_id: currentAnalogCamera,
            x: x,
            y: y,
            frame_width: frameWidth,
            frame_height: frameHeight
        });
        
        if (response.success) {
            console.log('Click registered:', response.message);
            
            pendingCalibrationClick = {
                x: x,
                y: y,
                message: response.message,
                timestamp: Date.now()
            };
            
            showValueInputModal();
        } else {
            alert('Calibration click failed: ' + response.message);
        }
        
    } catch (error) {
        console.error('Calibration click error:', error);
        alert('Error registering calibration click: ' + error.message);
    }
}

function updateCalibrationRequirements() {
    const select = document.getElementById('calibrationPointsCount');
    calibrationPointsRequired = parseInt(select.value) || 4;
    updateCalibrationProgress();
    updateCalibrationButtons();
}

function undoLastPoint() {
    if (analogCalibrationPoints.length === 0) {
        alert('No points to undo');
        return;
    }
    
    const removedPoint = analogCalibrationPoints.pop();
    calibrationHistory.pop();
    
    updateCalibrationPoints();
    updateCalibrationStatus();
    updateCalibrationProgress();
    updateCalibrationButtons();
    
    console.log('Last calibration point removed:', removedPoint);
    
    const statusText = document.getElementById('calibrationStatusText');
    if (statusText) {
        const remaining = calibrationPointsRequired - analogCalibrationPoints.length;
        statusText.textContent = `Point removed (${remaining} more needed)`;
        statusText.style.color = '#ca8a04';
        setTimeout(() => {
            statusText.style.color = '';
            updateCalibrationStatus();
        }, 2000);
    }
}

function resetAllPoints() {
    if (analogCalibrationPoints.length === 0) {
        alert('No points to reset');
        return;
    }
    
    const confirmText = `Are you sure you want to reset all ${analogCalibrationPoints.length} calibration points?`;
    
    if (confirm(confirmText)) {
        analogCalibrationPoints = [];
        calibrationHistory = [];
        
        updateCalibrationPoints();
        updateCalibrationStatus();
        updateCalibrationProgress();
        updateCalibrationButtons();
        
        console.log('All calibration points reset');
    }
}

function updateCalibrationButtons() {
    const elements = {
        undoBtn: document.getElementById('undoBtn'),
        resetBtn: document.getElementById('resetBtn'),
        completeBtn: document.getElementById('completeCalibrationBtn')
    };
    
    const hasPoints = analogCalibrationPoints.length > 0;
    const hasMinimumPoints = analogCalibrationPoints.length >= 2;
    
    if (elements.undoBtn) elements.undoBtn.disabled = !hasPoints;
    if (elements.resetBtn) elements.resetBtn.disabled = !hasPoints;
    if (elements.completeBtn) elements.completeBtn.disabled = !hasMinimumPoints;
}

function updateCalibrationProgress() {
    const collected = analogCalibrationPoints.length;
    const progress = Math.min(100, (collected / calibrationPointsRequired) * 100);
    const progressText = `${collected}/${calibrationPointsRequired}`;
    
    const elements = {
        progressBar: document.getElementById('calibrationProgressBar'),
        progressDisplay: document.getElementById('calibrationProgress')
    };
    
    if (elements.progressBar) {
        elements.progressBar.style.width = `${progress}%`;
    }
    
    if (elements.progressDisplay) {
        elements.progressDisplay.textContent = `${Math.round(progress)}%`;
    }
}

function updateCalibrationPoints() {
    const pointsContainer = document.getElementById('calibrationPoints');
    if (!pointsContainer) return;
    
    if (analogCalibrationPoints.length === 0) {
        pointsContainer.innerHTML = '<div class="log-placeholder">No calibration points yet - click on gauge numbers to start</div>';
        return;
    }
    
    const pointsHTML = analogCalibrationPoints.map((point, index) => {
        const timestamp = calibrationHistory[index] 
            ? new Date(calibrationHistory[index]).toLocaleTimeString() 
            : 'Unknown';
        
        return `
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 8px; margin: 4px 0; display: flex; justify-content: space-between; align-items: center;">
                <span>
                    <strong>Point ${index + 1}:</strong> 
                    (${point.position[0].toFixed(1)}, ${point.position[1].toFixed(1)}) = 
                    <strong style="color: #059669;">${point.value}</strong>
                    <small style="color: #64748b; margin-left: 8px;">${timestamp}</small>
                </span>
            </div>
        `;
    }).join('');
    
    pointsContainer.innerHTML = pointsHTML;
}

function updateCalibrationStatus() {
    const elements = {
        statusText: document.getElementById('calibrationStatusText'),
        pointsCollected: document.getElementById('pointsCollected'),
        pointsRequiredDisplay: document.getElementById('pointsRequiredDisplay'),
        estimatedRange: document.getElementById('estimatedRange')
    };
    
    const collected = analogCalibrationPoints.length;
    
    if (elements.pointsCollected) {
        elements.pointsCollected.textContent = collected;
    }
    
    if (elements.pointsRequiredDisplay) {
        elements.pointsRequiredDisplay.textContent = ` / ${calibrationPointsRequired}`;
    }
    
    if (elements.estimatedRange && collected >= 2) {
        const values = analogCalibrationPoints.map(p => p.value);
        const min = Math.min(...values);
        const max = Math.max(...values);
        elements.estimatedRange.textContent = `${min.toFixed(1)} - ${max.toFixed(1)}`;
    } else if (elements.estimatedRange) {
        elements.estimatedRange.textContent = '--';
    }
    
    if (elements.statusText) {
        if (collected === 0) {
            elements.statusText.textContent = 'Ready for calibration - click on gauge numbers';
            elements.statusText.style.color = '#0369a1';
        } else if (collected < 2) {
            elements.statusText.textContent = `Need ${2 - collected} more point(s) for minimum calibration`;
            elements.statusText.style.color = '#dc2626';
        } else if (collected < calibrationPointsRequired) {
            const remaining = calibrationPointsRequired - collected;
            elements.statusText.textContent = `${remaining} more point(s) needed for completion`;
            elements.statusText.style.color = '#ca8a04';
        } else {
            elements.statusText.textContent = 'Sufficient points collected - ready to complete!';
            elements.statusText.style.color = '#16a34a';
        }
    }
}

function completeAnalogCalibration() {
    if (analogCalibrationPoints.length < 2) {
        alert('At least 2 calibration points are required');
        return;
    }
    
    console.log('Completing calibration with', analogCalibrationPoints.length, 'points');
    
    hideAnalogCalibrationModal();
    renderCameras();
}

function startAnalogStatusUpdates() {
    if (analogStatusInterval) {
        clearInterval(analogStatusInterval);
    }
    
    analogStatusInterval = setInterval(async () => {
        if (currentAnalogCamera) {
            try {
                const status = await apiCall(`/api/analog-status/${currentAnalogCamera}`);
                updateAnalogStatusDisplay(status);
            } catch (error) {
                console.error('Status update error:', error);
            }
        }
    }, 1000);
}

function stopAnalogStatusUpdates() {
    if (analogStatusInterval) {
        clearInterval(analogStatusInterval);
        analogStatusInterval = null;
    }
}

function updateAnalogStatusDisplay(status) {
    const elements = {
        currentReading: document.getElementById('currentAnalogReading'),
        statusText: document.getElementById('calibrationStatusText'),
        pointsCollected: document.getElementById('pointsCollected')
    };
    
    if (elements.currentReading && status.reading !== null && status.reading !== undefined) {
        elements.currentReading.textContent = status.reading.toFixed(2);
    }
    
    if (elements.statusText && status.status && analogCalibrationPoints.length === 0) {
        elements.statusText.textContent = status.status;
    }
    
    if (elements.pointsCollected && status.calibration_points !== undefined) {
        if (status.calibration_points !== analogCalibrationPoints.length) {
            console.log(`Server reports ${status.calibration_points} points, client has ${analogCalibrationPoints.length}`);
        }
    }
}

// ================================
// MODAL FUNCTIONS
// ================================

function showAddCameraModal() {
    document.getElementById('cameraModal').classList.add('active');
    document.body.style.overflow = 'hidden';
}

function hideAddCameraModal() {
    document.getElementById('cameraModal').classList.remove('active');
    document.body.style.overflow = 'auto';
    document.getElementById('cameraForm').reset();
}

function showConfigModal(cameraIndex) {
    currentConfigCamera = cameraIndex;
    const camera = cameras[cameraIndex];
    
    document.getElementById('configCameraAddress').value = camera.address || '';
    document.getElementById('configGaugeType').value = camera.type || '';
    document.getElementById('configDetectorType').value = camera.detector_type || 'digital';
    document.getElementById('configCameraLocation').value = camera.location || '';
    document.getElementById('configThresholdValue').value = camera.threshold || '';
    document.getElementById('configDemoMode').checked = camera.demo_mode || false;
    
    const configDemoModeCheckbox = document.getElementById('configDemoMode');
    const configAddressGroup = document.getElementById('configAddressGroup');
    const configCameraAddressInput = document.getElementById('configCameraAddress');
    
    if (camera.demo_mode) {
        configAddressGroup.style.opacity = '0.5';
        configCameraAddressInput.required = false;
    } else {
        configAddressGroup.style.opacity = '1';
        configCameraAddressInput.required = true;
    }
    
    document.getElementById('configModal').classList.add('active');
    document.body.style.overflow = 'hidden';
}

function hideConfigModal() {
    document.getElementById('configModal').classList.remove('active');
    document.body.style.overflow = 'auto';
    currentConfigCamera = null;
}

function showAddTeamModal() {
    updateGaugeCheckboxes();
    document.getElementById('teamModal').classList.add('active');
    document.body.style.overflow = 'hidden';
}

function hideAddTeamModal() {
    document.getElementById('teamModal').classList.remove('active');
    document.body.style.overflow = 'auto';
    document.getElementById('teamForm').reset();
}

function showValueInputModal() {
    if (!pendingCalibrationClick) return;
    
    const pointNumber = analogCalibrationPoints.length + 1;
    
    document.getElementById('clickPositionText').textContent = 
        `Enter the value shown at position (${pendingCalibrationClick.x.toFixed(1)}, ${pendingCalibrationClick.y.toFixed(1)}):`;
    
    document.getElementById('currentPointNumber').textContent = pointNumber;
    document.getElementById('totalPointsNeededText').textContent = ` of ${calibrationPointsRequired}`;
    
    document.getElementById('gaugeValueInput').value = '';
    document.getElementById('valueInputModal').classList.add('active');
    
    setTimeout(() => {
        const input = document.getElementById('gaugeValueInput');
        if (input) {
            input.focus();
        }
    }, 100);
}

function hideValueInputModal() {
    document.getElementById('valueInputModal').classList.remove('active');
    pendingCalibrationClick = null;
}

function cancelValueInput() {
    hideValueInputModal();
    console.log('Calibration point input cancelled');
}

function skipThisPoint() {
    hideValueInputModal();
    console.log('Calibration point skipped');
}

async function submitGaugeValue() {
    const valueInput = document.getElementById('gaugeValueInput');
    const value = valueInput.value.trim();
    
    if (!value || !currentAnalogCamera) {
        alert('Please enter a valid number');
        return;
    }
    
    const numericValue = parseFloat(value);
    if (isNaN(numericValue)) {
        alert('Please enter a valid numeric value');
        return;
    }
    
    const roundedValue = Math.round(numericValue * 100) / 100;
    
    const tolerance = 0.001;
    const existingValues = analogCalibrationPoints.map(p => p.value);
    const isDuplicate = existingValues.some(existing => Math.abs(existing - roundedValue) < tolerance);
    
    if (isDuplicate) {
        if (!confirm(`Value ${roundedValue} is very close to an existing calibration point. Add anyway?`)) {
            return;
        }
    }
    
    try {
        const response = await apiCall('/api/analog-calibration-value', 'POST', {
            camera_id: currentAnalogCamera,
            value: roundedValue
        });
        
        if (response.success) {
            console.log('Calibration point added:', response.message);
            
            analogCalibrationPoints.push({
                value: roundedValue,
                position: [pendingCalibrationClick.x, pendingCalibrationClick.y]
            });
            calibrationHistory.push(Date.now());
            
            updateCalibrationPoints();
            updateCalibrationStatus();
            updateCalibrationProgress();
            updateCalibrationButtons();
            
            if (analogCalibrationPoints.length >= calibrationPointsRequired) {
                setTimeout(() => {
                    if (response.completed || confirm(`You've collected ${analogCalibrationPoints.length} of ${calibrationPointsRequired} points. Complete calibration now?`)) {
                        completeAnalogCalibration();
                    }
                }, 500);
            } else if (response.completed) {
                setTimeout(() => {
                    alert('Calibration completed successfully! The gauge is now ready for use.');
                    hideAnalogCalibrationModal();
                    renderCameras();
                }, 500);
            }
            
            hideValueInputModal();
            
        } else {
            alert('Error adding calibration point: ' + response.message);
        }
        
    } catch (error) {
        console.error('Submit value error:', error);
        alert('Error submitting calibration value: ' + error.message);
    }
}

function handleDetectorTypeChange() {
    const detectorType = document.getElementById('detectorType').value;
    const calibrationInfo = document.getElementById('calibrationInfo');
    
    if (detectorType === 'analog') {
        if (calibrationInfo) {
            calibrationInfo.style.display = 'block';
            calibrationInfo.innerHTML = `
                <div style="background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px; padding: 8px; margin-top: 8px; font-size: 12px; color: #92400e;">
                    ℹ️ Analog gauges require manual calibration after setup. You'll need to click on visible numbers on the gauge.
                </div>
            `;
        }
    } else {
        if (calibrationInfo) {
            calibrationInfo.style.display = 'none';
        }
    }
}

// ================================
// CHART FUNCTIONS
// ================================

function createChartsGrid() {
    const chartsContainer = document.getElementById('chartsGrid');
    
    chartsContainer.innerHTML = '';
    individualCharts = {};
    
    const activeCameras = cameras.filter(camera => 
        (camera.address && chartData[camera.id]) || 
        (camera.demo_mode) ||
        (camera.address)
    );
    
    console.log('Creating charts grid:', {
        totalCameras: cameras.length,
        activeCameras: activeCameras.length,
        activeCameraIds: activeCameras.map(c => c.id)
    });
    
    if (activeCameras.length === 0) {
        chartsContainer.innerHTML = `
            <div class="no-charts-message">
                <div class="icon">📊</div>
                <div><strong>No cameras with data</strong></div>
                <div>Add cameras and wait for data collection to see charts</div>
            </div>
        `;
        return;
    }
    
    activeCameras.forEach(camera => {
        console.log(`Creating chart container for ${camera.id}`);
        
        const chartContainer = document.createElement('div');
        chartContainer.className = 'individual-chart';
        const modeIndicator = camera.demo_mode ? '🎭 DEMO' : (camera.address ? '🔴 LIVE' : 'OFFLINE');
        const statusClass = camera.demo_mode ? 'status-demo' : (camera.address ? 'status-online' : 'status-warning');
        
        chartContainer.innerHTML = `
            <div class="chart-header">
                <div class="chart-title">${getTypeIcon(camera.type)} ${camera.id} - ${camera.location}</div>
                <div class="chart-status ${statusClass}">${modeIndicator}</div>
            </div>
            <div class="chart-container">
                <canvas id="chart-${camera.id}" width="400" height="180"></canvas>
            </div>
            <div class="chart-info">
                <span>Current: <span class="current-reading" id="current-${camera.id}">0${gaugeConfigs[camera.type].unit}</span></span>
                <span>Threshold: ${camera.threshold}${gaugeConfigs[camera.type].unit}</span>
                <span>Status: <span id="chart-status-${camera.id}" class="status-normal-reading">NORMAL</span></span>
            </div>
        `;
        
        chartsContainer.appendChild(chartContainer);
        
        setTimeout(() => {
            initializeIndividualChart(camera);
        }, 100);
    });
}

function initializeIndividualChart(camera) {
    const canvas = document.getElementById(`chart-${camera.id}`);
    if (!canvas) {
        console.error(`Canvas not found for camera ${camera.id}`);
        return;
    }
    
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        console.error(`Could not get 2D context for camera ${camera.id}`);
        return;
    }
    
    const container = canvas.parentElement;
    const containerWidth = container.offsetWidth || 400;
    const containerHeight = 180;
    
    canvas.width = containerWidth;
    canvas.height = containerHeight;
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = containerHeight + 'px';
    
    console.log(`Initializing chart for ${camera.id}:`, {
        canvasWidth: canvas.width,
        canvasHeight: canvas.height,
        containerWidth: containerWidth,
        offsetWidth: canvas.offsetWidth
    });
    
    individualCharts[camera.id] = {
        canvas: canvas,
        ctx: ctx,
        width: canvas.width,
        height: canvas.height,
        padding: { top: 15, right: 20, bottom: 30, left: 40 },
        camera: camera,
        color: getRandomColor()
    };
    
    if (camera.demo_mode && !chartData[camera.id]) {
        console.log(`Initializing demo data for ${camera.id}`);
        chartData[camera.id] = [];
        gaugeReadings[camera.id] = generateRandomReading(camera);
    }
    
    requestAnimationFrame(() => {
        updateIndividualChart(camera.id);
    });
}

function getRandomColor() {
    const colors = [
        '#dc2626', '#3b82f6', '#10b981', '#f59e0b', 
        '#8b5cf6', '#ef4444', '#06b6d4', '#84cc16'
    ];
    return colors[Math.floor(Math.random() * colors.length)];
}

function updateIndividualChart(cameraId) {
    const chart = individualCharts[cameraId];
    if (!chart) {
        console.log(`No chart found for camera ${cameraId}`);
        return;
    }
    
    const camera = chart.camera;
    const { ctx, width, height, padding, color } = chart;
    const data = chartData[cameraId] || [];
    
    console.log(`Updating chart for ${cameraId}:`, {
        dataPoints: data.length,
        latestValue: data.length > 0 ? data[data.length - 1].value : 'no data',
        chartExists: !!chart,
        canvasSize: { width, height },
        currentReading: gaugeReadings[cameraId]
    });
    
    ctx.clearRect(0, 0, width, height);
    
    if (data.length === 0) {
        ctx.fillStyle = '#64748b';
        ctx.font = '14px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Collecting data...', width / 2, height / 2);
        console.log(`No data available for ${cameraId}, showing placeholder`);
        return;
    }
    
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    
    const values = data.map(point => point.value);
    let minY = Math.min(...values);
    let maxY = Math.max(...values);
    
    const yPadding = (maxY - minY) * 0.1;
    minY -= yPadding;
    maxY += yPadding;
    
    if (maxY - minY < 1) {
        minY -= 0.5;
        maxY += 0.5;
    }
    
    // Draw grid lines
    ctx.strokeStyle = '#f1f5f9';
    ctx.lineWidth = 1;
    
    for (let i = 0; i <= 4; i++) {
        const y = padding.top + (chartHeight / 4) * i;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(padding.left + chartWidth, y);
        ctx.stroke();
    }
    
    for (let i = 0; i <= 6; i++) {
        const x = padding.left + (chartWidth / 6) * i;
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, padding.top + chartHeight);
        ctx.stroke();
    }
    
    // Draw axes
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 2;
    
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top + chartHeight);
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.stroke();
    
    // Draw Y-axis labels
    ctx.fillStyle = '#64748b';
    ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    
    for (let i = 0; i <= 4; i++) {
        const value = maxY - ((maxY - minY) / 4) * i;
        const y = padding.top + (chartHeight / 4) * i;
        ctx.fillText(value.toFixed(1), padding.left - 5, y);
    }
    
    // Draw X-axis labels
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    
    for (let i = 0; i <= 6; i++) {
        const secondsAgo = 30 - (30 / 6) * i;
        const x = padding.left + (chartWidth / 6) * i;
        ctx.fillText(`-${Math.round(secondsAgo)}s`, x, padding.top + chartHeight + 5);
    }
    
    // Draw threshold line
    const threshold = camera.threshold || gaugeConfigs[camera.type]?.normalMax || 0;
    const thresholdY = padding.top + chartHeight - ((threshold - minY) / (maxY - minY)) * chartHeight;
    
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(padding.left, thresholdY);
    ctx.lineTo(padding.left + chartWidth, thresholdY);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Draw data line
    if (data.length >= 1) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        
        const points = data.map((point, pointIndex) => {
            let timePosition;
            
            if (data.length === 1) {
                timePosition = 0.5;
            } else {
                const timeRange = data[data.length - 1].timestamp - data[0].timestamp;
                timePosition = timeRange > 0 ? 
                    (point.timestamp - data[0].timestamp) / timeRange : 
                    pointIndex / (data.length - 1);
            }
            
            const x = padding.left + chartWidth * timePosition;
            const y = padding.top + chartHeight - ((point.value - minY) / (maxY - minY)) * chartHeight;
            return { x, y, value: point.value };
        });
        
        if (data.length > 1) {
            ctx.beginPath();
            ctx.moveTo(points[0].x, points[0].y);
            
            if (points.length === 2) {
                ctx.lineTo(points[1].x, points[1].y);
            } else {
                for (let i = 0; i < points.length - 1; i++) {
                    const current = points[i];
                    const next = points[i + 1];
                    
                    if (i === 0) {
                        const controlX = current.x + (next.x - current.x) * 0.5;
                        const controlY = current.y;
                        ctx.quadraticCurveTo(controlX, controlY, next.x, next.y);
                    } else if (i === points.length - 2) {
                        const controlX = current.x + (next.x - current.x) * 0.5;
                        const controlY = next.y;
                        ctx.quadraticCurveTo(controlX, controlY, next.x, next.y);
                    } else {
                        const previous = points[i - 1];
                        const nextNext = points[i + 2] || next;
                        
                        const cp1x = current.x + (next.x - previous.x) * 0.15;
                        const cp1y = current.y + (next.y - previous.y) * 0.15;
                        const cp2x = next.x - (nextNext.x - current.x) * 0.15;
                        const cp2y = next.y - (nextNext.y - current.y) * 0.15;
                        
                        ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, next.x, next.y);
                    }
                }
            }
            
            ctx.stroke();
        }
        
        ctx.fillStyle = color;
        points.forEach(point => {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 4, 0, 2 * Math.PI);
            ctx.fill();
            
            if (point.value > threshold) {
                ctx.strokeStyle = '#ef4444';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(point.x, point.y, 7, 0, 2 * Math.PI);
                ctx.stroke();
                
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
            }
        });
    }
    
    const currentValue = gaugeReadings[cameraId];
    if (currentValue !== undefined && currentValue !== null) {
        const chartCurrentElement = document.getElementById(`current-${cameraId}`);
        const chartStatusElement = document.getElementById(`chart-status-${cameraId}`);
        
        if (chartCurrentElement) {
            chartCurrentElement.textContent = `${currentValue.toFixed ? currentValue.toFixed(1) : currentValue}${gaugeConfigs[camera.type].unit}`;
        }
        
        if (chartStatusElement) {
            const status = getReadingStatus(currentValue, camera);
            chartStatusElement.className = `status-${status}-reading`;
            chartStatusElement.textContent = status.toUpperCase();
        }
    }
}

function updateAllCharts() {
    Object.keys(individualCharts).forEach(cameraId => {
        updateIndividualChart(cameraId);
    });
}

// ================================
// DATA MANAGEMENT FUNCTIONS
// ================================

function updateGaugeReadings() {
   cameras.forEach(camera => {
        if (camera.demo_mode) {
            // Demo mode - always generate random readings
            const newReading = generateRandomReading(camera);
            const oldReading = gaugeReadings[camera.id];
            gaugeReadings[camera.id] = newReading;
            
            if (!chartData[camera.id]) {
                chartData[camera.id] = [];
            }
            
            const now = Date.now();
            chartData[camera.id].push({
                timestamp: now,
                value: newReading
            });
            
            const cutoffTime = now - 30000;
            chartData[camera.id] = chartData[camera.id].filter(point => point.timestamp > cutoffTime);
            
            updateCameraDisplay(camera.id, newReading);
            
            const threshold = camera.threshold || gaugeConfigs[camera.type]?.normalMax || 0;
            const wasAlerting = oldReading > threshold;
            const isAlerting = newReading > threshold;
            
            if (isAlerting && !wasAlerting) {
                const assignedMembers = teamMembers
                    .filter(member => member.assignedGauges && member.assignedGauges.includes(camera.id))
                    .map(member => member.name);
                
                addAlertLog(camera.id, newReading, threshold, assignedMembers);
            }
        } else {
            // Real detector mode - readings come from fetchCameraReadings()
            // But ensure we always have a valid display value
            if (gaugeReadings[camera.id] === undefined) {
                gaugeReadings[camera.id] = 0;
            }
            
            // Update display with current reading (even if it's 0)
            updateCameraDisplay(camera.id, gaugeReadings[camera.id]);
        }
    });
    
    updateAllCharts();
    updateDashboardStats();
}

function updateDashboardStats() {
    const totalCameras = cameras.length;
    const onlineCameras = cameras.filter(c => c.address || c.demo_mode).length;
    const alertingCameras = cameras.filter(c => {
        const reading = gaugeReadings[c.id];
        const threshold = c.threshold || gaugeConfigs[c.type]?.normalMax || 0;
        return reading > threshold;
    }).length;
    
    document.getElementById('total-cameras').textContent = totalCameras;
    document.getElementById('cameras-online').textContent = onlineCameras;
    document.getElementById('threshold-alerts').textContent = alertingCameras;
    
    document.getElementById('online-count').textContent = onlineCameras;
    document.getElementById('offline-count').textContent = totalCameras - onlineCameras;
    document.getElementById('alert-count').textContent = alertingCameras;
    document.getElementById('normal-count').textContent = onlineCameras - alertingCameras;
    
    document.getElementById('last-update').textContent = formatTime(new Date());
}

function checkAlerts() {
    cameras.forEach(camera => {
        if (camera.address || camera.demo_mode) {
            const reading = gaugeReadings[camera.id] || 0;
            const threshold = camera.threshold || gaugeConfigs[camera.type]?.normalMax || 0;
            
            if (reading > threshold * 1.1) {
                if (camera.demo_mode && Math.random() < 0.002) {
                    camera.address = '';
                    camera.demo_mode = false;
                    addActivityLog(`Camera ${camera.id} went OFFLINE at ${camera.location}`, 'warning');
                    renderCameras();
                    createChartsGrid();
                }
            }
        }
    });
}

function getDetailedStatus(value, camera) {
    const config = gaugeConfigs[camera.type];
    const threshold = camera.threshold || config.normalMax;
    
    const dangerHigh = threshold * 1.2;
    const warningHigh = threshold * 1.05;
    const warningLow = threshold * 0.8;
    const dangerLow = threshold * 0.6;
    
    if (value >= dangerHigh) return { status: 'danger', color: '#ef4444', level: 'high' };
    if (value >= warningHigh) return { status: 'warning', color: '#f59e0b', level: 'high' };
    if (value <= dangerLow) return { status: 'danger', color: '#ef4444', level: 'low' };
    if (value <= warningLow) return { status: 'warning', color: '#f59e0b', level: 'low' };
    return { status: 'normal', color: '#10b981', level: 'normal' };
}

function predictNextValue(dataPoints) {
    if (dataPoints.length < 3) return dataPoints[dataPoints.length - 1]?.value || 0;
    
    const recentPoints = dataPoints.slice(-5);
    const n = recentPoints.length;
    
    const sumX = recentPoints.reduce((sum, _, i) => sum + i, 0);
    const sumY = recentPoints.reduce((sum, point) => sum + point.value, 0);
    const sumXY = recentPoints.reduce((sum, point, i) => sum + i * point.value, 0);
    const sumXX = recentPoints.reduce((sum, _, i) => sum + i * i, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    return slope * n + intercept;
}

// ================================
// TEAM MANAGEMENT FUNCTIONS
// ================================

function getInitials(name) {
    return name.split(' ').map(n => n[0]).join('').toUpperCase().substring(0, 2);
}

function renderTeams() {
    const container = document.getElementById('teamsGrid');
    
    container.innerHTML = '';
    
    if (teamMembers.length === 0) {
        container.innerHTML = `
            <div class="empty-teams-message">
                <span class="icon">👥</span>
                <div><strong>No team members added</strong></div>
                <div>Click "Add Team Member" button to add notification recipients</div>
            </div>
        `;
    } else {
        const teamsHTML = teamMembers.map((member, index) => {
            const initials = getInitials(member.name);
            const assignedGauges = member.assignedGauges || [];
            
            return `
                <div class="team-card">
                    <div class="team-header" style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center;">
                            <div class="team-avatar">${initials}</div>
                            <div class="team-info">
                                <h4>${member.name}</h4>
                                <p>${member.email}</p>
                            </div>
                        </div>
                        <button class="remove-btn" onclick="removeTeamMember(${index})" title="Remove Member">🗑️</button>
                    </div>
                    <div class="team-details">
                        <div class="detail-item">
                            <span class="detail-label">Contact:</span>
                            <span class="detail-value">${member.contact}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label">Assigned Gauges:</span>
                            <span class="detail-value">${assignedGauges.length} gauge(s)</span>
                        </div>
                        <div class="gauge-tags">
                            ${assignedGauges.map(gaugeId => {
                                const camera = cameras.find(c => c.id === gaugeId);
                                return `<span class="gauge-tag">${camera ? camera.id : gaugeId}</span>`;
                            }).join('')}
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        container.innerHTML = teamsHTML;
    }
}

function removeTeamMember(index) {
    if (confirm('Are you sure you want to remove this team member?')) {
        const removedMember = teamMembers[index];
        
        addActivityLog(`Team member ${removedMember.name} removed from system`, 'warning');
        
        teamMembers.splice(index, 1);
        renderTeams();
    }
}

function updateGaugeCheckboxes() {
    const container = document.getElementById('gaugeCheckboxes');
    
    if (cameras.length === 0) {
        container.innerHTML = '<div style="text-align: center; color: #64748b; padding: 1rem;">No cameras available. Add cameras first.</div>';
        return;
    }
    
    container.innerHTML = cameras.map((camera, index) => `
        <div class="checkbox-item">
            <input type="checkbox" id="gauge_${index}" value="${camera.id}">
            <label for="gauge_${index}">${camera.id} - ${camera.location}</label>
        </div>
    `).join('');
}

// ================================
// ACTIVITY AND ALERT LOG FUNCTIONS
// ================================

function addActivityLog(message, type = 'activity') {
    const entry = {
        id: Date.now(),
        message: message,
        type: type,
        timestamp: new Date()
    };
    
    activityLog.unshift(entry);
    
    if (activityLog.length > 50) {
        activityLog = activityLog.slice(0, 50);
    }
    
    renderActivityLog();
}

function addAlertLog(cameraId, currentValue, threshold, assignedMembers) {
    const camera = cameras.find(c => c.id === cameraId);
    if (!camera) return;

    const entry = {
        id: Date.now(),
        cameraId: cameraId,
        location: camera.location,
        currentValue: currentValue,
        threshold: threshold,
        assignedMembers: assignedMembers || [],
        timestamp: new Date(),
        type: currentValue > threshold * 1.2 ? 'danger' : 'warning'
    };
    
    alertLog.unshift(entry);
    
    if (alertLog.length > 100) {
        alertLog = alertLog.slice(0, 100);
    }
    
    renderAlertLog();
    updateRecentAlertsTable();
}

function renderActivityLog() {
    const container = document.getElementById('activity-log');
    const countElement = document.getElementById('activity-count');
    
    countElement.textContent = `${activityLog.length} entries`;
    
    if (activityLog.length === 0) {
        container.innerHTML = `
            <div class="empty-log">
                <div class="icon">📝</div>
                <div><strong>No activity yet</strong></div>
                <div>System activities will appear here</div>
            </div>
        `;
    } else {
        const logsHTML = activityLog.map(entry => {
            const iconClass = entry.type === 'warning' ? 'warning' : 
                             entry.type === 'success' ? 'success' : 'activity';
            const icon = entry.type === 'warning' ? '⚠️' : 
                        entry.type === 'success' ? '✅' : 'ℹ️';
            
            return `
                <div class="log-entry">
                    <div class="log-icon ${iconClass}">${icon}</div>
                    <div class="log-time">${formatTime(entry.timestamp)}</div>
                    <div class="log-message">${entry.message}</div>
                </div>
            `;
        }).join('');
        
        container.innerHTML = logsHTML;
    }
}

function renderAlertLog() {
    const container = document.getElementById('alert-log');
    const countElement = document.getElementById('alert-log-count');
    
    countElement.textContent = `${alertLog.length} alerts`;
    
    if (alertLog.length === 0) {
        container.innerHTML = `
            <div class="empty-log">
                <div class="icon">🔔</div>
                <div><strong>No alerts yet</strong></div>
                <div>Threshold alerts will appear here</div>
            </div>
        `;
    } else {
        const logsHTML = alertLog.map(entry => {
            const icon = entry.type === 'danger' ? '🚨' : '⚠️';
            const membersText = entry.assignedMembers.length > 0 ? 
                `Notified: ${entry.assignedMembers.join(', ')}` : 'No members assigned';
            
            return `
                <div class="log-entry">
                    <div class="log-icon alert">${icon}</div>
                    <div class="log-time">${formatTime(entry.timestamp)}</div>
                    <div class="log-message">
                        <strong>${entry.cameraId}</strong> at ${entry.location}<br>
                        Threshold exceeded: ${entry.currentValue} > ${entry.threshold}<br>
                        <small style="color: #64748b;">${membersText}</small>
                    </div>
                </div>
            `;
        }).join('');
        
        container.innerHTML = logsHTML;
    }
}

function updateRecentAlertsTable() {
    const container = document.getElementById('recent-alerts-list');
    const recentAlerts = alertLog.slice(0, 5);
    
    if (recentAlerts.length === 0) {
        container.innerHTML = `
            <div class="table-row" style="text-align: center; color: #64748b; grid-column: 1 / -1; padding: 2rem;">
                No recent alerts
            </div>
        `;
    } else {
        const alertsHTML = recentAlerts.map(alert => {
            const config = gaugeConfigs[cameras.find(c => c.id === alert.cameraId)?.type];
            const unit = config ? config.unit : '';
            
            return `
                <div class="table-row">
                    <div class="camera-info">
                        <div class="camera-avatar">${alert.cameraId.split('-')[1]}</div>
                        <div class="camera-details">
                            <h5>${alert.cameraId}</h5>
                            <p>${alert.location}</p>
                        </div>
                    </div>
                    <div class="time-info">${formatDateTime(alert.timestamp)}</div>
                    <div class="duration-info">Active</div>
                    <div class="threshold-info">${alert.threshold}${unit} → ${alert.currentValue}${unit}</div>
                    <div><a href="#" class="details-btn" onclick="switchToPage('alerts')">Details</a></div>
                </div>
            `;
        }).join('');
        
        container.innerHTML = alertsHTML;
    }
}

// ================================
// AI PREDICTION FUNCTIONS
// ================================

function generateAIPredictions() {
    const predictionsContainer = document.getElementById('predictions-grid');
    
    if (cameras.length === 0) {
        predictionsContainer.innerHTML = `
            <div style="grid-column: 1 / -1; text-align: center; color: #64748b; padding: 2rem;">
                No cameras available for predictions
            </div>
        `;
        return;
    }
    
    const predictions = cameras.filter(camera => (camera.address || camera.demo_mode) && chartData[camera.id]).map(camera => {
        const data = chartData[camera.id];
        if (!data || data.length < 3) return null;
        
        const prediction = predictNextValue(data);
        const currentValue = gaugeReadings[camera.id] || 0;
        const predictionStatus = getDetailedStatus(prediction, camera);
        const confidence = calculatePredictionConfidence(data);
        const trend = calculateTrend(data);
        
        return {
            camera,
            prediction,
            currentValue,
            status: predictionStatus,
            confidence,
            trend
        };
    }).filter(Boolean);
    
    if (predictions.length === 0) {
        predictionsContainer.innerHTML = `
            <div style="grid-column: 1 / -1; text-align: center; color: #64748b; padding: 2rem;">
                Collecting data for predictions...
            </div>
        `;
        return;
    }
    
    const predictionsHTML = predictions.map(pred => {
        const config = gaugeConfigs[pred.camera.type];
        const trendIcon = pred.trend.direction === 'up' ? '↗️' : 
                        pred.trend.direction === 'down' ? '↘️' : '➡️';
        const trendClass = pred.trend.direction === 'up' ? 'trend-up' : 
                        pred.trend.direction === 'down' ? 'trend-down' : 'trend-stable';
        
        return `
            <div class="prediction-card ${pred.status.status}">
                <div class="prediction-header">
                    <div class="prediction-title">${pred.camera.id} - ${pred.camera.location}</div>
                    <div class="prediction-confidence">${pred.confidence}% confidence</div>
                </div>
                <div class="prediction-values">
                    <div class="value-item">
                        <div class="value-label">Current</div>
                        <div class="value-number">${pred.currentValue.toFixed(1)}${config.unit}</div>
                    </div>
                    <div class="value-item">
                        <div class="value-label">Predicted</div>
                        <div class="value-number">${pred.prediction.toFixed(1)}${config.unit}</div>
                    </div>
                </div>
                <div class="prediction-trend">
                    <span class="trend-arrow ${trendClass}">${trendIcon}</span>
                    <span>Trending ${pred.trend.direction} (${pred.trend.rate.toFixed(1)}% change)</span>
                </div>
            </div>
        `;
    }).join('');
    
    predictionsContainer.innerHTML = predictionsHTML;
}

function calculatePredictionConfidence(dataPoints) {
    if (dataPoints.length < 3) return 50;
    
    const recent = dataPoints.slice(-5);
    const values = recent.map(p => p.value);
    const mean = values.reduce((a, b) => a + b) / values.length;
    const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2)) / values.length;
    const stdDev = Math.sqrt(variance);
    
    const confidence = Math.max(60, Math.min(95, 95 - (stdDev * 10)));
    return Math.round(confidence);
}

function calculateTrend(dataPoints) {
    if (dataPoints.length < 3) return { direction: 'stable', rate: 0 };
    
    const recent = dataPoints.slice(-3);
    const firstValue = recent[0].value;
    const lastValue = recent[recent.length - 1].value;
    const change = ((lastValue - firstValue) / firstValue) * 100;
    
    return {
        direction: change > 2 ? 'up' : change < -2 ? 'down' : 'stable',
        rate: Math.abs(change)
    };
}

function generateAISummary() {
    const summaryContent = document.getElementById('summary-content');
    const analysisSection = document.getElementById('analysis-summary');
    
    analysisSection.style.display = 'block';
    
    summaryContent.innerHTML = `
        <div style="text-align: center; padding: 2rem;">
            <div class="loading-spinner"></div>
            <div style="margin-top: 1rem; color: #64748b;">Generating AI analysis...</div>
        </div>
    `;
    
    setTimeout(() => {
        const stats = calculateSystemStats();
        const summary = generateSystemSummary(stats);
        
        summaryContent.innerHTML = `
            <div class="summary-stats">
                <div class="stat-box">
                    <div class="stat-number">${stats.totalReadings}</div>
                    <div class="stat-label">Total Readings</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">${stats.warningCount}</div>
                    <div class="stat-label">Warning Events</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">${stats.dangerCount}</div>
                    <div class="stat-label">Danger Events</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">${stats.systemHealth}%</div>
                    <div class="stat-label">System Health</div>
                </div>
            </div>
            <div>${summary}</div>
        `;
    }, 2000);
}

function calculateSystemStats() {
    const totalReadings = Object.values(chartData).reduce((sum, data) => sum + data.length, 0);
    let warningCount = 0;
    let dangerCount = 0;
    
    cameras.forEach(camera => {
        const data = chartData[camera.id] || [];
        data.forEach(point => {
            const status = getDetailedStatus(point.value, camera);
            if (status.status === 'warning') warningCount++;
            if (status.status === 'danger') dangerCount++;
        });
    });
    
    const systemHealth = Math.max(0, 100 - (warningCount * 2) - (dangerCount * 5));
    
    return { totalReadings, warningCount, dangerCount, systemHealth };
}

function generateSystemSummary(stats) {
    let summary = "<h4>System Performance Analysis</h4>";
    
    if (stats.systemHealth >= 90) {
        summary += "<p style='color: #16a34a;'>✅ <strong>Excellent:</strong> Your system is performing optimally with minimal alerts. All cameras are operating within normal parameters.</p>";
    } else if (stats.systemHealth >= 70) {
        summary += "<p style='color: #ca8a04;'>⚠️ <strong>Good:</strong> System performance is generally stable, but there have been some warning events that should be monitored.</p>";
    } else {
        summary += "<p style='color: #dc2626;'>🚨 <strong>Attention Required:</strong> The system has experienced multiple alerts. Immediate review and potential maintenance may be needed.</p>";
    }
    
    summary += `
        <h4>Key Insights</h4>
        <ul style="margin-left: 1rem; color: #374151;">
            <li>Processed ${stats.totalReadings} total readings across all cameras</li>
            <li>${stats.warningCount} warning events detected (values approaching thresholds)</li>
            <li>${stats.dangerCount} critical events recorded (values exceeding safety limits)</li>
            <li>Overall system reliability: ${stats.systemHealth}%</li>
        </ul>
        
        <h4>Recommendations</h4>
        <ul style="margin-left: 1rem; color: #374151;">
            <li>Monitor cameras with frequent threshold violations</li>
            <li>Consider adjusting thresholds if false positives are common</li>
            <li>Schedule maintenance for cameras showing irregular patterns</li>
            <li>Review team member assignments for critical gauges</li>
        </ul>
    `;
    
    return summary;
}

// ================================
// UTILITY FUNCTIONS
// ================================

function formatTime(date) {
    return date.toLocaleTimeString('en-US', { 
        hour12: false, 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit' 
    });
}

function formatDateTime(date) {
    return date.toLocaleString('en-US', { 
        month: 'short', 
        day: '2-digit', 
        year: 'numeric',
        hour: '2-digit', 
        minute: '2-digit',
        hour12: false
    });
}

// ================================