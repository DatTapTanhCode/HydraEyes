#"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"
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, Response
import os
from datetime import datetime
import threading
import time
import cv2
import numpy as np
from digital_detector import GaugeDetector
from analog_detector_web import WebAnalogGaugeProcessor
import json
import base64
import logging
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app with explicit static folder configuration
app = Flask(__name__, 
            static_folder='templates/static',
            static_url_path='/static',
            template_folder='templates')
app.secret_key = 'vision-ui-secret-key-change-in-production'

OPENAI_API_KEY = ""  # Replace with your actual API key

# Initialize detectors
digital_detector = GaugeDetector('best.pt')

# Analog detector initialization (with model paths)
analog_models = {
    'detection': 'gauge_reader_web/models/gauge_detection_model.pt',
    'keypoint': 'gauge_reader_web/models/keypoint_model.pt', 
    'segmentation': 'gauge_reader_web/models/needle_segmentation_model.pt'
}

# Global variables for camera management
active_cameras = {}  # Store camera capture objects
camera_threads = {}  # Store camera processing threads
camera_readings = {}  # Store latest readings
analog_processors = {}  # Store analog processors
detector_enabled = True

# Approved accounts list (In production, use a database)
APPROVED_ACCOUNTS = [
    {'email': 'admin@visionui.com', 'password': 'admin123', 'name': 'Administrator'},
    {'email': 'user@visionui.com', 'password': 'user123', 'name': 'Regular User'},
    {'email': 'demo@visionui.com', 'password': 'demo123', 'name': 'Demo User'},
    {'email': 'test@visionui.com', 'password': 'test123', 'name': 'Test User'},
    {'email': 'oktavia@visionui.com', 'password': 'oktavia123', 'name': 'Oktavia'}
]

# Sample camera data
CAMERA_DATA = {
    'cameras_online': 8,
    'alerts_threshold': 3,
    'total_cameras': 12,
    'system_uptime': 99.7,
    'recent_alerts': []
}

# Store camera configurations
camera_configs = {}

class CameraProcessor:
    def __init__(self, camera_id, address, demo_mode=False, detector_type="digital"):
        self.camera_id = camera_id
        self.address = address
        self.demo_mode = demo_mode
        self.detector_type = detector_type  # "digital" or "analog"
        self.running = False
        self.cap = None
        self.thread = None
        
    def start(self):
        """Start camera processing"""
        if self.demo_mode:
            logger.info(f"Camera {self.camera_id} started in DEMO mode ({self.detector_type})")
            return True
            
        try:
            # Initialize camera capture
            if self.address.startswith('http://') or self.address.startswith('https://'):
                self.cap = cv2.VideoCapture(self.address)
            else:
                if self.address.isdigit():
                    self.cap = cv2.VideoCapture(int(self.address))
                else:
                    rtsp_url = f"rtsp://{self.address}/stream"
                    self.cap = cv2.VideoCapture(rtsp_url)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id} at {self.address}")
                return False
            
            # Initialize analog processor if needed
            if self.detector_type == "analog":
                try:
                    analog_processor = WebAnalogGaugeProcessor(
                        analog_models['detection'],
                        analog_models['keypoint'],
                        analog_models['segmentation']
                    )
                    analog_processors[self.camera_id] = analog_processor
                    logger.info(f"✅ Analog processor initialized for {self.camera_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize analog processor: {e}")
                    return False
            
            self.running = True
            self.thread = threading.Thread(target=self._process_frames, daemon=True)
            self.thread.start()
            logger.info(f"Camera {self.camera_id} started successfully ({self.detector_type})")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera {self.camera_id}: {e}")
            return False
    
    def _process_frames(self):
        """Process camera frames continuously"""
        while self.running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret:
                    if self.detector_type == "digital":
                        # Use digital detector
                        digital_detector.detect_gauge_reading(frame, self.camera_id)
                        reading_data = digital_detector.get_camera_reading(self.camera_id)
                        if reading_data:
                            camera_readings[self.camera_id] = {
                                'value': reading_data.get('reading', 0),
                                'confidence': reading_data.get('confidence', 0),
                                'timestamp': time.time(),
                                'status': 'online',
                                'type': 'digital'
                            }
                    
                    elif self.detector_type == "analog":
                        # Use analog detector
                        if self.camera_id in analog_processors:
                            processor = analog_processors[self.camera_id]
                            results = processor.process_frame(frame)
                            
                            # Extract reading
                            reading_value = 0
                            if results.get('reading'):
                                try:
                                    reading_value = float(results['reading'])
                                except:
                                    reading_value = 0
                            
                            camera_readings[self.camera_id] = {
                                'value': reading_value,
                                'confidence': 95 if results.get('reading') else 0,
                                'timestamp': time.time(),
                                'status': 'online',
                                'type': 'analog',
                                'angle': results.get('angle'),
                                'calibrated': processor.manual_calibrated,
                                'waiting_for_click': processor.waiting_for_click
                            }
                else:
                    time.sleep(0.1)
                    
                time.sleep(0.1)  # 10 FPS processing
                
            except Exception as e:
                logger.error(f"Error processing frame for {self.camera_id}: {e}")
                time.sleep(1)
    
    def stop(self):
        """Stop camera processing"""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.thread:
            self.thread.join(timeout=2)
        
        # Clean up processors
        if self.detector_type == "digital":
            digital_detector.clear_camera_data(self.camera_id)
        elif self.detector_type == "analog" and self.camera_id in analog_processors:
            del analog_processors[self.camera_id]
        
        if self.camera_id in camera_readings:
            del camera_readings[self.camera_id]
        logger.info(f"Camera {self.camera_id} stopped")

def validate_user(email, password):
    """Validate user credentials"""
    for account in APPROVED_ACCOUNTS:
        if account['email'] == email and account['password'] == password:
            return account
    return None

@app.route('/')
def index():
    """Serve the main page"""
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    """Handle login form submission"""
    email = request.form.get('email', '').strip()
    password = request.form.get('password', '').strip()
    
    if not email or not password:
        flash('Please enter both email and password', 'error')
        return redirect(url_for('index'))
    
    user = validate_user(email, password)
    
    if user:
        session['user'] = {
            'email': user['email'],
            'name': user['name'],
            'login_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return redirect(url_for('dashboard'))
    else:
        return f"""
        <html>
        <head>
            <title>Login Failed</title>
            <style>
                body {{
                    font-family: 'Segoe UI', sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    text-align: center;
                }}
                .error-container {{
                    background: rgba(255, 255, 255, 0.1);
                    padding: 2rem;
                    border-radius: 15px;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }}
                .error-title {{ font-size: 2rem; margin-bottom: 1rem; color: #ff6b6b; }}
                .error-message {{ font-size: 1.1rem; margin-bottom: 2rem; }}
                .back-btn {{
                    background: linear-gradient(135deg, #4A90E2, #357ABD);
                    color: white;
                    padding: 12px 24px;
                    border: none;
                    border-radius: 8px;
                    text-decoration: none;
                    font-weight: 600;
                    transition: all 0.3s ease;
                }}
                .back-btn:hover {{ transform: translateY(-2px); }}
            </style>
        </head>
        <body>
            <div class="error-container">
                <h1 class="error-title">❌ Login Failed</h1>
                <p class="error-message">Invalid email or password. Please try again.</p>
                <a href="/" class="back-btn">← Back to Login</a>
            </div>
            <script>
                setTimeout(() => {{ window.location.href = '/'; }}, 3000);
            </script>
        </body>
        </html>
        """

@app.route('/dashboard')
def dashboard():
    """Dashboard page (protected route)"""
    if 'user' not in session:
        return redirect(url_for('index'))
    
    user = session['user']
    return render_template('main.html', user=user, camera_data=CAMERA_DATA)

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    return redirect(url_for('index'))

@app.route('/api/camera-data')
def get_camera_data():
    """API endpoint to get current camera data"""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(CAMERA_DATA)

@app.route('/api/camera-readings')
def get_camera_readings():
    """API endpoint to get real-time camera readings"""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(camera_readings)

@app.route('/api/add-camera', methods=['POST'])
def add_camera():
    """API endpoint to add a new camera"""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        logger.info(f"Received camera data: {data}")
        
        camera_id = data.get('camera_id')
        address = data.get('address', '').strip()
        demo_mode = data.get('demo_mode', False)
        gauge_type = data.get('gauge_type')
        detector_type = data.get('detector_type', 'digital')  # Default to digital if not specified
        location = data.get('location', '').strip()
        threshold = data.get('threshold')
        
        # Validation
        if not camera_id:
            return jsonify({'error': 'Camera ID is required'}), 400
        if not gauge_type:
            return jsonify({'error': 'Gauge type is required'}), 400
        if not location:
            return jsonify({'error': 'Location is required'}), 400
        if threshold is None or threshold == '':
            return jsonify({'error': 'Threshold value is required'}), 400
        if detector_type not in ['digital', 'analog']:
            return jsonify({'error': 'Invalid detector type. Must be "digital" or "analog"'}), 400
            
        try:
            threshold = float(threshold)
        except (ValueError, TypeError):
            return jsonify({'error': 'Threshold must be a valid number'}), 400
            
        if not demo_mode and not address:
            return jsonify({'error': 'Camera address is required for detector mode'}), 400
        
        # Store camera configuration
        camera_configs[camera_id] = {
            'address': address,
            'demo_mode': demo_mode,
            'gauge_type': gauge_type,
            'detector_type': detector_type,  # Store detector type
            'location': location,
            'threshold': threshold,
            'created_at': datetime.now().isoformat()
        }
        
        # Start camera processor
        if not demo_mode and address:
            processor = CameraProcessor(camera_id, address, demo_mode, detector_type)
            if processor.start():
                active_cameras[camera_id] = processor
                return jsonify({'success': True, 'message': f'Camera {camera_id} started successfully ({detector_type})'})
            else:
                return jsonify({'error': f'Failed to start camera {camera_id}'}), 500
        else:
            # Demo mode
            camera_readings[camera_id] = {
                'value': 0,
                'confidence': 95,
                'timestamp': time.time(),
                'status': 'demo',
                'type': detector_type
            }
            return jsonify({'success': True, 'message': f'Camera {camera_id} added in demo mode ({detector_type})'})
            
    except Exception as e:
        logger.error(f"Error in add_camera: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/remove-camera', methods=['POST'])
def remove_camera():
    """API endpoint to remove a camera"""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        camera_id = data.get('camera_id')
        
        if not camera_id:
            return jsonify({'error': 'Camera ID required'}), 400
        
        # Stop camera processor if exists
        if camera_id in active_cameras:
            active_cameras[camera_id].stop()
            del active_cameras[camera_id]
        
        # Remove from configurations
        if camera_id in camera_configs:
            del camera_configs[camera_id]
        
        # Remove readings
        if camera_id in camera_readings:
            del camera_readings[camera_id]
        
        return jsonify({'success': True, 'message': f'Camera {camera_id} removed successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update-camera', methods=['POST'])
def update_camera():
    """API endpoint to update camera configuration"""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        camera_id = data.get('camera_id')
        
        if not camera_id or camera_id not in camera_configs:
            return jsonify({'error': 'Camera not found'}), 404
        
        config = camera_configs[camera_id]
        old_address = config.get('address')
        old_demo_mode = config.get('demo_mode', False)
        old_detector_type = config.get('detector_type', 'digital')
        
        # Update configuration
        config.update({
            'address': data.get('address', config.get('address')),
            'demo_mode': data.get('demo_mode', config.get('demo_mode', False)),
            'gauge_type': data.get('gauge_type', config.get('gauge_type')),
            'detector_type': data.get('detector_type', config.get('detector_type', 'digital')),  # Include detector_type
            'location': data.get('location', config.get('location')),
            'threshold': data.get('threshold', config.get('threshold')),
            'updated_at': datetime.now().isoformat()
        })
        
        new_address = config.get('address')
        new_demo_mode = config.get('demo_mode', False)
        new_detector_type = config.get('detector_type', 'digital')
        
        # Check if we need to restart camera processor
        if (old_address != new_address or 
            old_demo_mode != new_demo_mode or 
            old_detector_type != new_detector_type):
            
            # Stop existing processor
            if camera_id in active_cameras:
                active_cameras[camera_id].stop()
                del active_cameras[camera_id]
            
            # Start new processor if not demo mode
            if not new_demo_mode and new_address:
                processor = CameraProcessor(camera_id, new_address, new_demo_mode, new_detector_type)
                if processor.start():
                    active_cameras[camera_id] = processor
                else:
                    return jsonify({'error': f'Failed to restart camera {camera_id}'}), 500
            else:
                # Switch to demo mode
                camera_readings[camera_id] = {
                    'value': 0,
                    'confidence': 95,
                    'timestamp': time.time(),
                    'status': 'demo',
                    'type': new_detector_type
                }
        
        return jsonify({'success': True, 'message': f'Camera {camera_id} updated successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera-configs')
def get_camera_configs():
    """API endpoint to get all camera configurations"""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    logger.info(f"Current camera configs: {camera_configs}")
    return jsonify(camera_configs)

# NEW ANALOG-SPECIFIC ROUTES

@app.route('/api/analog-status/<camera_id>')
def get_analog_status(camera_id):
    """Get status of analog gauge processor"""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if camera_id in analog_processors:
        processor = analog_processors[camera_id]
        status = processor.get_current_status()
        return jsonify(status)
    else:
        return jsonify({'error': 'Analog processor not found'}), 404

@app.route('/api/analog-calibration-click', methods=['POST'])
def handle_analog_calibration_click():
    """Handle calibration click for analog gauge"""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        camera_id = data.get('camera_id')
        x = data.get('x')
        y = data.get('y')
        frame_width = data.get('frame_width', 640)
        frame_height = data.get('frame_height', 480)
        
        if camera_id not in analog_processors:
            return jsonify({'error': 'Analog processor not found'}), 404
        
        processor = analog_processors[camera_id]
        success, message, completed = processor.handle_calibration_click(x, y, frame_width, frame_height)
        
        return jsonify({
            'success': success,
            'message': message,
            'completed': completed
        })
        
    except Exception as e:
        logger.error(f"Calibration click error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analog-calibration-value', methods=['POST'])
def add_analog_calibration_value():
    """Add calibration value for analog gauge"""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        camera_id = data.get('camera_id')
        value = data.get('value')
        
        if camera_id not in analog_processors:
            return jsonify({'error': 'Analog processor not found'}), 404
        
        processor = analog_processors[camera_id]
        success, message, completed = processor.add_calibration_value(value)
        
        return jsonify({
            'success': success,
            'message': message,
            'completed': completed
        })
        
    except Exception as e:
        logger.error(f"Calibration value error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/video-feed/<camera_id>')
def video_feed(camera_id):
    """Stream video feed for analog gauge"""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    def generate_frames():
        while camera_id in analog_processors:
            try:
                processor = analog_processors[camera_id]
                frame_base64 = processor.get_frame_as_base64()
                
                if frame_base64:
                    # Convert base64 to bytes for streaming
                    frame_data = frame_base64.split(',')[1]  # Remove data:image/jpeg;base64,
                    frame_bytes = base64.b64decode(frame_data)
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Video feed error: {e}")
                time.sleep(0.5)
    
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/toggle-detector', methods=['POST'])
def toggle_detector():
    """API endpoint to enable/disable detector globally"""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    global detector_enabled
    data = request.get_json()
    detector_enabled = data.get('enabled', True)
    
    return jsonify({'success': True, 'detector_enabled': detector_enabled})


@app.route('/api/generate-ai-summary', methods=['POST'])
def generate_ai_summary():
    """API endpoint to generate AI summary using OpenAI"""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        selected_cameras = data.get('selectedCameras', [])
        stats = data.get('stats', [])
        
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        if not selected_cameras:
            return jsonify({'error': 'No cameras selected'}), 400
        
        logger.info(f"Generating AI summary for cameras: {selected_cameras}")
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Create the chat completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an AI assistant specialized in analyzing industrial monitoring data. Provide clear, actionable insights for system operators."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        summary = response.choices[0].message.content
        
        # Log the AI summary generation
        logger.info(f"AI summary generated successfully for {len(selected_cameras)} cameras")
        
        return jsonify({
            'success': True,
            'summary': summary,
            'cameras_analyzed': selected_cameras,
            'stats_summary': {
                'total_cameras': len(stats),
                'total_warnings': sum(stat.get('warningCount', 0) for stat in stats),
                'total_dangers': sum(stat.get('dangerCount', 0) for stat in stats),
                'generated_at': datetime.now().isoformat()
            }
        })
        
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {e}")
        return jsonify({'error': f'OpenAI API error: {str(e)}'}), 500
        
    except openai.APIConnectionError as e:
        logger.error(f"OpenAI connection error: {e}")
        return jsonify({'error': 'Failed to connect to OpenAI service'}), 500
        
    except openai.RateLimitError as e:
        logger.error(f"OpenAI rate limit error: {e}")
        return jsonify({'error': 'OpenAI rate limit exceeded. Please try again later.'}), 429
        
    except openai.APITimeoutError as e:
        logger.error(f"OpenAI timeout error: {e}")
        return jsonify({'error': 'OpenAI request timed out. Please try again.'}), 408
        
    except Exception as e:
        logger.error(f"Error generating AI summary: {e}")
        return jsonify({'error': f'Failed to generate AI summary: {str(e)}'}), 500

@app.route('/api/ai-analysis-config', methods=['GET', 'POST'])
def ai_analysis_config():
    """API endpoint to get/set AI analysis configuration"""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if request.method == 'GET':
        # Return current AI analysis configuration
        config = {
            'enabled': True,
            'auto_analysis_interval': 300,  # 5 minutes
            'include_technical_details': True,
            'max_cameras_per_analysis': 10,
            'analysis_history_limit': 50
        }
        return jsonify(config)
    
    elif request.method == 'POST':
        # Update AI analysis configuration
        try:
            data = request.get_json()
            
            # Here you would typically save to database
            # For now, just return success
            logger.info(f"AI analysis config updated: {data}")
            
            return jsonify({
                'success': True,
                'message': 'AI analysis configuration updated successfully'
            })
            
        except Exception as e:
            logger.error(f"Error updating AI config: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/api/ai-analysis-history', methods=['GET'])
def ai_analysis_history():
    """API endpoint to get AI analysis history"""
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # In a real implementation, you would fetch from database
        # For now, return empty history
        history = []
        
        return jsonify({
            'success': True,
            'history': history,
            'total_analyses': len(history)
        })
        
    except Exception as e:
        logger.error(f"Error fetching AI analysis history: {e}")
        return jsonify({'error': str(e)}), 500
    
def cleanup_cameras():
    """Cleanup function to stop all cameras on app shutdown"""
    for camera_id, processor in active_cameras.items():
        processor.stop()

if __name__ == "__main__":
    logger.info("Starting Hydra Eyes Camera Monitoring System...")
    logger.info("Static folder: " + str(app.static_folder))
    logger.info("Template folder: " + str(app.template_folder))
    
    # Check if analog models exist
    for model_name, model_path in analog_models.items():
        if os.path.exists(model_path):
            logger.info(f"✅ {model_name.title()} model found: {model_path}")
        else:
            logger.warning(f"⚠️ {model_name.title()} model not found: {model_path}")
    
    import atexit
    atexit.register(cleanup_cameras)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        cleanup_cameras()