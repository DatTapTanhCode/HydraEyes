#!/usr/bin/env python3
"""
Web-Compatible Analog Gauge Detector
Modified from still2.py to work with Flask web application
"""

import cv2
import numpy as np
import time
import os
import sys
import logging
from datetime import datetime
import torch
import json
import base64
from io import BytesIO
from PIL import Image
import threading

# Tắt logging của ultralytics
os.environ['YOLO_VERBOSE'] = 'False'
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Thêm path để import các module của pipeline
sys.path.append('gauge_reader_web')

from gauge_reader_web.angle_reading_fit.angle_converter import AngleConverter
from gauge_reader_web.angle_reading_fit.line_fit import line_fit, line_fit_ransac
from gauge_reader_web.geometry.ellipse import get_point_from_angle, get_theta_middle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NumberLabel:
    """Class to store OCR number with ellipse angle"""
    def __init__(self, number, position, theta):
        self.number = number
        self.position = position  # (x, y) position
        self.theta = theta  # angle on ellipse

class WebAnalogGaugeProcessor:
    """
    Web-compatible version of FullGaugeProcessor
    Designed to work with Flask and provide real-time updates
    """
    
    def __init__(self, detection_model_path, key_point_model_path, segmentation_model_path):
        self.detection_model_path = detection_model_path
        self.key_point_model_path = key_point_model_path
        self.segmentation_model_path = segmentation_model_path
        
        # Processing control
        self.frame_skip = 3
        self.keypoint_skip = 9
        self.frame_counter = 0
        
        # Cache results
        self.last_gauge_box = None
        self.last_keypoints = None
        self.last_needle_line = None
        self.last_frame = None
        self.last_processed_frame = None
        
        # Temporal filtering
        self.reading_history = []
        self.angle_history = []
        self.ellipse_history = []
        self.history_size = 8
        
        # Stability tracking
        self.stable_gauge_count = 0
        self.stable_threshold = 3
        
        # Calibration state
        self.manual_calibration_mode = True
        self.manual_calibrated = False
        self.calibration_completed = False
        self.required_calibration_points = 4
        self.manual_points = []
        self.current_calibration_step = 0
        self.waiting_for_click = False
        self.calibration_instruction = "Click on first number on gauge"
        
        # Permanent calibration data
        self.permanent_scale_mapping = None
        self.permanent_ellipse_params = None
        self.permanent_unit = None
        self.permanent_zero_point = None
        
        # Current reading data
        self.current_reading = None
        self.current_angle = None
        self.current_status = "Initializing"
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Feature toggles
        self.needle_enabled = True
        self.ocr_enabled = False  # Force disable OCR
        self.one_time_calibration_mode = False  # Force disable
        self.calibration_disabled_calculation = True
        
        # Initialize models
        self._initialize_models()
        
        logger.info("✅ WebAnalogGaugeProcessor initialized")
    
    def _initialize_models(self):
        """Initialize detection models"""
        try:
            # Initialize keypoint model
            self.key_point_inferencer = None
            if os.path.exists(self.key_point_model_path):
                try:
                    from gauge_reader_web.key_point_detection.key_point_inference import KeyPointInference
                    self.key_point_inferencer = KeyPointInference(self.key_point_model_path)
                    logger.info("✅ Keypoint model loaded")
                except Exception as e:
                    logger.warning(f"⚠️ Keypoint model failed: {e}")
                    self.keypoint_skip = 99999  # Effectively disable
            else:
                logger.warning(f"⚠️ Keypoint model not found: {self.key_point_model_path}")
                
            # Check other model paths
            if not os.path.exists(self.detection_model_path):
                logger.warning(f"⚠️ Detection model not found: {self.detection_model_path}")
            if not os.path.exists(self.segmentation_model_path):
                logger.warning(f"⚠️ Segmentation model not found: {self.segmentation_model_path}")
                
        except Exception as e:
            logger.error(f"❌ Model initialization error: {e}")
    
    def process_frame(self, frame):
        """
        Process a single frame and return results
        Thread-safe processing with web-compatible output
        """
        with self.lock:
            return self._process_frame_internal(frame)
    
    def _process_frame_internal(self, frame):
        """Internal frame processing"""
        try:
            self.frame_counter += 1
            self.last_frame = frame.copy()
            
            results = {
                'processed_frame': frame.copy(),
                'gauge_detected': False,
                'gauge_box': None,
                'keypoints': None,
                'needle_line': None,
                'reading': None,
                'status': 'Processing',
                'calibration_ready': False
            }
            
            # Step 1: Gauge Detection
            if self.frame_counter % self.frame_skip == 0:
                results = self._detect_gauge(frame, results)
            else:
                # Reuse cached gauge
                if self.last_gauge_box is not None:
                    results['gauge_detected'] = True
                    results['gauge_box'] = self.last_gauge_box
            
            # Step 2: Keypoint Detection (when stable)
            if (results['gauge_detected'] and 
                self.stable_gauge_count >= self.stable_threshold and
                self.frame_counter % self.keypoint_skip == 0 and
                self.key_point_inferencer is not None):
                
                results = self._detect_keypoints(frame, results)
            else:
                # Reuse cached keypoints
                if self.last_keypoints is not None:
                    results['keypoints'] = self.last_keypoints
            
            # Step 3: Manual Calibration Check
            if (self.manual_calibration_mode and
                results['gauge_detected'] and
                self.stable_gauge_count >= self.stable_threshold and
                not self.manual_calibrated):
                
                if not self.waiting_for_click and len(self.manual_points) == 0:
                    self.waiting_for_click = True
                    results['calibration_ready'] = True
                    results['status'] = 'Ready for calibration - click on gauge numbers'
            
            # Step 4: Needle Detection and Reading Calculation
            if (results['gauge_detected'] and
                self.stable_gauge_count >= self.stable_threshold and
                self.frame_counter % self.keypoint_skip == 0):
                
                results = self._detect_needle(frame, results)
                
                # Calculate reading if everything is available
                if results.get('needle_line') and results.get('keypoints'):
                    results = self._calculate_reading(results)
            else:
                # Reuse cached needle
                if self.last_needle_line is not None:
                    results['needle_line'] = self.last_needle_line
            
            # Update status
            self._update_status(results)
            
            # Draw results
            self._draw_results(results)
            self.last_processed_frame = results['processed_frame']
            
            return results
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return {
                'processed_frame': frame,
                'gauge_detected': False,
                'status': f'Error: {str(e)}',
                'reading': None
            }
    
    # Web Interface Methods
    def handle_calibration_click(self, x, y, frame_width, frame_height):
        """
        Handle calibration click from web interface
        Returns: (success, message, completed)
        """
        try:
            if not self.manual_calibration_mode or not self.waiting_for_click:
                return False, "Not in calibration mode", False
            
            if self.last_gauge_box is None:
                return False, "No gauge detected", False
            
            # Convert click coordinates to gauge coordinates
            gauge_box = self.last_gauge_box
            x1, y1, x2, y2 = gauge_box
            gauge_width = x2 - x1
            gauge_height = y2 - y1
            
            # Convert to gauge coordinates (0-448 space)
            gauge_x = ((x - x1) / gauge_width) * 448
            gauge_y = ((y - y1) / gauge_height) * 448
            
            # Check if click is inside gauge
            if 0 <= gauge_x <= 448 and 0 <= gauge_y <= 448:
                # Store click point (value will be provided separately)
                click_point = {
                    'position': (gauge_x, gauge_y),
                    'screen_pos': (x, y),
                    'step': self.current_calibration_step,
                    'timestamp': time.time()
                }
                
                # Temporary storage until value is provided
                self.pending_click = click_point
                
                message = f"Click registered at ({gauge_x:.1f}, {gauge_y:.1f}). Please provide the value."
                return True, message, False
            else:
                return False, "Click outside gauge area", False
                
        except Exception as e:
            logger.error(f"Calibration click error: {e}")
            return False, f"Error: {str(e)}", False
    
    def add_calibration_value(self, value):
        """
        Add value for the last clicked point
        Returns: (success, message, completed)
        """
        try:
            if not hasattr(self, 'pending_click'):
                return False, "No pending click", False
            
            # Validate value
            try:
                float_value = float(value)
            except ValueError:
                return False, "Invalid number", False
            
            # Add to manual points
            click_point = self.pending_click
            self.manual_points.append((click_point['position'], float_value))
            self.current_calibration_step += 1
            
            # Remove pending click
            delattr(self, 'pending_click')
            
            message = f"Point {self.current_calibration_step}: ({click_point['position'][0]:.1f}, {click_point['position'][1]:.1f}) = {float_value}"
            logger.info(message)
            
            # Check if calibration is complete
            if self.current_calibration_step >= self.required_calibration_points:
                success = self._complete_manual_calibration()
                if success:
                    return True, "Calibration completed successfully!", True
                else:
                    return False, "Calibration completion failed", False
            else:
                remaining = self.required_calibration_points - self.current_calibration_step
                return True, f"Point added. {remaining} more points needed.", False
                
        except Exception as e:
            logger.error(f"Add calibration value error: {e}")
            return False, f"Error: {str(e)}", False
    
    def get_current_status(self):
        """Get current processor status for web interface"""
        with self.lock:
            return {
                'reading': self.current_reading,
                'angle': np.degrees(self.current_angle) if self.current_angle else None,
                'status': self.current_status,
                'calibrated': self.manual_calibrated,
                'waiting_for_click': self.waiting_for_click,
                'calibration_points': len(self.manual_points),
                'required_points': self.required_calibration_points,
                'stable_count': self.stable_gauge_count,
                'gauge_detected': self.last_gauge_box is not None
            }
    
    def get_frame_as_base64(self):
        """Get current processed frame as base64 string for web display"""
        try:
            if self.last_processed_frame is not None:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', self.last_processed_frame, 
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                return f"data:image/jpeg;base64,{frame_base64}"
            return None
        except Exception as e:
            logger.error(f"Frame encoding error: {e}")
            return None
    
    # Keep all original processing methods unchanged
    def _detect_gauge(self, frame, results):
        """Gauge detection with web-compatible error handling"""
        try:
            from gauge_reader_web.gauge_detection.detection_inference import detection_gauge_face
            
            detection_result = detection_gauge_face(frame, self.detection_model_path)
            
            # Handle different return formats
            if isinstance(detection_result, tuple):
                box, all_boxes = detection_result
            else:
                box = detection_result
            
            # Extract coordinates safely
            if box is not None:
                box_coords = self._extract_box_coordinates(box)
                
                if box_coords is not None and len(box_coords) == 4:
                    results['gauge_detected'] = True
                    results['gauge_box'] = box_coords
                    
                    # Check stability
                    if self._is_gauge_stable(box_coords):
                        self.stable_gauge_count += 1
                    else:
                        self.stable_gauge_count = 0
                    
                    self.last_gauge_box = box_coords
                else:
                    self.stable_gauge_count = 0
            else:
                self.stable_gauge_count = 0
                
        except Exception as e:
            logger.error(f"Gauge detection error: {e}")
            self.stable_gauge_count = 0
        
        return results
    
    def _detect_keypoints(self, frame, results):
        """Keypoint detection"""
        try:
            gauge_box = results['gauge_box']
            cropped_img = self._crop_image(frame, gauge_box)
            cropped_resized_img = cv2.resize(cropped_img, (448, 448), interpolation=cv2.INTER_CUBIC)
            
            from gauge_reader_web.key_point_detection.key_point_inference import detect_key_points
            
            heatmaps = self.key_point_inferencer.predict_heatmaps(cropped_resized_img)
            
            # Convert heatmaps to numpy if needed
            if torch.is_tensor(heatmaps):
                if heatmaps.is_cuda:
                    heatmaps = heatmaps.detach().cpu().numpy()
                else:
                    heatmaps = heatmaps.detach().numpy()
            
            key_point_list = detect_key_points(heatmaps)
            
            if len(key_point_list) >= 3:
                # Convert keypoints to numpy if needed
                processed_keypoints = []
                for kp in key_point_list:
                    if torch.is_tensor(kp):
                        if kp.is_cuda:
                            kp = kp.detach().cpu().numpy()
                        else:
                            kp = kp.detach().numpy()
                    processed_keypoints.append(kp)
                
                results['keypoints'] = processed_keypoints
                self.last_keypoints = processed_keypoints
                logger.debug(f"✅ Keypoints detected: {[len(kp) for kp in processed_keypoints]}")
            
        except Exception as e:
            logger.error(f"Keypoint detection error: {e}")
        
        return results
    
    def _detect_needle(self, frame, results):
        """Needle segmentation"""
        try:
            gauge_box = results['gauge_box']
            cropped_img = self._crop_image(frame, gauge_box)
            cropped_resized_img = cv2.resize(cropped_img, (448, 448), interpolation=cv2.INTER_CUBIC)
            
            from gauge_reader_web.segmentation.segmenation_inference import segment_gauge_needle
            from gauge_reader_web.segmentation.segmenation_inference import get_fitted_line, get_start_end_line
            
            needle_mask_x, needle_mask_y = segment_gauge_needle(
                cropped_resized_img, self.segmentation_model_path
            )
            
            if len(needle_mask_x) > 0 and len(needle_mask_y) > 0:
                needle_line_coeffs, needle_error = get_fitted_line(needle_mask_x, needle_mask_y)
                needle_line_start_x, needle_line_end_x = get_start_end_line(needle_mask_x)
                needle_line_start_y, needle_line_end_y = get_start_end_line(needle_mask_y)
                
                needle_line = {
                    'coeffs': needle_line_coeffs,
                    'start': (needle_line_start_x, needle_line_start_y),
                    'end': (needle_line_end_x, needle_line_end_y),
                    'error': needle_error,
                    'mask_x': needle_mask_x,
                    'mask_y': needle_mask_y
                }
                
                results['needle_line'] = needle_line
                self.last_needle_line = needle_line
                logger.debug(f"✅ Needle detected (error: {needle_error:.3f})")
            
        except Exception as e:
            logger.error(f"Needle detection error: {e}")
        
        return results
    
    def _calculate_reading(self, results):
        """Calculate gauge reading"""
        try:
            if not (self.manual_calibrated or self.calibration_completed):
                return results
            
            keypoints = results['keypoints']
            needle_line = results['needle_line']
            
            if len(keypoints) >= 3 and needle_line is not None:
                # Get ellipse parameters
                if self.permanent_ellipse_params is not None:
                    ellipse_params = self.permanent_ellipse_params
                else:
                    ellipse_params = self._get_stable_ellipse(keypoints)
                
                if ellipse_params is None:
                    return results
                
                # Find intersection
                needle_coeffs = needle_line['coeffs']
                needle_start_x, needle_end_x = needle_line['start'][0], needle_line['end'][0]
                
                intersection_point = self._find_needle_ellipse_intersection(
                    needle_coeffs, [needle_start_x, needle_end_x], ellipse_params
                )
                
                if intersection_point is None:
                    return results
                
                # Calculate angle
                angle = self._calculate_needle_angle(intersection_point, ellipse_params)
                
                # Convert to reading
                raw_reading = self._angle_to_actual_reading(angle)
                filtered_reading = self._apply_temporal_filter(raw_reading, angle)
                
                if filtered_reading is not None:
                    unit = self.permanent_unit or 'units'
                    results['reading'] = f"{filtered_reading:.2f}"
                    results['unit'] = unit
                    results['angle'] = angle
                    
                    # Update current reading
                    self.current_reading = filtered_reading
                    self.current_angle = angle
        
        except Exception as e:
            logger.error(f"Reading calculation error: {e}")
        
        return results
    
    # Keep all other original methods with minimal modifications
    def _tensor_to_numpy(self, tensor):
        """Safely convert tensor to numpy, handling CUDA tensors"""
        if torch.is_tensor(tensor):
            if tensor.is_cuda:
                return tensor.detach().cpu().numpy()
            else:
                return tensor.detach().numpy()
        return tensor
    
    def _extract_box_coordinates(self, box):
        """Safely extract box coordinates"""
        if box is None:
            return None
        
        if torch.is_tensor(box):
            if box.is_cuda:
                box_np = box.detach().cpu().numpy()
            else:
                box_np = box.detach().numpy()
        else:
            box_np = np.array(box)
        
        if len(box_np.shape) > 1:
            box_np = box_np.flatten()
        
        if len(box_np) >= 4:
            return box_np[:4].astype(int).tolist()
        return None
    
    def _is_gauge_stable(self, current_box):
        """Check if gauge detection is stable"""
        if self.last_gauge_box is None:
            return False
        
        try:
            prev_center = [(self.last_gauge_box[0] + self.last_gauge_box[2]) / 2,
                          (self.last_gauge_box[1] + self.last_gauge_box[3]) / 2]
            curr_center = [(current_box[0] + current_box[2]) / 2,
                          (current_box[1] + current_box[3]) / 2]
            
            distance = np.sqrt((prev_center[0] - curr_center[0]) ** 2 +
                             (prev_center[1] - curr_center[1]) ** 2)
            
            return distance < 20  # pixels
        except Exception as e:
            logger.error(f"Stability check error: {e}")
            return False
    
    def _crop_image(self, img, box):
        """Crop image using pipeline logic"""
        try:
            img = np.copy(img)
            x1, y1, x2, y2 = [int(coord) for coord in box[:4]]
            
            cropped_img = img[y1:y2, x1:x2, :]
            height = int(y2 - y1)
            width = int(x2 - x1)
            
            # Make square with padding
            if height > width:
                delta = height - width
                left, right = delta // 2, delta - (delta // 2)
                top = bottom = 0
            else:
                delta = width - height
                top, bottom = delta // 2, delta - (delta // 2)
                left = right = 0
            
            pad_color = [0, 0, 0]
            new_img = cv2.copyMakeBorder(
                cropped_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
            )
            
            return new_img
        
        except Exception as e:
            logger.error(f"Crop error: {e}")
            return img
    
    def _update_status(self, results):
        """Update current status based on results"""
        if not results['gauge_detected']:
            self.current_status = "No gauge detected"
        elif self.stable_gauge_count < self.stable_threshold:
            self.current_status = f"Detecting gauge... ({self.stable_gauge_count}/{self.stable_threshold})"
        elif self.waiting_for_click:
            points_remaining = self.required_calibration_points - len(self.manual_points)
            self.current_status = f"Calibration: Click on gauge numbers ({points_remaining} remaining)"
        elif not (self.manual_calibrated or self.calibration_completed):
            self.current_status = "Ready for calibration"
        elif results.get('reading'):
            self.current_status = "Reading gauge"
        else:
            self.current_status = "Processing..."
    
    # Add other helper methods as needed from still_pipeline.py
    # (I'll include the most important ones for basic functionality)
    
    def _get_stable_ellipse(self, keypoints):
        """Get stable ellipse parameters using history"""
        try:
            new_ellipse = self._fit_ellipse_from_keypoints(keypoints)
            
            if new_ellipse is not None:
                self.ellipse_history.append(new_ellipse)
                if len(self.ellipse_history) > self.history_size:
                    self.ellipse_history.pop(0)
                
                if len(self.ellipse_history) >= 3:
                    avg_ellipse = np.mean(self.ellipse_history, axis=0)
                    return avg_ellipse
                else:
                    return new_ellipse
            
            if len(self.ellipse_history) > 0:
                return self.ellipse_history[-1]
            
            return None
        
        except Exception as e:
            logger.error(f"Stable ellipse error: {e}")
            return None
    
    def _fit_ellipse_from_keypoints(self, keypoints):
        """Fit ellipse from keypoint coordinates"""
        try:
            from gauge_reader_web.geometry.ellipse import fit_ellipse, cart_to_pol
            
            all_points = []
            for kp_group in keypoints:
                if hasattr(kp_group, 'shape') and kp_group.shape[0] > 0:
                    for point in kp_group:
                        all_points.append([point[0], point[1]])
            
            if len(all_points) < 5:
                return None
            
            all_points = np.array(all_points)
            x_coords = all_points[:, 0]
            y_coords = all_points[:, 1]
            
            ellipse_coeffs = fit_ellipse(x_coords, y_coords)
            ellipse_params = cart_to_pol(ellipse_coeffs)
            
            return ellipse_params
        
        except Exception as e:
            logger.error(f"Ellipse fitting error: {e}")
            return None
    
    def _find_needle_ellipse_intersection(self, needle_coeffs, needle_x_range, ellipse_params):
        """Find intersection between needle and ellipse"""
        try:
            from gauge_reader_web.geometry.ellipse import get_line_ellipse_point
            return get_line_ellipse_point(needle_coeffs, needle_x_range, ellipse_params)
        except Exception as e:
            logger.error(f"Intersection error: {e}")
            return None
    
    def _calculate_needle_angle(self, intersection_point, ellipse_params):
        """Calculate needle angle"""
        try:
            from gauge_reader_web.geometry.ellipse import get_polar_angle
            return get_polar_angle(intersection_point, ellipse_params)
        except Exception as e:
            logger.error(f"Angle calculation error: {e}")
            return 0
    
    def _angle_to_actual_reading(self, angle):
        """Convert angle to actual reading using calibration"""
        try:
            if self.permanent_scale_mapping is not None and self.permanent_ellipse_params is not None:
                return self._interpolate_from_permanent_calibration(angle)
            else:
                # Generic fallback
                normalized_angle = angle % (2 * np.pi)
                angle_degrees = np.degrees(normalized_angle)
                percentage = (angle_degrees / 360) * 100
                return max(0, min(100, percentage))
        except Exception as e:
            logger.error(f"Reading conversion error: {e}")
            return 0
    
    def _interpolate_from_permanent_calibration(self, angle):
        """Interpolate using permanent calibration"""
        try:
            from gauge_reader_web.geometry.ellipse import get_polar_angle, project_point
            from gauge_reader_web.angle_reading_fit.angle_converter import AngleConverter
            from gauge_reader_web.angle_reading_fit.line_fit import line_fit_ransac, line_fit
            
            # Convert markers to NumberLabel objects
            number_labels = []
            for marker in self.permanent_scale_mapping:
                marker_pos = np.array(marker['position'])
                proj_point = project_point(marker_pos, self.permanent_ellipse_params)
                marker_theta = get_polar_angle(proj_point, self.permanent_ellipse_params)
                
                number_label = NumberLabel(
                    number=marker['value'],
                    position=marker['position'],
                    theta=marker_theta
                )
                number_labels.append(number_label)
            
            if len(number_labels) < 2:
                return 0
            
            # Use permanent zero point if available
            theta_zero = self.permanent_zero_point if self.permanent_zero_point else np.pi
            
            # Convert angles
            angle_converter = AngleConverter(theta_zero)
            angle_number_list = []
            for number_label in number_labels:
                converted_angle = angle_converter.convert_angle(number_label.theta)
                angle_number_list.append((converted_angle, number_label.number))
            
            angle_number_arr = np.array(angle_number_list)
            
            # Fit line
            try:
                reading_line_coeff, inlier_mask, outlier_mask = line_fit_ransac(
                    angle_number_arr[:, 0], angle_number_arr[:, 1]
                )
            except:
                reading_line_coeff = line_fit(angle_number_arr[:, 0], angle_number_arr[:, 1])
            
            # Calculate reading
            reading_line = np.poly1d(reading_line_coeff)
            needle_angle_conv = angle_converter.convert_angle(angle)
            reading = reading_line(needle_angle_conv)
            
            return reading
        
        except Exception as e:
            logger.error(f"Permanent interpolation error: {e}")
            return 0
    
    def _apply_temporal_filter(self, raw_reading, angle):
        """Apply temporal filtering"""
        try:
            self.reading_history.append(raw_reading)
            self.angle_history.append(angle)
            
            if len(self.reading_history) > self.history_size:
                self.reading_history.pop(0)
            if len(self.angle_history) > self.history_size:
                self.angle_history.pop(0)
            
            if len(self.reading_history) < 2:
                return raw_reading
            
            # Adaptive filtering based on change rate
            if len(self.reading_history) >= 3:
                recent_change = abs(self.reading_history[-1] - self.reading_history[-3])
                
                if recent_change > 0.5:
                    filter_window = 2
                    weight = 0.8
                elif recent_change > 0.2:
                    filter_window = 3
                    weight = 0.6
                else:
                    filter_window = min(5, len(self.reading_history))
                    weight = 0.4
                
                filtered_reading = np.median(self.reading_history[-filter_window:])
                final_reading = weight * raw_reading + (1 - weight) * filtered_reading
            else:
                final_reading = raw_reading
            
            return final_reading
        
        except Exception as e:
            logger.error(f"Temporal filter error: {e}")
            return raw_reading
    
    def _complete_manual_calibration(self):
        """Complete manual calibration"""
        try:
            if len(self.manual_points) < 2:
                return False
            
            # Convert to scale markers
            scale_markers = []
            for position, value in self.manual_points:
                scale_markers.append({
                    'value': float(value),
                    'position': list(position),
                    'confidence': 1.0
                })
            
            # Store permanently
            self.permanent_scale_mapping = scale_markers
            
            # Get ellipse params
            if hasattr(self, 'last_keypoints') and self.last_keypoints is not None:
                fitted_ellipse = self._fit_ellipse_from_keypoints(self.last_keypoints)
                if fitted_ellipse is not None:
                    if isinstance(fitted_ellipse, tuple):
                        self.permanent_ellipse_params = np.array(fitted_ellipse)
                    else:
                        self.permanent_ellipse_params = fitted_ellipse.copy()
                    
                    # Calculate permanent zero point
                    self.permanent_zero_point = self._get_permanent_zero_point()
                else:
                    return False
            else:
                return False
            
            # Set unit
            self.permanent_unit = "units"
            
            # Mark as completed
            self.manual_calibrated = True
            self.calibration_completed = True
            self.waiting_for_click = False
            
            logger.info(f"✅ Manual calibration completed with {len(scale_markers)} points")
            return True
        
        except Exception as e:
            logger.error(f"Calibration completion error: {e}")
            return False
    
    def _get_permanent_zero_point(self):
        """Calculate permanent zero point"""
        try:
            if hasattr(self, 'last_keypoints') and self.last_keypoints is not None:
                keypoints = self.last_keypoints
                if len(keypoints) >= 2:
                    start_points = keypoints[0]
                    end_points = keypoints[-1]
                    
                    if len(start_points) > 0 and len(end_points) > 0:
                        start_point = np.mean(start_points, axis=0)
                        end_point = np.mean(end_points, axis=0)
                        
                        from gauge_reader_web.geometry.ellipse import get_polar_angle, get_theta_middle
                        theta_start = get_polar_angle(start_point, self.permanent_ellipse_params)
                        theta_end = get_polar_angle(end_point, self.permanent_ellipse_params)
                        
                        return get_theta_middle(theta_start, theta_end)
            
            return np.pi  # Bottom middle fallback
        
        except Exception as e:
            logger.error(f"Zero point calculation error: {e}")
            return np.pi
    
    def _draw_results(self, results):
        """Draw results on frame"""
        frame = results['processed_frame']
        
        # Draw gauge bounding box
        if results['gauge_detected'] and results['gauge_box']:
            try:
                box = results['gauge_box']
                x1, y1, x2, y2 = [int(coord) for coord in box[:4]]
                
                if self.stable_gauge_count >= self.stable_threshold:
                    color = (0, 255, 0)  # Green - stable
                    status = "STABLE"
                else:
                    color = (0, 255, 255)  # Yellow - detecting
                    status = "DETECTING"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'GAUGE {status}', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except Exception as e:
                logger.error(f"Draw gauge box error: {e}")
        
        # Draw keypoints
        if results.get('keypoints') and results['gauge_box']:
            self._draw_keypoints(frame, results['keypoints'], results['gauge_box'])
        
        # Draw needle
        if results.get('needle_line') and results['gauge_box']:
            self._draw_needle_line(frame, results['needle_line'], results['gauge_box'])
        
        # Draw reading
        if results.get('reading'):
            unit = results.get('unit', 'units')
            text = f"Reading: {results['reading']} {unit}"
            cv2.putText(frame, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw calibration status
        cal_text = f"Calibration: {len(self.manual_points)}/{self.required_calibration_points}"
        if self.manual_calibrated:
            cal_color = (0, 255, 0)
            cal_text += " COMPLETED"
        elif self.waiting_for_click:
            cal_color = (0, 255, 255)
            cal_text += " CLICK ON NUMBERS"
        else:
            cal_color = (128, 128, 128)
            cal_text += " WAITING"
        
        cv2.putText(frame, cal_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, cal_color, 1)
    
    def _draw_keypoints(self, frame, key_point_list, gauge_box):
        """Draw keypoints on frame"""
        try:
            if len(key_point_list) >= 3:
                x1, y1, x2, y2 = [int(coord) for coord in gauge_box[:4]]
                gauge_width = x2 - x1
                gauge_height = y2 - y1
                
                scale_x = gauge_width / 448
                scale_y = gauge_height / 448
                
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
                labels = ['Start', 'Middle', 'End']
                
                for i, points in enumerate(key_point_list):
                    color = colors[i % 3]
                    
                    if torch.is_tensor(points):
                        if points.is_cuda:
                            points = points.detach().cpu().numpy()
                        else:
                            points = points.detach().numpy()
                    
                    if hasattr(points, 'shape') and points.shape[0] > 0:
                        for point in points:
                            try:
                                x = int(x1 + point[0] * scale_x)
                                y = int(y1 + point[1] * scale_y)
                                cv2.circle(frame, (x, y), 4, color, -1)
                                cv2.putText(frame, labels[i], (x + 5, y - 5),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            except:
                                continue
        except Exception as e:
            logger.error(f"Draw keypoints error: {e}")
    
    def _draw_needle_line(self, frame, needle_line, gauge_box):
        """Draw needle line on frame"""
        try:
            x1, y1, x2, y2 = [int(coord) for coord in gauge_box[:4]]
            gauge_width = x2 - x1
            gauge_height = y2 - y1
            
            scale_x = gauge_width / 448
            scale_y = gauge_height / 448
            
            # Draw needle pixels as dots
            if 'mask_x' in needle_line and 'mask_y' in needle_line:
                mask_x = needle_line['mask_x']
                mask_y = needle_line['mask_y']
                
                for px, py in zip(mask_x, mask_y):
                    x = int(x1 + px * scale_x)
                    y = int(y1 + py * scale_y)
                    cv2.circle(frame, (x, y), 1, (0, 165, 255), -1)
            
            # Draw fitted line
            if 'start' in needle_line and 'end' in needle_line:
                start_x = int(x1 + needle_line['start'][0] * scale_x)
                start_y = int(y1 + needle_line['start'][1] * scale_y)
                end_x = int(x1 + needle_line['end'][0] * scale_x)
                end_y = int(y1 + needle_line['end'][1] * scale_y)
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 165, 255), 2)
                cv2.putText(frame, 'NEEDLE', (start_x + 5, start_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        
        except Exception as e:
            logger.error(f"Draw needle line error: {e}")