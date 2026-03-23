"""
Reward and Safety Modules for Sovereign-v5.0
Vision-based rewards and adaptive safety constraints
"""

import numpy as np
import cv2
import logging
from collections import deque
from typing import Tuple, Optional, Dict
from config import Config

logger = logging.getLogger(__name__)


class TemporalVisionReward:
    """
    Vision-based reward calculation with temporal stability
    Uses edge detection and contour analysis
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.cap = cv2.VideoCapture(config.hardware.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        self.history = deque(maxlen=4)
        self.frame_count = 0
        self.frame_skip = 3  # Process every Nth frame to reduce computation
        
        logger.info(f"Vision reward initialized on camera {config.hardware.camera_id}")

    def _analyze_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Analyze frame for print quality metrics
        Args:
            frame: BGR image from camera
        Returns:
            dict with metrics
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 80, 180)
        edge_density = float(np.mean(edges) / 255.0)
        
        # Blob detection (via contours)
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_blobs = len(contours)
        blob_score = 1.0 if 3 < num_blobs < 60 else 0.2
        
        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = float(np.clip(laplacian_var / 1000.0, 0, 1))
        
        return {
            'edge_density': edge_density,
            'num_blobs': num_blobs,
            'blob_score': blob_score,
            'sharpness': sharpness,
        }

    def get(self) -> float:
        """
        Get reward based on current frame
        Returns:
            reward in range [-1, 1]
        """
        self.frame_count += 1
        
        # Skip frames to reduce computation
        if self.frame_count % self.frame_skip != 0:
            return 0.0
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read camera frame")
            return -0.5
        
        # Analyze frame
        metrics = self._analyze_frame(frame)
        self.history.append(metrics)
        
        # Compute quality score
        quality = (
            metrics['edge_density'] * 0.6 +
            metrics['blob_score'] * 0.4
        )
        
        # Apply temporal stability bonus
        if len(self.history) > 1:
            edge_densities = [m['edge_density'] for m in self.history]
            stability = 1.0 - np.std(edge_densities)
            quality *= stability
        
        # Map to reward range [-1, 1]
        reward = np.clip((quality - 0.45) * 2.5, -1.0, 1.0)
        
        logger.debug(f"Reward: {reward:.3f} | Quality: {quality:.3f} | Blobs: {metrics['num_blobs']}")
        
        return float(reward)

    def close(self):
        """Release camera resources"""
        if self.cap.isOpened():
            self.cap.release()


class AdaptiveSafety:
    """
    Hardware safety layer with adaptive learning
    Tracks failures and prevents dangerous parameter combinations
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.unsafe_temps = deque(maxlen=200)
        self.consecutive_failures = 0
        self.emergency_stop = False
        self.total_failures = 0
        
        logger.info("Safety layer initialized")

    def validate(
        self,
        action_id: int,
        current_temp: float,
        current_speed: float
    ) -> Tuple[bool, Optional[Dict[str, float]]]:
        """
        Validate action against hardware constraints
        Args:
            action_id: 0=lower_temp, 1=raise_temp, 2=slower, 3=faster
            current_temp: current nozzle temperature
            current_speed: current extrusion speed
        Returns:
            (is_safe, command_dict)
        """
        if self.emergency_stop:
            logger.critical("Emergency stop is active")
            return False, None
        
        # Build candidate action
        candidate = {
            'temp': float(current_temp),
            'speed': float(current_speed)
        }
        
        if action_id == 0:      # Lower temp
            candidate['temp'] -= 5
        elif action_id == 1:    # Raise temp
            candidate['temp'] += 5
        elif action_id == 2:    # Slower
            candidate['speed'] -= 10
        elif action_id == 3:    # Faster
            candidate['speed'] += 10
        else:
            logger.error(f"Invalid action_id: {action_id}")
            return False, None
        
        # Clamp to safe ranges
        candidate['temp'] = np.clip(
            candidate['temp'],
            self.config.hardware.nozzle_min,
            self.config.hardware.nozzle_max
        )
        candidate['speed'] = np.clip(
            candidate['speed'],
            self.config.hardware.speed_min,
            self.config.hardware.speed_max
        )
        
        # Check against historical failures
        if self.unsafe_temps:
            mean_unsafe = float(np.mean(self.unsafe_temps))
            if abs(candidate['temp'] - mean_unsafe) < self.config.hardware.safe_margin:
                self.consecutive_failures += 1
                logger.warning(
                    f"Unsafe temperature: {candidate['temp']:.1f}°C "
                    f"near failure zone {mean_unsafe:.1f}°C "
                    f"({self.consecutive_failures} consecutive failures)"
                )
                
                if self.consecutive_failures >= 5:
                    self.emergency_stop = True
                    logger.critical("EMERGENCY STOP: Too many consecutive failures")
                    return False, None
                
                return False, None
        
        # Safe action
        self.consecutive_failures = 0
        return True, candidate

    def report_failure(self, temp: float):
        """Record a failure temperature"""
        self.unsafe_temps.append(float(temp))
        self.total_failures += 1
        logger.error(f"Failure recorded at {temp:.1f}°C (total: {self.total_failures})")

    def reset(self):
        """Reset safety state (use after successful recovery)"""
        self.consecutive_failures = 0
        self.emergency_stop = False
        logger.info("Safety state reset")

    def get_stats(self) -> Dict:
        """Get safety statistics"""
        return {
            'emergency_stop': self.emergency_stop,
            'consecutive_failures': self.consecutive_failures,
            'total_failures': self.total_failures,
            'unsafe_temps': list(self.unsafe_temps),
        }
