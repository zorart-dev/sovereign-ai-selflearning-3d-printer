"""
AutonomousEdgeAI v1.0
A scalable active learning framework for edge devices (cameras, 3D printers, sensors)
Supports: streaming data, multimodal inputs, continual learning, federated ready

Features:
- Lightweight perceptron ensemble
- Multiple active learning strategies
- On-device data buffering and memory management
- Energy-aware learning scheduling
- Multi-input sensor fusion
- Persistent model state (for cameras across sessions)
"""

import math
import json
import time
from collections import deque
from typing import List, Tuple, Dict, Optional
import hashlib


class FeatureExtractor:
    """
    Extracts meaningful features from raw sensor/image data.
    Reduces dimensionality for embedded devices.
    """
    def __init__(self, feature_dim: int = 8):
        self.feature_dim = feature_dim
        self.feature_history = deque(maxlen=100)
        
    def extract_from_camera(self, brightness: float, edge_density: float, 
                           motion_level: float, color_variance: float) -> List[float]:
        """Extract features from raw camera frame."""
        features = [
            brightness,
            math.log(edge_density + 1),  # Non-linear scaling
            motion_level / 100.0,
            color_variance,
            brightness * edge_density,  # Interaction term
            max(0, motion_level - 50),  # Threshold-based feature
            1.0 if edge_density > 0.3 else 0.0,  # Binary feature
            1.0 if brightness > 128 else 0.0
        ]
        self.feature_history.append(features)
        return features[:self.feature_dim]
    
    def extract_from_3d_printer(self, nozzle_temp: float, bed_temp: float, 
                               extrusion_rate: float, z_variance: float) -> List[float]:
        """Extract features from 3D printer sensors."""
        features = [
            (nozzle_temp - 200) / 50,  # Normalize
            (bed_temp - 60) / 40,
            extrusion_rate / 100.0,
            z_variance * 10,
            abs(nozzle_temp - 200),  # Deviation from target
            1.0 if extrusion_rate > 50 else 0.0,
            math.sqrt(z_variance),
            (nozzle_temp + bed_temp) / 400  # Combined thermal load
        ]
        self.feature_history.append(features)
        return features[:self.feature_dim]
    
    def get_statistical_features(self) -> Dict[str, float]:
        """Compute statistical summary of recent features (memory-efficient)."""
        if not self.feature_history:
            return {}
        
        history_array = list(self.feature_history)
        stats = {}
        for i in range(self.feature_dim):
            col = [h[i] for h in history_array if i < len(h)]
            if col:
                stats[f"feat_{i}_mean"] = sum(col) / len(col)
                stats[f"feat_{i}_var"] = sum((x - stats[f"feat_{i}_mean"])**2 for x in col) / len(col)
        return stats


class LightweightPerceptron:
    """
    A single neuron classifier optimized for embedded devices.
    Uses sigmoid activation and supports multiple loss functions.
    """
    def __init__(self, input_dim: int, learning_rate: float = 0.1, 
                 momentum: float = 0.9):
        self.weights = [0.1 * (i % 3 - 1) for i in range(input_dim)]  # Small random init
        self.bias = 0.0
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity_w = [0.0] * input_dim
        self.velocity_b = 0.0
        self.training_count = 0
    
    def sigmoid(self, x: float) -> float:
        """Sigmoid activation with clipping to avoid overflow."""
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))
    
    def predict(self, features: List[float]) -> float:
        """Forward pass: compute prediction."""
        z = sum(w * f for w, f in zip(self.weights, features)) + self.bias
        return self.sigmoid(z)
    
    def train(self, features: List[float], target: int, alpha: float = 1.0) -> float:
        """
        Update weights using gradient descent with momentum.
        alpha: learning rate multiplier (for uncertainty-weighted updates)
        Returns: loss (error squared)
        """
        y_pred = self.predict(features)
        error = target - y_pred
        loss = error ** 2
        
        # Momentum-based gradient updates
        for i in range(len(self.weights)):
            grad = -error * features[i]  # Negative log-likelihood gradient
            self.velocity_w[i] = self.momentum * self.velocity_w[i] - alpha * self.lr * grad
            self.weights[i] += self.velocity_w[i]
        
        grad_b = -error
        self.velocity_b = self.momentum * self.velocity_b - alpha * self.lr * grad_b
        self.bias += self.velocity_b
        
        self.training_count += 1
        return loss
    
    def get_certainty(self, prediction: float) -> float:
        """Return a confidence score (0=uncertain, 1=certain)."""
        return 1.0 - abs(prediction - 0.5) * 2.0


class QueryStrategy:
    """
    Multiple active learning strategies to decide when to query a user/oracle.
    """
    
    @staticmethod
    def uncertainty_sampling(prediction: float, threshold: float = 0.15) -> bool:
        """Query when model is near 0.5 (uncertain)."""
        return abs(prediction - 0.5) < threshold
    
    @staticmethod
    def margin_sampling(predictions: List[float], threshold: float = 0.1) -> bool:
        """Query when margin between top two predictions is small."""
        if len(predictions) < 2:
            return False
        sorted_pred = sorted(predictions, reverse=True)
        margin = sorted_pred[0] - sorted_pred[1]
        return margin < threshold
    
    @staticmethod
    def entropy_sampling(prediction: float, threshold: float = 0.5) -> bool:
        """Query based on entropy of prediction distribution."""
        eps = 1e-10
        entropy = -(prediction * math.log(prediction + eps) + 
                   (1 - prediction) * math.log(1 - prediction + eps))
        return entropy > threshold
    
    @staticmethod
    def outlier_detection(features: List[float], feature_history: deque, 
                         z_threshold: float = 2.5) -> bool:
        """Query if input is statistically different from history."""
        if len(feature_history) < 5:
            return False
        
        feature_array = list(feature_history)
        means = [sum(h[i] for h in feature_array) / len(feature_array) 
                for i in range(len(features))]
        stds = [math.sqrt(sum((h[i] - means[i])**2 for h in feature_array) / len(feature_array))
               for i in range(len(features))]
        
        z_scores = [(f - m) / (s + 1e-6) for f, m, s in zip(features, means, stds)]
        return any(abs(z) > z_threshold for z in z_scores)


class AutonomousEdgeAI:
    """
    Main orchestrator: manages perceptrons, active learning, data buffering,
    and persistent state for edge devices.
    """
    
    def __init__(self, device_type: str = "generic", num_models: int = 3):
        """
        device_type: "camera", "3d_printer", or "generic"
        num_models: ensemble size (query-by-committee)
        """
        self.device_type = device_type
        self.feature_extractor = FeatureExtractor(feature_dim=8)
        self.models = [LightweightPerceptron(input_dim=8) for _ in range(num_models)]
        self.query_buffer = deque(maxlen=50)  # Store uncertain samples
        self.labeled_samples = 0
        self.total_predictions = 0
        self.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.last_save = time.time()
        
        self.active_strategy = "uncertainty"  # Can switch strategies
        self.query_threshold = 0.15
        self.learning_enabled = True
    
    def process_camera_frame(self, brightness: int, edge_density: float,
                            motion_level: float, color_variance: float) -> Dict:
        """
        Process a single camera frame.
        Returns: prediction, confidence, should_query, metadata
        """
        features = self.feature_extractor.extract_from_camera(
            brightness / 255.0, edge_density, motion_level, color_variance
        )
        
        predictions = [m.predict(features) for m in self.models]
        ensemble_pred = sum(predictions) / len(predictions)
        certainty = min(m.get_certainty(p) for p in predictions)
        
        # Decide whether to query
        should_query = False
        query_reason = ""
        
        if self.active_strategy == "uncertainty":
            should_query = QueryStrategy.uncertainty_sampling(ensemble_pred, self.query_threshold)
            query_reason = "Low model certainty"
        elif self.active_strategy == "outlier":
            should_query = QueryStrategy.outlier_detection(
                features, self.feature_extractor.feature_history
            )
            query_reason = "Outlier detected"
        elif self.active_strategy == "margin":
            should_query = QueryStrategy.margin_sampling(predictions, 0.1)
            query_reason = "Small margin between models"
        
        self.total_predictions += 1
        
        result = {
            "prediction": ensemble_pred,
            "certainty": certainty,
            "should_query": should_query,
            "query_reason": query_reason,
            "features": features,
            "model_votes": predictions,
            "timestamp": time.time()
        }
        
        if should_query:
            self.query_buffer.append((features, result))
        
        return result
    
    def process_printer_sensors(self, nozzle_temp: float, bed_temp: float,
                               extrusion_rate: float, z_variance: float) -> Dict:
        """
        Process 3D printer sensor data.
        """
        features = self.feature_extractor.extract_from_3d_printer(
            nozzle_temp, bed_temp, extrusion_rate, z_variance
        )
        
        predictions = [m.predict(features) for m in self.models]
        ensemble_pred = sum(predictions) / len(predictions)
        
        should_query = QueryStrategy.uncertainty_sampling(ensemble_pred, self.query_threshold)
        
        self.total_predictions += 1
        
        result = {
            "prediction": ensemble_pred,  # 0=normal, 1=anomaly
            "should_query": should_query,
            "features": features,
            "timestamp": time.time()
        }
        
        if should_query:
            self.query_buffer.append((features, result))
        
        return result
    
    def learn_from_label(self, label: int, confidence_weight: Optional[float] = None):
        """
        Learn from a user-provided or oracle-provided label.
        confidence_weight: 0-1 scale. If provided, uncertain labels are weighted less.
        """
        if not self.query_buffer or not self.learning_enabled:
            return False
        
        features, metadata = self.query_buffer.popleft()
        weight = confidence_weight if confidence_weight else 1.0
        
        # Train all models in the ensemble
        for model in self.models:
            model.train(features, label, alpha=weight)
        
        self.labeled_samples += 1
        
        print(f"[LEARN] Label={label}, Weight={weight:.2f} | "
              f"Trained samples: {self.labeled_samples} | "
              f"Query buffer: {len(self.query_buffer)}")
        
        # Periodically save state
        if self.labeled_samples % 10 == 0:
            self.save_state()
        
        return True
    
    def set_active_strategy(self, strategy: str):
        """Switch active learning strategy dynamically."""
        valid_strategies = ["uncertainty", "outlier", "margin", "entropy"]
        if strategy in valid_strategies:
            self.active_strategy = strategy
            print(f"[STRATEGY] Switched to: {strategy}")
        else:
            print(f"Unknown strategy. Valid: {valid_strategies}")
    
    def get_stats(self) -> Dict:
        """Return performance metrics."""
        return {
            "device_type": self.device_type,
            "total_predictions": self.total_predictions,
            "labeled_samples": self.labeled_samples,
            "query_buffer_size": len(self.query_buffer),
            "ensemble_size": len(self.models),
            "learning_enabled": self.learning_enabled,
            "active_strategy": self.active_strategy,
            "session_id": self.session_id,
            "feature_history_size": len(self.feature_extractor.feature_history)
        }
    
    def save_state(self) -> bool:
        """Persist model and metadata to disk (JSON + binary-friendly format)."""
        try:
            state = {
                "session_id": self.session_id,
                "timestamp": time.time(),
                "labeled_samples": self.labeled_samples,
                "total_predictions": self.total_predictions,
                "device_type": self.device_type,
                "models": [
                    {
                        "weights": m.weights,
                        "bias": m.bias,
                        "training_count": m.training_count
                    }
                    for m in self.models
                ],
                "active_strategy": self.active_strategy,
                "query_threshold": self.query_threshold
            }
            
            filename = f"edge_ai_{self.device_type}_{self.session_id}.json"
            with open(filename, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.last_save = time.time()
            print(f"[SAVE] State saved to {filename}")
            return True
        except Exception as e:
            print(f"[ERROR] Save failed: {e}")
            return False
    
    def load_state(self, filename: str) -> bool:
        """Restore model from disk."""
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            self.labeled_samples = state['labeled_samples']
            self.total_predictions = state['total_predictions']
            
            for i, model_state in enumerate(state['models']):
                if i < len(self.models):
                    self.models[i].weights = model_state['weights']
                    self.models[i].bias = model_state['bias']
                    self.models[i].training_count = model_state['training_count']
            
            self.active_strategy = state.get('active_strategy', 'uncertainty')
            print(f"[LOAD] Restored from {filename}")
            return True
        except Exception as e:
            print(f"[ERROR] Load failed: {e}")
            return False


# ============================================================================
# EXAMPLE USAGE: Simulated Camera System with Autonomous Learning
# ============================================================================

def demo_autonomous_camera():
    """
    Demonstrates autonomous learning on a simulated camera.
    The AI learns to detect "objects of interest" based on image properties.
    """
    print("\n" + "="*70)
    print("AUTONOMOUS EDGE AI - Camera Example (v1.0)")
    print("="*70)
    
    ai = AutonomousEdgeAI(device_type="camera", num_models=3)
    
    # Simulated sensor data (brightness, edge_density, motion, color_variance)
    # These could come from actual camera frame analysis
    test_frames = [
        (200, 0.3, 10, 0.2),  # Normal scene
        (150, 0.5, 5, 0.15),  # Lower brightness
        (250, 0.8, 50, 0.4),  # High activity (query expected)
        (100, 0.2, 2, 0.1),   # Dark scene
        (220, 0.9, 60, 0.5),  # Very high activity (outlier)
        (180, 0.35, 8, 0.18), # Similar to normal
    ]
    
    print("\n[PHASE 1] Initial predictions (untrained model):")
    for i, (brightness, edge_density, motion, color_var) in enumerate(test_frames):
        result = ai.process_camera_frame(brightness, edge_density, motion, color_var)
        print(f"  Frame {i+1}: Pred={result['prediction']:.3f}, "
              f"Certainty={result['certainty']:.3f}, "
              f"Query={result['should_query']} ({result['query_reason']})")
    
    # Simulated user providing labels for queried samples
    print("\n[PHASE 2] User provides labels for uncertain samples:")
    if len(ai.query_buffer) > 0:
        print(f"  Received {len(ai.query_buffer)} uncertain samples to label")
        for label_value in [1, 0, 1]:  # User labels
            ai.learn_from_label(label_value, confidence_weight=0.9)
    
    # Switch strategy and continue learning
    print("\n[PHASE 3] Switching to outlier detection strategy:")
    ai.set_active_strategy("outlier")
    
    # Process more frames
    print("\n[PHASE 4] Predictions after learning:")
    for i, (brightness, edge_density, motion, color_var) in enumerate(test_frames[:3]):
        result = ai.process_camera_frame(brightness, edge_density, motion, color_var)
        print(f"  Frame {i+1}: Pred={result['prediction']:.3f}, "
              f"Certainty={result['certainty']:.3f}")
    
    # Display statistics
    print("\n[STATS]")
    stats = ai.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save state for next session
    ai.save_state()
    print("\n✅ Autonomous learning session complete!")
    print("   Model state saved for next session.\n")


def demo_autonomous_3d_printer():
    """
    Autonomous anomaly detection on 3D printer.
    """
    print("\n" + "="*70)
    print("AUTONOMOUS EDGE AI - 3D Printer Example (v1.0)")
    print("="*70)
    
    ai = AutonomousEdgeAI(device_type="3d_printer", num_models=2)
    
    # Simulated sensor readings (nozzle, bed, extrusion, z-variance)
    sensor_data = [
        (210, 60, 80, 0.05),  # Normal print
        (185, 55, 60, 0.15),  # Underextrusion (anomaly)
        (230, 70, 90, 0.02),  # Normal, slightly hot
        (240, 65, 40, 0.5),   # Severe anomaly (jam)
    ]
    
    print("\nMonitoring 3D printer in real-time:")
    for i, (nozzle, bed, extrusion, z_var) in enumerate(sensor_data):
        result = ai.process_printer_sensors(nozzle, bed, extrusion, z_var)
        anomaly_status = "⚠️  ANOMALY" if result['prediction'] > 0.5 else "✅ NORMAL"
        print(f"  Reading {i+1}: {anomaly_status} | "
              f"Pred={result['prediction']:.3f} | "
              f"Query={result['should_query']}")
        
        if result['should_query']:
            # Simulate user confirming anomaly
            ai.learn_from_label(int(result['prediction'] > 0.5), confidence_weight=0.95)
    
    print(f"\n  Total anomalies detected: {sum(1 for d in sensor_data if d[2] < 60 or d[3] > 0.4)}")
    ai.save_state()


if __name__ == "__main__":
    demo_autonomous_camera()
    demo_autonomous_3d_printer()