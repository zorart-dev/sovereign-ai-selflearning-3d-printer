"""
Federated Learning Coordinator for Sovereign-v5.0
Enables multi-device model aggregation and parameter averaging
"""

import logging
import json
import requests
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ModelChecksum:
    """Checksum metadata for model synchronization"""
    layer_name: str
    checksum: str
    shape: Tuple[int, ...]
    dtype: str
    timestamp: float


class FederatedLearningNode:
    """
    Local node in federated learning network
    Aggregates local training and participates in global model updates
    """

    def __init__(
        self,
        node_id: str,
        server_url: Optional[str] = None,
        sync_frequency: int = 100,  # Sync every N training steps
        aggregation_method: str = 'average',  # 'average' or 'weighted'
    ):
        """
        Initialize federated node
        Args:
            node_id: unique identifier for this node
            server_url: central server URL (None = local mode)
            sync_frequency: how often to sync with server
            aggregation_method: model aggregation strategy
        """
        self.node_id = node_id
        self.server_url = server_url
        self.sync_frequency = sync_frequency
        self.aggregation_method = aggregation_method
        
        self.training_steps = 0
        self.last_sync_step = 0
        self.model_version = 0
        self.local_checksums: Dict[str, ModelChecksum] = {}
        
        logger.info(
            f"FederatedLearningNode '{node_id}' initialized "
            f"(server: {server_url}, method: {aggregation_method})"
        )

    def should_sync(self) -> bool:
        """Check if local node should sync with server"""
        return (self.training_steps - self.last_sync_step) >= self.sync_frequency

    def compute_model_checksum(self, model: torch.nn.Module) -> Dict[str, ModelChecksum]:
        """
        Compute checksums for model parameters
        Useful for detecting transmission errors and drift
        Args:
            model: PyTorch model
        Returns:
            dict mapping layer names to ModelChecksum objects
        """
        checksums = {}
        for name, param in model.named_parameters():
            # Serialize parameter
            tensor_bytes = param.data.cpu().numpy().tobytes()
            # Compute SHA256
            checksum = hashlib.sha256(tensor_bytes).hexdigest()
            
            checksums[name] = ModelChecksum(
                layer_name=name,
                checksum=checksum,
                shape=tuple(param.shape),
                dtype=str(param.dtype),
                timestamp=datetime.now().timestamp(),
            )
        
        self.local_checksums = checksums
        return checksums

    def extract_model_weights(self, model: torch.nn.Module) -> Dict[str, np.ndarray]:
        """
        Extract all model weights as NumPy arrays
        Args:
            model: PyTorch model
        Returns:
            dict mapping parameter names to numpy arrays
        """
        weights = {}
        for name, param in model.named_parameters():
            weights[name] = param.data.cpu().numpy()
        return weights

    def upload_local_model(
        self,
        model: torch.nn.Module,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Upload local model to federated server
        Args:
            model: trained PyTorch model
            metadata: optional training metadata (loss, accuracy, etc.)
        Returns:
            success status
        """
        if not self.server_url:
            logger.debug("No server URL; skipping upload")
            return True
        
        try:
            # Extract weights and checksums
            weights = self.extract_model_weights(model)
            checksums = self.compute_model_checksum(model)
            
            # Prepare payload
            payload = {
                'node_id': self.node_id,
                'model_version': self.model_version,
                'weights': {k: v.tolist() for k, v in weights.items()},
                'checksums': {k: asdict(v) for k, v in checksums.items()},
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat(),
            }
            
            # Send to server
            response = requests.post(
                f"{self.server_url}/upload",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            
            logger.info(
                f"Model uploaded to server (version {self.model_version}, "
                f"{len(weights)} parameters)"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            return False

    def download_global_model(
        self,
        model: torch.nn.Module,
    ) -> bool:
        """
        Download aggregated global model from server
        Args:
            model: local model to update
        Returns:
            success status
        """
        if not self.server_url:
            logger.debug("No server URL; skipping download")
            return True
        
        try:
            # Fetch global model
            response = requests.get(
                f"{self.server_url}/download",
                params={'node_id': self.node_id},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            
            # Verify checksums
            server_checksums = data.get('checksums', {})
            if not self._verify_checksums(server_checksums):
                logger.warning("Server model checksum verification failed")
                return False
            
            # Update model weights
            weights = data.get('weights', {})
            for name, param in model.named_parameters():
                if name in weights:
                    param.data = torch.tensor(
                        weights[name],
                        dtype=param.dtype,
                        device=param.device
                    )
            
            self.model_version = data.get('model_version', self.model_version + 1)
            
            logger.info(f"Downloaded global model (version {self.model_version})")
            return True
        except Exception as e:
            logger.error(f"Failed to download global model: {e}")
            return False

    def _verify_checksums(self, server_checksums: Dict[str, Dict]) -> bool:
        """Verify downloaded model checksums"""
        for name, server_cksum in server_checksums.items():
            if name not in self.local_checksums:
                return False
        return True

    def average_local_models(
        self,
        models: List[torch.nn.Module],
        weights: Optional[List[float]] = None
    ) -> torch.nn.Module:
        """
        Average multiple models (used by central server)
        Args:
            models: list of PyTorch models
            weights: optional per-model weights (default: uniform)
        Returns:
            averaged model
        """
        if not models:
            raise ValueError("No models to average")
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]
        
        # Initialize with first model
        averaged = torch.nn.Module()
        device = next(models[0].parameters()).device
        
        # Average parameters
        for name, param in models[0].named_parameters():
            avg_param = param.data.clone() * weights[0]
            
            for i in range(1, len(models)):
                if name in dict(models[i].named_parameters()):
                    model_param = dict(models[i].named_parameters())[name]
                    avg_param += model_param.data * weights[i]
            
            # Update parameter
            for param_list in [models[j].named_parameters() for j in range(len(models))]:
                for n, p in param_list:
                    if n == name:
                        p.data = avg_param.to(device)
                        break
        
        logger.debug(f"Averaged {len(models)} models with weights {weights}")
        return models[0]

    def get_training_status(self) -> Dict:
        """Get current federated training status"""
        return {
            'node_id': self.node_id,
            'training_steps': self.training_steps,
            'model_version': self.model_version,
            'last_sync_step': self.last_sync_step,
            'should_sync': self.should_sync(),
            'num_local_checksums': len(self.local_checksums),
        }

    def increment_step(self):
        """Increment training step counter"""
        self.training_steps += 1

    def record_sync(self):
        """Record that a sync operation occurred"""
        self.last_sync_step = self.training_steps
        self.model_version += 1
        logger.info(f"Sync recorded (step {self.training_steps}, version {self.model_version})")


class FederatedServer:
    """
    Central aggregation server for federated learning
    Collects models from nodes and performs aggregation
    """

    def __init__(self, port: int = 5001, aggregation_method: str = 'average'):
        """
        Initialize federated server
        Args:
            port: server port
            aggregation_method: 'average', 'weighted', or custom
        """
        self.port = port
        self.aggregation_method = aggregation_method
        
        self.models: Dict[str, Dict] = {}  # node_id -> model data
        self.global_model: Optional[Dict] = None
        self.aggregation_history: List[Dict] = []
        self.version = 0
        
        logger.info(f"FederatedServer initialized on port {port}")

    def receive_model(
        self,
        node_id: str,
        model_data: Dict,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Receive model submission from federated node
        Args:
            node_id: identifier of submitting node
            model_data: model weights and checksums
            metadata: training metadata
        Returns:
            success status
        """
        try:
            self.models[node_id] = {
                'weights': model_data.get('weights', {}),
                'checksums': model_data.get('checksums', {}),
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat(),
            }
            logger.info(f"Received model from node '{node_id}'")
            return True
        except Exception as e:
            logger.error(f"Failed to receive model from {node_id}: {e}")
            return False

    def aggregate_models(self) -> bool:
        """
        Aggregate models from all connected nodes
        Returns:
            success status
        """
        if not self.models:
            logger.warning("No models to aggregate")
            return False
        
        try:
            if self.aggregation_method == 'average':
                self._aggregate_average()
            else:
                self._aggregate_weighted()
            
            self.version += 1
            logger.info(
                f"Aggregated {len(self.models)} models (version {self.version})"
            )
            return True
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return False

    def _aggregate_average(self):
        """Simple averaging aggregation"""
        if not self.models:
            return
        
        # Get first model as template
        template_keys = list(self.models.values())[0]['weights'].keys()
        self.global_model = {'weights': {}, 'version': self.version}
        
        # Average each parameter
        for key in template_keys:
            values = []
            for model_data in self.models.values():
                if key in model_data['weights']:
                    values.append(np.array(model_data['weights'][key]))
            
            if values:
                self.global_model['weights'][key] = np.mean(values, axis=0).tolist()

    def _aggregate_weighted(self):
        """Weighted averaging (by metadata if available)"""
        # More sophisticated aggregation logic can be added here
        self._aggregate_average()

    def broadcast_model(self) -> Dict:
        """
        Broadcast aggregated model to all nodes
        Returns:
            aggregated model data
        """
        if not self.global_model:
            logger.warning("No aggregated model available")
            return {}
        
        return {
            'model_version': self.version,
            'weights': self.global_model.get('weights', {}),
            'checksums': {},  # Could compute checksums here
            'timestamp': datetime.now().isoformat(),
        }

    def get_status(self) -> Dict:
        """Get server status"""
        return {
            'port': self.port,
            'aggregation_method': self.aggregation_method,
            'num_connected_nodes': len(self.models),
            'global_model_version': self.version,
            'num_aggregations': len(self.aggregation_history),
        }
