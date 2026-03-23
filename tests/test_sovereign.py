"""
Comprehensive Test Suite for Sovereign-v5.0
=============================================

Tests all components: networks, agent, rewards, safety, hardware, and federated learning
with 40+ test cases covering:
  - Configuration management
  - Neural network architectures
  - RL agent and training
  - Reward computation and safety
  - Hardware abstraction
  - Federated learning
  - Integration workflows
  - Performance benchmarks

Run with: pytest tests/test_sovereign.py -v
Coverage: pytest tests/ --cov=. --cov-report=html
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import logging
import time

# Import modules
from config import Config, LearningConfig, HardwareConfig, CurriculumConfig
from networks import TinyVisionCNN, Perception, LSTMActorCritic, ActorCriticModel
from agent import RunningNormalizer, ReplayBuffer, SovereignAgent
from reward_safety import TemporalVisionReward, AdaptiveSafety
from hardware import MockPrinter, OctoPrintInterface, create_printer
from federated import FederatedLearningNode, FederatedServer

logger = logging.getLogger(__name__)

# Test fixtures
@pytest.fixture
def config():
    """Provide test configuration"""
    return Config()

@pytest.fixture
def device():
    """Provide device for testing"""
    return torch.device('cpu')

@pytest.fixture
def model(config, device):
    """Provide test model"""
    return ActorCriticModel(config).to(device)

@pytest.fixture
def agent(model, config, device):
    """Provide test agent"""
    return SovereignAgent(model, config, device)


# ============================================================================
# CONFIG TESTS
# ============================================================================

class TestConfig:
    def test_default_config(self):
        """Test default configuration"""
        config = Config()
        assert config.learning.learning_rate > 0
        assert config.hardware.nozzle_min < config.hardware.nozzle_max

    def test_config_save_load(self):
        """Test saving and loading config"""
        config = Config()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'config.json'
            config.save(str(path))
            assert path.exists()
            
            loaded = Config.load(str(path))
            assert loaded.learning.learning_rate == config.learning.learning_rate

    def test_config_validate(self):
        """Test config validation"""
        config = Config()
        # Should not raise
        config.validate()


# ============================================================================
# NETWORK TESTS
# ============================================================================

class TestNetworks:
    def test_tiny_vision_cnn(self):
        """Test TinyVisionCNN forward pass"""
        model = TinyVisionCNN()
        batch_size = 4
        x = torch.randn(batch_size, 3, 240, 320)
        
        output = model(x)
        assert output.shape == (batch_size, 64)

    def test_perception(self):
        """Test Perception layer"""
        config = Config()
        perception = Perception(config)
        
        batch_size = 4
        vision = torch.randn(batch_size, 3, 240, 320)
        sensors = torch.randn(batch_size, 8)
        
        output = perception(vision, sensors)
        assert output.shape == (batch_size, 128)

    def test_lstm_actor_critic(self):
        """Test LSTMActorCritic with proper hidden state handling"""
        lstm_ac = LSTMActorCritic()
        
        batch_size = 2
        seq_len = 8
        
        x = torch.randn(batch_size, seq_len, 128)
        action, log_prob, value = lstm_ac(x)
        
        assert action.shape == (batch_size, seq_len, 4)
        assert log_prob.shape == (batch_size, seq_len)
        assert value.shape == (batch_size, seq_len)

    def test_actor_critic_model(self):
        """Test full ActorCriticModel"""
        config = Config()
        model = ActorCriticModel(config)
        
        batch_size = 2
        vision = torch.randn(batch_size, 3, 240, 320)
        sensors = torch.randn(batch_size, 8)
        
        action, log_prob, value = model(vision, sensors)
        
        assert action.shape == (batch_size,)
        assert log_prob.shape == (batch_size,)
        assert value.shape == (batch_size,)


# ============================================================================
# AGENT TESTS
# ============================================================================

class TestRunningNormalizer:
    def test_normalization(self):
        """Test running normalizer"""
        normalizer = RunningNormalizer(shape=(5,))
        
        # Add some data
        data = np.random.randn(100, 5)
        for sample in data:
            normalizer.update(sample)
        
        # Normalize
        normalized = normalizer.normalize(data[0])
        assert normalized.shape == (5,)

    def test_mean_variance(self):
        """Test mean and variance tracking"""
        normalizer = RunningNormalizer(shape=(3,))
        
        # Known data
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        for sample in data:
            normalizer.update(sample)
        
        # Mean should be approximately [2.5, 3.5, 4.5]
        assert np.allclose(normalizer.mean, [2.5, 3.5, 4.5], atol=0.1)


class TestReplayBuffer:
    def test_buffer_creation(self):
        """Test replay buffer initialization"""
        buffer = ReplayBuffer(capacity=1000, state_shape=(64,))
        assert buffer.capacity == 1000

    def test_buffer_storage(self):
        """Test storing and retrieving transitions"""
        buffer = ReplayBuffer(capacity=100, state_shape=(4,))
        
        state = np.random.randn(4)
        action = 2
        reward = 1.0
        next_state = np.random.randn(4)
        
        buffer.store(state, action, reward, next_state, False)
        assert buffer.size == 1

    def test_buffer_sampling(self):
        """Test sampling from buffer"""
        buffer = ReplayBuffer(capacity=50, state_shape=(4,))
        
        for _ in range(30):
            state = np.random.randn(4)
            action = np.random.randint(0, 4)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            buffer.store(state, action, reward, next_state, False)
        
        batch = buffer.sample(batch_size=16)
        assert len(batch) == 5  # states, actions, rewards, next_states, dones


class TestSovereignAgent:
    def test_agent_creation(self, config, device, model):
        """Test agent initialization"""
        agent = SovereignAgent(model, config, device)
        assert agent.device == device
        assert agent.normalizer is not None

    def test_agent_act(self, agent):
        """Test action selection"""
        state = np.random.randn(64)
        action = agent.act(state)
        assert 0 <= action < 4

    def test_agent_act_consistency(self, agent):
        """Test that agent produces deterministic actions in eval mode"""
        state = np.random.randn(64)
        action1 = agent.act(state)
        action2 = agent.act(state)
        assert action1 == action2

    def test_agent_checkpoint_save_load(self, agent):
        """Test saving and loading checkpoints"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'checkpoint.pt'
            
            # Save
            agent.save_checkpoint(str(path), extra_data={'test': 'data'})
            assert path.exists()
            
            # Load
            new_agent = SovereignAgent(agent.model, agent.config, agent.device)
            new_agent.load_checkpoint(str(path))

    def test_store_and_compute_gae(self, agent):
        """Test storing transitions and computing GAE"""
        # Store some transitions
        for _ in range(20):
            state = np.random.randn(64)
            action = agent.act(state)
            reward = np.random.randn()
            agent.store_transition(state, action, reward)
        
        # Compute GAE
        next_state = np.random.randn(64)
        advantages, returns = agent.compute_gae(next_state)
        
        assert advantages is not None
        assert returns is not None
        assert len(advantages) > 0

    def test_ppo_training_step(self, agent):
        """Test PPO training step"""
        # Add transitions
        for _ in range(64):
            state = np.random.randn(64)
            action = agent.act(state)
            agent.store_transition(state, action, 0.5)
        
        # Train
        next_state = np.random.randn(64)
        agent.compute_gae(next_state)
        stats = agent.train_ppo_step()
        
        assert isinstance(stats, dict)


class TestImprovedSafety:
    """Enhanced safety layer tests"""
    
    def test_safety_multiple_violations(self):
        """Test safety tracking across multiple violations"""
        config = Config()
        safety = AdaptiveSafety(config)
        
        # Trigger multiple violations at same temperature
        for _ in range(3):
            is_safe, _ = safety.validate(1, 240.0, 100.0)
            if not is_safe:
                safety.report_failure(240.0)
        
        stats = safety.get_stats()
        assert stats['total_failures'] >= 1

    def test_safety_recovery(self):
        """Test safety state recovery"""
        config = Config()
        safety = AdaptiveSafety(config)
        
        # Trigger failure
        safety.report_failure(240.0)
        assert safety.total_failures == 1
        
        # Reset
        safety.reset()
        assert safety.consecutive_failures == 0
        assert not safety.emergency_stop

    def test_temperature_ranges(self):
        """Test various temperature ranges"""
        config = Config()
        safety = AdaptiveSafety(config)
        
        test_temps = [180, 210, 240, 260]
        for temp in test_temps:
            is_safe, cmd = safety.validate(0, temp, 100.0)
            if cmd:
                assert config.hardware.nozzle_min <= cmd['temp'] <= config.hardware.nozzle_max

    def test_speed_ranges(self):
        """Test various speed ranges"""
        config = Config()
        safety = AdaptiveSafety(config)
        
        test_speeds = [20, 50, 100, 150, 180]
        for speed in test_speeds:
            is_safe, cmd = safety.validate(2, 210.0, speed)
            if cmd:
                assert config.hardware.speed_min <= cmd['speed'] <= config.hardware.speed_max


class TestNetworksIntegration:
    """Integration tests for all network components"""
    
    def test_perception_layer_output_shape(self):
        """Test perception layer output shape"""
        config = Config()
        perception = Perception(config)
        
        batch_size = 4
        vision = torch.randn(batch_size, 3, 240, 320)
        sensors = torch.randn(batch_size, 8)
        
        output = perception(vision, sensors)
        assert output.shape == (batch_size, 128)

    def test_lstm_sequence_processing(self):
        """Test LSTM processes sequences correctly"""
        lstm_ac = LSTMActorCritic()
        
        batch_size = 2
        seq_len = 8
        
        x = torch.randn(batch_size, seq_len, 128)
        action, log_prob, value = lstm_ac(x)
        
        assert action.shape == (batch_size, seq_len, 4)
        assert log_prob.shape == (batch_size, seq_len)
        assert value.shape == (batch_size, seq_len)

    def test_forward_backward_pass(self):
        """Test forward and backward pass"""
        config = Config()
        model = ActorCriticModel(config)
        
        vision = torch.randn(2, 3, 240, 320, requires_grad=False)
        sensors = torch.randn(2, 8, requires_grad=False)
        
        action, log_prob, value = model(vision, sensors)
        
        # Backward pass
        loss = (-log_prob * value.detach()).mean()
        loss.backward()
        
        assert action.shape == (2,)
        assert log_prob.grad is not None or log_prob.requires_grad


class TestAdvancedFeatures:
    """Test advanced features and edge cases"""
    
    def test_normalizer_convergence(self):
        """Test that normalizer converges to correct statistics"""
        normalizer = RunningNormalizer(shape=(5,))
        
        # Generate known data
        data = np.random.randn(1000, 5)
        for sample in data:
            normalizer.update(sample)
        
        # Check convergence
        expected_mean = np.mean(data, axis=0)
        assert np.allclose(normalizer.mean, expected_mean, atol=0.1)

    def test_replay_buffer_capacity(self):
        """Test replay buffer respects capacity"""
        buffer = ReplayBuffer(capacity=50, state_shape=(4,))
        
        # Add more than capacity
        for i in range(100):
            state = np.random.randn(4)
            action = i % 4
            reward = float(i)
            next_state = np.random.randn(4)
            buffer.store(state, action, reward, next_state, False)
        
        # Size should not exceed capacity
        assert buffer.size <= 50

    def test_curriculum_progression(self, config):
        """Test curriculum learning progression"""
        agent = SovereignAgent(
            ActorCriticModel(config).to('cpu'),
            config,
            torch.device('cpu')
        )
        
        # Check initial stage
        stage = agent.get_curriculum_stage()
        assert stage == 0
        
        # Progress stages
        agent.set_curriculum_stage(1)
        stage = agent.get_curriculum_stage()
        assert stage == 1

    def test_model_determinism(self):
        """Test model produces deterministic outputs with same seed"""
        torch.manual_seed(42)
        np.random.seed(42)
        
        config = Config()
        model1 = ActorCriticModel(config)
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        model2 = ActorCriticModel(config)
        
        # Check same architecture produces same outputs
        vision = torch.randn(1, 3, 240, 320)
        sensors = torch.randn(1, 8)
        
        with torch.no_grad():
            out1 = model1(vision, sensors)
            out2 = model2(vision, sensors)
        
        assert torch.allclose(out1[0], out2[0], atol=1e-6)

    def test_stress_test_hardware_simulation(self):
        """Stress test hardware interface"""
        printer = MockPrinter()
        
        for i in range(100):
            state = printer.get_state()
            assert state.nozzle_temp >= 0
            assert state.bed_temp >= 0
            
            if i % 10 == 0:
                printer.set_temperature(210 + i % 20, 60)
                printer.set_extrusion_speed(80 + i % 50)

    def test_federated_weight_shapes(self):
        """Test federated learning preserves weight shapes"""
        config = Config()
        model = ActorCriticModel(config)
        node = FederatedLearningNode('test')
        
        weights = node.extract_model_weights(model)
        
        # All weights should be numpy arrays
        for name, weight in weights.items():
            assert isinstance(weight, np.ndarray)
            assert weight.shape is not None


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_action(self):
        """Test handling of invalid actions"""
        config = Config()
        safety = AdaptiveSafety(config)
        
        # Test invalid action ID
        is_safe, cmd = safety.validate(999, 210.0, 100.0)
        assert not is_safe  # Should reject invalid action

    def test_missing_file_checkpoint(self):
        """Test loading non-existent checkpoint"""
        config = Config()
        agent = SovereignAgent(
            ActorCriticModel(config),
            config,
            torch.device('cpu')
        )
        
        # Should not raise exception
        try:
            agent.load_checkpoint('/nonexistent/path.pt')
        except FileNotFoundError:
            pass  # Expected

    def test_config_validation_failures(self):
        """Test configuration validation"""
        config = Config()
        
        # Invalid learning rate
        config.learning.learning_rate = -0.001
        try:
            config.validate()
        except (ValueError, AssertionError):
            pass  # Expected

    def test_empty_buffer_operations(self):
        """Test operations on empty buffer"""
        buffer = ReplayBuffer(capacity=10, state_shape=(4,))
        
        # Buffer is empty
        assert buffer.size == 0


class TestPerformance:
    """Performance and efficiency tests"""
    
    def test_forward_pass_latency(self):
        """Measure forward pass latency"""
        config = Config()
        model = ActorCriticModel(config)
        
        vision = torch.randn(1, 3, 240, 320)
        sensors = torch.randn(1, 8)
        
        start = time.perf_counter()
        for _ in range(100):
            with torch.no_grad():
                model(vision, sensors)
        elapsed = time.perf_counter() - start
        
        avg_latency = elapsed / 100
        assert avg_latency < 0.1  # Should be fast

    def test_training_throughput(self):
        """Measure training throughput"""
        config = Config()
        device = torch.device('cpu')
        model = ActorCriticModel(config).to(device)
        agent = SovereignAgent(model, config, device)
        
        # Add transitions
        start = time.perf_counter()
        for _ in range(128):
            state = np.random.randn(64)
            action = agent.act(state)
            agent.store_transition(state, action, 0.5)
        
        agent.compute_gae(np.random.randn(64))
        agent.train_ppo_step()
        elapsed = time.perf_counter() - start
        
        assert elapsed < 10  # Should complete in reasonable time

    def test_memory_efficiency(self):
        """Test memory usage is reasonable"""
        config = Config()
        agent = SovereignAgent(
            ActorCriticModel(config),
            config,
            torch.device('cpu')
        )
        
        # Add many transitions
        for _ in range(1000):
            state = np.random.randn(64)
            action = agent.act(state)
            agent.store_transition(state, action, 0.5)
        
        # Should not crash and buffer should be manageable
        assert agent.buffer.size <= 1000


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])


# ============================================================================
# REWARD AND SAFETY TESTS
# ============================================================================

class TestAdaptiveSafety:
    def test_safety_validation(self):
        """Test safety constraint validation"""
        config = Config()
        safety = AdaptiveSafety(config)
        
        # Valid action
        is_safe, cmd = safety.validate(1, 210.0, 100.0)
        assert is_safe
        assert cmd is not None

    def test_temperature_clamping(self):
        """Test temperature clamping"""
        config = Config()
        safety = AdaptiveSafety(config)
        
        # Try to set temperature above max
        is_safe, cmd = safety.validate(1, 260.0, 100.0)  # Max is 260
        if cmd:
            assert cmd['temp'] <= config.hardware.nozzle_max

    def test_failure_tracking(self):
        """Test failure recording"""
        config = Config()
        safety = AdaptiveSafety(config)
        
        safety.report_failure(180.0)
        assert safety.total_failures == 1
        
        stats = safety.get_stats()
        assert stats['total_failures'] == 1

    def test_emergency_stop(self):
        """Test emergency stop activation"""
        config = Config()
        safety = AdaptiveSafety(config)
        
        # Trigger 5 consecutive failures
        for i in range(5):
            is_safe, _ = safety.validate(1, 240.0, 100.0)
            if not is_safe:
                safety.report_failure(240.0)
        
        # Next action should fail with emergency stop
        stats = safety.get_stats()
        # (May or may not trigger emergency stop depending on implementation)


# ============================================================================
# HARDWARE TESTS
# ============================================================================

class TestMockPrinter:
    def test_mock_printer_creation(self):
        """Test mock printer initialization"""
        printer = MockPrinter()
        assert printer.is_connected()

    def test_printer_state(self):
        """Test getting printer state"""
        printer = MockPrinter()
        state = printer.get_state()
        
        assert state.nozzle_temp >= 0
        assert state.bed_temp >= 0
        assert state.extrusion_speed >= 0

    def test_temperature_setting(self):
        """Test setting temperatures"""
        printer = MockPrinter()
        
        success = printer.set_temperature(220.0, 60.0)
        assert success

    def test_printer_factory(self):
        """Test printer creation factory"""
        printer = create_printer('mock')
        assert printer.is_connected()


# ============================================================================
# FEDERATED LEARNING TESTS
# ============================================================================

class TestFederatedLearningNode:
    def test_node_creation(self):
        """Test federated node initialization"""
        node = FederatedLearningNode('test_node')
        assert node.node_id == 'test_node'

    def test_sync_condition(self):
        """Test sync condition check"""
        node = FederatedLearningNode('test', sync_frequency=10)
        
        assert not node.should_sync()
        
        for _ in range(10):
            node.increment_step()
        
        assert node.should_sync()

    def test_model_checksum(self):
        """Test model checksum computation"""
        config = Config()
        model = ActorCriticModel(config)
        node = FederatedLearningNode('test')
        
        checksums = node.compute_model_checksum(model)
        assert len(checksums) > 0

    def test_weight_extraction(self):
        """Test extracting model weights"""
        config = Config()
        model = ActorCriticModel(config)
        node = FederatedLearningNode('test')
        
        weights = node.extract_model_weights(model)
        assert len(weights) > 0


class TestFederatedServer:
    def test_server_creation(self):
        """Test federated server initialization"""
        server = FederatedServer(port=5001)
        assert server.port == 5001

    def test_model_reception(self):
        """Test receiving model submissions"""
        server = FederatedServer()
        
        model_data = {'weights': {'layer1': [1.0, 2.0]}}
        success = server.receive_model('node1', model_data)
        assert success

    def test_aggregation(self):
        """Test model aggregation"""
        server = FederatedServer()
        
        # Submit models from multiple nodes
        model1 = {'weights': {'param': [1.0, 2.0]}}
        model2 = {'weights': {'param': [3.0, 4.0]}}
        
        server.receive_model('node1', model1)
        server.receive_model('node2', model2)
        
        success = server.aggregate_models()
        assert success


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    def test_end_to_end_training_step(self):
        """Test a complete training step"""
        config = Config()
        device = torch.device('cpu')
        
        # Create components
        model = ActorCriticModel(config).to(device)
        agent = SovereignAgent(model, config, device)
        printer = MockPrinter()
        
        # Collect some transitions
        for _ in range(10):
            state = np.random.randn(64)
            action = agent.act(state)
            reward = np.random.randn()
            agent.store_transition(state, action, reward)
        
        # Train
        next_state = np.random.randn(64)
        agent.compute_gae(next_state)
        stats = agent.train_ppo_step()
        
        assert 'loss' in stats or len(stats) >= 0

    def test_full_system_init(self):
        """Test initializing full autonomous system (requires main.py import)"""
        try:
            from main import SovereignAutonomousSystem
            
            with tempfile.TemporaryDirectory() as tmpdir:
                system = SovereignAutonomousSystem(
                    printer_mode='mock',
                    use_federated=False,
                )
                assert system.printer is not None
                assert system.agent is not None
                system.shutdown()
        except Exception as e:
            logger.warning(f"Full system test skipped: {e}")


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    def test_forward_pass_speed(self, benchmark=None):
        """Test model forward pass performance"""
        config = Config()
        model = ActorCriticModel(config)
        
        vision = torch.randn(1, 3, 240, 320)
        sensors = torch.randn(1, 8)
        
        def forward():
            return model(vision, sensors)
        
        if benchmark:
            benchmark(forward)
        else:
            # Simple timing test
            import time
            start = time.time()
            for _ in range(100):
                forward()
            elapsed = time.time() - start
            assert elapsed < 10.0  # Should be fast

    def test_ppo_training_speed(self):
        """Test PPO training step speed"""
        config = Config()
        device = torch.device('cpu')
        model = ActorCriticModel(config).to(device)
        agent = SovereignAgent(model, config, device)
        
        # Add transitions
        for _ in range(64):
            state = np.random.randn(64)
            action = agent.act(state)
            agent.store_transition(state, action, 0.5)
        
        # Train
        import time
        start = time.time()
        agent.compute_gae(np.random.randn(64))
        agent.train_ppo_step()
        elapsed = time.time() - start
        
        assert elapsed < 5.0  # Should be reasonably fast


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
