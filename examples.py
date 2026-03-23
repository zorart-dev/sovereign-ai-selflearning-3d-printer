"""
Example scripts demonstrating Sovereign-v5.0 usage patterns
Ready-to-run examples for common use cases
"""

import logging
from pathlib import Path
from main import SovereignAutonomousSystem
from config import Config
from hardware import create_printer
from agent import SovereignAgent
from networks import ActorCriticModel
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# EXAMPLE 1: Quick Start with Mock Printer
# ============================================================================

def example_quick_start():
    """
    Simplest possible example: train for a few episodes with mock printer
    
    Usage:
        python -c "from examples import example_quick_start; example_quick_start()"
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Quick Start with Mock Printer")
    logger.info("=" * 80)
    
    # Create system with default config
    system = SovereignAutonomousSystem(
        printer_mode='mock',
        use_federated=False,
    )
    
    # Train for 10 episodes
    logger.info("Starting 10-episode training...")
    system.train(num_episodes=10, save_interval=5)
    
    logger.info("Training complete! Check sovereign.log for details.")


# ============================================================================
# EXAMPLE 2: Resume from Checkpoint
# ============================================================================

def example_resume_training():
    """
    Resume training from a previously saved checkpoint
    
    Useful for:
    - Continuing long training runs
    - Fine-tuning on different hardware
    - Recovering from interruptions
    
    Usage:
        python -c "from examples import example_resume_training; example_resume_training()"
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 2: Resume Training from Checkpoint")
    logger.info("=" * 80)
    
    checkpoint_path = 'checkpoints/best_model.pt'
    
    if not Path(checkpoint_path).exists():
        logger.warning(f"Checkpoint not found at {checkpoint_path}")
        logger.info("Train first with example_quick_start()")
        return
    
    # Create system and load checkpoint
    system = SovereignAutonomousSystem(
        printer_mode='mock',
        checkpoint_path=checkpoint_path,
    )
    
    logger.info(f"Resumed from checkpoint: {checkpoint_path}")
    logger.info(f"Starting from episode {system.episode}, step {system.step}")
    
    # Continue training for more episodes
    system.train(num_episodes=50, save_interval=10)


# ============================================================================
# EXAMPLE 3: Custom Configuration
# ============================================================================

def example_custom_config():
    """
    Train with a custom configuration
    
    Demonstrates:
    - Modifying learning rates
    - Adjusting hardware constraints
    - Changing curriculum settings
    
    Usage:
        python -c "from examples import example_custom_config; example_custom_config()"
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 3: Custom Configuration")
    logger.info("=" * 80)
    
    # Load default config and modify
    config = Config()
    
    # Increase learning rate for faster convergence
    config.learning.learning_rate = 1e-3
    
    # Relax temperature constraints for exploration
    config.hardware.nozzle_min = 170
    config.hardware.nozzle_max = 270
    
    # Reduce curriculum steps
    config.learning.curriculum_steps_per_stage = 500
    
    # Save custom config
    config_path = 'custom_config.json'
    config.save(config_path)
    logger.info(f"Custom config saved to {config_path}")
    
    # Create system with custom config
    system = SovereignAutonomousSystem(
        config_path=config_path,
        printer_mode='mock',
    )
    
    logger.info("Training with custom configuration...")
    system.train(num_episodes=20, save_interval=5)


# ============================================================================
# EXAMPLE 4: Direct Agent Usage
# ============================================================================

def example_agent_direct_usage():
    """
    Use the agent directly for inference or custom training loops
    
    Demonstrates:
    - Creating agent instances
    - Inference/action selection
    - Manual training loops
    
    Usage:
        python -c "from examples import example_agent_direct_usage; example_agent_direct_usage()"
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 4: Direct Agent Usage")
    logger.info("=" * 80)
    
    # Setup
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and agent
    model = ActorCriticModel(config).to(device)
    agent = SovereignAgent(model, config, device)
    
    logger.info(f"Agent created on device: {device}")
    
    # Collect some transitions
    logger.info("Collecting 100 transitions...")
    total_reward = 0.0
    
    for step in range(100):
        # Create random state
        state = np.random.randn(64)
        
        # Select action
        action = agent.act(state)
        
        # Simulate reward
        reward = np.random.randn() * 0.5
        total_reward += reward
        
        # Store transition
        agent.store_transition(state, action, reward)
        
        if (step + 1) % 20 == 0:
            logger.info(f"  Step {step+1}: Action={action}, Cumulative Reward={total_reward:.3f}")
    
    # Perform training update
    logger.info("Performing PPO training update...")
    next_state = np.random.randn(64)
    agent.compute_gae(next_state)
    train_stats = agent.train_ppo_step()
    
    logger.info(f"Training complete. Stats: {train_stats}")
    
    # Save checkpoint
    agent.save_checkpoint('examples/agent_checkpoint.pt')
    logger.info("Checkpoint saved to examples/agent_checkpoint.pt")


# ============================================================================
# EXAMPLE 5: Hardware Interface Testing
# ============================================================================

def example_hardware_testing():
    """
    Test different hardware interfaces
    
    Demonstrates:
    - Mock printer for testing
    - Printer state management
    - Command execution
    
    Usage:
        python -c "from examples import example_hardware_testing; example_hardware_testing()"
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 5: Hardware Interface Testing")
    logger.info("=" * 80)
    
    # Create mock printer
    printer = create_printer('mock')
    logger.info("Created mock printer")
    
    # Simulate print sequence
    logger.info("\nSimulating print sequence:")
    
    for step in range(10):
        # Get state
        state = printer.get_state()
        logger.info(
            f"Step {step}: "
            f"Nozzle={state.nozzle_temp:.1f}°C, "
            f"Bed={state.bed_temp:.1f}°C, "
            f"Progress={state.print_progress:.1%}"
        )
        
        # Adjust temperatures
        if state.nozzle_temp < 210:
            printer.set_temperature(220, 60)
        
        # Adjust speed
        if step % 3 == 0:
            printer.set_extrusion_speed(100 + step * 5)
    
    logger.info("Hardware test complete")


# ============================================================================
# EXAMPLE 6: Federated Learning Setup
# ============================================================================

def example_federated_learning():
    """
    Setup and test federated learning
    
    Demonstrates:
    - Multi-node federated training
    - Model synchronization
    - Aggregation
    
    Usage:
        python -c "from examples import example_federated_learning; example_federated_learning()"
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 6: Federated Learning Setup")
    logger.info("=" * 80)
    
    from federated import FederatedLearningNode, FederatedServer
    
    # Create federated server
    server = FederatedServer(port=5001, aggregation_method='average')
    logger.info(f"Federated server created: {server.get_status()}")
    
    # Create multiple federated nodes
    nodes = []
    for i in range(3):
        node = FederatedLearningNode(
            node_id=f'printer_{i}',
            sync_frequency=10,
        )
        nodes.append(node)
        logger.info(f"Created federated node: {node.node_id}")
    
    # Simulate training and sync
    logger.info("\nSimulating federated training...")
    
    config = Config()
    
    for sync_round in range(3):
        logger.info(f"\n--- Federated Sync Round {sync_round + 1} ---")
        
        # Each node trains locally
        for node in nodes:
            # Simulate training steps
            for _ in range(10):
                node.increment_step()
            
            logger.info(
                f"Node {node.node_id}: "
                f"Steps={node.training_steps}, "
                f"Version={node.model_version}"
            )
        
        # Simulate model aggregation
        logger.info("Aggregating models...")
        server.aggregate_models()
        logger.info(f"Server status: {server.get_status()}")
    
    logger.info("Federated learning demo complete")


# ============================================================================
# EXAMPLE 7: Metrics and Monitoring
# ============================================================================

def example_monitoring():
    """
    Setup monitoring and metrics collection
    
    Demonstrates:
    - Accessing training metrics
    - Safety statistics
    - Performance monitoring
    
    Usage:
        python -c "from examples import example_monitoring; example_monitoring()"
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 7: Monitoring and Metrics")
    logger.info("=" * 80)
    
    # Create system
    system = SovereignAutonomousSystem(printer_mode='mock')
    
    logger.info("Training with metrics collection...")
    
    metrics_history = []
    
    for episode in range(5):
        stats = system.train_episode()
        
        # Extract key metrics
        metrics = {
            'episode': stats['episode'],
            'return': stats['episode_return'],
            'best_return': stats['best_return'],
            'length': stats['episode_length'],
            'mean_reward': stats['mean_reward'],
        }
        
        metrics_history.append(metrics)
        
        logger.info(
            f"Episode {episode}: "
            f"Return={metrics['return']:.3f}, "
            f"Length={metrics['length']}, "
            f"Best={metrics['best_return']:.3f}"
        )
    
    # Calculate statistics
    returns = [m['return'] for m in metrics_history]
    logger.info(f"\nMetrics Summary:")
    logger.info(f"  Mean Return: {np.mean(returns):.3f}")
    logger.info(f"  Std Return:  {np.std(returns):.3f}")
    logger.info(f"  Max Return:  {np.max(returns):.3f}")
    logger.info(f"  Min Return:  {np.min(returns):.3f}")
    
    system.shutdown()


# ============================================================================
# EXAMPLE 8: Inference Mode
# ============================================================================

def example_inference():
    """
    Load trained model and run inference
    
    Demonstrates:
    - Loading checkpoints
    - Inference without training
    - Action evaluation
    
    Usage:
        python -c "from examples import example_inference; example_inference()"
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 8: Inference Mode")
    logger.info("=" * 80)
    
    checkpoint_path = 'checkpoints/best_model.pt'
    
    if not Path(checkpoint_path).exists():
        logger.warning(f"Checkpoint not found at {checkpoint_path}")
        logger.info("Train first with example_quick_start()")
        return
    
    # Setup
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and load model
    model = ActorCriticModel(config).to(device)
    agent = SovereignAgent(model, config, device)
    agent.load_checkpoint(checkpoint_path)
    
    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    
    # Run inference
    logger.info("\nRunning inference on 20 random states...")
    
    with torch.no_grad():
        for i in range(20):
            state = np.random.randn(64)
            action = agent.act(state)
            logger.info(f"State {i}: Action={action}")


# ============================================================================
# EXAMPLE 9: Configuration Generation
# ============================================================================

def example_config_generation():
    """
    Generate different configurations for different scenarios
    
    Demonstrates:
    - High-precision mode
    - Fast convergence mode
    - Conservative mode
    
    Usage:
        python -c "from examples import example_config_generation; example_config_generation()"
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 9: Configuration Generation")
    logger.info("=" * 80)
    
    # High precision mode
    config_precision = Config()
    config_precision.learning.learning_rate = 1e-4
    config_precision.learning.entropy_coef = 0.001
    config_precision.hardware.safe_margin = 5.0
    config_precision.save('config_precision.json')
    logger.info("Saved: config_precision.json (high precision mode)")
    
    # Fast convergence mode
    config_fast = Config()
    config_fast.learning.learning_rate = 5e-3
    config_fast.learning.entropy_coef = 0.05
    config_fast.hardware.safe_margin = 15.0
    config_fast.save('config_fast.json')
    logger.info("Saved: config_fast.json (fast convergence mode)")
    
    # Conservative mode
    config_conservative = Config()
    config_conservative.learning.learning_rate = 1e-5
    config_conservative.learning.entropy_coef = 0.01
    config_conservative.hardware.safe_margin = 2.0
    config_conservative.hardware.nozzle_min = 190
    config_conservative.hardware.nozzle_max = 240
    config_conservative.save('config_conservative.json')
    logger.info("Saved: config_conservative.json (conservative mode)")


# ============================================================================
# EXAMPLE 10: Batch Evaluation
# ============================================================================

def example_batch_evaluation():
    """
    Evaluate model performance across multiple episodes
    
    Demonstrates:
    - Running multiple episodes
    - Collecting statistics
    - Performance analysis
    
    Usage:
        python -c "from examples import example_batch_evaluation; example_batch_evaluation()"
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 10: Batch Evaluation")
    logger.info("=" * 80)
    
    checkpoint_path = 'checkpoints/best_model.pt'
    
    if not Path(checkpoint_path).exists():
        logger.warning(f"Checkpoint not found at {checkpoint_path}")
        logger.info("Train first with example_quick_start()")
        return
    
    # Load system
    system = SovereignAutonomousSystem(
        printer_mode='mock',
        checkpoint_path=checkpoint_path,
    )
    
    logger.info("Running 10 evaluation episodes...")
    
    returns = []
    lengths = []
    
    for episode in range(10):
        episode_return, stats = system.collect_trajectory(max_steps=200)
        returns.append(episode_return)
        lengths.append(stats['episode_length'])
        
        logger.info(
            f"Episode {episode}: "
            f"Return={episode_return:.3f}, "
            f"Length={stats['episode_length']}"
        )
    
    # Statistics
    logger.info("\n=== Evaluation Results ===")
    logger.info(f"Mean Return:  {np.mean(returns):.3f} ± {np.std(returns):.3f}")
    logger.info(f"Mean Length:  {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    logger.info(f"Best Return:  {np.max(returns):.3f}")
    logger.info(f"Worst Return: {np.min(returns):.3f}")
    
    system.shutdown()


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == '__main__':
    import sys
    
    examples = {
        '1': ('Quick Start', example_quick_start),
        '2': ('Resume Training', example_resume_training),
        '3': ('Custom Config', example_custom_config),
        '4': ('Agent Direct Usage', example_agent_direct_usage),
        '5': ('Hardware Testing', example_hardware_testing),
        '6': ('Federated Learning', example_federated_learning),
        '7': ('Monitoring', example_monitoring),
        '8': ('Inference', example_inference),
        '9': ('Config Generation', example_config_generation),
        '10': ('Batch Evaluation', example_batch_evaluation),
    }
    
    print("\n" + "=" * 80)
    print("SOVEREIGN-V5.0 EXAMPLES")
    print("=" * 80)
    print("\nAvailable examples:\n")
    
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    print("\nUsage:")
    print("  python examples.py <number>")
    print("  python -c \"from examples import example_quick_start; example_quick_start()\"")
    print("\n" + "=" * 80 + "\n")
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num in examples:
            name, func = examples[example_num]
            logger.info(f"Running: {name}")
            func()
        else:
            logger.error(f"Unknown example: {example_num}")
            sys.exit(1)
    else:
        logger.info("Run with example number: python examples.py <1-10>")
