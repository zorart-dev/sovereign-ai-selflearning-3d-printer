"""
Main Autonomous Training Loop for Sovereign-v5.0
Orchestrates all components: perception, policy, rewards, safety, and federated learning
"""

import logging
import torch
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import sys

from config import Config
from networks import ActorCriticModel
from agent import SovereignAgent
from reward_safety import TemporalVisionReward, AdaptiveSafety
from hardware import create_printer, PrinterInterface
from federated import FederatedLearningNode

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/claude/sovereign_v5_final/sovereign.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SovereignAutonomousSystem:
    """
    Main autonomous training system integrating all components
    """

    def __init__(
        self,
        config_path: str = 'config.json',
        printer_mode: str = 'mock',
        use_federated: bool = False,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize autonomous system
        Args:
            config_path: path to config file
            printer_mode: 'mock', 'octoprint', or 'serial'
            use_federated: enable federated learning
            checkpoint_path: path to resume from checkpoint
        """
        logger.info("=" * 80)
        logger.info("SOVEREIGN AUTONOMOUS SYSTEM v5.0 INITIALIZATION")
        logger.info("=" * 80)
        
        # Load configuration
        self.config = Config.load(config_path) if Path(config_path).exists() else Config()
        logger.info(f"Configuration loaded from {config_path}")
        
        # Initialize printer interface
        self.printer = create_printer(printer_mode)
        logger.info(f"Printer interface initialized in {printer_mode} mode")
        
        # Initialize reward and safety
        self.reward = TemporalVisionReward(self.config)
        self.safety = AdaptiveSafety(self.config)
        
        # Initialize neural network and agent
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        model = ActorCriticModel(self.config).to(device)
        self.agent = SovereignAgent(model, self.config, device)
        
        logger.info(f"Agent initialized (device: {device})")
        
        # Initialize federated learning if enabled
        self.federated = None
        if use_federated:
            self.federated = FederatedLearningNode(
                node_id=self.config.federated.node_id,
                server_url=self.config.federated.server_url,
                sync_frequency=self.config.federated.sync_frequency,
            )
            logger.info("Federated learning enabled")
        
        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_checkpoint(checkpoint_path)
        
        # Training state
        self.episode = 0
        self.step = 0
        self.best_return = float('-inf')
        self.training_start = datetime.now()
        
        logger.info("Autonomous system initialization complete")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        try:
            self.agent.load_checkpoint(checkpoint_path)
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

    def collect_trajectory(self, max_steps: int = 200) -> Tuple[float, Dict]:
        """
        Collect one trajectory/episode
        Args:
            max_steps: maximum steps per episode
        Returns:
            (episode_return, stats_dict)
        """
        state_dict = self.printer.get_state()
        nozzle_temp = state_dict.nozzle_temp
        speed = state_dict.extrusion_speed
        
        # Get vision frame
        frame = self.reward.get_state()  # Returns normalized frame
        
        episode_return = 0.0
        episode_length = 0
        episode_rewards = []
        episode_actions = []
        
        for step_idx in range(max_steps):
            # Stack state
            state = np.concatenate([
                frame.flatten() / 255.0,  # Normalize vision
                np.array([nozzle_temp / 300.0, speed / 180.0])  # Normalize sensors
            ])
            
            # Select action
            action = self.agent.act(state)
            episode_actions.append(action)
            
            # Validate with safety layer
            is_safe, command = self.safety.validate(action, nozzle_temp, speed)
            
            if not is_safe:
                self.safety.report_failure(nozzle_temp)
                reward = -1.0
            else:
                # Execute action
                if command:
                    self.printer.set_temperature(command['temp'], state_dict.bed_temp)
                    self.printer.set_extrusion_speed(command['speed'])
                
                # Get reward
                reward = self.reward.get()
            
            episode_rewards.append(reward)
            episode_return += reward
            
            # Store transition for PPO
            self.agent.store_transition(state, action, reward)
            
            # Update state
            state_dict = self.printer.get_state()
            nozzle_temp = state_dict.nozzle_temp
            speed = state_dict.extrusion_speed
            
            episode_length += 1
            self.step += 1
        
        # Compute GAE and train
        if episode_length > 0:
            self.agent.compute_gae(state)  # Final value estimate
            train_stats = self.agent.train_ppo_step()
        else:
            train_stats = {}
        
        return episode_return, {
            'episode_length': episode_length,
            'episode_rewards': episode_rewards,
            'episode_actions': episode_actions,
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'train_stats': train_stats,
            'safety_stats': self.safety.get_stats(),
        }

    def update_curriculum(self):
        """Update curriculum stage based on training progress"""
        current_stage = self.agent.get_curriculum_stage()
        target_stage = min(
            len(self.config.learning.curriculum_stages) - 1,
            self.step // self.config.learning.curriculum_steps_per_stage
        )
        
        if target_stage > current_stage:
            self.agent.set_curriculum_stage(target_stage)
            logger.info(f"Curriculum advanced to stage {target_stage}")

    def federated_sync(self):
        """Perform federated learning synchronization if enabled"""
        if not self.federated or not self.federated.should_sync():
            return
        
        try:
            # Upload local model
            self.federated.upload_local_model(self.agent.model)
            
            # Download global model
            self.federated.download_global_model(self.agent.model)
            
            self.federated.record_sync()
            logger.info(f"Federated sync complete (step {self.step})")
        except Exception as e:
            logger.error(f"Federated sync failed: {e}")

    def train_episode(self) -> Dict:
        """Train one episode and return statistics"""
        logger.info(f"Episode {self.episode} starting (step {self.step})")
        
        episode_return, stats = self.collect_trajectory()
        
        # Update curriculum
        self.update_curriculum()
        
        # Federated sync
        self.federated_sync()
        
        # Update best return
        if episode_return > self.best_return:
            self.best_return = episode_return
            logger.info(f"New best return: {self.best_return:.3f}")
            self._save_checkpoint(f'checkpoints/best_model.pt')
        
        # Periodic checkpoint
        if self.episode % self.config.learning.checkpoint_interval == 0:
            self._save_checkpoint(f'checkpoints/episode_{self.episode}.pt')
        
        # Log statistics
        self._log_episode_stats(episode_return, stats)
        
        self.episode += 1
        
        return {
            'episode': self.episode,
            'total_steps': self.step,
            'episode_return': episode_return,
            'best_return': self.best_return,
            'time_elapsed': (datetime.now() - self.training_start).total_seconds(),
            **stats,
        }

    def _save_checkpoint(self, path: str):
        """Save training checkpoint"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.agent.save_checkpoint(
                path,
                extra_data={
                    'episode': self.episode,
                    'step': self.step,
                    'best_return': self.best_return,
                    'timestamp': datetime.now().isoformat(),
                    'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
                }
            )
            logger.info(f"Checkpoint saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _log_episode_stats(self, episode_return: float, stats: Dict):
        """Log episode statistics"""
        mean_reward = stats.get('mean_reward', 0.0)
        episode_length = stats.get('episode_length', 0)
        safety_stats = stats.get('safety_stats', {})
        
        logger.info(
            f"Episode {self.episode} finished | "
            f"Return: {episode_return:.3f} | "
            f"Mean Reward: {mean_reward:.3f} | "
            f"Length: {episode_length} | "
            f"Failures: {safety_stats.get('total_failures', 0)}"
        )

    def train(
        self,
        num_episodes: int = 1000,
        save_interval: int = 10,
    ):
        """
        Main training loop
        Args:
            num_episodes: total episodes to train
            save_interval: save checkpoint every N episodes
        """
        logger.info(f"Starting training for {num_episodes} episodes")
        
        try:
            while self.episode < num_episodes:
                stats = self.train_episode()
                
                # Log to file
                if self.episode % save_interval == 0:
                    self._save_training_log(stats)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            self.shutdown()

    def _save_training_log(self, stats: Dict):
        """Save training statistics to log file"""
        log_file = Path('/home/claude/sovereign_v5_final/training_log.json')
        try:
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append({
                'timestamp': datetime.now().isoformat(),
                **stats,
            })
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save training log: {e}")

    def shutdown(self):
        """Cleanup resources"""
        logger.info("Shutting down autonomous system")
        
        # Save final checkpoint
        self._save_checkpoint('checkpoints/final_model.pt')
        
        # Close hardware
        if hasattr(self.printer, 'close'):
            self.printer.close()
        
        # Close camera
        if hasattr(self.reward, 'close'):
            self.reward.close()
        
        logger.info("Shutdown complete")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sovereign Autonomous System v5.0')
    parser.add_argument('--config', type=str, default='config.json', help='Config file path')
    parser.add_argument('--printer', type=str, default='mock', help='Printer mode: mock/octoprint/serial')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--federated', action='store_true', help='Enable federated learning')
    
    args = parser.parse_args()
    
    # Initialize system
    system = SovereignAutonomousSystem(
        config_path=args.config,
        printer_mode=args.printer,
        use_federated=args.federated,
        checkpoint_path=args.checkpoint,
    )
    
    # Train
    system.train(num_episodes=args.episodes)


if __name__ == '__main__':
    main()
