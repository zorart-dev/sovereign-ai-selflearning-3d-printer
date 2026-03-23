"""
Hardware Interface for Sovereign-v5.0
Supports mock, OctoPrint, and serial-based 3D printer control
"""

import logging
import json
import requests
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class PrinterState:
    """Current state of the 3D printer"""
    nozzle_temp: float
    bed_temp: float
    extrusion_speed: float
    print_progress: float
    is_printing: bool
    error_state: Optional[str] = None


class PrinterInterface(ABC):
    """Abstract base class for printer control"""

    @abstractmethod
    def get_state(self) -> PrinterState:
        """Get current printer state"""
        pass

    @abstractmethod
    def set_temperature(self, nozzle: float, bed: float) -> bool:
        """Set nozzle and bed temperatures"""
        pass

    @abstractmethod
    def set_extrusion_speed(self, speed: float) -> bool:
        """Set extrusion speed as percentage (0-200)"""
        pass

    @abstractmethod
    def emergency_stop(self) -> bool:
        """Emergency stop"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to printer"""
        pass

    def close(self):
        """Cleanup resources"""
        pass


class MockPrinter(PrinterInterface):
    """Mock printer for testing without hardware"""

    def __init__(self):
        self.nozzle_temp = 210.0
        self.bed_temp = 60.0
        self.extrusion_speed = 100.0
        self.print_progress = 0.0
        self.is_printing = False
        self._target_nozzle = 210.0
        self._target_bed = 60.0
        self._connected = True
        self.step_count = 0
        
        logger.info("MockPrinter initialized")

    def get_state(self) -> PrinterState:
        """Simulate temperature gradual changes"""
        # Simulate thermal dynamics
        self.nozzle_temp += (self._target_nozzle - self.nozzle_temp) * 0.05
        self.bed_temp += (self._target_bed - self.bed_temp) * 0.02
        
        # Simulate print progress
        if self.is_printing:
            self.print_progress += 0.001
            if self.print_progress >= 1.0:
                self.is_printing = False
                self.print_progress = 1.0
        
        self.step_count += 1
        
        return PrinterState(
            nozzle_temp=self.nozzle_temp,
            bed_temp=self.bed_temp,
            extrusion_speed=self.extrusion_speed,
            print_progress=self.print_progress,
            is_printing=self.is_printing,
        )

    def set_temperature(self, nozzle: float, bed: float) -> bool:
        """Set target temperatures"""
        self._target_nozzle = float(nozzle)
        self._target_bed = float(bed)
        logger.debug(f"Set temperatures: nozzle={nozzle:.1f}, bed={bed:.1f}")
        return True

    def set_extrusion_speed(self, speed: float) -> bool:
        """Set extrusion speed"""
        self.extrusion_speed = float(speed)
        logger.debug(f"Set extrusion speed: {speed:.1f}%")
        return True

    def emergency_stop(self) -> bool:
        """Stop printing and cool down"""
        self.is_printing = False
        self._target_nozzle = 25.0
        self._target_bed = 25.0
        logger.warning("Emergency stop triggered")
        return True

    def is_connected(self) -> bool:
        """Always connected in mock mode"""
        return self._connected


class OctoPrintInterface(PrinterInterface):
    """OctoPrint API interface for networked printers"""

    def __init__(self, url: str, api_key: str, timeout: int = 10):
        """
        Initialize OctoPrint connection
        Args:
            url: OctoPrint server URL (e.g., http://192.168.1.100:5000)
            api_key: OctoPrint API key
            timeout: request timeout in seconds
        """
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {
            'X-API-Key': api_key,
            'Content-Type': 'application/json',
        }
        
        self._connected = False
        self._check_connection()
        
        if self._connected:
            logger.info(f"OctoPrintInterface connected to {url}")
        else:
            logger.error(f"Failed to connect to OctoPrint at {url}")

    def _check_connection(self) -> bool:
        """Verify connection to OctoPrint"""
        try:
            response = requests.get(
                f"{self.url}/api/version",
                headers=self.headers,
                timeout=self.timeout
            )
            self._connected = response.status_code == 200
            return self._connected
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection check failed: {e}")
            self._connected = False
            return False

    def get_state(self) -> PrinterState:
        """Fetch current printer state from OctoPrint"""
        try:
            response = requests.get(
                f"{self.url}/api/printer",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            state = data.get('state', {})
            temps = data.get('temperature', {})
            progress = data.get('progress', {})
            
            return PrinterState(
                nozzle_temp=temps.get('tool0', {}).get('actual', 0.0),
                bed_temp=temps.get('bed', {}).get('actual', 0.0),
                extrusion_speed=100.0,  # Not directly available in OctoPrint
                print_progress=progress.get('completion', 0.0) / 100.0,
                is_printing=state.get('flags', {}).get('printing', False),
            )
        except Exception as e:
            logger.error(f"Failed to get printer state: {e}")
            return PrinterState(
                nozzle_temp=0.0, bed_temp=0.0, extrusion_speed=0.0,
                print_progress=0.0, is_printing=False, error_state=str(e)
            )

    def set_temperature(self, nozzle: float, bed: float) -> bool:
        """Set nozzle and bed temperatures"""
        try:
            payload = {
                'command': 'target',
                'targets': {
                    'tool0': float(nozzle),
                    'bed': float(bed)
                }
            }
            response = requests.post(
                f"{self.url}/api/printer/tool",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.debug(f"Temperature set: nozzle={nozzle:.1f}, bed={bed:.1f}")
            return True
        except Exception as e:
            logger.error(f"Failed to set temperature: {e}")
            return False

    def set_extrusion_speed(self, speed: float) -> bool:
        """Set extrusion speed via feedrate"""
        try:
            payload = {
                'command': 'feedrate',
                'factor': float(speed) / 100.0  # Convert percentage to factor
            }
            response = requests.post(
                f"{self.url}/api/printer/tool",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.debug(f"Extrusion speed set: {speed:.1f}%")
            return True
        except Exception as e:
            logger.error(f"Failed to set extrusion speed: {e}")
            return False

    def emergency_stop(self) -> bool:
        """Perform emergency stop"""
        try:
            payload = {'command': 'emergency'}
            response = requests.post(
                f"{self.url}/api/printer/printhead",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.warning("Emergency stop sent to OctoPrint")
            return True
        except Exception as e:
            logger.error(f"Failed to execute emergency stop: {e}")
            return False

    def is_connected(self) -> bool:
        """Check if connected to OctoPrint"""
        return self._check_connection()


class SerialPrinter(PrinterInterface):
    """Direct serial interface for Marlin-based printers (not fully implemented)"""

    def __init__(self, port: str, baudrate: int = 115200):
        """
        Initialize serial printer connection
        Args:
            port: serial port (e.g., /dev/ttyUSB0 or COM3)
            baudrate: communication speed
        """
        self.port = port
        self.baudrate = baudrate
        self._connected = False
        
        try:
            import serial
            self.serial = serial.Serial(port, baudrate, timeout=1)
            self._connected = True
            logger.info(f"Serial printer connected on {port}")
        except ImportError:
            logger.error("pyserial not installed. Cannot use SerialPrinter.")
            self.serial = None
        except Exception as e:
            logger.error(f"Failed to connect to serial port {port}: {e}")
            self.serial = None

    def _send_command(self, command: str) -> bool:
        """Send G-code command to printer"""
        if not self._connected or not self.serial:
            return False
        
        try:
            self.serial.write(f"{command}\n".encode())
            return True
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False

    def get_state(self) -> PrinterState:
        """Read current printer state (M105 = temp report)"""
        # This is a simplified stub
        return PrinterState(
            nozzle_temp=0.0, bed_temp=0.0, extrusion_speed=100.0,
            print_progress=0.0, is_printing=False,
            error_state="SerialPrinter state reading not fully implemented"
        )

    def set_temperature(self, nozzle: float, bed: float) -> bool:
        """Set temperatures via G-code"""
        self._send_command(f"M104 S{int(nozzle)}")  # Set nozzle
        self._send_command(f"M140 S{int(bed)}")      # Set bed
        return True

    def set_extrusion_speed(self, speed: float) -> bool:
        """Set feedrate via G-code"""
        self._send_command(f"M220 S{int(speed)}")
        return True

    def emergency_stop(self) -> bool:
        """Send emergency stop command"""
        return self._send_command("M112")

    def is_connected(self) -> bool:
        """Check connection status"""
        return self._connected

    def close(self):
        """Close serial connection"""
        if self.serial and self.serial.is_open:
            self.serial.close()
            logger.info("Serial connection closed")


def create_printer(mode: str, **kwargs) -> PrinterInterface:
    """
    Factory function to create appropriate printer interface
    Args:
        mode: 'mock', 'octoprint', or 'serial'
        **kwargs: additional arguments for the specific printer type
    Returns:
        PrinterInterface instance
    """
    if mode == 'mock':
        return MockPrinter()
    elif mode == 'octoprint':
        return OctoPrintInterface(
            url=kwargs.get('url', 'http://localhost:5000'),
            api_key=kwargs.get('api_key', ''),
            timeout=kwargs.get('timeout', 10)
        )
    elif mode == 'serial':
        return SerialPrinter(
            port=kwargs.get('port', '/dev/ttyUSB0'),
            baudrate=kwargs.get('baudrate', 115200)
        )
    else:
        raise ValueError(f"Unknown printer mode: {mode}")
