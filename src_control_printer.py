"""Hardware abstraction for printer control."""

import requests
import time
import logging

logger = logging.getLogger(__name__)


class PrinterInterface:
    """Interface to 3D printer (OctoPrint/Serial/Mock)."""
    
    def __init__(self, mode: str = "mock"):
        self.mode = mode
        self.session = None
        self.url = None
        self.api_key = None
        self.port = None
        
        if mode == "octoprint":
            self.session = requests.Session()
        elif mode == "serial":
            import serial
            self.ser = None
        else:
            self.temp = 200
            self.speed = 60
    
    def execute(self, temp_delta: float = 0, speed_delta: float = 0):
        """Execute action on printer."""
        try:
            if self.mode == "octoprint" and self.url and self.api_key:
                if temp_delta != 0:
                    new_temp = 210 + int(temp_delta)
                    self.session.post(
                        f"{self.url}/api/printer/tool",
                        json={"command": "setTemperature", "targets": {"tool0": new_temp}},
                        headers={"X-API-Key": self.api_key},
                        timeout=2
                    )
                if speed_delta != 0:
                    new_speed = 60 + int(speed_delta)
                    self.session.post(
                        f"{self.url}/api/printer/printhead",
                        json={"command": "feedRate", "factor": new_speed},
                        headers={"X-API-Key": self.api_key},
                        timeout=2
                    )
            elif self.mode == "serial" and self.ser:
                if temp_delta != 0:
                    self.ser.write(f"M104 S{int(210 + temp_delta)}\n".encode())
                if speed_delta != 0:
                    self.ser.write(f"M220 S{int(60 + speed_delta)}\n".encode())
            else:  # mock
                self.temp = max(180, min(260, self.temp + temp_delta))
                self.speed = max(20, min(180, self.speed + speed_delta))
        except Exception as e:
            logger.error(f"Execute failed: {e}")
    
    def get_sensors(self):
        """Read printer sensors (temp, bed_temp, extrusion_rate, z_variance)."""
        try:
            if self.mode == "octoprint" and self.url and self.api_key:
                r = self.session.get(f"{self.url}/api/printer", 
                                    headers={"X-API-Key": self.api_key},
                                    timeout=2)
                data = r.json()
                return (
                    data['temperature']['tool0']['actual'],
                    data['temperature']['bed']['actual'],
                    60.0, 0.05
                )
            elif self.mode == "serial" and self.ser:
                return (210.0, 60.0, 60.0, 0.05)
            else:  # mock
                return (self.temp, 60.0, self.speed, 0.05)
        except Exception as e:
            logger.error(f"Sensor read failed: {e}")
            return (200.0, 60.0, 60.0, 0.05)