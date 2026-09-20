import time
import os
import sys
from typing import Optional
import psutil


class RAPLMeter:
    """
    Measures true CPU energy consumption via Intel RAPL sysfs interface on Linux.
    """
    RAPL_PATH = "/sys/class/powercap/intel-rapl:0/energy_uj"

    def __init__(self):
        self.available = os.path.exists(self.RAPL_PATH)
        self.start_uj = 0
        self.end_uj = 0
        self.start_t = 0.0
        self.end_t = 0.0

    def _read_uj(self) -> int:
        if not self.available:
            return 0
        try:
            with open(self.RAPL_PATH, "r") as f:
                return int(f.read().strip())
        except Exception:
            return 0

    def __enter__(self):
        self.start_uj = self._read_uj()
        self.start_t = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_uj = self._read_uj()
        self.end_t = time.perf_counter()

    @property
    def energy_joules(self) -> float:
        if not self.available:
            return 0.0
        diff = self.end_uj - self.start_uj
        # Handle wrap around
        if diff < 0:
            diff += 2**32
        return float(diff) / 1_000_000.0


class CPUEnergyMeter:
    """
    Cross-platform Energy Proxy Meter (Windows / macOS / Linux).
    Estimates energy via CPU utilization and user-specified TDP:
        Energy (Joules) = Duration (s) * (Avg_CPU_Percent / 100.0) * TDP (Watts)
    Also tracks peak RAM usage (MB).
    """
    def __init__(self, tdp_watts: float = 28.0):
        self.tdp_watts = tdp_watts
        self.process = psutil.Process(os.getpid())
        self.cpu_samples = []
        self.ram_samples = []
        self.start_t = 0.0
        self.end_t = 0.0
        # Initialize psutil cpu measurement
        psutil.cpu_percent(interval=None)

    def __enter__(self):
        self.cpu_samples.clear()
        self.ram_samples.clear()
        self.start_t = time.perf_counter()
        psutil.cpu_percent(interval=None)
        return self

    def record_step(self):
        """Records CPU and RAM at the end of each frame cycle."""
        try:
            cpu = psutil.cpu_percent(interval=None)
            ram = self.process.memory_info().rss / (1024 * 1024)
            self.cpu_samples.append(cpu)
            self.ram_samples.append(ram)
        except Exception:
            pass

    def __exit__(self, *args):
        self.end_t = time.perf_counter()

    @property
    def duration_seconds(self) -> float:
        return max(0.001, self.end_t - self.start_t)

    @property
    def avg_cpu_percent(self) -> float:
        if not self.cpu_samples:
            return 0.0
        return float(sum(self.cpu_samples) / len(self.cpu_samples))

    @property
    def peak_ram_mb(self) -> float:
        if not self.ram_samples:
            return self.process.memory_info().rss / (1024 * 1024)
        return float(max(self.ram_samples))

    @property
    def energy_joules(self) -> float:
        avg_power_watts = (self.avg_cpu_percent / 100.0) * self.tdp_watts
        return float(avg_power_watts * self.duration_seconds)
