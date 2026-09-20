"""
sim_engine — Real-Time Adaptive Cascade Simulation Package for CashVision
"""

from .packages import LightPackage, FullPackage, DetectorOnlyPackage
from .controllers import CascadeController, SingleShotTrigger, CascadeState
from .frame_source import FrameSource, DatasetSequenceGenerator
from .energy_meter import RAPLMeter, CPUEnergyMeter
from .experiment_runner import run_session, run_matrix_benchmark
from .stats_analyzer import (
    compute_session_summary,
    compute_system_aggregate_metrics,
    compute_condition_breakdown,
    compute_denomination_breakdown,
    run_wilcoxon_tests,
    generate_latex_table,
    generate_pareto_data
)
