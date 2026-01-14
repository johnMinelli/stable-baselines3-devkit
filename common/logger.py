import os
import sys
from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime
from multiprocessing import Process
from time import time
from typing import Any, Dict, TextIO, Tuple, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from stable_baselines3.common.logger import (
    CSVOutputFormat,
    Figure,
    HParam,
    HumanOutputFormat,
    Image,
    JSONOutputFormat,
    KVWriter,
)
from stable_baselines3.common.logger import Logger as SB3Logger
from stable_baselines3.common.logger import TensorBoardOutputFormat, Video
from tqdm import tqdm

from common.utils import dump_yaml

wandb = None


class InlineOutputFormat(HumanOutputFormat):
    def __init__(self, filename_or_file: Union[str, TextIO], max_length: int = 36):
        self.progress_bar = None
        super().__init__(filename_or_file, max_length)

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Tuple[str, ...]], step: int = 0) -> None:
        output_parts = []

        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):
            if (excluded is not None and ("stdout" in excluded or "log" in excluded)) or "time/" in key:
                continue
            if any(isinstance(value, t) for t in (Video, Figure, Image, HParam)):
                continue

            value_str = f"{value:.3g}" if isinstance(value, float) else str(value)
            key = key.split("/")[-1].strip() if "/" in key else key
            output_parts.append(f"{key}={value_str}")

        # Create tab-separated output
        output = "\t| ".join(output_parts)

        time_info = {k: v for k, v in key_values.items() if "time/" in k}
        timer_info = {k.replace("timers/", ""): f"{v:.3f}s" for k, v in key_values.items() if "timers/" in k}

        if time_info or timer_info:
            info_parts = []
            info_parts.extend([f"{k}: {v}" for k, v in timer_info.items()])
            if "time/fps" in time_info:
                info_parts.append(f"SimFPS: {time_info['time/fps']}")
            if "time/time_elapsed" in time_info:
                info_parts.append(f"ATD: {time_info['time/time_elapsed']} s")
            if info_parts:
                output = " (" + ", ".join(info_parts) + ")\t" + output

        if tqdm is not None and hasattr(self.file, "name") and self.file.name == "<stdout>":
            # Write on the same line as tqdm
            tqdm.write(output + "\n", file=sys.stdout, end="")
        else:
            self.file.write(output + "\n")

        self.file.flush()


class WandbOutputFormat(KVWriter):
    """
    Dumps key/value pairs into Wandb dashboard.
    """

    def __init__(self):
        self._is_closed = False

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Tuple[str, ...]], step: int = 0) -> None:
        assert not self._is_closed, "The SummaryWriter was closed, please re-create one."
        metrics = {}
        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):
            if excluded is not None and "wandb" in excluded:
                continue
            if isinstance(value, (torch.Tensor, np.ndarray)):
                continue
            if isinstance(value, Image):
                continue
            if isinstance(value, np.ScalarType) and not isinstance(value, str):
                metrics[key] = value

        wandb.log(metrics, step=step)

    def close(self) -> None:
        """
        closes the file
        """
        self._is_closed = True


def make_output_format(_format: str, log_dir: str, log_suffix: str = "") -> KVWriter:
    """
    return a logger for the requested format

    :param _format: the requested format to log to ('stdout', 'log', 'json' or 'csv' or 'tensorboard')
    :param log_dir: the logging directory
    :param log_suffix: the suffix for the log file
    :return: the logger
    """
    os.makedirs(log_dir, exist_ok=True)
    if _format == "stdout":
        return InlineOutputFormat(sys.stdout)
    if _format == "log":
        return HumanOutputFormat(os.path.join(log_dir, f"log{log_suffix}.txt"))
    if _format == "json":
        return JSONOutputFormat(os.path.join(log_dir, f"progress{log_suffix}.json"))
    if _format == "csv":
        return CSVOutputFormat(os.path.join(log_dir, f"progress{log_suffix}.csv"))
    if _format == "tensorboard":
        return TensorBoardOutputFormat(log_dir)
    if _format == "wandb":
        return WandbOutputFormat()

    raise ValueError(f"Unknown format specified: {_format}")


class Logger(SB3Logger):
    logger_context: ContextVar = ContextVar("logger", default=None)

    @staticmethod
    def get_logger():
        logger = Logger.logger_context.get()
        if logger is None:
            print("Logger not initialized")
            logger = Logger()
        return logger

    def __init__(self, args_cli=None, log_root="logs", log_folder=None):
        # self.dt = args_cli.dt
        self.device = args_cli.device if args_cli else "cpu"
        self.tb = args_cli.tensorboard if args_cli else False
        self.wb = args_cli.wandb if args_cli else False
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.num_episodes = 0
        self.timers: Dict[str, float] = {}
        self.timer_results: Dict[str, float] = {}
        self.log_dir = os.path.join(
            log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if log_folder is None else log_folder
        )

        # setup remote loggers
        format_strings = ["stdout"]
        if self.tb:
            format_strings.append("tensorboard")
        if self.wb:
            format_strings.append("wandb")

            try:
                global wandb
                import wandb
            except ImportError:
                assert wandb is not None, "wandb is not installed, you can use `pip install wandb` to do so"

            resume_id = None
            if os.path.exists("wandb") and args_cli.resume:
                wandb_logs = sorted(
                    filter(lambda x: "run-" in x, os.listdir("wandb")),
                    key=lambda x: os.path.getmtime(os.path.join("wandb", x)),
                )
                if args_cli.wandb_run == "-1" and len(wandb_logs) > 0:
                    resume_id = wandb_logs[-1].split("-")[-1]
                elif args_cli.wandb_run != "-1":
                    resume_id = args_cli.wandb_run
            wandb.init(settings=wandb.Settings(start_method="thread"), project="sb3-devkit", id=resume_id, resume=resume_id is not None, allow_val_change=True, name=args_cli.experiment_name)

        # setup superclass (StableBaseline3 Logger)
        output_formats = [make_output_format(f, self.log_dir, "") for f in format_strings]
        super().__init__(self.log_dir, output_formats)

        self.logger_context.set(self)

    def clear(self):
        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()
        self.timer_results.clear()

    def log_hp(self, hp_set: dict, path: str):
        dump_yaml(path, hp_set)
        if self.wb:
            wandb.config.update(hp_set, allow_val_change=True)

    def start_timer(self, id: str):
        """Start a timer with the given ID"""
        self.timers[id] = time()

    def stop_timer(self, id: str, div=1.0):
        """Stop the timer with the given ID and store the elapsed time"""
        if id not in self.timers:
            raise KeyError(f"Timer '{id}' was never started")
        elapsed = time() - self.timers[id]
        self.timer_results[id] = elapsed / div
        del self.timers[id]
        return elapsed

    def dump(self, step: int = 0):
        # edit the rew and state dict at necessity
        for timer_id, elapsed in self.timer_results.items():
            self.record(f"timers/{timer_id}", elapsed)

        return super().dump(step)

    def dump_async(self, step: int = 0):
        return super().dump(step)

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        nb_rows = 3
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        # plot joint targets and measured positions
        a = axs[1, 0]
        if log["dof_pos"]:
            a.plot(time, log["dof_pos"], label="measured")
        if log["dof_pos_target"]:
            a.plot(time, log["dof_pos_target"], label="target")
        a.set(xlabel="time [s]", ylabel="Position [rad]", title="DOF Position")
        a.legend()
        # plot joint velocity
        a = axs[1, 1]
        if log["dof_vel"]:
            a.plot(time, log["dof_vel"], label="measured")
        if log["dof_vel_target"]:
            a.plot(time, log["dof_vel_target"], label="target")
        a.set(xlabel="time [s]", ylabel="Velocity [rad/s]", title="Joint Velocity")
        a.legend()
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]:
            a.plot(time, log["base_vel_x"], label="measured")
        if log["command_x"]:
            a.plot(time, log["command_x"], label="commanded")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity x")
        a.legend()
        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]:
            a.plot(time, log["base_vel_y"], label="measured")
        if log["command_y"]:
            a.plot(time, log["command_y"], label="commanded")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity y")
        a.legend()
        # plot base vel yaw
        a = axs[0, 2]
        if log["base_vel_yaw"]:
            a.plot(time, log["base_vel_yaw"], label="measured")
        if log["command_yaw"]:
            a.plot(time, log["command_yaw"], label="commanded")
        a.set(xlabel="time [s]", ylabel="base ang vel [rad/s]", title="Base velocity yaw")
        a.legend()
        # plot base vel z
        a = axs[1, 2]
        if log["base_vel_z"]:
            a.plot(time, log["base_vel_z"], label="measured")
        a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity z")
        a.legend()
        # plot contact forces
        a = axs[2, 0]
        if log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f"force {i}")
        a.set(xlabel="time [s]", ylabel="Forces z [N]", title="Vertical Contact forces")
        a.legend()
        # plot torque/vel curves
        a = axs[2, 1]
        if log["dof_vel"] != [] and log["dof_torque"] != []:
            a.plot(log["dof_vel"], log["dof_torque"], "x", label="measured")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        a.legend()
        # plot torques
        a = axs[2, 2]
        if log["dof_torque"] != []:
            a.plot(time, log["dof_torque"], label="measured")
        a.set(xlabel="time [s]", ylabel="Joint Torque [Nm]", title="Torque")
        a.legend()
        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
