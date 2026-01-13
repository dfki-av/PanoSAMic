"""
Author: Mahdi Chamseddine
"""

import math
import os
import subprocess


def is_slurm_job() -> bool:
    if os.environ.get("SLURM_JOB_ID", None):
        return True
    return False


def parse_slurm_time_left(time_str: str) -> float:
    if time_str in ["NOT_SET", "UNLIMITED"]:
        return math.inf

    seconds = 0

    # Days component, 86400 is number of seconds in a day
    time_list = time_str.split("-")
    seconds += int(time_list[0]) * 86400 if len(time_list) > 1 else 0

    # Time components, then reverse to get seconds first
    time_list = reversed(time_list[-1].split(":"))

    for i, item in enumerate(time_list):
        seconds += int(item) * (60**i)

    return seconds


def job_time_left() -> float:
    """returns the slurm job time remaining in seconds or math.inf if not a slurm job"""
    if not is_slurm_job():
        return math.inf

    result = subprocess.run(
        args="squeue -h -j $SLURM_JOB_ID -O TimeLeft",
        shell=True,
        text=True,
        capture_output=True,
    )

    return parse_slurm_time_left(result.stdout)
