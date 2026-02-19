"""Training logger with CSV export for experiment tracking."""
import os
import csv
import logging
import time
from datetime import datetime


class TrainingLogger:
    """Logs metrics to console + CSV + file.

    Creates in log_dir:
        - {exp_name}_{timestamp}.log  — full text log
        - {exp_name}_{timestamp}.csv  — epoch metrics for plotting
    """

    def __init__(self, log_dir="logs", exp_name="experiment"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"{exp_name}_{ts}.csv")
        self.log_path = os.path.join(log_dir, f"{exp_name}_{ts}.log")

        # Setup logger
        self.logger = logging.getLogger(f"{exp_name}_{ts}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        fh = logging.FileHandler(self.log_path)
        fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        self.logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(sh)

        self._csv_init = False
        self.start_time = time.time()

    def log(self, msg):
        self.logger.info(msg)

    def log_epoch(self, metrics):
        """Log one epoch's metrics to CSV and console."""
        if not self._csv_init:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                writer.writeheader()
            self._csv_init = True

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writerow(metrics)

        parts = " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        )
        self.logger.info(parts)

    def elapsed(self):
        return time.time() - self.start_time
