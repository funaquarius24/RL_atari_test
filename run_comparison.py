import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_comparison")

MODELS = ["dqn", "dueling_dqn", "nec"]
BASE_ARGS = [
    "--mode", "train",
    "--env", "BreakoutNoFrameskip-v4",
    "--seed", "42",
    "--lr", "1e-4",
    "--batch-size", "32",
    "--buffer-capacity", "100000",
    "--total-steps", "10000000",
    "--time-limit", "3",
]


def run_agent(model: str, log_dir: str, checkpoint_dir: str, python: str) -> float:
    args = [
        python, "main.py",
        "--model", model,
        "--log-dir", f"{log_dir}/{model}",
        "--checkpoint-dir", checkpoint_dir,
    ] + BASE_ARGS[2:]

    logger.info("=" * 72)
    logger.info("Starting %s  %s", model, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 72)
    logger.info("Command: %s", " ".join(args))
    logger.info("")

    start = time.time()
    result = subprocess.run(args, capture_output=False)
    elapsed = time.time() - start

    if result.returncode == 0:
        logger.info("%s finished in %.0fs (%.2fh)", model, elapsed, elapsed / 3600)
    else:
        logger.error("%s FAILED after %.0fs (returncode=%d)", model, elapsed, result.returncode)

    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Run DQN / Dueling DQN / NEC comparison")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter")
    parser.add_argument("--log-dir", default="runs", help="TensorBoard root directory")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint root directory")
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_dir = f"{args.log_dir}/comparison_{run_timestamp}"
    checkpoint_dir = f"{args.checkpoint_dir}/comparison_{run_timestamp}"

    logger.info("Starting comparison run at %s", run_timestamp)
    logger.info("Models: %s", ", ".join(MODELS))
    logger.info("Log dir: %s", log_dir)
    logger.info("Checkpoint dir: %s", checkpoint_dir)
    logger.info("")

    total_start = time.time()
    times = {}

    for model in MODELS:
        elapsed = run_agent(model, log_dir, checkpoint_dir, args.python)
        times[model] = elapsed

    total_elapsed = time.time() - total_start

    logger.info("")
    logger.info("=" * 72)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 72)
    for model in MODELS:
        logger.info("  %-12s  %.0fs (%.2fh)", model, times[model], times[model] / 3600)
    logger.info("  %-12s  %.0fs (%.2fh)", "TOTAL", total_elapsed, total_elapsed / 3600)


if __name__ == "__main__":
    main()
