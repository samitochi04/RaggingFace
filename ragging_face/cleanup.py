import os
import time

DATA_ROOT = os.path.join(os.path.dirname(__file__), "datasets", "results")


def clean_old_data(days: int = 1):
    """Remove files older than `days` from the datasets/results folder."""
    cutoff = time.time() - days * 86400
    for root, dirs, files in os.walk(DATA_ROOT):
        for fname in files:
            path = os.path.join(root, fname)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime < cutoff:
                try:
                    os.remove(path)
                    print(f"Removed old file: {path}")
                except OSError:
                    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cleanup old uploaded data from Ragging Face dataset directories.")
    parser.add_argument("--days", type=int, default=1, help="Remove files older than this many days")
    args = parser.parse_args()
    clean_old_data(days=args.days)
