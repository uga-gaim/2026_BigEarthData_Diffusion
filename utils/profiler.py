import time

class Profiler:
    def __init__(self):
        self.checkpoints = []  # List of (label, timestamp)
        self.start_time = time.time()
        self.last_time = self.start_time

    def checkpoint(self, label):
        now = time.time()
        self.checkpoints.append((label, now))
        self.last_time = now

    def print_summary(self):
        print("--- Profiler Summary ---")
        prev_time = self.start_time
        if not self.checkpoints:
            print("No checkpoints recorded.")
            return
        total = self.checkpoints[-1][1] - self.start_time
        for label, t in self.checkpoints:
            duration = t - prev_time
            percent = (duration / total * 100) if total > 0 else 0
            print(f"{label}: {duration:.3f} seconds ({percent:.1f}%)")
            prev_time = t
        print(f"Total time: {total:.3f} seconds") 