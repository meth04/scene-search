"""Optional driver VRAM sampling; baseline does not depend on NVML."""

import threading


class MemorySampler:
    def __init__(self):
        self.peak_mb = None
        self.stop_event = threading.Event()
        self.thread = None
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.read = lambda: pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20
            self.peak_mb = self.read()
        except Exception:
            return
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        while not self.stop_event.wait(0.1):
            try:
                self.peak_mb = max(self.peak_mb, self.read())
            except Exception:
                break

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)
        return self.peak_mb
