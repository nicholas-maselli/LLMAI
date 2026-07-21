import json
from pathlib import Path
from typing import Any


class MetricsLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(metrics) + "\n")

        readable = " | ".join(
            f"{name} {value:.4g}" if isinstance(value, float) else f"{name} {value}"
            for name, value in metrics.items()
        )
        print(readable, flush=True)
