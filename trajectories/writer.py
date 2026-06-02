"""Export trajectories to JSONL, CSV, and optional Parquet."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class TrajectoryWriter:
    def __init__(self, output_dir: str = "results/trajectories") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(
        self,
        steps: List[Dict[str, Any]],
        filename: str,
    ) -> str:
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for step in steps:
                f.write(json.dumps(step, default=str) + "\n")
        return str(path)

    def write_csv(
        self,
        steps: List[Dict[str, Any]],
        filename: str,
    ) -> str:
        path = self.output_dir / filename
        if not steps:
            path.touch()
            return str(path)

        flat_rows = []
        for step in steps:
            flat_rows.append({
                "week": step.get("week"),
                "agent_role": step.get("agent_role"),
                "action": step.get("action"),
                "reward": step.get("reward"),
                "done": step.get("done"),
                "policy_type": step.get("policy_type"),
                "model_name": step.get("model_name"),
                "tool_order": step.get("tool_order"),
                "llm_order": step.get("llm_order"),
                "difference": step.get("difference"),
                "consensus_gap": step.get("consensus_gap"),
                "state_json": json.dumps(step.get("state", {})),
                "next_state_json": json.dumps(step.get("next_state", {})),
                "info_json": json.dumps(step.get("info", {})),
            })

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=flat_rows[0].keys())
            writer.writeheader()
            writer.writerows(flat_rows)
        return str(path)

    def write_parquet(
        self,
        steps: List[Dict[str, Any]],
        filename: str,
    ) -> Optional[str]:
        try:
            import pandas as pd
        except ImportError:
            return None

        path = self.output_dir / filename
        df = pd.json_normalize(steps)
        df.to_parquet(path, index=False)
        return str(path)

    def write_all(
        self,
        steps: List[Dict[str, Any]],
        stem: str,
    ) -> Dict[str, str]:
        paths = {
            "jsonl": self.write_jsonl(steps, f"{stem}.jsonl"),
            "csv": self.write_csv(steps, f"{stem}.csv"),
        }
        parquet_path = self.write_parquet(steps, f"{stem}.parquet")
        if parquet_path:
            paths["parquet"] = parquet_path
        return paths
