from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml


class DefenseType(StrEnum):
    NODE_HARDENING = "node_hardening"
    EDGE_RESTRICTION = "edge_restriction"
    NODE_ISOLATION = "node_isolation"


@dataclass(frozen=True)
class DefenseAction:
    id: str
    action_type: DefenseType
    target: str | tuple[str, str]
    cost: float
    effect: float
    name: str
    response_delay: int = 0


def load_actions_from_config(path: str | Path) -> list[DefenseAction]:
    """Load defensible, offline simulation actions from a YAML config file."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    records = payload["actions"] if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("Defense action config must contain an actions list")
    actions: list[DefenseAction] = []
    for record in records:
        action_type = DefenseType(record["action_type"])
        target: str | tuple[str, str]
        if action_type == DefenseType.EDGE_RESTRICTION:
            raw_target = record["target"]
            if not isinstance(raw_target, list) or len(raw_target) != 2:
                raise ValueError(f"Edge restriction {record['id']} requires a two-node target")
            target = (str(raw_target[0]), str(raw_target[1]))
        else:
            target = str(record["target"])
        actions.append(
            DefenseAction(
                id=str(record["id"]),
                action_type=action_type,
                target=target,
                cost=float(record["cost"]),
                effect=float(record["effect"]),
                name=str(record["name"]),
                response_delay=int(record.get("response_delay", 0)),
            )
        )
    if any(action.response_delay < 0 for action in actions):
        raise ValueError("response_delay must be non-negative")
    if len({action.id for action in actions}) != len(actions):
        raise ValueError("Defense action IDs must be unique")
    return actions


def reference_actions() -> list[DefenseAction]:
    return [
        DefenseAction("D01", DefenseType.NODE_HARDENING, "Web-01", 2, 0.55, "Patch Web-01"),
        DefenseAction("D02", DefenseType.EDGE_RESTRICTION, ("API-01", "DB-01"), 2, 0.70, "Restrict API-01 → DB-01"),
        DefenseAction("D03", DefenseType.NODE_HARDENING, "VPN-01", 2, 0.48, "Harden VPN-01"),
        DefenseAction("D04", DefenseType.EDGE_RESTRICTION, ("Office-01", "Admin-01"), 2, 0.62, "Restrict Office-01 → Admin-01"),
        DefenseAction("D05", DefenseType.NODE_ISOLATION, "Office-01", 4, 1.0, "Isolate Office-01"),
        DefenseAction("D06", DefenseType.NODE_HARDENING, "Admin-01", 3, 0.45, "Harden Admin-01"),
        DefenseAction("D07", DefenseType.EDGE_RESTRICTION, ("Admin-01", "DB-01"), 2, 0.72, "Restrict Admin-01 → DB-01"),
        DefenseAction("D08", DefenseType.NODE_HARDENING, "Gateway-01", 1, 0.25, "Harden Gateway-01"),
    ]
