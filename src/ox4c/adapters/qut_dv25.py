from __future__ import annotations

import pandas as pd

from .base import SecurityDataAdapter, normalize_binary


class QUTDV25Adapter(SecurityDataAdapter):
    domain = "software_supply_chain_security"
    target_candidates = ("Labels", "Level", "Label", "label", "Class", "Target")
    id_candidates = ("Package_Name", "Package Name", "package_name", "Package")
    timestamp_candidates = ("Timestamp", "Date", "Collection_Date", "Release_Date")

    VIEW_TERMS = {
        "file_io": ("read_process", "write_process", "data_transfer", "file_access_process"),
        "dependencies": ("dependenc", "install"),
        "directory_access": ("dir_access", "root_dir", "temp_dir", "home_dir", "user_dir", "sys_dir", "etc_dir"),
        "network": ("state_transition", "ip", "port", "tcp", "network"),
        "system_calls": ("operations", "syscall", "security_operation", "process_operation"),
        "behavior_patterns": ("pattern_", "pattern "),
    }

    def label(self, frame: pd.DataFrame) -> tuple[pd.Series, str]:
        target = self._find(frame.columns, self.target_candidates)
        if not target:
            raise ValueError(f"QUT-DV25 label not found; expected one of {self.target_candidates}")
        return normalize_binary(frame[target]), target

    def views(self, feature_names: list[str]) -> dict[str, list[str]]:
        views = {}
        for view, terms in self.VIEW_TERMS.items():
            views[view] = [name for name in feature_names if any(term in name.lower() for term in terms)]
        assigned = {feature for features in views.values() for feature in features}
        remainder = [feature for feature in feature_names if feature not in assigned]
        if remainder:
            views["other_behavior"] = remainder
        return views
