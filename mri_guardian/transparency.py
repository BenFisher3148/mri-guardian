"""File-backed transparency adapter for deterministic ICSD evidence logs."""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from icsd.core.serialise import canonical_json_dumps, hash_canonical_json

_ADAPTER_NAME = "jsonl_hash_chain"
_LOCK_POLL_INTERVAL_SECONDS = 0.01
_LOCK_REGISTRY_GUARD = threading.Lock()
_LOCK_REGISTRY: dict[str, threading.Lock] = {}


def _lock_for_path(lock_path: Path) -> threading.Lock:
    key = str(lock_path.absolute())
    with _LOCK_REGISTRY_GUARD:
        lock = _LOCK_REGISTRY.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCK_REGISTRY[key] = lock
        return lock


class JsonlHashChainTransparencyLog:
    """Persist evidence hashes in an append-only JSONL hash chain."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, *, evidence_payload: dict[str, Any]) -> dict[str, Any]:
        # Serialize the full read/compute/write cycle so correctness never depends
        # on platform append atomicity.
        with self._serialised_log_access(create_parent=True):
            entries = self._read_entries()
            previous_entry_hash = entries[-1]["entry_hash"] if entries else None
            entry = {
                "index": len(entries),
                "previous_entry_hash": previous_entry_hash,
                "evidence_hash": hash_canonical_json(evidence_payload),
            }
            entry["entry_hash"] = hash_canonical_json(entry)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(f"{canonical_json_dumps(entry)}\n")
                handle.flush()
                # Flush the JSONL file before releasing the lock. Directory fsync is
                # intentionally out of scope for this adapter.
                os.fsync(handle.fileno())
        return {
            "adapter": _ADAPTER_NAME,
            "reference": self._reference(entry),
            "details": {
                "index": entry["index"],
                "entry_hash": entry["entry_hash"],
                "previous_entry_hash": previous_entry_hash,
            },
        }

    def verify(
        self,
        *,
        evidence_payload: dict[str, Any],
        receipt: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        evidence_hash = hash_canonical_json(evidence_payload)
        # Reads hold the same lock as writers so verification observes one
        # serialized snapshot of the log rather than a live stream.
        with self._serialised_log_access(create_parent=False):
            entries = self._read_entries()
        chain_error = self._verify_chain(entries)
        if chain_error is not None:
            return {
                "verified": False,
                "reason": chain_error,
                "details": {"path": str(self.path)},
            }

        expected_reference = None
        if receipt is not None:
            expected_reference = str(receipt.reference if hasattr(receipt, "reference") else receipt["reference"])

        matched_entry: dict[str, Any] | None = None
        for entry in entries:
            if entry["evidence_hash"] != evidence_hash:
                continue
            if expected_reference is not None and self._reference(entry) != expected_reference:
                continue
            matched_entry = entry
            break

        if matched_entry is None:
            return {
                "verified": False,
                "reason": "evidence hash not present in transparency log.",
                "details": {"path": str(self.path)},
            }

        checkpoint = entries[-1]["entry_hash"] if entries else None
        return {
            "verified": True,
            "checkpoint": checkpoint,
            "details": {
                "index": matched_entry["index"],
                "reference": self._reference(matched_entry),
                "path": str(self.path),
            },
        }

    def _read_entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    msg = (
                        f"invalid JSON at line {line_number}: "
                        f"{exc.msg} (column {exc.colno})."
                    )
                    raise ValueError(msg) from exc
                if not isinstance(entry, dict):
                    msg = f"transparency log line {line_number} is not a JSON object."
                    raise ValueError(msg)
                entries.append(entry)
        return entries

    def _lock_path(self) -> Path:
        return self.path.parent / f"{self.path.name}.lock"

    @contextmanager
    def _serialised_log_access(self, *, create_parent: bool) -> Iterator[None]:
        lock_path = self._lock_path()
        if create_parent:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
        elif not lock_path.parent.exists():
            yield
            return

        with _lock_for_path(lock_path):
            # The sidecar file may remain on disk after use. Its presence alone is
            # not a lock signal; the OS lock state is authoritative.
            with lock_path.open("a+b") as handle:
                self._acquire_os_lock(handle)
                try:
                    yield
                finally:
                    self._release_os_lock(handle)

    @staticmethod
    def _acquire_os_lock(handle: Any) -> None:
        if os.name == "nt":
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    return
                except OSError:
                    time.sleep(_LOCK_POLL_INTERVAL_SECONDS)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

    @staticmethod
    def _release_os_lock(handle: Any) -> None:
        if os.name == "nt":
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _reference(entry: dict[str, Any]) -> str:
        return f"{entry['index']}:{entry['entry_hash']}"

    @staticmethod
    def _verify_chain(entries: list[dict[str, Any]]) -> str | None:
        previous_entry_hash = None
        for expected_index, entry in enumerate(entries):
            if entry.get("index") != expected_index:
                return "transparency log index sequence is not contiguous."
            if entry.get("previous_entry_hash") != previous_entry_hash:
                return "transparency log previous_entry_hash chain is broken."
            candidate = {
                "index": entry.get("index"),
                "previous_entry_hash": entry.get("previous_entry_hash"),
                "evidence_hash": entry.get("evidence_hash"),
            }
            expected_entry_hash = hash_canonical_json(candidate)
            if entry.get("entry_hash") != expected_entry_hash:
                return "transparency log entry_hash does not match canonical contents."
            previous_entry_hash = expected_entry_hash
        return None
