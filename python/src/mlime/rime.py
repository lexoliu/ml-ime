"""Measure RIME on an eval slice through librime's C API.

Every number the project has is relative to its own trigram or to the GPT-6
ceiling; issue #51 asks where the engines a user can install today sit. RIME
is the one that runs headless: ``librime`` is a dynamic library with a C API
(``rime_api.h``, a struct of function pointers returned by ``rime_get_api``),
and its schemas deploy into a directory with ``rime_deployer``. This module
binds that API with ``ctypes``, so nothing is compiled, feeds each record's
keystrokes to a fresh session and accepts the first candidate until the
input is consumed, which is what a user pressing space gets.

RIME reads no context and, with the user dictionary disabled in the schema
patch, learns nothing between records. The records are the ones a
``fused-eval --dump`` file names, so the slice is the same as every other
experiment's and the beam's own hypotheses ride along for the comparison.
"""

from __future__ import annotations

import ctypes
import json
from collections.abc import Iterator, Sequence
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    c_char_p,
    c_int,
    c_size_t,
    c_void_p,
    sizeof,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlime.generate import characters_right
from mlime.logging import log
from mlime.rescore import DumpedRecord

Bool = c_int
SessionId = c_size_t

#: The function pointers of ``rime_api_t`` in header order (librime 1.17).
#: Only the ones this module calls get a prototype; the rest keep the layout.
API_FUNCTIONS = (
    "setup", "set_notification_handler", "initialize", "finalize",
    "start_maintenance", "is_maintenance_mode", "join_maintenance_thread",
    "deployer_initialize", "prebuild", "deploy", "deploy_schema", "deploy_config_file",
    "sync_user_data", "create_session", "find_session", "destroy_session",
    "cleanup_stale_sessions", "cleanup_all_sessions", "process_key", "commit_composition",
    "clear_composition", "get_commit", "free_commit", "get_context", "free_context",
    "get_status", "free_status", "set_option", "get_option", "set_property", "get_property",
    "get_schema_list", "free_schema_list", "get_current_schema", "select_schema",
    "schema_open", "config_open", "config_close", "config_get_bool", "config_get_int",
    "config_get_double", "config_get_string", "config_get_cstring", "config_update_signature",
    "config_begin_map", "config_next", "config_end", "simulate_key_sequence",
    "register_module", "find_module", "run_task", "get_shared_data_dir", "get_user_data_dir",
    "get_sync_dir", "get_user_id", "get_user_data_sync_dir", "config_init",
    "config_load_string", "config_set_bool", "config_set_int", "config_set_double",
    "config_set_string", "config_get_item", "config_set_item", "config_clear",
    "config_create_list", "config_create_map", "config_list_size", "config_begin_list",
    "get_input", "get_caret_pos", "select_candidate", "get_version", "set_caret_pos",
    "select_candidate_on_current_page", "candidate_list_begin", "candidate_list_next",
    "candidate_list_end", "user_config_open", "candidate_list_from_index",
    "get_prebuilt_data_dir", "get_staging_dir", "commit_proto", "context_proto",
    "status_proto", "get_state_label", "delete_candidate", "delete_candidate_on_current_page",
    "get_state_label_abbreviated", "set_input", "get_shared_data_dir_s",
    "get_user_data_dir_s", "get_prebuilt_data_dir_s", "get_staging_dir_s", "get_sync_dir_s",
    "highlight_candidate", "highlight_candidate_on_current_page", "change_page",
)  # fmt: skip


class RimeTraits(Structure):
    _fields_ = [
        ("data_size", c_int),
        ("shared_data_dir", c_char_p),
        ("user_data_dir", c_char_p),
        ("distribution_name", c_char_p),
        ("distribution_code_name", c_char_p),
        ("distribution_version", c_char_p),
        ("app_name", c_char_p),
        ("modules", POINTER(c_char_p)),
        ("min_log_level", c_int),
        ("log_dir", c_char_p),
        ("prebuilt_data_dir", c_char_p),
        ("staging_dir", c_char_p),
    ]


class RimeComposition(Structure):
    _fields_ = [
        ("length", c_int),
        ("cursor_pos", c_int),
        ("sel_start", c_int),
        ("sel_end", c_int),
        ("preedit", c_char_p),
    ]


class RimeCandidate(Structure):
    _fields_ = [("text", c_char_p), ("comment", c_char_p), ("reserved", c_void_p)]


class RimeMenu(Structure):
    _fields_ = [
        ("page_size", c_int),
        ("page_no", c_int),
        ("is_last_page", Bool),
        ("highlighted_candidate_index", c_int),
        ("num_candidates", c_int),
        ("candidates", POINTER(RimeCandidate)),
        ("select_keys", c_char_p),
    ]


class RimeCommit(Structure):
    _fields_ = [("data_size", c_int), ("text", c_char_p)]


class RimeContext(Structure):
    _fields_ = [
        ("data_size", c_int),
        ("composition", RimeComposition),
        ("menu", RimeMenu),
        ("commit_text_preview", c_char_p),
        ("select_labels", POINTER(c_char_p)),
    ]


class RimeApi(Structure):
    _fields_ = [("data_size", c_int), *((name, c_void_p) for name in API_FUNCTIONS)]


def _sized[T: Structure](kind: type[T]) -> T:
    """``RIME_STRUCT_INIT``: a zeroed struct whose ``data_size`` excludes the size field."""
    value = kind()
    value.data_size = sizeof(kind) - sizeof(c_int)
    return value


#: Prototypes of the functions this module calls: name -> (restype, argtypes).
PROTOTYPES: dict[str, tuple[Any, tuple[Any, ...]]] = {
    "setup": (None, (POINTER(RimeTraits),)),
    "initialize": (None, (POINTER(RimeTraits),)),
    "finalize": (None, ()),
    "start_maintenance": (Bool, (Bool,)),
    "is_maintenance_mode": (Bool, ()),
    "join_maintenance_thread": (None, ()),
    "create_session": (SessionId, ()),
    "destroy_session": (Bool, (SessionId,)),
    "process_key": (Bool, (SessionId, c_int, c_int)),
    "get_commit": (Bool, (SessionId, POINTER(RimeCommit))),
    "free_commit": (Bool, (POINTER(RimeCommit),)),
    "get_context": (Bool, (SessionId, POINTER(RimeContext))),
    "free_context": (Bool, (POINTER(RimeContext),)),
    "select_schema": (Bool, (SessionId, c_char_p)),
    "select_candidate_on_current_page": (Bool, (SessionId, c_size_t)),
    "get_version": (c_char_p, ()),
}


class Rime:
    """One librime instance over a deployed data directory."""

    def __init__(self, library: Path, data_dir: Path):
        self._lib = ctypes.CDLL(str(library))
        self._lib.rime_get_api.restype = POINTER(RimeApi)
        api = self._lib.rime_get_api().contents
        self._call: dict[str, Any] = {}
        for name, (restype, argtypes) in PROTOTYPES.items():
            address = getattr(api, name)
            if not address:
                raise RuntimeError(f"librime at {library} has no {name}")
            self._call[name] = CFUNCTYPE(restype, *argtypes)(address)
        self._data_dir = data_dir
        # ctypes keeps no reference to the strings a struct points at, so the
        # traits and their bytes live as long as the instance.
        self._traits = _sized(RimeTraits)
        self._traits.shared_data_dir = self._traits.user_data_dir = bytes(data_dir)
        self._traits.staging_dir = bytes(data_dir / "build")
        self._traits.log_dir = bytes(data_dir / "log")
        self._traits.distribution_name = b"mlime"
        self._traits.distribution_code_name = b"mlime"
        self._traits.distribution_version = b"0"
        self._traits.app_name = b"rime.mlime"
        self._traits.min_log_level = 2
        (data_dir / "log").mkdir(exist_ok=True)
        self._call["setup"](ctypes.byref(self._traits))
        self._call["initialize"](ctypes.byref(self._traits))
        if self._call["start_maintenance"](0) and self._call["is_maintenance_mode"]():
            self._call["join_maintenance_thread"]()
        log.info("rime", version=self.version, data_dir=str(data_dir))

    @property
    def version(self) -> str:
        """The librime version string."""
        return str(self._call["get_version"]().decode())

    def close(self) -> None:
        """Release the library's sessions and data."""
        self._call["finalize"]()

    def type_sentence(self, schema: str, keys: str) -> tuple[str, list[str]]:
        """Type *keys* into a fresh session of *schema* and accept the first candidate until
        the input is consumed. Returns the committed text and the candidates the first
        page showed after the last key, which is what a user sees before choosing."""
        session = self._call["create_session"]()
        if not session:
            raise RuntimeError("librime created no session")
        try:
            if not self._call["select_schema"](session, schema.encode()):
                raise RuntimeError(f"librime has no schema {schema!r}")
            for key in keys:
                if not self._call["process_key"](session, ord(key), 0):
                    raise RuntimeError(f"librime rejected the key {key!r} of {keys!r}")
            first_page = list(self._candidates(session))
            committed = ""
            # Each acceptance consumes at least one syllable, so the input's
            # length bounds the loop; a menu with nothing to accept ends it.
            for _ in range(len(keys)):
                if not self._composing(session):
                    break
                if not self._call["select_candidate_on_current_page"](session, 0):
                    break
                committed += self._commit(session)
            return committed, first_page
        finally:
            self._call["destroy_session"](session)

    def _candidates(self, session: int) -> Iterator[str]:
        context = _sized(RimeContext)
        if not self._call["get_context"](session, ctypes.byref(context)):
            raise RuntimeError("librime returned no context")
        try:
            menu = context.menu
            for index in range(menu.num_candidates):
                yield menu.candidates[index].text.decode()
        finally:
            self._call["free_context"](ctypes.byref(context))

    def _composing(self, session: int) -> bool:
        context = _sized(RimeContext)
        if not self._call["get_context"](session, ctypes.byref(context)):
            raise RuntimeError("librime returned no context")
        try:
            return bool(context.composition.length > 0 and context.menu.num_candidates > 0)
        finally:
            self._call["free_context"](ctypes.byref(context))

    def _commit(self, session: int) -> str:
        commit = _sized(RimeCommit)
        if not self._call["get_commit"](session, ctypes.byref(commit)):
            return ""
        try:
            return str(commit.text.decode())
        finally:
            self._call["free_commit"](ctypes.byref(commit))


@dataclass(frozen=True)
class Typed:
    """What RIME produced for a record."""

    record: DumpedRecord
    committed: str
    first_page: tuple[str, ...]


@dataclass(frozen=True)
class RimeResult:
    """How RIME did on one slice, next to the beam it was measured against."""

    records: int
    rime_top1: float
    first_page_exact: float
    beam_top1: float
    characters: int
    characters_right: float
    length_mismatches: int

    def as_dict(self) -> dict[str, object]:
        """A JSON-friendly view."""
        return {
            "records": self.records,
            "rime_top1": self.rime_top1,
            "first_page_exact": self.first_page_exact,
            "beam_top1": self.beam_top1,
            "characters": self.characters,
            "characters_right": self.characters_right,
            "length_mismatches": self.length_mismatches,
        }


def evaluate(typed: Sequence[Typed]) -> RimeResult:
    """RIME's default sentence against the expected text, next to the beam's top-1."""
    total = len(typed)
    characters = sum(len(t.record.text) for t in typed)
    return RimeResult(
        records=total,
        rime_top1=sum(t.committed == t.record.text for t in typed) / total,
        first_page_exact=sum(t.record.text in t.first_page for t in typed) / total,
        beam_top1=sum(t.record.hypotheses[0].text == t.record.text for t in typed) / total,
        characters=characters,
        characters_right=sum(characters_right(t.record.text, t.committed) for t in typed)
        / characters,
        length_mismatches=sum(len(t.committed) != len(t.record.text) for t in typed),
    )


@dataclass(frozen=True)
class RimeReport:
    """What the measurement found on one slice of one eval set."""

    version: str
    schema: str
    result: RimeResult
    typed: tuple[Typed, ...]

    def as_dict(self) -> dict[str, object]:
        """A JSON-friendly view, with every sentence so the answers can be inspected."""
        return {
            "librime": self.version,
            "schema": self.schema,
            "result": self.result.as_dict(),
            "typed": [
                {
                    "record": t.record.index,
                    "expected": t.record.text,
                    "committed": t.committed,
                    "first_page": list(t.first_page),
                }
                for t in self.typed
            ],
        }

    def render(self) -> str:
        """A few lines for the terminal."""
        r = self.result
        return "\n".join(
            [
                f"librime {self.version}, schema {self.schema}",
                f"{r.records} records: rime top-1 {r.rime_top1:.4f}, "
                f"first page exact {r.first_page_exact:.4f}, beam top-1 {r.beam_top1:.4f}",
                f"characters right {r.characters_right:.4f} of {r.characters}, "
                f"length mismatches {r.length_mismatches}",
            ]
        )


def measure(
    records: Sequence[DumpedRecord], library: Path, data_dir: Path, schema: str, every: int = 1000
) -> RimeReport:
    """Type every record into RIME and report the slice."""
    rime = Rime(library, data_dir)
    try:
        typed: list[Typed] = []
        for record in records:
            committed, first_page = rime.type_sentence(schema, record.pinyin)
            typed.append(Typed(record=record, committed=committed, first_page=tuple(first_page)))
            if len(typed) % every == 0:
                log.info("typed", records=len(typed), of=len(records))
        return RimeReport(
            version=rime.version, schema=schema, result=evaluate(typed), typed=tuple(typed)
        )
    finally:
        rime.close()


def write_report(report: RimeReport, out: Path) -> None:
    """Write the report, with every typed sentence, as JSON."""
    out.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n", "utf-8")
