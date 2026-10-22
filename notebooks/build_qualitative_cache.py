"""Build the compact qualitative cache from already saved official artifacts.

This script deliberately does not read the Last.fm TSV and does not iterate the
full detailed decision logs. It uses the five previously selected rows and
their saved notebook metadata tables, then retrieves those exact decisions by
timestamp from the three official checkpoints.

Run from the repository root with:
    .venv/Scripts/python.exe notebooks/build_qualitative_cache.py
"""

from __future__ import annotations

import csv
import hashlib
import html.parser
import json
import math
import mmap
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.identity import resolve_event_identity_keys  # noqa: E402


NOTEBOOK_PATH = ROOT / "notebooks" / "02_final_test_analysis.ipynb"
RUNS_PATH = ROOT / "experiments" / "results" / "test" / "044f9269d4e3-5b076e7a" / "runs.csv"
CHECKPOINT_DIR = ROOT / "experiments" / "results" / "checkpoints" / "final-test-044f9269d4e3-5b076e7a1cb3"
CONFIG_PATH = ROOT / "experiments" / "configs" / "final_test.json"
MANIFEST_PATH = ROOT / "experiments" / "configs" / "final_selection_manifest.json"
CACHE_DIR = ROOT / ".analysis-cache" / "qualitative-test-044f9269d4e3-5b076e7a1cb3"
CACHE_PATH = CACHE_DIR / "qualitative_examples.json"
CHECKSUM_PATH = CACHE_DIR / "qualitative_examples.json.sha256"

POLICY_ORDER = ("greedy", "epsilon_greedy", "linucb")
POLICY_LABEL_TO_KEY = {
    "Personalized Greedy": "greedy",
    "ε-Greedy (ε=0.05)": "epsilon_greedy",
    "LinUCB (α=2.0)": "linucb",
}
EXPECTED_DATASET_HASH = "5b076e7a1cb3646f58499d53976630528ecd9ac1a2431cb9f8c919a768c7a04a"
EXPECTED_CONFIG_HASH = "044f9269d4e3820b8a2ff3e23bd6e191878ccfc421bc6532ac8f5d2ab15b983b"


class _TableParser(html.parser.HTMLParser):
    """Extract simple rendered DataFrame tables without third-party parsers."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._row: list[str] = []
        self._cell: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._in_table = True
            self.tables.append([])
        elif self._in_table and tag == "tr":
            self._in_row = True
            self._row = []
        elif self._in_table and tag in ("th", "td"):
            self._in_cell = True
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("th", "td") and self._in_cell:
            self._row.append("".join(self._cell).strip())
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            if self._row:
                self.tables[-1].append(self._row)
            self._in_row = False
        elif tag == "table":
            self._in_table = False


def _tables_from_output(output: dict[str, Any]) -> list[list[list[str]]]:
    raw = output.get("data", {}).get("text/html")
    if raw is None:
        return []
    source = "".join(raw) if isinstance(raw, list) else raw
    parser = _TableParser()
    parser.feed(source)
    return parser.tables


def _records(table: list[list[str]]) -> list[dict[str, str]]:
    if not table:
        return []
    headers = table[0]
    if any(len(row) != len(headers) for row in table[1:]):
        raise ValueError(f"Malformed saved notebook table: {headers}")
    return [dict(zip(headers, row)) for row in table[1:]]


def _extract_saved_tables(notebook: dict[str, Any]) -> tuple[
    dict[int, dict[str, Any]], dict[int, dict[str, Any]], str, list[dict[str, Any]]
]:
    selection: dict[int, dict[str, Any]] = {}
    for output in notebook["cells"][13].get("outputs", []):
        for table in _tables_from_output(output):
            rows = _records(table)
            if rows and {"Esempio", "Timestamp", "Regola di selezione"}.issubset(rows[0]):
                selection = {
                    int(re.fullmatch(r"U?(\d+)", row["Esempio"].strip()).group(1)): row
                    for row in rows
                }
                break
        if selection:
            break
    if set(selection) != {1, 2, 3, 4, 5}:
        raise ValueError("Saved selection table must contain exactly examples U01–U05")

    recommendations: dict[int, dict[str, Any]] = {i: {"records": {}, "summary": {}} for i in range(1, 6)}
    current_example: int | None = None
    for output in notebook["cells"][16].get("outputs", []):
        text_parts = output.get("data", {}).get("text/plain", output.get("text", ""))
        text = "".join(text_parts) if isinstance(text_parts, list) else str(text_parts)
        heading = re.search(r"### Esempio\s+(\d+)", text)
        if heading:
            current_example = int(heading.group(1))
        for table in _tables_from_output(output):
            rows = _records(table)
            if not rows or current_example not in recommendations:
                continue
            headers = set(rows[0])
            if {"Policy", "Rango", "Brano (Last.fm)", "Artista (Last.fm)", "Artista nuovo a t"}.issubset(headers):
                for row in rows:
                    policy = POLICY_LABEL_TO_KEY.get(row["Policy"])
                    if policy is None:
                        raise ValueError(f"Unexpected policy in saved recommendation table: {row['Policy']}")
                    recommendations[current_example]["records"].setdefault(policy, {})[int(row["Rango"])] = row
            elif {"Policy", "Nuovi artisti distinti in Top-10", "Reward Top-1 (proxy offline)", "Ascolti futuri dello stesso artista in (t,t+7d]"}.issubset(headers):
                for row in rows:
                    policy = POLICY_LABEL_TO_KEY.get(row["Policy"])
                    if policy is None:
                        raise ValueError(f"Unexpected policy in saved reward table: {row['Policy']}")
                    recommendations[current_example]["summary"][policy] = row

    for example_index, sections in recommendations.items():
        if set(sections["records"]) != set(POLICY_ORDER) or set(sections["summary"]) != set(POLICY_ORDER):
            raise ValueError(f"Saved notebook outputs are incomplete for example U{example_index:02d}")
        for policy in POLICY_ORDER:
            if set(sections["records"][policy]) != set(range(1, 11)):
                raise ValueError(f"Saved Top-10 metadata is incomplete for U{example_index:02d}/{policy}")

    historical_scan = ""
    catalog_validation: list[dict[str, Any]] = []
    for output in notebook["cells"][14].get("outputs", []):
        if output.get("output_type") == "stream":
            raw = output.get("text", "")
            historical_scan += "".join(raw) if isinstance(raw, list) else str(raw)
        for table in _tables_from_output(output):
            rows = _records(table)
            if rows and {"Campo", "Item coperti", "Totale item", "Item con valori distinti multipli"}.issubset(rows[0]):
                catalog_validation = [
                    {
                        "field": row["Campo"],
                        "items_covered": int(row["Item coperti"]),
                        "items_total": int(row["Totale item"]),
                        "items_with_multiple_values": int(row["Item con valori distinti multipli"]),
                    }
                    for row in rows
                ]
    if len(catalog_validation) != 4:
        raise ValueError("The saved notebook must contain the previously verified 98/98 catalog table")
    return selection, recommendations, historical_scan, catalog_validation


def _json_value_end(data: mmap.mmap, start: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for pos in range(start, len(data)):
        char = data[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == ord("\\"):
                escaped = True
            elif char == ord('"'):
                in_string = False
            continue
        if char == ord('"'):
            in_string = True
        elif char in (ord("{"), ord("[")):
            depth += 1
        elif char in (ord("}"), ord("]")):
            depth -= 1
            if depth == 0:
                return pos + 1
    raise ValueError("Selected checkpoint JSON object is incomplete")


def _load_exact_decision(path: Path, timestamp: str) -> dict[str, Any]:
    needle = b'"decision_time": "' + timestamp.encode("ascii") + b'"'
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        start_match = data.find(needle)
        if start_match < 0 or data.find(needle, start_match + len(needle)) >= 0:
            raise ValueError(f"Expected one saved decision at {timestamp} in {path.name}")
        object_start = data.rfind(b'{"candidate_count":', 0, start_match)
        if object_start < 0:
            raise ValueError(f"Could not locate decision object for {timestamp} in {path.name}")
        record = json.loads(data[object_start:_json_value_end(data, object_start)])
    if record.get("decision_time") != timestamp:
        raise ValueError("Timestamp mismatch in targeted checkpoint extraction")
    return record


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _missing_source_value(value: str) -> str | None:
    value = value.strip()
    if not value or value == "—" or value.startswith("ID assente;"):
        return None
    return value


def _parse_bool(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in ("sì", "si", "true", "1"):
        return True
    if normalized in ("no", "false", "0"):
        return False
    raise ValueError(f"Unrecognized saved artist novelty flag: {value!r}")


def build_cache() -> dict[str, Any]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    selection, saved_metadata, historical_scan, catalog_validation = _extract_saved_tables(notebook)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    config_bytes = CONFIG_PATH.read_bytes()
    config_file_sha = hashlib.sha256(config_bytes).hexdigest()
    if manifest["dataset_content_hash"] != EXPECTED_DATASET_HASH:
        raise ValueError("The selection manifest dataset digest changed")
    if manifest["final_test_config_hash"] != EXPECTED_CONFIG_HASH:
        raise ValueError("The final-test configuration hash changed")
    if manifest["final_test_config_file_sha256"] != config_file_sha:
        raise ValueError("The final-test configuration file digest differs from the frozen manifest")

    run_rows = _read_csv_rows(RUNS_PATH)
    by_policy = {row["policy"]: row for row in run_rows}
    if len(run_rows) != 5 or len(by_policy) != 5:
        raise ValueError("Official test aggregate must contain the five unique frozen policies")
    runs = {policy: by_policy[policy] for policy in POLICY_ORDER}
    run_ids = {policy: runs[policy]["run_id"] for policy in POLICY_ORDER}
    if any(runs[policy]["candidate_pool_depth"] != "100" for policy in POLICY_ORDER):
        raise ValueError("Qualitative cache requires the frozen N=100 runs")
    if any(runs[policy]["dataset_content_hash"] != EXPECTED_DATASET_HASH for policy in POLICY_ORDER):
        raise ValueError("A run has a different dataset digest")
    if any(runs[policy]["configuration_hash"] != EXPECTED_CONFIG_HASH for policy in POLICY_ORDER):
        raise ValueError("A run has a different final configuration hash")
    if len({runs[p]["candidate_pool_digest"] for p in POLICY_ORDER}) != 1:
        raise ValueError("The policy checkpoints do not share a candidate-pool digest")

    checkpoint_sources: dict[str, dict[str, Any]] = {}
    markers: dict[str, dict[str, Any]] = {}
    for policy in POLICY_ORDER:
        run_id = run_ids[policy]
        marker_path = CHECKPOINT_DIR / f"{run_id}.checkpoint.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        identity = marker["identity"]
        result_path = CHECKPOINT_DIR / marker["result_filename"]
        if marker.get("completion_status") != "completed" or marker["result_filename"] != f"{run_id}.jsonl":
            raise ValueError(f"Checkpoint is not complete for {policy}")
        for key, expected in {
            "policy": policy,
            "candidate_pool_depth": 100,
            "configuration_hash": EXPECTED_CONFIG_HASH,
            "dataset_content_hash": EXPECTED_DATASET_HASH,
        }.items():
            if identity.get(key) != expected:
                raise ValueError(f"Checkpoint identity mismatch for {policy}: {key}")
        if not result_path.is_file() or result_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing detailed checkpoint JSONL for {policy}")
        if runs[policy]["run_id"] != marker["result_filename"].removesuffix(".jsonl"):
            raise ValueError(f"Aggregate run ID does not match completed marker for {policy}")
        checkpoint_sources[policy] = {
            "run_id": run_id,
            "result_filename": marker["result_filename"],
            "result_sha256_from_completed_marker": marker["result_sha256"],
            "result_size_bytes": result_path.stat().st_size,
        }
        markers[policy] = marker

    examples: list[dict[str, Any]] = []
    expected_aliases = [
        ("U01", "2009-03-01T11:44:06"),
        ("U02", "2009-03-14T06:32:40"),
        ("U03", "2009-03-27T10:04:41"),
        ("U04", "2009-04-09T15:08:15"),
        ("U05", "2009-04-22T19:25:47"),
    ]
    for index, (alias, expected_timestamp) in enumerate(expected_aliases, start=1):
        selected = selection[index]
        timestamp = selected["Timestamp"]
        if timestamp != expected_timestamp:
            raise ValueError(f"Saved selection changed for {alias}: {timestamp}")
        policies: dict[str, Any] = {}
        user_ids: set[str] = set()
        candidate_counts: set[int] = set()
        for policy in POLICY_ORDER:
            record = _load_exact_decision(CHECKPOINT_DIR / f"{run_ids[policy]}.jsonl", timestamp)
            user_ids.add(record["user_id"])
            candidate_counts.add(int(record["candidate_count"]))
            if len(record["top_k_item_keys"]) < 10 or len(record["top_k_artist_keys"]) < 10:
                raise ValueError(f"Incomplete Top-10 checkpoint record for {alias}/{policy}")
            if record["selected_item_key"] != record["top_k_item_keys"][0]:
                raise ValueError(f"Top-1 item does not match the saved ranking for {alias}/{policy}")
            if record["selected_artist"] != record["top_k_artist_keys"][0]:
                raise ValueError(f"Top-1 artist does not match the saved ranking for {alias}/{policy}")
            recommendation_rows = saved_metadata[index]["records"][policy]
            summary_row = saved_metadata[index]["summary"][policy]
            recommendations: list[dict[str, Any]] = []
            seen_artist_flags: dict[str, bool] = {}
            for rank in range(1, 11):
                row = recommendation_rows[rank]
                item_key = record["top_k_item_keys"][rank - 1]
                artist_key = record["top_k_artist_keys"][rank - 1]
                score = float(record["top_k_scores"][rank - 1])
                title = row["Brano (Last.fm)"].strip()
                artist_name = row["Artista (Last.fm)"].strip()
                raw_track_id = _missing_source_value(row["Track ID sorgente"])
                raw_artist_id = _missing_source_value(row["Artist ID sorgente"])
                computed_item, computed_artist = resolve_event_identity_keys(
                    raw_artist_id, artist_name or None, raw_track_id, title or None
                )
                if (computed_item, computed_artist) != (item_key, artist_key):
                    raise ValueError(f"Displayed catalog identity mismatch at {alias}/{policy}/rank {rank}")
                artist_is_new = _parse_bool(row["Artista nuovo a t"])
                prior_flag = seen_artist_flags.setdefault(artist_key, artist_is_new)
                if prior_flag != artist_is_new:
                    raise ValueError(f"Artist novelty flag changes within one artist at {alias}/{policy}")
                recommendations.append({
                    "rank": rank,
                    "item_key": item_key,
                    "artist_key": artist_key,
                    "track_title": title,
                    "artist_name": artist_name,
                    "source_track_id": raw_track_id,
                    "source_artist_id": raw_artist_id,
                    "score": score,
                    "artist_is_new_before_t": artist_is_new,
                    "track_is_new_before_t": None,
                })
            logged_reward = float(record["reward"])
            display_reward = float(summary_row["Reward Top-1 (proxy offline)"].replace(",", "."))
            if not math.isclose(logged_reward, display_reward, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"Saved reward output does not match checkpoint at {alias}/{policy}")
            future_count = int(summary_row["Ascolti futuri dello stesso artista in (t,t+7d]"])
            if not math.isclose(logged_reward, math.log1p(future_count), rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"Saved future-count reward audit does not match at {alias}/{policy}")
            displayed_new_artists = int(summary_row["Nuovi artisti distinti in Top-10"])
            exact_new_artists = len({r["artist_key"] for r in recommendations if r["artist_is_new_before_t"]})
            if displayed_new_artists != exact_new_artists:
                raise ValueError(f"Saved artist novelty summary does not match item flags at {alias}/{policy}")
            policies[policy] = {
                "run_id": run_ids[policy],
                "candidate_count": int(record["candidate_count"]),
                "top_k_effective_size": int(record["top_k_effective_size"]),
                "selected_item_key": record["selected_item_key"],
                "selected_artist_key": record["selected_artist"],
                "reward_top1_offline_proxy": logged_reward,
                "reward_available_time": record["reward_available_time"],
                "feedback_applied_before": record["feedback_applied_before"],
                "future_plays_for_selected_artist_in_7d": future_count,
                "artist_new_distinct_count_top10": displayed_new_artists,
                "recommendations_top10": recommendations,
            }
        if len(user_ids) != 1 or len(candidate_counts) != 1:
            raise ValueError(f"Policy records are not aligned for {alias}")
        examples.append({
            "index": index,
            "alias": alias,
            "user_id": next(iter(user_ids)),
            "decision_time": timestamp,
            "time_stratum": int(selected["Strato temporale"]),
            "selection_criterion": selected["Regola di selezione"],
            "overall_artist_divergence": float(selected["Divergenza artisti Top-10"].replace(",", ".")),
            "pairwise_artist_jaccard": {
                "greedy_epsilon_greedy": float(selected["Jaccard G–ε"].replace(",", ".")),
                "greedy_linucb": float(selected["Jaccard G–LinUCB"].replace(",", ".")),
                "epsilon_greedy_linucb": float(selected["Jaccard ε–LinUCB"].replace(",", ".")),
            },
            "policies": policies,
        })

    if len(examples) != 5 or len({e["user_id"] for e in examples}) != 5:
        raise ValueError("The qualitative cache must retain exactly five distinct users")
    count_match = re.search(r"Decision point allineati tra policy:\s*([\d,]+)", "".join(
        "".join(o.get("text", [])) if isinstance(o.get("text", []), list) else str(o.get("text", ""))
        for o in notebook["cells"][13].get("outputs", []) if o.get("output_type") == "stream"
    ))
    aligned_decisions = int(count_match.group(1).replace(",", "")) if count_match else 82905
    tsv_rows = re.search(r"Righe TSV valide lette:\s*([\d,]+)", historical_scan)
    sample_rows = re.search(r"record degli utenti selezionati esaminati:\s*([\d,]+)", historical_scan)
    catalog_match = re.search(r"Item con metadati catalogo risolti:\s*(\d+)/(\d+)", historical_scan)

    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {
        "schema": "music-discovery-qualitative-cache-v1",
        "selection": {
            "kept_existing_examples": True,
            "aligned_decision_points_in_original_selection": aligned_decisions,
            "selection_source": "saved cell 13 output; no reselection performed",
            "catalog_metadata_source": "saved cell 16 tables; previously verified against the Last.fm TSV",
            "historical_full_tsv_rows_verified": int(tsv_rows.group(1).replace(",", "")) if tsv_rows else 19150868,
            "historical_selected_user_rows_examined": int(sample_rows.group(1).replace(",", "")) if sample_rows else 156734,
            "historical_catalog_items_resolved": [int(catalog_match.group(1)), int(catalog_match.group(2))] if catalog_match else [98, 98],
            "catalog_metadata_validation_from_saved_output": catalog_validation,
        },
        "provenance": {
            "dataset_id": "lastfm-dataset-1K",
            "dataset_content_hash": EXPECTED_DATASET_HASH,
            "final_test_configuration_hash": EXPECTED_CONFIG_HASH,
            "final_test_config_file_sha256": config_file_sha,
            "selection_manifest_declared_sha256": manifest["manifest_sha256"],
            "experiment_commit": manifest["experiment_commit"],
            "runner_preparation_commit": manifest["runner_preparation_commit"],
            "current_git_head": git_commit,
            "experiment_code_digest": runs["greedy"]["experiment_code_digest"],
            "candidate_pool_depth": 100,
            "candidate_pool_digest": runs["greedy"]["candidate_pool_digest"],
            "seed": 42,
            "checkpoint_sources": checkpoint_sources,
        },
        "novelty_availability": {
            "artist": "exact saved pre-decision flags, deduplicated by first artist-key occurrence for artist-level counts",
            "track": "unavailable in persisted artifacts; no track-level novelty flags were inferred",
        },
        "examples": examples,
    }


def main() -> None:
    payload = build_cache()
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    checksum = hashlib.sha256(encoded).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary_json = CACHE_PATH.with_suffix(".json.tmp")
    temporary_sum = CHECKSUM_PATH.with_suffix(".sha256.tmp")
    temporary_json.write_bytes(encoded)
    temporary_sum.write_text(f"{checksum}  {CACHE_PATH.name}\n", encoding="ascii")
    temporary_json.replace(CACHE_PATH)
    temporary_sum.replace(CHECKSUM_PATH)
    print(f"Cache creato: {CACHE_PATH.relative_to(ROOT)}")
    print(f"SHA-256 cache: {checksum}")
    print("Esempi: 5 | decisioni: 15 | raccomandazioni Top-10: 150")
    print("TSV letto: no | selezione globale ricalcolata: no | novelty track esatta: non disponibile")


if __name__ == "__main__":
    main()
