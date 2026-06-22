"""
Shared data loading utilities for TF-IDF analysis scripts.
Handles fuzzy matching between annotation track names and manifest file paths.
"""
from __future__ import annotations
import csv
import os
from difflib import SequenceMatcher

BASE = os.path.dirname(os.path.abspath(__file__))


def _stem(path: str) -> str:
    """Strip directory and extension from a path or bare filename."""
    return os.path.splitext(os.path.basename(path.strip()))[0].strip()


def _normalise(s: str) -> str:
    """Lowercase and collapse whitespace for fuzzy comparison."""
    return " ".join(s.lower().split())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalise(a), _normalise(b)).ratio()


def build_stem_map(*manifest_paths: str, fuzzy_threshold: float = 0.82) -> dict[str, str]:
    """
    Build {track_stem → group} from one or more manifest CSVs.

    Priority: earlier manifests win on conflict.
    Fuzzy matching is applied at lookup time (see match_track).
    """
    stem_map: dict[str, str] = {}          # exact stem → group
    for path in manifest_paths:
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                grp = row.get("group", "").strip()
                if not grp or grp.lower() == "group":
                    continue
                s = _stem(row.get("path", ""))
                if s and s not in stem_map:
                    stem_map[s] = grp
    return stem_map


def match_track(track: str, stem_map: dict[str, str],
                fuzzy_threshold: float = 0.82) -> str | None:
    """
    Return the group for a track name, using:
      1. Exact stem match
      2. Case-insensitive exact match
      3. Best fuzzy match above threshold
    Returns None if no match found.
    """
    s = _stem(track)

    # 1. Exact
    if s in stem_map:
        return stem_map[s]

    # 2. Case-insensitive
    s_low = _normalise(s)
    for key, grp in stem_map.items():
        if _normalise(key) == s_low:
            return grp

    # 3. Fuzzy
    best_score, best_grp = 0.0, None
    for key, grp in stem_map.items():
        score = _similarity(s, key)
        if score > best_score:
            best_score, best_grp = score, grp

    if best_score >= fuzzy_threshold:
        return best_grp

    return None


def load_annotations(path: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            t = row.get("track", "").strip()
            l = row.get("label", "").strip()
            if t and l:
                records.append((t, l))
    return records


def build_group_counts(annotations: list[tuple[str, str]],
                       stem_map: dict[str, str],
                       fuzzy_threshold: float = 0.82
                       ) -> tuple[dict, list[str]]:
    """
    Returns (group_pattern_counts, unmatched_tracks).
    group_pattern_counts: {group: {pattern: count}}
    """
    from collections import defaultdict
    counts: dict = defaultdict(lambda: defaultdict(int))
    unmatched: list[str] = []
    seen_unmatched: set[str] = set()

    for track, label in annotations:
        grp = match_track(track, stem_map, fuzzy_threshold)
        if grp:
            counts[grp][label] += 1
        else:
            s = _stem(track)
            if s not in seen_unmatched:
                unmatched.append(s)
                seen_unmatched.add(s)

    return dict(counts), unmatched
