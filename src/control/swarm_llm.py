#!/usr/bin/env python3
"""Sprint 21: Human-Swarm Teaming — LLM/SLM intent parser → SwarmIntent.

SAR-only. Pipeline::

    Operator text ("search building 3 for victims")
      → few-shot prompt → LLM/SLM (or local regex fallback) → JSON intent
      → schema validation → SwarmIntent {task_graph, roles, constraints}

The local regex parser is the default (no network, <500 ms budget);
:func:`parse_with_llm_backend` tries an OpenAI-compatible endpoint first
and falls back to :func:`parse_swarm_command` on any failure.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Schema defaults (spec § Swarm Intent Schema)
# ---------------------------------------------------------------------------

DEFAULT_CONSTRAINTS = {
    "max_drones": 6,
    "max_time_min": 30,
    "safety_margin_m": 2.0,
    "comms_range_m": 500,
}

DEFAULT_ROLES = {"scout": 2, "searcher": 3, "relay": 1}

SWARM_INTENTS = (
    "search_and_rescue",
    "deliver_payload",
    "survey_area",
    "patrol",
    "return_home",
)

SAR_OBJECTIVE_TYPES = ("search", "assist", "deliver", "survey", "patrol", "return")

# Few-shot prompt templates grounding the parser (SAR-only).
FEW_SHOT_EXAMPLES: List[Tuple[str, str]] = [
    ("search building 3 for victims",
     '{"intent": "search_and_rescue", "area": {"building": "building_3", '
     '"floors": [1, 2, 3]}}'),
    ("scan floor 2 of building 1",
     '{"intent": "search_and_rescue", "area": {"building": "building_1", '
     '"floors": [2]}}'),
    ("deliver medkit to room 2 in building 3",
     '{"intent": "deliver_payload", "area": {"building": "building_3", '
     '"floors": [1]}}'),
    ("survey building 2 rooftops",
     '{"intent": "survey_area", "area": {"building": "building_2", '
     '"floors": [1]}}'),
    ("all drones return home",
     '{"intent": "return_home", "area": {"building": "base", "floors": []}}'),
    ("patrol building 1 floors 1 to 2",
     '{"intent": "patrol", "area": {"building": "building_1", '
     '"floors": [1, 2]}}'),
]

PROMPT_TEMPLATE = (
    "Parse the SAR swarm command into JSON with keys "
    "intent, area {{building, floors}}, objectives, constraints, roles.\n"
    "Allowed intents: search_and_rescue, deliver_payload, survey_area, "
    "patrol, return_home. SAR-only.\n"
    "Examples:\n{few_shot}\nCommand: {text}\nJSON:"
)


def build_swarm_prompt(text: str) -> str:
    """Render the few-shot prompt for `text` (LLM/SLM backends)."""
    lines = [f'- "{ex}" -> {js}' for ex, js in FEW_SHOT_EXAMPLES]
    return PROMPT_TEMPLATE.format(few_shot="\n".join(lines), text=text)


# ---------------------------------------------------------------------------
# SwarmIntent
# ---------------------------------------------------------------------------

@dataclass
class SwarmIntent:
    intent: str = "search_and_rescue"
    area: Dict = field(default_factory=lambda: {"building": "building_1",
                                                "floors": [1]})
    objectives: List[Dict] = field(default_factory=list)
    constraints: Dict = field(default_factory=lambda: dict(DEFAULT_CONSTRAINTS))
    roles: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_ROLES))
    task_graph: List[Dict] = field(default_factory=list)
    priority: int = 1
    latency_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "intent": self.intent,
            "area": dict(self.area),
            "objectives": [dict(o) for o in self.objectives],
            "constraints": dict(self.constraints),
            "roles": dict(self.roles),
            "task_graph": [dict(t) for t in self.task_graph],
            "priority": int(self.priority),
        }


def validate_swarm_intent(data: Dict) -> Dict:
    """Validate + normalize a raw intent dict. Raises ValueError if bad."""
    if not isinstance(data, dict):
        raise ValueError("intent must be a dict")
    intent = str(data.get("intent", "search_and_rescue"))
    if intent not in SWARM_INTENTS:
        raise ValueError(f"non-SAR intent: {intent!r}")
    area = dict(data.get("area", {}))
    building = str(area.get("building", "building_1"))
    floors = list(area.get("floors", [1]))
    floors = [int(f) for f in floors][:10] or [1]
    objectives = list(data.get("objectives", []))
    for o in objectives:
        if not isinstance(o, dict) or o.get("type") not in SAR_OBJECTIVE_TYPES:
            raise ValueError(f"non-SAR objective: {o!r}")
    constraints = dict(DEFAULT_CONSTRAINTS)
    constraints.update(dict(data.get("constraints", {}) or {}))
    constraints["max_drones"] = max(1, min(10, int(constraints["max_drones"])))
    constraints["max_time_min"] = max(1, int(constraints["max_time_min"]))
    roles = dict(data.get("roles", {}) or DEFAULT_ROLES)
    roles = {str(k): max(0, int(v)) for k, v in roles.items()}
    if sum(roles.values()) > constraints["max_drones"]:
        # Scale down proportionally so roles fit the drone cap.
        total = sum(roles.values()) or 1
        cap = constraints["max_drones"]
        scaled = {k: max(1 if v > 0 else 0,
                         int(round(v * cap / total))) for k, v in roles.items()}
        while sum(scaled.values()) > cap:
            k = max(scaled, key=lambda kk: scaled[kk])
            scaled[k] -= 1
        roles = scaled
    priority = int(data.get("priority", 1))
    return {"intent": intent,
            "area": {"building": building, "floors": floors},
            "objectives": objectives, "constraints": constraints,
            "roles": roles, "priority": priority}


# ---------------------------------------------------------------------------
# Local rule-based parser (default; deterministic, <500 ms)
# ---------------------------------------------------------------------------

_BUILDING_RE = re.compile(r"building\s*(\d+|[a-z])", re.I)
_FLOOR_RE = re.compile(r"floor[s]?\s*(\d+)(?:\s*(?:to|[-–])\s*(\d+))?", re.I)
_ROOM_RE = re.compile(r"room\s*(\d+)", re.I)
_FLOORS_ALL_RE = re.compile(r"all\s+floors|every\s+floor|each\s+floor", re.I)
_MAX_DRONES_RE = re.compile(r"(\d+)\s+drones?", re.I)
_MAX_TIME_RE = re.compile(r"(\d+)\s*min", re.I)


def _extract_building(text: str) -> str:
    m = _BUILDING_RE.search(text)
    if m:
        return f"building_{m.group(1).lower()}"
    if re.search(r"\b(base|home)\b", text, re.I):
        return "base"
    return "building_1"


def _extract_floors(text: str) -> List[int]:
    if _FLOORS_ALL_RE.search(text):
        return [1, 2, 3]
    m = _FLOOR_RE.search(text)
    if m:
        a, b = int(m.group(1)), m.group(2)
        if b is not None:
            lo, hi = sorted((a, int(b)))
            return list(range(lo, min(hi, lo + 9) + 1)) or [a]
        return [a]
    if _ROOM_RE.search(text) or _BUILDING_RE.search(text):
        # "search building 3" with no floor → sweep floors 1-3.
        if re.search(r"\b(search|scan|victims?|survivors?)\b", text, re.I):
            return [1, 2, 3]
        return [1]
    return [1]


def _classify_intent(text: str) -> str:
    low = text.lower()
    if re.search(r"\b(return|recall|land|rtb|come back|abort)\b", low):
        return "return_home"
    if re.search(r"\b(deliver|drop|dropoff|medkit|payload|suppl\w*|medicine)\b", low):
        return "deliver_payload"
    if re.search(r"\b(survey|map|inspect|photograph|rooftop)\b", low):
        return "survey_area"
    if re.search(r"\b(patrol|monitor|guard|watch|orbit)\b", low):
        return "patrol"
    return "search_and_rescue"


_OBJECTIVE_FOR_INTENT = {
    "search_and_rescue": "search",
    "deliver_payload": "deliver",
    "survey_area": "survey",
    "patrol": "patrol",
    "return_home": "return",
}


def intent_to_task_graph(intent: SwarmIntent | Dict) -> List[Dict]:
    """Map SwarmIntent objectives → ordered per-floor task-graph nodes."""
    d = intent.to_dict() if isinstance(intent, SwarmIntent) else intent
    objectives = list(d.get("objectives", []))
    graph: List[Dict] = []
    for i, o in enumerate(objectives):
        graph.append({
            "node_id": f"t{i}",
            "type": str(o.get("type", "search")),
            "area": str(o.get("area", "")),
            "priority": int(o.get("priority", i + 1)),
            "depends_on": [f"t{i - 1}"] if i > 0 else [],
            "skill": {"search": "navigate_to", "assist": "drop_payload",
                      "deliver": "drop_payload", "survey": "navigate_to",
                      "patrol": "navigate_to",
                      "return": "return_home"}.get(str(o.get("type", "search")),
                                                   "navigate_to"),
        })
    return graph


def parse_swarm_command(text: str, max_drones: int = 6) -> SwarmIntent:
    """Parse operator text → validated SwarmIntent (local, <500 ms)."""
    t0 = time.perf_counter()
    intent_name = _classify_intent(text)
    building = _extract_building(text)
    floors = _extract_floors(text)
    if building == "base":
        floors = []
    obj_type = _OBJECTIVE_FOR_INTENT[intent_name]
    room_m = _ROOM_RE.search(text)
    if intent_name == "return_home":
        objectives = [{"type": "return", "area": "base", "priority": 1}]
        roles = {"scout": 0, "searcher": 0, "relay": min(max_drones, 1)}
    elif room_m and intent_name == "deliver_payload":
        objectives = [{"type": "deliver", "area": f"room_{room_m.group(1)}",
                       "priority": 1}]
        roles = {"scout": 1, "searcher": min(max_drones - 2, 2),
                 "relay": 1}
    else:
        objectives = [{"type": obj_type, "area": f"floor_{f}", "priority": i + 1}
                      for i, f in enumerate(floors)] or \
            [{"type": obj_type, "area": building, "priority": 1}]
        roles = {"scout": min(2, max_drones - 2) if max_drones >= 3 else 1,
                 "searcher": max(1, max_drones - 3) if max_drones >= 3 else 1,
                 "relay": 1}
        while sum(roles.values()) > max_drones:
            k = max(roles, key=lambda kk: roles[kk])
            roles[k] -= 1
    m_drones = _MAX_DRONES_RE.search(text)
    m_time = _MAX_TIME_RE.search(text)
    constraints = dict(DEFAULT_CONSTRAINTS)
    constraints["max_drones"] = int(m_drones.group(1)) if m_drones \
        else int(max_drones)
    constraints["max_drones"] = max(1, min(10, constraints["max_drones"]))
    if m_time:
        constraints["max_time_min"] = max(1, int(m_time.group(1)))
    # Re-fit roles to an explicit drone cap from text.
    total = sum(roles.values()) or 1
    cap = constraints["max_drones"]
    if total > cap and intent_name != "return_home":
        scale = cap / total
        roles = {k: max(1 if v > 0 else 0, int(round(v * scale)))
                 for k, v in roles.items()}
        while sum(roles.values()) > cap:
            roles[max(roles, key=lambda kk: roles[kk])] -= 1
    priority = 1
    if re.search(r"\b(urgent|emergency|immediate|asap|critical)\b", text, re.I):
        priority = 1
    validated = validate_swarm_intent({
        "intent": intent_name,
        "area": {"building": building, "floors": floors},
        "objectives": objectives,
        "constraints": constraints,
        "roles": roles,
        "priority": priority,
    })
    graph = intent_to_task_graph(validated)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return SwarmIntent(intent=validated["intent"], area=validated["area"],
                       objectives=validated["objectives"],
                       constraints=validated["constraints"],
                       roles=validated["roles"], task_graph=graph,
                       priority=validated["priority"],
                       latency_ms=latency_ms)


def parse_with_llm_backend(text: str, timeout_s: float = 10.0,
                           max_drones: int = 6) -> SwarmIntent:
    """Try an OpenAI-compatible LLM/SLM, else local parser (SAR-only)."""
    import json as _json
    import os as _os
    import urllib.request as _req

    base = _os.environ.get("OPENAI_API_URL", "").strip()
    key = _os.environ.get("OPENAI_API_KEY", "").strip()
    model = _os.environ.get("OPENAI_MODEL", "slm-local").strip() or "slm-local"
    if not base or not key:
        return parse_swarm_command(text, max_drones=max_drones)
    try:
        payload = _json.dumps({
            "model": model,
            "messages": [{"role": "user",
                          "content": build_swarm_prompt(text)}],
            "temperature": 0.0,
            "max_tokens": 512,
        }).encode("utf-8")
        req = _req.Request(base.rstrip("/") + "/chat/completions", data=payload,
                           headers={"Content-Type": "application/json",
                                    "Authorization": f"Bearer {key}"})
        with _req.urlopen(req, timeout=timeout_s) as resp:
            body = _json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        s, e = content.find("{"), content.rfind("}")
        data = _json.loads(content[s:e + 1] if s >= 0 and e > s else content)
        validated = validate_swarm_intent(data)
        graph = intent_to_task_graph(validated)
        return SwarmIntent(intent=validated["intent"], area=validated["area"],
                           objectives=validated["objectives"],
                           constraints=validated["constraints"],
                           roles=validated["roles"], task_graph=graph,
                           priority=validated["priority"])
    except Exception:
        return parse_swarm_command(text, max_drones=max_drones)


def estimate_parser_accuracy(commands: List[str],
                             expected: List[str]) -> float:
    """Fraction of commands whose parsed intent matches `expected`."""
    if not commands:
        return 1.0
    hits = sum(1 for c, e in zip(commands, expected)
               if parse_swarm_command(c).intent == e)
    return hits / len(commands)
