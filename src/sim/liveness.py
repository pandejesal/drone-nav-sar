#!/usr/bin/env python3
"""Sprint 24: LTL model checking — liveness, safety, response, fairness.

Finite-trace (LTLf) semantics SPIN/NuSMV-style: parse an LTL formula into an
AST, evaluate it over a trace of proposition valuations, and report a
counterexample index on failure. Also ships an explicit Büchi automaton
fragment for G/F-class properties. NumPy-free stdlib core, SAR-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# -- LTL AST ---------------------------------------------------------------
class Node:
    pass


@dataclass
class Prop(Node):
    name: str


@dataclass
class Const(Node):
    value: bool


@dataclass
class Not(Node):
    child: Node


@dataclass
class And(Node):
    left: Node
    right: Node


@dataclass
class Or(Node):
    left: Node
    right: Node


@dataclass
class Next(Node):
    child: Node


@dataclass
class Eventually(Node):
    child: Node


@dataclass
class Always(Node):
    child: Node


@dataclass
class Until(Node):
    left: Node
    right: Node


class LTLParser:
    """Tokens: F/<> (eventually), G/[] (always), X (next), U, !, &, |, ->."""

    def __init__(self, text: str):
        self.toks = self._tokenize(text)
        self.pos = 0

    static_aliases = {"<>": "F", "[]": "G"}

    def _tokenize(self, text: str) -> List[str]:
        t = text.replace("<>", " F ").replace("[]", " G ").replace("->", " -> ")
        for ch in "()!&|XUGF":
            t = t.replace(ch, f" {ch} ")
        # 'U' inside identifiers would split; re-join alnum runs below is
        # unnecessary since proposition names are single alnum/underscore runs
        # and 'U' as operator is standalone. Keep simple: split whitespace.
        out: List[str] = []
        for tok in t.split():
            # re-split glued single letters, e.g. "FG" -> F G (rare, best-effort)
            if len(tok) > 1 and all(c in "XUGF" for c in tok):
                out.extend(list(tok))
            else:
                out.append(tok)
        return out

    def parse(self) -> Node:
        n = self._implies()
        if self.pos != len(self.toks):
            raise ValueError(f"Trailing tokens: {self.toks[self.pos:]}")
        return n

    def _peek(self) -> Optional[str]:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def _eat(self, tok: str) -> None:
        if self._peek() != tok:
            raise ValueError(f"Expected {tok!r}, got {self._peek()!r}")
        self.pos += 1

    def _implies(self) -> Node:
        left = self._or()
        while self._peek() == "->":
            self._eat("->")
            left = Or(Not(left), self._or())
        return left

    def _or(self) -> Node:
        left = self._and()
        while self._peek() == "|":
            self._eat("|")
            left = Or(left, self._and())
        return left

    def _and(self) -> Node:
        left = self._unary()
        while self._peek() == "&":
            self._eat("&")
            left = And(left, self._unary())
        return left

    def _unary(self) -> Node:
        tok = self._peek()
        if tok == "!":
            self._eat("!")
            return Not(self._unary())
        if tok == "X":
            self._eat("X")
            return Next(self._unary())
        if tok in ("F", "G"):
            self._eat(tok)
            child = self._unary()
            return Eventually(child) if tok == "F" else Always(child)
        return self._until()

    def _until(self) -> Node:
        left = self._atom()
        if self._peek() == "U":
            self._eat("U")
            return Until(left, self._until())
        return left

    def _atom(self) -> Node:
        tok = self._peek()
        if tok == "(":
            self._eat("(")
            n = self._implies()
            self._eat(")")
            return n
        if tok in ("true", "True"):
            self.pos += 1
            return Const(True)
        if tok in ("false", "False"):
            self.pos += 1
            return Const(False)
        if tok is None or tok in (")", "&", "|", "->", "U"):
            raise ValueError(f"Unexpected token {tok!r} in {self.toks}")
        self.pos += 1
        return Prop(tok)


def parse_ltl(formula: str) -> Node:
    return LTLParser(formula).parse()


def eval_ltl(node: Node, trace: List[Dict[str, bool]], i: int = 0) -> bool:
    """Finite-trace semantics; out-of-range Next is False, Always vacuous."""
    n = len(trace)
    if isinstance(node, Const):
        return node.value
    if isinstance(node, Prop):
        return bool(trace[i].get(node.name, False)) if 0 <= i < n else False
    if isinstance(node, Not):
        return not eval_ltl(node.child, trace, i)
    if isinstance(node, And):
        return eval_ltl(node.left, trace, i) and eval_ltl(node.right, trace, i)
    if isinstance(node, Or):
        return eval_ltl(node.left, trace, i) or eval_ltl(node.right, trace, i)
    if isinstance(node, Next):
        return eval_ltl(node.child, trace, i + 1) if i + 1 < n else False
    if isinstance(node, Eventually):
        return any(eval_ltl(node.child, trace, k) for k in range(i, n))
    if isinstance(node, Always):
        return all(eval_ltl(node.child, trace, k) for k in range(i, n))
    if isinstance(node, Until):
        for k in range(i, n):
            if eval_ltl(node.right, trace, k):
                if all(eval_ltl(node.left, trace, j) for j in range(i, k)):
                    return True
        return False
    raise TypeError(f"Unknown LTL node {node!r}")


@dataclass
class BuchiAutomaton:
    """Explicit-state Büchi fragment for G/F-class LTL (SPIN-style)."""
    name: str
    states: List[str] = field(default_factory=list)
    initial: str = "q0"
    accepting: List[str] = field(default_factory=list)
    transitions: List[Tuple[str, str, str]] = field(default_factory=list)

    @staticmethod
    def for_eventually(prop: str) -> "BuchiAutomaton":
        return BuchiAutomaton(
            name=f"F {prop}", states=["q0", "q1"], initial="q0",
            accepting=["q1"],
            transitions=[("q0", f"!{prop}", "q0"), ("q0", prop, "q1"),
                         ("q1", "true", "q1")])

    @staticmethod
    def for_always(prop: str) -> "BuchiAutomaton":
        return BuchiAutomaton(
            name=f"G {prop}", states=["q0", "qrej"], initial="q0",
            accepting=["q0"],
            transitions=[("q0", prop, "q0"), ("q0", f"!{prop}", "qrej"),
                         ("qrej", "true", "qrej")])


@dataclass
class LivenessResult:
    satisfied: bool
    formula: str
    counterexample_step: Optional[int] = None
    automaton: Optional[BuchiAutomaton] = None


class LivenessVerifier:
    """SPIN/NuSMV-style LTL model checker over finite proposition traces."""

    def check(self, trace: List[Dict[str, bool]], formula: str) -> LivenessResult:
        node = parse_ltl(formula)
        if eval_ltl(node, trace):
            auto = self._automaton_for(formula)
            return LivenessResult(True, formula, None, auto)
        # Minimal counterexample: first prefix from which it already fails.
        cex: Optional[int] = None
        for k in range(len(trace)):
            if not eval_ltl(node, trace[: k + 1]):
                # report the first index where failure is already decided for
                # safety (G) formulas; for liveness report end of trace.
                cex = k
                if isinstance(node, (Always, Not)):
                    break
        else:
            cex = len(trace) - 1
        return LivenessResult(False, formula, cex, self._automaton_for(formula))

    def check_eventually(self, trace: List[Dict[str, bool]], prop: str) -> LivenessResult:
        return self.check(trace, f"F {prop}")

    def check_always(self, trace: List[Dict[str, bool]], prop: str) -> LivenessResult:
        return self.check(trace, f"G {prop}")

    def check_response(self, trace: List[Dict[str, bool]],
                       request: str, response: str) -> LivenessResult:
        return self.check(trace, f"G ({request} -> F {response})")

    def check_fairness(self, trace: List[Dict[str, bool]],
                       enabled: str, taken: str) -> LivenessResult:
        return self.check(trace, f"(G F {enabled}) -> (G F {taken})")

    def _automaton_for(self, formula: str) -> Optional[BuchiAutomaton]:
        f = formula.strip()
        if f.startswith("F ") and len(f.split()) == 2:
            return BuchiAutomaton.for_eventually(f.split()[1])
        if f.startswith("G ") and len(f.split()) == 2:
            return BuchiAutomaton.for_always(f.split()[1])
        return None


def propositions_from_episode(positions, goal, obstacles=(),
                              goal_radius: float = 1.0,
                              obstacle_margin: float = 1.0,
                              obstacle_radius: float = 1.0):
    """Build a per-step {goal, safe} proposition trace from positions."""
    import numpy as np
    P = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    G = np.asarray(goal, dtype=np.float64).reshape(3)
    trace: List[Dict[str, bool]] = []
    for p in P:
        at_goal = bool(np.linalg.norm(p - G) <= goal_radius)
        safe = True
        for o in obstacles:
            if float(np.linalg.norm(p - np.asarray(o).reshape(3))) < obstacle_radius + obstacle_margin:
                safe = False
                break
        trace.append({"goal": at_goal, "safe": safe,
                      "obstacle": not safe, "request": False, "response": at_goal})
    return trace
