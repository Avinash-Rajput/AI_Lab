#!/usr/bin/env python3
"""Propositional Logic Medical Diagnosis Simulator - interactive GUI.

A visual simulator in the style of a "circuit board":
  * Left canvas : symptom propositions (click to toggle) wired into diagnosis rules.
  * Right panel : Run / Pause / Step / Reset, speed slider and tabs
                  (Symptoms, Rules, Results, History, Truth Table, Logic).
  * Bottom      : live evaluation log and a permanent safety notice.

Only the Python standard library is used (tkinter ships with Python).
Run with:  python3 medical_diagnosis_gui.py
(On Ubuntu/Debian, if tkinter is missing:  sudo apt install python3-tk)
"""

from __future__ import annotations

import itertools
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Sequence, Set

try:
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import messagebox, ttk
except ImportError:  # pragma: no cover
    print("tkinter is not available. On Ubuntu/Debian install it with:\n"
          "    sudo apt install python3-tk")
    sys.exit(1)


URGENT_MESSAGE = (
    "You reported BOTH chest pain AND breathing difficulty. This combination "
    "can be serious. Please seek urgent medical evaluation right away "
    "(contact a doctor or your local emergency services). Do NOT treat any "
    "result from this simulator as a diagnosis."
)


# --------------------------------------------------------------------------
# Logical expression system (AND / OR / NOT)
# --------------------------------------------------------------------------
Facts = Dict[str, bool]


class Expression(ABC):
    @abstractmethod
    def evaluate(self, facts: Facts) -> bool:
        """Evaluate against a dictionary of Boolean facts."""

    @abstractmethod
    def render(self, leaf: Callable[["Proposition"], str]) -> str:
        """Render the expression, printing each proposition with `leaf`."""

    @abstractmethod
    def propositions(self) -> List["Proposition"]:
        """Unique propositions used, in order of appearance."""

    def conditions(self) -> List["Expression"]:
        return [self]

    def to_text(self) -> str:
        return self.render(lambda p: p.name)

    def to_symbols(self) -> str:
        return self.render(lambda p: p.symbol)

    def to_values(self, facts: Facts) -> str:
        return self.render(lambda p: "TRUE" if facts.get(p.symbol, False) else "FALSE")

    @staticmethod
    def _wrap_child(child: "Expression", leaf: Callable[["Proposition"], str]) -> str:
        text = child.render(leaf)
        return f"({text})" if isinstance(child, (And, Or)) else text


class Proposition(Expression):
    def __init__(self, symbol: str, name: str) -> None:
        self.symbol = symbol
        self.name = name

    def evaluate(self, facts: Facts) -> bool:
        return bool(facts.get(self.symbol, False))

    def render(self, leaf: Callable[["Proposition"], str]) -> str:
        return leaf(self)

    def propositions(self) -> List["Proposition"]:
        return [self]


class Not(Expression):
    def __init__(self, operand: Expression) -> None:
        self.operand = operand

    def evaluate(self, facts: Facts) -> bool:
        return not self.operand.evaluate(facts)

    def render(self, leaf: Callable[[Proposition], str]) -> str:
        return "NOT " + self._wrap_child(self.operand, leaf)

    def propositions(self) -> List[Proposition]:
        return self.operand.propositions()


class _NaryOperator(Expression):
    operator = ""

    def __init__(self, *operands: Expression) -> None:
        if len(operands) < 2:
            raise ValueError(f"{self.operator} needs at least two operands")
        self.operands: List[Expression] = list(operands)

    def render(self, leaf: Callable[[Proposition], str]) -> str:
        parts = [self._wrap_child(op, leaf) for op in self.operands]
        return f" {self.operator} ".join(parts)

    def propositions(self) -> List[Proposition]:
        seen: Dict[str, Proposition] = {}
        for operand in self.operands:
            for prop in operand.propositions():
                seen.setdefault(prop.symbol, prop)
        return list(seen.values())


class And(_NaryOperator):
    operator = "AND"

    def evaluate(self, facts: Facts) -> bool:
        return all(op.evaluate(facts) for op in self.operands)

    def conditions(self) -> List[Expression]:
        return list(self.operands)


class Or(_NaryOperator):
    operator = "OR"

    def evaluate(self, facts: Facts) -> bool:
        return any(op.evaluate(facts) for op in self.operands)


AND, OR, NOT = And, Or, Not


def negated_symbols(expression: Expression) -> Set[str]:
    """Symbols that appear directly under a NOT (used to draw inverter wires)."""
    if isinstance(expression, Not):
        if isinstance(expression.operand, Proposition):
            return {expression.operand.symbol}
        return negated_symbols(expression.operand)
    if isinstance(expression, _NaryOperator):
        found: Set[str] = set()
        for operand in expression.operands:
            found |= negated_symbols(operand)
        return found
    return set()


# --------------------------------------------------------------------------
# Propositions, rules, engine
# --------------------------------------------------------------------------
Fever = Proposition("F", "Fever")
Cough = Proposition("C", "Cough")
BreathingDifficulty = Proposition("B", "Breathing Difficulty")
ChestPain = Proposition("P", "Chest Pain")
SoreThroat = Proposition("S", "Sore Throat")
RunnyNose = Proposition("R", "Runny Nose")
Headache = Proposition("H", "Headache")
Wheezing = Proposition("W", "Wheezing")
Fatigue = Proposition("T", "Fatigue")

SYMPTOMS: List[Proposition] = [
    Fever, Cough, BreathingDifficulty, ChestPain, SoreThroat,
    RunnyNose, Headache, Wheezing, Fatigue,
]


@dataclass(frozen=True)
class Rule:
    name: str
    expression: Expression
    explanation: str


@dataclass
class DiagnosisResult:
    rule: Rule
    satisfied: int
    total: int
    is_match: bool

    @property
    def percent(self) -> int:
        return int(round(100 * self.satisfied / self.total)) if self.total else 0


def build_rules() -> List[Rule]:
    return [
        Rule("Pneumonia", AND(Fever, Cough, BreathingDifficulty),
             "Fever, cough and breathing difficulty together satisfy this rule."),
        Rule("Flu", AND(Fever, Cough, Fatigue),
             "Fever, cough and fatigue together satisfy this rule."),
        Rule("Common Cold", AND(Cough, RunnyNose, SoreThroat),
             "Cough, runny nose and sore throat together satisfy this rule."),
        Rule("Bronchitis", AND(Cough, BreathingDifficulty, Wheezing),
             "Cough, breathing difficulty and wheezing together satisfy this rule."),
        Rule("COVID-like Respiratory Infection",
             AND(Fever, Cough, BreathingDifficulty, Fatigue),
             "Fever, cough, breathing difficulty and fatigue all hold."),
        Rule("Migraine", AND(Headache, NOT(Fever)),
             "Headache is present and fever is absent."),
        Rule("Angina-like Condition", AND(ChestPain, BreathingDifficulty),
             "Chest pain and breathing difficulty hold together."),
        Rule("Respiratory Infection", AND(Cough, Fever),
             "Cough and fever hold together."),
        Rule("Severe Respiratory Condition", AND(BreathingDifficulty, ChestPain),
             "Breathing difficulty and chest pain hold together."),
        Rule("Upper Respiratory Irritation", AND(SoreThroat, OR(Cough, RunnyNose)),
             "Sore throat holds and at least one of cough or runny nose holds."),
    ]


class LogicEngine:
    def __init__(self, rules: Sequence[Rule]) -> None:
        self.rules = list(rules)

    @staticmethod
    def evaluate_rule(rule: Rule, facts: Facts) -> DiagnosisResult:
        conditions = rule.expression.conditions()
        satisfied = sum(1 for c in conditions if c.evaluate(facts))
        return DiagnosisResult(rule, satisfied, len(conditions),
                               rule.expression.evaluate(facts))

    def evaluate_all(self, facts: Facts) -> List[DiagnosisResult]:
        return [self.evaluate_rule(rule, facts) for rule in self.rules]

    @staticmethod
    def matches(results: Sequence[DiagnosisResult]) -> List[DiagnosisResult]:
        found = [r for r in results if r.is_match]
        return sorted(found, key=lambda r: (-r.percent, -r.total))

    @staticmethod
    def unmet_conditions(rule: Rule, facts: Facts) -> List[str]:
        """Conditions of a rule that are still FALSE for the given facts."""
        return [c.to_text() for c in rule.expression.conditions() if not c.evaluate(facts)]

    @staticmethod
    def supporting_symptoms(rule: Rule, facts: Facts) -> List[str]:
        """Symptoms present (TRUE) that this rule uses positively."""
        negated = negated_symbols(rule.expression)
        return [p.name for p in rule.expression.propositions()
                if facts.get(p.symbol, False) and p.symbol not in negated]

    @staticmethod
    def closest(results: Sequence[DiagnosisResult], facts: Facts,
                limit: int = 3) -> List[DiagnosisResult]:
        candidates = []
        for r in results:
            if r.is_match or r.satisfied == 0:
                continue
            if not any(facts.get(p.symbol, False) for p in r.rule.expression.propositions()):
                continue
            candidates.append(r)
        candidates.sort(key=lambda r: (-r.percent, -r.satisfied))
        return candidates[:limit]


@dataclass
class DiagnosisSession:
    timestamp: datetime
    facts: Facts
    results: List[DiagnosisResult]
    closest: List[DiagnosisResult] = field(default_factory=list)

    def symptoms_present(self) -> List[str]:
        return [s.name for s in SYMPTOMS if self.facts.get(s.symbol, False)]

    def matched(self) -> List[DiagnosisResult]:
        return [r for r in self.results if r.is_match]


class DiagnosisHistory:
    def __init__(self) -> None:
        self._sessions: List[DiagnosisSession] = []

    def add(self, session: DiagnosisSession) -> None:
        self._sessions.append(session)

    def all(self) -> List[DiagnosisSession]:
        return list(self._sessions)


# --------------------------------------------------------------------------
# GUI layout constants
# --------------------------------------------------------------------------
CANVAS_W, CANVAS_H = 820, 560
NODE_X, NODE_R = 215, 17
RULE_X, RULE_W, RULE_H = 470, 335, 38

PRESETS = {
    "Flu-like (F, C, T)": ["F", "C", "T"],
    "Cold-like (C, R, S)": ["C", "R", "S"],
    "Pneumonia-like (F, C, B)": ["F", "C", "B"],
    "Bronchitis-like (C, B, W)": ["C", "B", "W"],
    "Migraine-like (H)": ["H"],
    "Chest pain + breathing (warning)": ["P", "B"],
}

COLOR_BG = "#e6f2fb"
COLOR_TRUE = "#2e9e4f"
COLOR_FALSE = "#d05a5a"
COLOR_ACTIVE = "#f39c12"
COLOR_IDLE = "#b9cbd9"


def node_y(index: int) -> int:
    return 100 + index * 46


def rule_y(index: int) -> int:
    return 76 + index * 45


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------
class SimulatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.engine = LogicEngine(build_rules())
        self.history = DiagnosisHistory()

        self.facts: Facts = {s.symbol: False for s in SYMPTOMS}
        self.snapshot: Facts = dict(self.facts)
        self.symptom_vars: Dict[str, tk.BooleanVar] = {}
        self.symbol_index = {s.symbol: i for i, s in enumerate(SYMPTOMS)}

        self.results: List[DiagnosisResult] = []
        self.active = False
        self.finished = False
        self.running = False
        self.after_id = None

        self.speed = tk.IntVar(value=500)
        self.status_var = tk.StringVar(value="Ready. Press Run or Step.")
        self.mono = tkfont.nametofont("TkFixedFont")

        self._build_ui()
        self.redraw()
        self.update_buttons()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        root = self.root
        root.title("Propositional Logic Medical Diagnosis Simulator - interactive")
        root.geometry("1290x820")
        root.minsize(1100, 700)
        try:
            ttk.Style().theme_use("clam")
        except tk.TclError:
            pass

        menubar = tk.Menu(root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        root.config(menu=menubar)

        tk.Label(root, text="Propositional Logic Medical Diagnosis Simulator",
                 font=("Helvetica", 18, "bold")).pack(pady=(8, 0))
        tk.Label(root, fg="#444",
                 text="Symptoms are Boolean propositions  \u2022  rules use AND / OR / NOT  "
                      "\u2022  every rule is evaluated, none stops at the first match"
                 ).pack(pady=(0, 6))

        main = tk.Frame(root)
        main.pack(fill="both", expand=True, padx=8)

        self.canvas = tk.Canvas(main, width=CANVAS_W, height=CANVAS_H,
                                bg=COLOR_BG, highlightthickness=1,
                                highlightbackground="#889")
        self.canvas.pack(side="left", anchor="n")
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        panel = tk.Frame(main)
        panel.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self._build_controls(panel)
        self._build_tabs(panel)

        log_frame = tk.Frame(root)
        log_frame.pack(fill="x", padx=8, pady=(6, 0))
        self.log_text = tk.Text(log_frame, height=7, font=self.mono, wrap="word",
                                state="disabled", bg="#fbfbfb")
        log_scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="x", expand=True)
        log_scroll.pack(side="left", fill="y")
        self.log_text.tag_configure("head", font=(self.mono.actual("family"), 10, "bold"))
        self.log_text.tag_configure("true", foreground=COLOR_TRUE)
        self.log_text.tag_configure("false", foreground="#777")

        tk.Label(root, textvariable=self.status_var,
                 font=("Helvetica", 11, "bold")).pack(pady=(4, 0))
        tk.Frame(root, height=6).pack()

    def _build_controls(self, parent: tk.Frame) -> None:
        bar = tk.Frame(parent)
        bar.pack(fill="x")
        self.run_btn = ttk.Button(bar, text="\u25B6 Run", command=self.run)
        self.pause_btn = ttk.Button(bar, text="\u23F8 Pause", command=self.pause)
        self.step_btn = ttk.Button(bar, text="\u23ED Step", command=self.step)
        self.reset_btn = ttk.Button(bar, text="\u27F2 Reset", command=self.reset)
        for button in (self.run_btn, self.pause_btn, self.step_btn, self.reset_btn):
            button.pack(side="left", padx=2, fill="x", expand=True)

        tk.Label(parent, text="Speed (fast \u2192 slow), ms per rule:",
                 font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(8, 0))
        tk.Scale(parent, from_=100, to=1500, resolution=50, orient="horizontal",
                 variable=self.speed).pack(fill="x")

    def _make_text(self, parent: tk.Widget, mono: bool = True) -> tk.Text:
        frame = tk.Frame(parent)
        frame.pack(fill="both", expand=True)
        text = tk.Text(frame, wrap="word", state="disabled", width=46,
                       font=self.mono if mono else None, bg="#fdfdfd")
        scroll = ttk.Scrollbar(frame, command=text.yview)
        text.config(yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")
        base = self.mono.actual("family")
        text.tag_configure("h1", font=(base, 12, "bold"))
        text.tag_configure("h2", font=(base, 10, "bold"), foreground="#124a7a")
        text.tag_configure("true", foreground=COLOR_TRUE, font=(base, 10, "bold"))
        text.tag_configure("false", foreground=COLOR_FALSE, font=(base, 10, "bold"))
        text.tag_configure("warn", foreground="white", background="#c0392b",
                           font=(base, 10, "bold"))
        text.tag_configure("note", foreground="#665", font=(base, 9, "italic"))
        text.tag_configure("safety", foreground="#5a4500", background="#fff4cc")
        return text

    @staticmethod
    def _put(widget: tk.Text, text: str, *tags: str) -> None:
        widget.insert("end", text, tags)

    def _build_tabs(self, parent: tk.Frame) -> None:
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True, pady=(8, 0))

        # --- Symptoms tab
        tab_symptoms = tk.Frame(self.notebook, padx=8, pady=6)
        self.notebook.add(tab_symptoms, text="Symptoms")
        box = ttk.LabelFrame(tab_symptoms, text="Do you have...  (tick = TRUE)")
        box.pack(fill="x")
        for symptom in SYMPTOMS:
            var = tk.BooleanVar(value=False)
            self.symptom_vars[symptom.symbol] = var
            ttk.Checkbutton(
                box, text=f"{symptom.symbol}   \u2192   {symptom.name}", variable=var,
                command=lambda s=symptom.symbol: self.on_symptom_changed(s)
            ).pack(anchor="w", padx=8, pady=2)
        row = tk.Frame(tab_symptoms)
        row.pack(fill="x", pady=6)
        ttk.Button(row, text="Select all", command=lambda: self.set_all(True)
                   ).pack(side="left", padx=2)
        ttk.Button(row, text="Clear all", command=lambda: self.set_all(False)
                   ).pack(side="left", padx=2)
        tk.Label(tab_symptoms, text="Example cases:").pack(anchor="w", pady=(6, 0))
        self.preset_combo = ttk.Combobox(tab_symptoms, values=list(PRESETS),
                                         state="readonly")
        self.preset_combo.pack(fill="x")
        self.preset_combo.bind("<<ComboboxSelected>>", self.on_preset)
        tk.Label(tab_symptoms, fg="#555", justify="left", wraplength=400,
                 text="Tip: you can also click a symptom circle on the board. "
                      "Then press Run (animated) or Step (one rule at a time)."
                 ).pack(anchor="w", pady=8)

        # --- Rules tab
        tab_rules = tk.Frame(self.notebook, padx=4, pady=4)
        self.notebook.add(tab_rules, text="Rules")
        rules_text = self._make_text(tab_rules)
        rules_text.config(state="normal")
        self._put(rules_text, "PROPOSITIONS\n", "h1")
        for s in SYMPTOMS:
            self._put(rules_text, f"{s.symbol:<3} -> {s.name}\n")
        self._put(rules_text, "\nPROPOSITIONAL RULES\n", "h1")
        for number, rule in enumerate(self.engine.rules, start=1):
            self._put(rules_text, f"\n{number}. {rule.name}\n", "h2")
            self._put(rules_text, f"   {rule.expression.to_text()}\n")
            self._put(rules_text, f"   ({rule.expression.to_symbols()})\n", "note")
        rules_text.config(state="disabled")

        # --- Results tab
        tab_results = tk.Frame(self.notebook, padx=4, pady=4)
        self.notebook.add(tab_results, text="Results")
        self.tab_results = tab_results
        self.results_text = self._make_text(tab_results)
        self._show_results_placeholder()

        # --- History tab
        tab_history = tk.Frame(self.notebook, padx=4, pady=4)
        self.notebook.add(tab_history, text="History")
        self.history_list = tk.Listbox(tab_history, height=6, exportselection=False)
        self.history_list.pack(fill="x")
        self.history_list.bind("<<ListboxSelect>>", self.on_history_select)
        self.history_text = self._make_text(tab_history)

        # --- Truth table tab
        tab_truth = tk.Frame(self.notebook, padx=4, pady=4)
        self.notebook.add(tab_truth, text="Truth Table")
        self.tab_truth = tab_truth
        tk.Label(tab_truth, text="Select a rule:").pack(anchor="w")
        self.truth_combo = ttk.Combobox(
            tab_truth, values=[r.name for r in self.engine.rules], state="readonly")
        self.truth_combo.pack(fill="x", pady=(0, 4))
        self.truth_combo.current(0)
        self.truth_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self.show_truth_table(self.truth_combo.current()))
        self.truth_text = self._make_text(tab_truth)
        self.show_truth_table(0, switch_tab=False)

        # --- Logic explanation tab
        tab_logic = tk.Frame(self.notebook, padx=4, pady=4)
        self.notebook.add(tab_logic, text="Logic")
        logic_text = self._make_text(tab_logic)
        self._fill_logic_explanation(logic_text)

    # -------------------------------------------------------- explanation
    def _fill_logic_explanation(self, text: tk.Text) -> None:
        text.config(state="normal")
        pneumonia = self.engine.rules[0]
        migraine = self.engine.rules[5]
        all_true = {"F": True, "C": True, "B": True}
        one_false = {"F": True, "C": True, "B": False}
        mig = {"H": True, "F": False}
        put = lambda s, *t: self._put(text, s, *t)  # noqa: E731
        put("HOW PROPOSITIONAL LOGIC IS USED\n", "h1")
        put("\nEach symptom is a proposition that is TRUE or FALSE:\n")
        put("  F = Fever\n  C = Cough\n  B = Breathing Difficulty\n  ...\n")
        put("\nOperators\n", "h2")
        put("  AND : TRUE only if every part is TRUE\n")
        put("  OR  : TRUE if at least one part is TRUE\n")
        put("  NOT : flips TRUE <-> FALSE\n")
        put("\nExample: Pneumonia\n", "h2")
        put(f"  {pneumonia.expression.to_symbols()}\n")
        put("  If F = TRUE, C = TRUE, B = TRUE then\n")
        put(f"  {pneumonia.expression.to_values(all_true)} = TRUE\n")
        put("  Therefore the rule is satisfied.\n")
        put("\n  If B = FALSE then\n")
        put(f"  {pneumonia.expression.to_values(one_false)} = FALSE\n")
        put("  Therefore the rule is NOT satisfied.\n")
        put("\nExample with NOT: Migraine\n", "h2")
        put(f"  {migraine.expression.to_symbols()}\n")
        put("  If H = TRUE and F = FALSE then\n")
        put(f"  {migraine.expression.to_values(mig)} = TRUE\n")
        put("\nRule match score\n", "h2")
        put("  satisfied conditions / total conditions of a rule\n"
            "  (e.g. 3/4 = 75%). It is only a logical rule-match score,\n"
            "  NOT a medical probability.\n")
        text.config(state="disabled")

    # ---------------------------------------------------------- symptoms
    def on_symptom_changed(self, symbol: str) -> None:
        self.facts[symbol] = self.symptom_vars[symbol].get()
        self.reset_evaluation("Symptoms changed. Press Run or Step.")

    def set_all(self, value: bool) -> None:
        for symbol, var in self.symptom_vars.items():
            var.set(value)
            self.facts[symbol] = value
        self.reset_evaluation("Symptoms updated. Press Run or Step.")

    def on_preset(self, _event=None) -> None:
        chosen = PRESETS.get(self.preset_combo.get())
        if chosen is None:
            return
        for symbol, var in self.symptom_vars.items():
            value = symbol in chosen
            var.set(value)
            self.facts[symbol] = value
        self.reset_evaluation(f"Loaded example: {self.preset_combo.get()}. Press Run or Step.")

    # ------------------------------------------------------------ canvas
    def redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        c.create_rectangle(0, 0, CANVAS_W, CANVAS_H, fill=COLOR_BG, outline="")
        c.create_rectangle(0, 0, CANVAS_W, 40, fill="#2c3e50", outline="")
        # small hospital cross
        c.create_rectangle(14, 10, 34, 30, fill="white", outline="")
        c.create_rectangle(22, 12, 26, 28, fill="#c0392b", outline="")
        c.create_rectangle(16, 18, 32, 22, fill="#c0392b", outline="")
        c.create_text(44, 20, anchor="w", fill="white", font=("Helvetica", 12, "bold"),
                      text="LOGIC BOARD")
        c.create_text(CANVAS_W - 12, 20, anchor="e", fill="#dfe8f0",
                      font=("Helvetica", 10),
                      text=f"Rules evaluated: {len(self.results)}/{len(self.engine.rules)}")
        c.create_text(NODE_X - 60, 58, text="SYMPTOMS (propositions)",
                      font=("Helvetica", 10, "bold"), fill="#1f4e79")
        c.create_text(RULE_X + RULE_W / 2, 58, text="DIAGNOSIS RULES (logic gates)",
                      font=("Helvetica", 10, "bold"), fill="#1f4e79")

        current = len(self.results) - 1 if (self.active and not self.finished) else -1
        self._draw_wires(current)
        self._draw_symptoms()
        self._draw_rules(current)
        self._draw_banner()

    def _draw_wires(self, current: int) -> None:
        c = self.canvas
        for j, rule in enumerate(self.engine.rules):
            props = rule.expression.propositions()
            negated = negated_symbols(rule.expression)
            evaluated = j < len(self.results)
            count = len(props)
            for k, prop in enumerate(props):
                sx = NODE_X + NODE_R
                sy = node_y(self.symbol_index[prop.symbol])
                ex = RULE_X
                ey = rule_y(j) + RULE_H * (k + 1) / (count + 1)
                if j == current:
                    color, width = COLOR_ACTIVE, 3
                elif evaluated:
                    on = self.snapshot.get(prop.symbol, False)
                    color, width = (COLOR_TRUE, 2) if on else ("#dcaaaa", 1)
                else:
                    color, width = COLOR_IDLE, 1
                dash = (4, 3) if prop.symbol in negated else None
                c.create_line(sx, sy, ex, ey, fill=color, width=width, dash=dash)
                if prop.symbol in negated:  # inverter bubble
                    c.create_oval(ex - 9, ey - 4, ex - 1, ey + 4, fill="white", outline=color)

    def _draw_symptoms(self) -> None:
        c = self.canvas
        for i, symptom in enumerate(SYMPTOMS):
            y = node_y(i)
            on = self.facts[symptom.symbol]
            c.create_text(NODE_X - NODE_R - 10, y, anchor="e", text=symptom.name,
                          font=("Helvetica", 10), fill="#223")
            c.create_oval(NODE_X - NODE_R, y - NODE_R, NODE_X + NODE_R, y + NODE_R,
                          fill=COLOR_TRUE if on else "white",
                          outline=COLOR_TRUE if on else "#8aa0b2", width=2)
            c.create_text(NODE_X, y, text=symptom.symbol,
                          font=("Helvetica", 12, "bold"),
                          fill="white" if on else "#445")

    def _draw_rules(self, current: int) -> None:
        c = self.canvas
        for j, rule in enumerate(self.engine.rules):
            x, y = RULE_X, rule_y(j)
            result = self.results[j] if j < len(self.results) else None
            if result is None:
                fill, outline, width = "white", COLOR_IDLE, 1
            elif result.is_match:
                fill, outline, width = "#c8f0cf", COLOR_TRUE, 2
            else:
                fill, outline, width = "#eceff1", "#9aa8b4", 1
            if j == current:
                outline, width = COLOR_ACTIVE, 4
            c.create_rectangle(x, y, x + RULE_W, y + RULE_H, fill=fill,
                               outline=outline, width=width)
            c.create_text(x + 8, y + 11, anchor="w", text=rule.name,
                          font=("Helvetica", 10, "bold"), fill="#123")
            c.create_text(x + 8, y + 27, anchor="w", text=rule.expression.to_symbols(),
                          font=("Helvetica", 9), fill="#456")
            if result is None:
                c.create_text(x + RULE_W - 8, y + 19, anchor="e", text="pending",
                              font=("Helvetica", 9, "italic"), fill="#889")
            else:
                c.create_text(x + RULE_W - 8, y + 11, anchor="e",
                              text="TRUE \u2713" if result.is_match else "FALSE",
                              font=("Helvetica", 10, "bold"),
                              fill=COLOR_TRUE if result.is_match else COLOR_FALSE)
                c.create_text(x + RULE_W - 8, y + 27, anchor="e",
                              text=f"{result.satisfied}/{result.total} \u2022 {result.percent}%",
                              font=("Helvetica", 9), fill="#334")

    def _draw_banner(self) -> None:
        c = self.canvas
        if self.facts["P"] and self.facts["B"]:
            c.create_rectangle(10, CANVAS_H - 44, CANVAS_W - 10, CANVAS_H - 8,
                               fill="#c0392b", outline="")
            c.create_text(CANVAS_W / 2, CANVAS_H - 26, fill="white",
                          font=("Helvetica", 11, "bold"),
                          text="\u26A0 URGENT: chest pain + breathing difficulty - "
                               "seek urgent medical evaluation")
        else:
            c.create_text(CANVAS_W / 2, CANVAS_H - 24, fill="#556",
                          font=("Helvetica", 9),
                          text="Click a symptom circle to toggle it  \u2022  "
                               "click a rule box to see its truth table  \u2022  "
                               "dashed wire with a circle = NOT")

    def on_canvas_click(self, event: tk.Event) -> None:
        for i, symptom in enumerate(SYMPTOMS):
            dx, dy = event.x - NODE_X, event.y - node_y(i)
            if dx * dx + dy * dy <= (NODE_R + 4) ** 2:
                var = self.symptom_vars[symptom.symbol]
                var.set(not var.get())
                self.on_symptom_changed(symptom.symbol)
                return
        for j in range(len(self.engine.rules)):
            if (RULE_X <= event.x <= RULE_X + RULE_W
                    and rule_y(j) <= event.y <= rule_y(j) + RULE_H):
                self.show_truth_table(j)
                return

    # ----------------------------------------------------------- control
    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def update_buttons(self) -> None:
        self.run_btn.state(["disabled"] if self.running else ["!disabled"])
        self.step_btn.state(["disabled"] if self.running else ["!disabled"])
        self.pause_btn.state(["!disabled"] if self.running else ["disabled"])

    def log(self, text: str, tag: str = "") -> None:
        self.log_text.config(state="normal")
        self.log_text.insert("end", text + "\n", (tag,) if tag else ())
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def clear_log(self) -> None:
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def cancel_timer(self) -> None:
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def reset_evaluation(self, message: str) -> None:
        self.cancel_timer()
        self.running = False
        self.active = False
        self.finished = False
        self.results = []
        self.redraw()
        self.update_buttons()
        self.set_status(message)

    def reset(self) -> None:
        self.reset_evaluation("Reset. Press Run or Step.")
        self.clear_log()
        self._show_results_placeholder()

    def begin_evaluation(self) -> None:
        self.cancel_timer()
        self.snapshot = dict(self.facts)
        self.results = []
        self.active = True
        self.finished = False
        self.clear_log()
        present = [s.symbol for s in SYMPTOMS if self.snapshot[s.symbol]]
        self.log(f"=== Evaluating {len(self.engine.rules)} rules ===", "head")
        facts_text = "  ".join(
            f"{s.symbol}={'TRUE' if self.snapshot[s.symbol] else 'FALSE'}" for s in SYMPTOMS)
        self.log("Facts: " + facts_text)
        if not present:
            self.log("(no symptoms selected - no rule can be satisfied)", "false")

    def run(self) -> None:
        if self.running:
            return
        if self.finished or not self.active:
            self.begin_evaluation()
        self.running = True
        self.update_buttons()
        self.set_status("Running...")
        self._tick()

    def _tick(self) -> None:
        if not self.running:
            return
        self.evaluate_next_rule()
        if self.finished:
            return
        self.after_id = self.root.after(self.speed.get(), self._tick)

    def pause(self) -> None:
        if not self.running:
            return
        self.cancel_timer()
        self.running = False
        self.update_buttons()
        self.set_status("Paused. Press Run to continue or Step for one rule.")

    def step(self) -> None:
        if self.running:
            return
        if self.finished or not self.active:
            self.begin_evaluation()
        self.evaluate_next_rule()

    def evaluate_next_rule(self) -> None:
        index = len(self.results)
        rule = self.engine.rules[index]
        result = self.engine.evaluate_rule(rule, self.snapshot)
        self.results.append(result)
        expression = rule.expression

        self.log(f"Rule {index + 1}: {rule.name} = {expression.to_text()}", "head")
        outcome = "TRUE" if result.is_match else "FALSE"
        self.log(f"    {expression.to_symbols()}  ->  {expression.to_values(self.snapshot)}"
                 f"  =  {outcome}   ({result.satisfied}/{result.total} conditions, "
                 f"{result.percent}%)", "true" if result.is_match else "false")
        self.set_status(f"Evaluated rule {index + 1}/{len(self.engine.rules)}: "
                        f"{rule.name} = {outcome}")
        self.redraw()
        if len(self.results) == len(self.engine.rules):
            self.finalize()

    # --------------------------------------------------------- finishing
    def finalize(self) -> None:
        self.finished = True
        self.running = False
        self.cancel_timer()
        matched = self.engine.matches(self.results)
        closest = [] if matched else self.engine.closest(self.results, self.snapshot)
        self.history.add(DiagnosisSession(datetime.now(), dict(self.snapshot),
                                          list(self.results), closest))
        self.refresh_history()
        self.render_results(matched, closest)
        self.notebook.select(self.tab_results)
        self.redraw()
        self.update_buttons()

        if matched:
            self.log(f"=== Done: {len(matched)} rule(s) satisfied ===", "head")
            self.set_status(f"Done. {len(matched)} possible rule(s) satisfied - "
                            "see the Results tab.")
        else:
            self.log("=== Done: no predefined condition matched ===", "head")
            if closest:
                top = closest[0]
                self.log(f"Prediction: most likely {top.rule.name} "
                         f"({top.percent}% rule match)", "head")
                self.set_status(f"No exact match. Prediction: {top.rule.name} "
                                f"({top.percent}% rule match) - see the Results tab.")
            else:
                self.set_status("Done. No predefined condition matched the entered symptoms.")

        if self.snapshot["P"] and self.snapshot["B"]:
            messagebox.showwarning("URGENT WARNING", URGENT_MESSAGE)

    def _show_results_placeholder(self) -> None:
        t = self.results_text
        t.config(state="normal")
        t.delete("1.0", "end")
        self._put(t, "No diagnosis yet.\n", "h1")
        self._put(t, "\nSelect symptoms, then press Run or Step.\n")
        t.config(state="disabled")

    def render_results(self, matched: List[DiagnosisResult],
                       closest: List[DiagnosisResult]) -> None:
        t = self.results_text
        facts = self.snapshot
        t.config(state="normal")
        t.delete("1.0", "end")
        put = lambda s, *tags: self._put(t, s, *tags)  # noqa: E731

        present = [s.name for s in SYMPTOMS if facts[s.symbol]]
        put("DIAGNOSIS RESULTS\n", "h1")
        put("Symptoms entered: " + (", ".join(present) if present else "none") + "\n")

        if matched:
            put(f"\n{len(matched)} rule(s) satisfied\n", "h2")
            for number, result in enumerate(matched, start=1):
                expression = result.rule.expression
                put(f"\n{number}. Possible Diagnosis: {result.rule.name}\n", "h2")
                put(f"   Rule:      {expression.to_text()}\n")
                put(f"   Symbolic:  {expression.to_symbols()}\n")
                put("   Evaluation:\n")
                for prop in expression.propositions():
                    put(f"      {prop.name} = ")
                    value = facts.get(prop.symbol, False)
                    put("TRUE\n" if value else "FALSE\n", "true" if value else "false")
                put(f"   Substituted: {expression.to_values(facts)} = TRUE\n")
                put("   Result:    ")
                put("TRUE\n", "true")
                put(f"   Match:     {result.satisfied}/{result.total} conditions "
                    f"satisfied -> {result.percent}%\n")
                put(f"   Why: {result.rule.explanation}\n", "note")
            put("\nRule Match Summary\n", "h2")
            put(f"{'Diagnosis':<32}{'Cond.':<8}{'Match'}\n")
            for r in matched:
                put(f"{r.rule.name:<32}{f'{r.satisfied}/{r.total}':<8}{r.percent}%\n")
            put("\nThe match score is a logical rule-match score, "
                "NOT a medical probability.\n", "note")
        else:
            put("\nNo predefined condition matched the entered symptoms.\n", "false")
            if not present:
                put("\nNo symptoms were reported, so no rule can be satisfied.\n")
            elif closest:
                put("\nPREDICTION - what could it be?\n", "h1")
                put("Based on the closest rules to your symptoms:\n", "note")
                top = closest[0]
                put(f"\nMost likely: {top.rule.name} ", "h2")
                put(f"({top.percent}% rule match)\n", "true")
                for rank, r in enumerate(closest, start=1):
                    put(f"\n{rank}. {r.rule.name}\n", "h2")
                    put(f"   Rule:        {r.rule.expression.to_text()}\n")
                    put(f"   Rule match:  {r.satisfied}/{r.total} conditions "
                        f"-> {r.percent}%\n")
                    supported = self.engine.supporting_symptoms(r.rule, facts)
                    unmet = self.engine.unmet_conditions(r.rule, facts)
                    put("   Supported by: " + (", ".join(supported) or "-") + "\n", "true")
                    put("   Still needed: " + (", ".join(unmet) or "-") + "\n", "false")
                put("\nThe score is a logical rule-match score, "
                    "not a medical probability.\n", "note")
            else:
                put("\nNo rule has any of its conditions satisfied.\n")

        if facts["P"] and facts["B"]:
            put("\n URGENT WARNING \n", "warn")
            put(URGENT_MESSAGE + "\n", "false")
        t.config(state="disabled")
        t.see("1.0")

    # ----------------------------------------------------------- history
    def refresh_history(self) -> None:
        self.history_list.delete(0, "end")
        for number, session in enumerate(self.history.all(), start=1):
            count = len(session.matched())
            self.history_list.insert(
                "end", f"Session {number}  {session.timestamp:%H:%M:%S}  -  {count} match(es)")
        last = self.history_list.size() - 1
        if last >= 0:
            self.history_list.selection_clear(0, "end")
            self.history_list.selection_set(last)
            self.history_list.see(last)
            self.show_session(last)

    def on_history_select(self, _event=None) -> None:
        selection = self.history_list.curselection()
        if selection:
            self.show_session(selection[0])

    def show_session(self, index: int) -> None:
        sessions = self.history.all()
        if not 0 <= index < len(sessions):
            return
        session = sessions[index]
        t = self.history_text
        t.config(state="normal")
        t.delete("1.0", "end")
        put = lambda s, *tags: self._put(t, s, *tags)  # noqa: E731
        put(f"Session {index + 1}  -  {session.timestamp:%Y-%m-%d %H:%M:%S}\n", "h1")
        symptoms = session.symptoms_present()
        put("Symptoms entered: " + (", ".join(symptoms) if symptoms else "none") + "\n")
        matched = session.matched()
        if matched:
            put("\nMatched conditions:\n", "h2")
            for r in sorted(matched, key=lambda r: (-r.percent, r.rule.name)):
                put(f"  - {r.rule.name}: {r.satisfied}/{r.total} conditions, "
                    f"match {r.percent}%\n")
        else:
            put("\nMatched conditions: none\n", "false")
            for r in session.closest:
                put(f"  closest: {r.rule.name}: {r.satisfied}/{r.total} conditions, "
                    f"match {r.percent}%\n")
        put("\nMatch scores are logical rule-match scores, not medical probabilities.\n",
            "note")
        t.config(state="disabled")

    # ------------------------------------------------------- truth table
    def show_truth_table(self, index: int, switch_tab: bool = True) -> None:
        if not 0 <= index < len(self.engine.rules):
            return
        rule = self.engine.rules[index]
        self.truth_combo.current(index)
        props = rule.expression.propositions()
        symbolic = rule.expression.to_symbols()

        t = self.truth_text
        t.config(state="normal")
        t.delete("1.0", "end")
        put = lambda s, *tags: self._put(t, s, *tags)  # noqa: E731
        put(f"Truth Table: {rule.name}\n", "h1")
        put(f"Rule:     {rule.expression.to_text()}\n")
        put(f"Symbolic: {symbolic}\n\n")
        header = "".join(f"{p.symbol:^7}|" for p in props) + f" {symbolic}"
        put(header + "\n")
        put("-" * len(header) + "\n")
        true_rows = 0
        for values in itertools.product([True, False], repeat=len(props)):
            row_facts = {p.symbol: v for p, v in zip(props, values)}
            outcome = rule.expression.evaluate(row_facts)
            true_rows += outcome
            put("".join(f"{('TRUE' if v else 'FALSE'):^7}|" for v in values) + " ")
            put("TRUE" if outcome else "FALSE", "true" if outcome else "false")
            if all(self.facts[p.symbol] == v for p, v in zip(props, values)):
                put("   <-- your current symptoms", "note")
            put("\n")
        put(f"\n{true_rows} of {2 ** len(props)} combinations make the rule TRUE.\n", "note")
        t.config(state="disabled")
        if switch_tab:
            self.notebook.select(self.tab_truth)

    # --------------------------------------------------------------- end
    def on_close(self) -> None:
        self.cancel_timer()
        self.root.destroy()


def main() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as error:
        print(f"Could not open a window (no display available?): {error}")
        sys.exit(1)
    SimulatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()