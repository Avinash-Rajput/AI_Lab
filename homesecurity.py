#!/usr/bin/env python3
"""Smart Home Security Simulator - propositional logic, interactive GUI.

Sensors (doors, windows, CCTV motion, smoke, gas, tamper...) are Boolean
propositions. Security rules are AND / OR / NOT expressions. Every rule is
evaluated against the sensor state and the highest severity among triggered
rules becomes the overall threat level.

  * Left   : logic board (sensors wired into rules) + evaluation log.
  * Right  : Run / Pause / Step / Reset, live house floor plan, and tabs
             (Sensors, Rules, Results, History, Truth Table, Logic).

Standard library only (tkinter ships with Python).
Run with:  python3 smart_home_security_gui.py
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
    """Symbols that appear directly under a NOT (drawn as inverter wires)."""
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
# Sensors (propositions)
# --------------------------------------------------------------------------
FrontDoor = Proposition("FD", "Front Door Open")
BackDoor = Proposition("BD", "Back Door Open")
GarageDoor = Proposition("GD", "Garage Door Open")
Window = Proposition("WN", "Window Open")
GlassBreak = Proposition("GB", "Glass-Break Sound")
CamFront = Proposition("CF", "CCTV Front Motion")
CamBack = Proposition("CB", "CCTV Back Motion")
CamIndoor = Proposition("CI", "CCTV Indoor Motion")
OwnerAway = Proposition("AW", "Owner Away")
NightTime = Proposition("NT", "Night Time")
AuthUnlock = Proposition("VK", "Authorized Unlock")
Smoke = Proposition("SM", "Smoke Detected")
HighTemp = Proposition("HT", "High Temperature")
GasLeak = Proposition("GL", "Gas Leak")
WaterLeak = Proposition("WL", "Water Leak")
Tamper = Proposition("TP", "Camera/Power Tamper")

SENSORS: List[Proposition] = [
    FrontDoor, BackDoor, GarageDoor, Window, GlassBreak,
    CamFront, CamBack, CamIndoor,
    OwnerAway, NightTime, AuthUnlock,
    Smoke, HighTemp, GasLeak, WaterLeak, Tamper,
]

SENSOR_GROUPS = [
    ("Entry points", ["FD", "BD", "GD", "WN", "GB"]),
    ("CCTV motion detection", ["CF", "CB", "CI"]),
    ("Context", ["AW", "NT", "VK"]),
    ("Hazards", ["SM", "HT", "GL", "WL"]),
    ("Tamper", ["TP"]),
]

LEVELS = ["SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
LEVEL_COLORS = ["#2e9e4f", "#c9a800", "#e67e22", "#e74c3c", "#7b1113"]


@dataclass(frozen=True)
class Rule:
    name: str
    expression: Expression
    explanation: str
    severity: int          # 1 LOW .. 4 CRITICAL
    action: str


@dataclass
class RuleResult:
    rule: Rule
    satisfied: int
    total: int
    is_match: bool

    @property
    def percent(self) -> int:
        return int(round(100 * self.satisfied / self.total)) if self.total else 0


def build_rules() -> List[Rule]:
    return [
        Rule("Front Door Burglary Attempt", AND(FrontDoor, OwnerAway, NOT(AuthUnlock)),
             "Front door is open while nobody is home and no authorized unlock was used.",
             4, "Sound siren, notify owner and police, save CCTV footage."),
        Rule("Back Door Intrusion", AND(BackDoor, NOT(AuthUnlock), OR(OwnerAway, NightTime)),
             "Back door is open without authorization while away or at night.",
             3, "Trigger alarm, notify owner, review back-yard camera."),
        Rule("Garage Breach", AND(GarageDoor, NOT(AuthUnlock), OR(OwnerAway, NightTime)),
             "Garage door is open without authorization while away or at night.",
             3, "Trigger alarm, notify owner, review garage footage."),
        Rule("Window Break-in", AND(OR(Window, GlassBreak), OR(OwnerAway, NightTime)),
             "Window opened or glass-break heard while away or at night.",
             3, "Trigger alarm, notify owner, check indoor camera."),
        Rule("Intruder Inside House", AND(CamIndoor, OwnerAway, OR(FrontDoor, BackDoor, GarageDoor, Window)),
             "Indoor motion while nobody is home and an entry point is open.",
             4, "Sound siren, call police, lock safe room doors."),
        Rule("Night Prowler Outside", AND(OR(CamFront, CamBack), NightTime, NOT(AuthUnlock)),
             "Outdoor camera motion at night without an authorized unlock.",
             2, "Switch on floodlights, start recording, notify owner."),
        Rule("Casing Pattern", AND(CamFront, CamBack),
             "Motion on both outdoor cameras - someone may be surveying the house.",
             2, "Increase recording quality, notify owner."),
        Rule("Camera / Power Tamper", AND(Tamper, OR(OwnerAway, NightTime)),
             "Tampering while away or at night can precede a break-in.",
             3, "Switch to battery backup, notify owner, alert neighbours."),
        Rule("Door Left Open at Night", AND(OR(FrontDoor, BackDoor, GarageDoor), NightTime, NOT(OwnerAway)),
             "A door is open at night while residents are home.",
             1, "Remind residents to close and lock the door."),
        Rule("Fire Emergency", AND(Smoke, HighTemp),
             "Smoke together with high temperature indicates a fire.",
             4, "Sound fire alarm, unlock exits, call the fire service."),
        Rule("Smoke Warning", AND(Smoke, NOT(HighTemp)),
             "Smoke detected but temperature is normal.",
             2, "Check appliances, ventilate, silence if false alarm."),
        Rule("Gas Leak Alert", GasLeak,
             "Gas sensor is triggered.",
             3, "Close gas valve, ventilate, avoid switches and flames."),
        Rule("Explosion Risk", AND(GasLeak, OR(HighTemp, Smoke)),
             "Gas leak combined with heat or smoke is explosive.",
             4, "Evacuate, cut power from outside, call emergency services."),
        Rule("Water Leak Alert", WaterLeak,
             "Water leak sensor is triggered.",
             2, "Close main water valve, check pipes and appliances."),
    ]


PRESETS = {
    "Burglary at front door, owner away": ["FD", "AW", "CF", "CI"],
    "Back-door intrusion at night": ["BD", "NT", "CB", "CI"],
    "Window break-in while away": ["WN", "GB", "AW", "CI"],
    "Prowler outside at night": ["CF", "CB", "NT"],
    "Camera tamper before break-in": ["TP", "AW", "NT"],
    "Kitchen fire": ["SM", "HT"],
    "Smoke only (burnt toast?)": ["SM"],
    "Gas leak + heat (explosion risk)": ["GL", "HT"],
    "Water leak": ["WL"],
    "Garage left open at night (residents home)": ["GD", "NT", "VK"],
    "Owner arrives (authorized)": ["FD", "VK", "CF"],
    "All clear": [],
}


class SecurityEngine:
    def __init__(self, rules: Sequence[Rule]) -> None:
        self.rules = list(rules)

    @staticmethod
    def evaluate_rule(rule: Rule, facts: Facts) -> RuleResult:
        conditions = rule.expression.conditions()
        satisfied = sum(1 for c in conditions if c.evaluate(facts))
        return RuleResult(rule, satisfied, len(conditions), rule.expression.evaluate(facts))

    @staticmethod
    def triggered(results: Sequence[RuleResult]) -> List[RuleResult]:
        found = [r for r in results if r.is_match]
        return sorted(found, key=lambda r: (-r.rule.severity, -r.percent, r.rule.name))

    @staticmethod
    def threat_level(results: Sequence[RuleResult]) -> int:
        return max((r.rule.severity for r in results if r.is_match), default=0)

    @staticmethod
    def closest(results: Sequence[RuleResult], facts: Facts, limit: int = 3) -> List[RuleResult]:
        candidates = []
        for r in results:
            if r.is_match or r.satisfied == 0:
                continue
            if not any(facts.get(p.symbol, False) for p in r.rule.expression.propositions()):
                continue
            candidates.append(r)
        candidates.sort(key=lambda r: (-r.percent, -r.rule.severity))
        return candidates[:limit]

    @staticmethod
    def unmet_conditions(rule: Rule, facts: Facts) -> List[str]:
        return [c.to_text() for c in rule.expression.conditions() if not c.evaluate(facts)]

    @staticmethod
    def supporting_sensors(rule: Rule, facts: Facts) -> List[str]:
        negated = negated_symbols(rule.expression)
        return [p.name for p in rule.expression.propositions()
                if facts.get(p.symbol, False) and p.symbol not in negated]


@dataclass
class Session:
    timestamp: datetime
    facts: Facts
    results: List[RuleResult]
    closest: List[RuleResult] = field(default_factory=list)

    def active_sensors(self) -> List[str]:
        return [s.name for s in SENSORS if self.facts.get(s.symbol, False)]

    def triggered(self) -> List[RuleResult]:
        return SecurityEngine.triggered(self.results)

    def level(self) -> int:
        return SecurityEngine.threat_level(self.results)


class SessionHistory:
    def __init__(self) -> None:
        self._sessions: List[Session] = []

    def add(self, session: Session) -> None:
        self._sessions.append(session)

    def all(self) -> List[Session]:
        return list(self._sessions)


# --------------------------------------------------------------------------
# GUI constants
# --------------------------------------------------------------------------
CANVAS_W, CANVAS_H = 820, 600
NODE_X, NODE_R = 205, 12
RULE_X, RULE_W, RULE_H = 450, 360, 30
HOUSE_W, HOUSE_H = 450, 270

COLOR_BG = "#e6f2fb"
COLOR_TRUE = "#2e9e4f"
COLOR_FALSE = "#d05a5a"
COLOR_ACTIVE = "#f39c12"
COLOR_IDLE = "#b9cbd9"


def node_y(index: int) -> int:
    return 96 + index * 30


def rule_y(index: int) -> int:
    return 74 + index * 34


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------
class SecurityApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.engine = SecurityEngine(build_rules())
        self.history = SessionHistory()

        self.facts: Facts = {s.symbol: False for s in SENSORS}
        self.snapshot: Facts = dict(self.facts)
        self.sensor_vars: Dict[str, tk.BooleanVar] = {}
        self.sensor_index = {s.symbol: i for i, s in enumerate(SENSORS)}

        self.results: List[RuleResult] = []
        self.active = False
        self.finished = False
        self.running = False
        self.threat = 0
        self.after_id = None

        self.speed = tk.IntVar(value=500)
        self.status_var = tk.StringVar(value="Ready. Choose sensors, then press Run or Step.")
        self.mono = tkfont.nametofont("TkFixedFont")

        self._build_ui()
        self.redraw()
        self.update_buttons()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ----------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        root = self.root
        root.title("Smart Home Security Simulator - propositional logic")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(1330, sw)}x{min(900, sh - 70)}")
        root.minsize(1100, 680)
        try:
            ttk.Style().theme_use("clam")
        except tk.TclError:
            pass

        menubar = tk.Menu(root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        root.config(menu=menubar)

        tk.Label(root, text="Smart Home Security Simulator",
                 font=("Helvetica", 18, "bold")).pack(pady=(8, 0))
        tk.Label(root, fg="#444",
                 text="Sensors are Boolean propositions  \u2022  rules use AND / OR / NOT  "
                      "\u2022  every rule is evaluated and the highest severity wins"
                 ).pack(pady=(0, 6))

        tk.Label(root, textvariable=self.status_var,
                 font=("Helvetica", 11, "bold")).pack(side="bottom", pady=(2, 6))

        body = tk.Frame(root)
        body.pack(fill="both", expand=True, padx=8)

        left = tk.Frame(body)
        left.pack(side="left", anchor="n")
        self.canvas = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H, bg=COLOR_BG,
                                highlightthickness=1, highlightbackground="#889")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        log_frame = tk.Frame(left)
        log_frame.pack(fill="x", pady=(6, 0))
        self.log_text = tk.Text(log_frame, height=6, width=1, font=self.mono,
                                wrap="word", state="disabled", bg="#fbfbfb")
        log_scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="x", expand=True)
        log_scroll.pack(side="left", fill="y")
        self.log_text.tag_configure("head", font=(self.mono.actual("family"), 10, "bold"))
        self.log_text.tag_configure("true", foreground=COLOR_TRUE)
        self.log_text.tag_configure("false", foreground="#777")

        right = tk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self._build_controls(right)
        self.house = tk.Canvas(right, width=HOUSE_W, height=HOUSE_H,
                               highlightthickness=1, highlightbackground="#889")
        self.house.pack(pady=(6, 0))
        self._build_tabs(right)

    def _build_controls(self, parent: tk.Frame) -> None:
        bar = tk.Frame(parent)
        bar.pack(fill="x")
        self.run_btn = ttk.Button(bar, text="\u25B6 Run", command=self.run)
        self.pause_btn = ttk.Button(bar, text="\u23F8 Pause", command=self.pause)
        self.step_btn = ttk.Button(bar, text="\u23ED Step", command=self.step)
        self.reset_btn = ttk.Button(bar, text="\u27F2 Reset", command=self.reset)
        for button in (self.run_btn, self.pause_btn, self.step_btn, self.reset_btn):
            button.pack(side="left", padx=2, fill="x", expand=True)

        speed_row = tk.Frame(parent)
        speed_row.pack(fill="x", pady=(4, 0))
        tk.Label(speed_row, text="Speed fast\u2192slow (ms):",
                 font=("Helvetica", 9, "bold")).pack(side="left")
        tk.Scale(speed_row, from_=100, to=1500, resolution=50, orient="horizontal",
                 variable=self.speed, length=200).pack(side="left", fill="x", expand=True)

    def _make_text(self, parent: tk.Widget) -> tk.Text:
        frame = tk.Frame(parent)
        frame.pack(fill="both", expand=True)
        text = tk.Text(frame, wrap="word", state="disabled", width=44,
                       height=10, font=self.mono, bg="#fdfdfd")
        scroll = ttk.Scrollbar(frame, command=text.yview)
        text.config(yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")
        base = self.mono.actual("family")
        text.tag_configure("h1", font=(base, 12, "bold"))
        text.tag_configure("h2", font=(base, 10, "bold"), foreground="#124a7a")
        text.tag_configure("true", foreground=COLOR_TRUE, font=(base, 10, "bold"))
        text.tag_configure("false", foreground=COLOR_FALSE, font=(base, 10, "bold"))
        text.tag_configure("note", foreground="#665", font=(base, 9, "italic"))
        text.tag_configure("lvl0", foreground=LEVEL_COLORS[0], font=(base, 11, "bold"))
        text.tag_configure("lvl1", foreground=LEVEL_COLORS[1], font=(base, 11, "bold"))
        text.tag_configure("lvl2", foreground=LEVEL_COLORS[2], font=(base, 11, "bold"))
        text.tag_configure("lvl3", foreground=LEVEL_COLORS[3], font=(base, 11, "bold"))
        text.tag_configure("lvl4", foreground="white", background=LEVEL_COLORS[4],
                           font=(base, 11, "bold"))
        return text

    @staticmethod
    def _put(widget: tk.Text, text: str, *tags: str) -> None:
        widget.insert("end", text, tags)

    def _build_tabs(self, parent: tk.Frame) -> None:
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True, pady=(6, 0))

        # --- Sensors tab
        tab = tk.Frame(self.notebook, padx=6, pady=4)
        self.notebook.add(tab, text="Sensors")
        top = tk.Frame(tab)
        top.pack(fill="x")
        ttk.Button(top, text="All on", command=lambda: self.set_all(True)).pack(side="left", padx=2)
        ttk.Button(top, text="All off", command=lambda: self.set_all(False)).pack(side="left", padx=2)
        self.preset_combo = ttk.Combobox(top, values=list(PRESETS), state="readonly")
        self.preset_combo.set("Example scenarios...")
        self.preset_combo.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.preset_combo.bind("<<ComboboxSelected>>", self.on_preset)

        by_symbol = {s.symbol: s for s in SENSORS}
        for group_name, symbols in SENSOR_GROUPS:
            box = ttk.LabelFrame(tab, text=group_name)
            box.pack(fill="x", pady=2)
            for position, symbol in enumerate(symbols):
                sensor = by_symbol[symbol]
                var = tk.BooleanVar(value=False)
                self.sensor_vars[symbol] = var
                ttk.Checkbutton(
                    box, text=f"{symbol} - {sensor.name}", variable=var,
                    command=lambda s=symbol: self.on_sensor_changed(s)
                ).grid(row=position // 2, column=position % 2, sticky="w", padx=6, pady=1)
            box.columnconfigure(0, weight=1)
            box.columnconfigure(1, weight=1)

        # --- Rules tab
        tab_rules = tk.Frame(self.notebook, padx=4, pady=4)
        self.notebook.add(tab_rules, text="Rules")
        rules_text = self._make_text(tab_rules)
        rules_text.config(state="normal")
        self._put(rules_text, "SENSORS (PROPOSITIONS)\n", "h1")
        for s in SENSORS:
            self._put(rules_text, f"{s.symbol:<3} -> {s.name}\n")
        self._put(rules_text, "\nSECURITY RULES\n", "h1")
        for number, rule in enumerate(self.engine.rules, start=1):
            self._put(rules_text, f"\n{number}. {rule.name}  ", "h2")
            self._put(rules_text, f"[{LEVELS[rule.severity]}]\n", f"lvl{rule.severity}")
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
        self.history_list = tk.Listbox(tab_history, height=4, exportselection=False)
        self.history_list.pack(fill="x")
        self.history_list.bind("<<ListboxSelect>>", self.on_history_select)
        self.history_text = self._make_text(tab_history)

        # --- Truth table tab
        tab_truth = tk.Frame(self.notebook, padx=4, pady=4)
        self.notebook.add(tab_truth, text="Truth Table")
        self.tab_truth = tab_truth
        self.truth_combo = ttk.Combobox(
            tab_truth, values=[r.name for r in self.engine.rules], state="readonly")
        self.truth_combo.pack(fill="x", pady=(0, 4))
        self.truth_combo.current(0)
        self.truth_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self.show_truth_table(self.truth_combo.current()))
        self.truth_text = self._make_text(tab_truth)
        self.show_truth_table(0, switch_tab=False)

        # --- Logic tab
        tab_logic = tk.Frame(self.notebook, padx=4, pady=4)
        self.notebook.add(tab_logic, text="Logic")
        self._fill_logic_explanation(self._make_text(tab_logic))

    def _fill_logic_explanation(self, text: tk.Text) -> None:
        text.config(state="normal")
        burglary = self.engine.rules[0]
        fire = self.engine.rules[9]
        put = lambda s, *t: self._put(text, s, *t)  # noqa: E731
        put("HOW PROPOSITIONAL LOGIC IS USED\n", "h1")
        put("\nEach sensor is a proposition: TRUE (triggered) or FALSE.\n")
        put("  FD = Front Door Open\n  AW = Owner Away\n  VK = Authorized Unlock\n")
        put("\nOperators\n", "h2")
        put("  AND : TRUE only if every part is TRUE\n")
        put("  OR  : TRUE if at least one part is TRUE\n")
        put("  NOT : flips TRUE <-> FALSE\n")
        put("\nExample: Front Door Burglary Attempt\n", "h2")
        put(f"  {burglary.expression.to_symbols()}\n")
        facts = {"FD": True, "AW": True, "VK": False}
        put("  If FD = TRUE, AW = TRUE, VK = FALSE then\n")
        put(f"  {burglary.expression.to_values(facts)} = TRUE\n")
        put("  -> rule triggered, severity CRITICAL.\n")
        facts = {"FD": True, "AW": True, "VK": True}
        put("\n  If the owner unlocked with a key (VK = TRUE):\n")
        put(f"  {burglary.expression.to_values(facts)} = FALSE\n")
        put("  -> rule NOT triggered.\n")
        put("\nExample: Fire Emergency\n", "h2")
        put(f"  {fire.expression.to_symbols()}\n")
        put("  Smoke alone only triggers 'Smoke Warning' (SM AND NOT HT).\n")
        put("\nThreat level\n", "h2")
        put("  Every rule is evaluated (none stops early). The overall\n"
            "  threat level is the highest severity among triggered rules:\n"
            "  SAFE < LOW < MEDIUM < HIGH < CRITICAL.\n")
        put("\nRule match score\n", "h2")
        put("  satisfied conditions / total conditions of a rule. When no\n"
            "  rule triggers, the closest rules are shown as a watch list.\n")
        text.config(state="disabled")

    # ------------------------------------------------------------ sensors
    def on_sensor_changed(self, symbol: str) -> None:
        self.facts[symbol] = self.sensor_vars[symbol].get()
        self.reset_evaluation("Sensors changed. Press Run or Step.")

    def set_all(self, value: bool) -> None:
        for symbol, var in self.sensor_vars.items():
            var.set(value)
            self.facts[symbol] = value
        self.reset_evaluation("Sensors updated. Press Run or Step.")

    def on_preset(self, _event=None) -> None:
        chosen = PRESETS.get(self.preset_combo.get())
        if chosen is None:
            return
        for symbol, var in self.sensor_vars.items():
            value = symbol in chosen
            var.set(value)
            self.facts[symbol] = value
        self.reset_evaluation(f"Scenario loaded: {self.preset_combo.get()}. Press Run or Step.")

    # ------------------------------------------------------------- drawing
    def redraw(self) -> None:
        self.draw_board()
        self.draw_house()

    def draw_board(self) -> None:
        c = self.canvas
        c.delete("all")
        c.create_rectangle(0, 0, CANVAS_W, CANVAS_H, fill=COLOR_BG, outline="")
        c.create_rectangle(0, 0, CANVAS_W, 40, fill="#2c3e50", outline="")
        # small padlock icon
        c.create_arc(15, 8, 29, 24, start=0, extent=180, style="arc", outline="white", width=3)
        c.create_rectangle(12, 17, 32, 32, fill="white", outline="")
        c.create_oval(20, 21, 24, 25, fill="#2c3e50", outline="")
        c.create_text(44, 20, anchor="w", fill="white", font=("Helvetica", 12, "bold"),
                      text="SECURITY LOGIC BOARD")
        c.create_text(CANVAS_W - 12, 20, anchor="e", fill="#dfe8f0", font=("Helvetica", 10),
                      text=f"Rules evaluated: {len(self.results)}/{len(self.engine.rules)}")
        c.create_text(NODE_X - 60, 58, text="SENSORS (propositions)",
                      font=("Helvetica", 10, "bold"), fill="#1f4e79")
        c.create_text(RULE_X + RULE_W / 2, 58, text="SECURITY RULES (logic gates)",
                      font=("Helvetica", 10, "bold"), fill="#1f4e79")

        current = len(self.results) - 1 if (self.active and not self.finished) else -1
        self._draw_wires(current)
        self._draw_sensors()
        self._draw_rules(current)
        self._draw_banner()

    def _draw_wires(self, current: int) -> None:
        c = self.canvas
        for j, rule in enumerate(self.engine.rules):
            props = rule.expression.propositions()
            negated = negated_symbols(rule.expression)
            evaluated = j < len(self.results)
            for k, prop in enumerate(props):
                sx = NODE_X + NODE_R
                sy = node_y(self.sensor_index[prop.symbol])
                ex = RULE_X
                ey = rule_y(j) + RULE_H * (k + 1) / (len(props) + 1)
                if j == current:
                    color, width = COLOR_ACTIVE, 3
                elif evaluated:
                    on = self.snapshot.get(prop.symbol, False)
                    color, width = (COLOR_TRUE, 2) if on else ("#dcaaaa", 1)
                else:
                    color, width = COLOR_IDLE, 1
                dash = (4, 3) if prop.symbol in negated else None
                c.create_line(sx, sy, ex, ey, fill=color, width=width, dash=dash)
                if prop.symbol in negated:
                    c.create_oval(ex - 9, ey - 4, ex - 1, ey + 4, fill="white", outline=color)

    def _draw_sensors(self) -> None:
        c = self.canvas
        for i, sensor in enumerate(SENSORS):
            y = node_y(i)
            on = self.facts[sensor.symbol]
            c.create_text(NODE_X - NODE_R - 8, y, anchor="e", text=sensor.name,
                          font=("Helvetica", 9), fill="#223")
            c.create_oval(NODE_X - NODE_R, y - NODE_R, NODE_X + NODE_R, y + NODE_R,
                          fill=COLOR_TRUE if on else "white",
                          outline=COLOR_TRUE if on else "#8aa0b2", width=2)
            c.create_text(NODE_X, y, text=sensor.symbol, font=("Helvetica", 8, "bold"),
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
                outline, width = COLOR_ACTIVE, 3
            c.create_rectangle(x, y, x + RULE_W, y + RULE_H, fill=fill, outline=outline, width=width)
            c.create_rectangle(x, y, x + 6, y + RULE_H, fill=LEVEL_COLORS[rule.severity], outline="")
            c.create_text(x + 12, y + 9, anchor="w", text=rule.name,
                          font=("Helvetica", 9, "bold"), fill="#123")
            c.create_text(x + 12, y + 22, anchor="w", text=rule.expression.to_symbols(),
                          font=("Helvetica", 8), fill="#456")
            if result is None:
                c.create_text(x + RULE_W - 8, y + 15, anchor="e", text="pending",
                              font=("Helvetica", 9, "italic"), fill="#889")
            else:
                c.create_text(x + RULE_W - 8, y + 9, anchor="e",
                              text="TRIGGERED \u2713" if result.is_match else "FALSE",
                              font=("Helvetica", 9, "bold"),
                              fill=COLOR_TRUE if result.is_match else COLOR_FALSE)
                c.create_text(x + RULE_W - 8, y + 22, anchor="e",
                              text=f"{result.satisfied}/{result.total} \u2022 {result.percent}%",
                              font=("Helvetica", 8), fill="#334")

    def _draw_banner(self) -> None:
        c = self.canvas
        if self.finished:
            c.create_rectangle(10, CANVAS_H - 34, CANVAS_W - 10, CANVAS_H - 6,
                               fill=LEVEL_COLORS[self.threat], outline="")
            triggered = len(self.engine.triggered(self.results))
            c.create_text(CANVAS_W / 2, CANVAS_H - 20, fill="white",
                          font=("Helvetica", 12, "bold"),
                          text=f"THREAT LEVEL: {LEVELS[self.threat]}   "
                               f"({triggered} rule(s) triggered)")
        else:
            c.create_text(CANVAS_W / 2, CANVAS_H - 20, fill="#556", font=("Helvetica", 9),
                          text="Click a sensor circle to toggle it  \u2022  click a rule box for its "
                               "truth table  \u2022  dashed wire with circle = NOT  \u2022  "
                               "stripe colour = severity")

    # --- live house floor plan
    @staticmethod
    def _person(c: tk.Canvas, x: int, y: int, color: str) -> None:
        c.create_oval(x - 4, y - 14, x + 4, y - 6, fill=color, outline="")
        c.create_line(x, y - 6, x, y + 6, fill=color, width=3)
        c.create_line(x - 6, y - 2, x + 6, y - 2, fill=color, width=2)
        c.create_line(x, y + 6, x - 5, y + 14, fill=color, width=2)
        c.create_line(x, y + 6, x + 5, y + 14, fill=color, width=2)

    @staticmethod
    def _camera_cone(c: tk.Canvas, origin, points, motion: bool) -> None:
        c.create_polygon(*origin, *points[0], *points[1],
                         fill="#ffcf8b" if motion else "#d9e6f2",
                         outline="#e67e22" if motion else "#a9bccd")

    @staticmethod
    def _camera_icon(c: tk.Canvas, x: int, y: int, label: str, motion: bool,
                     tamper: bool, ink: str) -> None:
        c.create_rectangle(x - 9, y - 5, x + 9, y + 5,
                           fill="#d62828" if tamper else "#2b2d42", outline="white")
        c.create_oval(x - 3, y - 3, x + 3, y + 3,
                      fill="#ff3b3b" if motion else "#7f8c8d", outline="")
        c.create_text(x + 14, y, anchor="w", font=("Helvetica", 7, "bold"),
                      fill="#d62828" if (motion or tamper) else ink,
                      text=label + (" MOTION!" if motion else "") + (" TAMPER" if tamper else ""))
        if tamper:
            c.create_line(x - 10, y - 7, x + 10, y + 7, fill="#d62828", width=2)
            c.create_line(x - 10, y + 7, x + 10, y - 7, fill="#d62828", width=2)

    @staticmethod
    def _door(c: tk.Canvas, start, end, swing, is_open: bool, label_at) -> None:
        if is_open:
            c.create_line(*start, *end, fill="#d62828", width=2, dash=(2, 2))
            c.create_line(*start, *swing, fill="#d62828", width=4)
            c.create_text(*label_at, text="OPEN", fill="#d62828", font=("Helvetica", 7, "bold"))
        else:
            c.create_line(*start, *end, fill="#2e9e4f", width=5)

    def draw_house(self) -> None:
        c = self.house
        c.delete("all")
        f = self.facts
        night = f["NT"]
        yard = "#16243a" if night else "#cdeacb"
        ink = "#e8eef5" if night else "#2b3a4a"
        c.create_rectangle(0, 0, HOUSE_W, HOUSE_H, fill=yard, outline="")
        if night:
            c.create_oval(405, 10, 429, 34, fill="#f6f2c9", outline="")
            c.create_oval(414, 7, 438, 31, fill=yard, outline="")
        else:
            c.create_oval(402, 10, 428, 36, fill="#ffd23f", outline="")

        # outdoor camera cones, house body, indoor cone, walls
        self._camera_cone(c, (125, 255), [(70, 196), (180, 196)], f["CF"])
        self._camera_cone(c, (125, 12), [(70, 49), (180, 49)], f["CB"])
        c.create_rectangle(70, 50, 370, 195, fill="#f6efe2", outline="#555", width=3)
        c.create_rectangle(290, 50, 370, 195, fill="#e3e6ea", outline="")
        self._camera_cone(c, (240, 127), [(205, 193), (275, 193)], f["CI"])
        for x1, y1, x2, y2 in [(190, 50, 190, 195), (70, 100, 190, 100),
                               (190, 125, 290, 125), (290, 50, 290, 195)]:
            c.create_line(x1, y1, x2, y2, fill="#888", width=2)
        for label, x, y in [("Kitchen", 165, 60), ("Living Room", 130, 108),
                            ("Bedroom", 240, 60), ("Hallway", 212, 138), ("Garage", 330, 60)]:
            c.create_text(x, y, text=label, fill="#7a7a7a", font=("Helvetica", 7))

        # doors and window
        self._door(c, (110, 195), (140, 195), (125, 226), f["FD"], (158, 214))
        self._door(c, (110, 50), (140, 50), (126, 24), f["BD"], (158, 36))
        self._door(c, (370, 80), (370, 165), (408, 96), f["GD"], (405, 112))
        if f["WN"]:
            c.create_line(70, 140, 70, 175, fill="#d62828", width=2, dash=(2, 2))
            c.create_line(70, 140, 52, 152, fill="#d62828", width=4)
            c.create_text(34, 165, text="OPEN", fill="#d62828", font=("Helvetica", 7, "bold"))
        else:
            c.create_line(70, 140, 70, 175, fill="#4aa3df", width=5)
        if f["GB"]:
            c.create_line(60, 136, 74, 148, 58, 158, 74, 168, 60, 180,
                          fill="#d62828", width=2)
            c.create_text(32, 188, text="CRASH!", fill="#d62828", font=("Helvetica", 8, "bold"))

        # cameras
        tamper = f["TP"]
        self._camera_icon(c, 125, 255, "CAM-F", f["CF"], tamper, ink)
        self._camera_icon(c, 125, 12, "CAM-B", f["CB"], tamper, ink)
        self._camera_icon(c, 240, 127, "CAM-I", f["CI"], tamper, ink)

        # people
        if not f["AW"]:
            self._person(c, 150, 160, "#2e86c1")
        if f["CF"]:
            self._person(c, 100, 232, "#d62828")
        if f["CB"]:
            self._person(c, 98, 32, "#d62828")
        if f["CI"]:
            self._person(c, 240, 168, "#d62828")
        if f["VK"]:
            c.create_rectangle(188, 224, 330, 242, fill="#2e9e4f", outline="")
            c.create_text(259, 233, text="\u2714 AUTHORIZED UNLOCK", fill="white",
                          font=("Helvetica", 8, "bold"))

        # hazards
        if f["SM"]:
            for dx, dy, r in [(0, 0, 9), (12, -6, 8), (24, 0, 9)]:
                c.create_oval(215 + dx - r, 80 + dy - r, 215 + dx + r, 80 + dy + r,
                              fill="#9a9a9a", outline="")
            c.create_text(228, 98, text="SMOKE", fill="#555", font=("Helvetica", 7, "bold"))
        if f["HT"]:
            c.create_polygon(268, 108, 261, 97, 265, 90, 268, 97, 272, 87, 276, 98, 274, 108,
                             fill="#ff6b00", outline="#c0392b")
            c.create_text(268, 116, text="HOT", fill="#c0392b", font=("Helvetica", 7, "bold"))
        if f["GL"]:
            for dx, dy, r in [(0, 0, 8), (11, -5, 7), (22, 1, 8)]:
                c.create_oval(145 + dx - r, 80 + dy - r, 145 + dx + r, 80 + dy + r,
                              fill="#7bd88f", outline="")
            c.create_text(160, 93, text="GAS", fill="#1e6b34", font=("Helvetica", 7, "bold"))
        if f["WL"]:
            c.create_oval(80, 80, 112, 92, fill="#4aa3df", outline="")
            c.create_text(96, 74, text="WATER", fill="#1f5f8b", font=("Helvetica", 7, "bold"))

        # mode label and alarm
        c.create_rectangle(6, 6, 104, 24, fill="#7d3c98" if f["AW"] else "#2e86c1", outline="")
        c.create_text(55, 15, text="MODE: AWAY" if f["AW"] else "MODE: HOME", fill="white",
                      font=("Helvetica", 8, "bold"))
        c.create_text(138, 15, text="NIGHT" if night else "DAY", fill=ink,
                      font=("Helvetica", 8, "bold"))
        if self.finished and self.threat >= 3:
            c.create_rectangle(3, 3, HOUSE_W - 3, HOUSE_H - 3, outline="#d62828", width=5)
            c.create_rectangle(280, 6, 380, 26, fill="#d62828", outline="")
            c.create_text(330, 16, text="\u26A0 ALARM!", fill="white",
                          font=("Helvetica", 9, "bold"))

    def on_canvas_click(self, event: tk.Event) -> None:
        for i, sensor in enumerate(SENSORS):
            dx, dy = event.x - NODE_X, event.y - node_y(i)
            if dx * dx + dy * dy <= (NODE_R + 4) ** 2:
                var = self.sensor_vars[sensor.symbol]
                var.set(not var.get())
                self.on_sensor_changed(sensor.symbol)
                return
        for j in range(len(self.engine.rules)):
            if (RULE_X <= event.x <= RULE_X + RULE_W
                    and rule_y(j) <= event.y <= rule_y(j) + RULE_H):
                self.show_truth_table(j)
                return

    # ------------------------------------------------------------ control
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
        self.threat = 0
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
        self.threat = 0
        self.clear_log()
        self.log(f"=== Evaluating {len(self.engine.rules)} security rules ===", "head")
        self.log("Sensors: " + "  ".join(
            f"{s.symbol}={'T' if self.snapshot[s.symbol] else 'F'}" for s in SENSORS))
        if not any(self.snapshot.values()):
            self.log("(all sensors clear - no rule can be triggered)", "false")
        self.redraw()

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

        self.log(f"Rule {index + 1}: {rule.name} [{LEVELS[rule.severity]}]", "head")
        outcome = "TRUE" if result.is_match else "FALSE"
        self.log(f"    {expression.to_symbols()}  ->  {expression.to_values(self.snapshot)}"
                 f"  =  {outcome}   ({result.satisfied}/{result.total}, {result.percent}%)",
                 "true" if result.is_match else "false")
        self.set_status(f"Evaluated rule {index + 1}/{len(self.engine.rules)}: "
                        f"{rule.name} = {outcome}")
        self.redraw()
        if len(self.results) == len(self.engine.rules):
            self.finalize()

    # ---------------------------------------------------------- finishing
    def finalize(self) -> None:
        self.finished = True
        self.running = False
        self.cancel_timer()
        triggered = self.engine.triggered(self.results)
        self.threat = self.engine.threat_level(self.results)
        closest = [] if triggered else self.engine.closest(self.results, self.snapshot)
        self.history.add(Session(datetime.now(), dict(self.snapshot),
                                 list(self.results), closest))
        self.refresh_history()
        self.render_results(triggered, closest)
        self.notebook.select(self.tab_results)
        self.redraw()
        self.update_buttons()

        if triggered:
            self.log(f"=== Done: {len(triggered)} rule(s) triggered - "
                     f"threat level {LEVELS[self.threat]} ===", "head")
            self.set_status(f"Done. Threat level {LEVELS[self.threat]} - "
                            f"{len(triggered)} rule(s) triggered. See the Results tab.")
        else:
            self.log("=== Done: no security rule triggered ===", "head")
            if closest:
                top = closest[0]
                self.log(f"Watch list: closest rule is {top.rule.name} "
                         f"({top.percent}% rule match)", "head")
                self.set_status(f"System SECURE - no rule triggered. Closest: "
                                f"{top.rule.name} ({top.percent}%).")
            else:
                self.set_status("System SECURE - no rule triggered.")

        if self.threat == 4:
            messagebox.showwarning(
                "CRITICAL SECURITY ALERT",
                "A CRITICAL rule was triggered:\n\n" +
                "\n".join(f"- {r.rule.name}: {r.rule.action}"
                          for r in triggered if r.rule.severity == 4))

    def _show_results_placeholder(self) -> None:
        t = self.results_text
        t.config(state="normal")
        t.delete("1.0", "end")
        self._put(t, "No analysis yet.\n", "h1")
        self._put(t, "\nSet the sensors, then press Run or Step.\n")
        t.config(state="disabled")

    def render_results(self, triggered: List[RuleResult], closest: List[RuleResult]) -> None:
        t = self.results_text
        facts = self.snapshot
        t.config(state="normal")
        t.delete("1.0", "end")
        put = lambda s, *tags: self._put(t, s, *tags)  # noqa: E731

        active = [s.name for s in SENSORS if facts[s.symbol]]
        put("SECURITY ANALYSIS\n", "h1")
        put("Active sensors: " + (", ".join(active) if active else "none") + "\n")
        put("\nOVERALL THREAT LEVEL: ", "h2")
        put(f" {LEVELS[self.threat]} \n", f"lvl{self.threat}")

        if triggered:
            put(f"\n{len(triggered)} rule(s) triggered (highest severity first)\n", "h2")
            for number, result in enumerate(triggered, start=1):
                rule = result.rule
                expression = rule.expression
                put(f"\n{number}. ", "h2")
                put(f" {LEVELS[rule.severity]} ", f"lvl{rule.severity}")
                put(f"  {rule.name}\n", "h2")
                put(f"   Rule:      {expression.to_text()}\n")
                put(f"   Symbolic:  {expression.to_symbols()}\n")
                put("   Evaluation:\n")
                for prop in expression.propositions():
                    value = facts.get(prop.symbol, False)
                    put(f"      {prop.name} = ")
                    put("TRUE\n" if value else "FALSE\n", "true" if value else "false")
                put(f"   Substituted: {expression.to_values(facts)} = TRUE\n")
                put("   Result:    ")
                put("TRUE\n", "true")
                put(f"   Match:     {result.satisfied}/{result.total} conditions "
                    f"-> {result.percent}%\n")
                put(f"   Why: {rule.explanation}\n", "note")
                put(f"   Action: {rule.action}\n", "h2")
            put("\nSummary\n", "h2")
            put(f"{'Alert':<30}{'Level':<10}{'Match'}\n")
            for r in triggered:
                put(f"{r.rule.name:<30}{LEVELS[r.rule.severity]:<10}{r.percent}%\n")
        else:
            put("\nNo predefined security rule was triggered.\n", "true")
            if not active:
                put("\nAll sensors are clear.\n")
            elif closest:
                put("\nWATCH LIST - possible developing threats\n", "h1")
                put("Closest rules to the current sensor state:\n", "note")
                top = closest[0]
                put(f"\nClosest: {top.rule.name} ", "h2")
                put(f"({top.percent}% rule match)\n", "true")
                for rank, r in enumerate(closest, start=1):
                    put(f"\n{rank}. {r.rule.name} ", "h2")
                    put(f"[{LEVELS[r.rule.severity]}]\n", f"lvl{r.rule.severity}")
                    put(f"   Rule:        {r.rule.expression.to_text()}\n")
                    put(f"   Rule match:  {r.satisfied}/{r.total} conditions -> {r.percent}%\n")
                    supported = self.engine.supporting_sensors(r.rule, facts)
                    unmet = self.engine.unmet_conditions(r.rule, facts)
                    put("   Supported by: " + (", ".join(supported) or "-") + "\n", "true")
                    put("   Still needed: " + (", ".join(unmet) or "-") + "\n", "false")
            else:
                put("\nNo rule has any of its conditions satisfied.\n")
        t.config(state="disabled")
        t.see("1.0")

    # ------------------------------------------------------------ history
    def refresh_history(self) -> None:
        self.history_list.delete(0, "end")
        for number, session in enumerate(self.history.all(), start=1):
            self.history_list.insert(
                "end", f"Session {number}  {session.timestamp:%H:%M:%S}  -  "
                       f"{LEVELS[session.level()]}  ({len(session.triggered())} rule(s))")
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
        sensors = session.active_sensors()
        put("Active sensors: " + (", ".join(sensors) if sensors else "none") + "\n")
        put("Threat level: ", "h2")
        put(f" {LEVELS[session.level()]} \n", f"lvl{session.level()}")
        triggered = session.triggered()
        if triggered:
            put("\nTriggered rules:\n", "h2")
            for r in triggered:
                put(f"  - [{LEVELS[r.rule.severity]}] {r.rule.name}: "
                    f"{r.satisfied}/{r.total}, {r.percent}%\n")
        else:
            put("\nTriggered rules: none\n", "true")
            for r in session.closest:
                put(f"  closest: {r.rule.name}: {r.satisfied}/{r.total}, {r.percent}%\n")
        t.config(state="disabled")

    # -------------------------------------------------------- truth table
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
                put("   <-- current sensors", "note")
            put("\n")
        put(f"\n{true_rows} of {2 ** len(props)} combinations trigger the rule.\n", "note")
        t.config(state="disabled")
        if switch_tab:
            self.notebook.select(self.tab_truth)

    # ---------------------------------------------------------------- end
    def on_close(self) -> None:
        self.cancel_timer()
        self.root.destroy()


def main() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as error:
        print(f"Could not open a window (no display available?): {error}")
        sys.exit(1)
    SecurityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()