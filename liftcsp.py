"""
CSP Lift Allotment Simulator  (Python + Tkinter)  -  multi-person version
-------------------------------------------------------------------------
What is new compared with the single-passenger version
  * Any number of people (max 8).  Every person has his/her OWN
        - start floor   (1 = Main Gate, or any other floor 2-5)
        - destination room (e.g. 521 = floor 5, room 21)
        - departure time (seconds) so people can start at different moments
    Use the "Number of people" box to add several identical people at once
    (handy to see the queueing / waiting effect).
  * Waiting time:  a lift carries ONE rider at a time.  If a lift is occupied
    the next person waits at its door until it is free.  Optionally the empty
    lift also needs travel time to come to the caller.  Waiting time per
    person and the total waiting time are reported.

Building
  * 5 floors, 25 rooms per floor  (room 521 -> floor 5, room no. 21)
  * Lift A stops at floors 1, 2, 3
  * Lift B stops at floors 2, 3, 4, 5
  * Lift C stops at floors 2, 4, 5
  * A passenger may change lifts at a common floor.

CSP model (solved separately for every person, animated one after another)
  Variables  : Leg1, Leg2, ... LegK   (K = max legs)
  Domain     : (lift, from_floor, to_floor)
  Constraints:
    C1  Leg1.from = the person's start floor
    C2  Leg(i).to = Leg(i+1).from                (continuous journey)
    C3  from and to must both be stops of the chosen lift
    C4  consecutive legs use different lifts     (otherwise it is one leg)
    C5  no floor is visited twice                (no useless loops)
    C6  the last leg must end at the destination floor
  Objective  : fewest legs (fewest lift changes), then fewest floors travelled.

Two phases when you press Run
  1. PLAN      - backtracking search finds the best route for each person
                 (search animation can be switched off in Options).
  2. SIMULATE  - all people move at the same time along their routes on a
                 common clock; lifts serve one rider at a time (first come,
                 first served) and people wait when a lift is occupied.
"""

import heapq
import random
import tkinter as tk
from tkinter import ttk

MODES = ["Find all solutions", "Stop at first solution", "Iterative deepening (fewest legs)"]
MAIN_GATE_FLOOR = 1
FLOORS = 5
ROOMS_PER_FLOOR = 25
MAX_PEOPLE = 8
BG = "#f5f5f7"

PERSON_COLORS = ["#e63946", "#3a86ff", "#2a9d8f", "#fb8500",
                 "#8338ec", "#d4a017", "#ef476f", "#118ab2"]

LIFTS = {
    "Lift A": {"stops": [1, 2, 3],    "color": "#3a86ff", "tint": "#e8f0ff", "x": 250},
    "Lift B": {"stops": [2, 3, 4, 5], "color": "#fb8500", "tint": "#fff1e0", "x": 340},
    "Lift C": {"stops": [2, 4, 5],    "color": "#2a9d8f", "tint": "#e3f5f2", "x": 430},
}

# ---------------- scene geometry ----------------
W, H = 840, 500
TOP, FLOOR_H = 58, 80
GROUND_Y = TOP + FLOORS * FLOOR_H          # bottom of the building
BLD_X0, BLD_X1 = 140, 800
GATE_X = 72
DEFAULT_CAB = {"Lift A": 1, "Lift B": 2, "Lift C": 2}
FRAME_MS = 25
WALK_SPEED = 60.0                           # pixels per simulated second


def yfl(f):
    """y coordinate of the floor surface for (possibly fractional) floor f."""
    return TOP + (FLOORS + 1 - f) * FLOOR_H - 6


def door_x(room_no):
    """Destination doors are drawn at an x position that follows the room number."""
    return 510 + (room_no - 1) * 10


def stand_x(room_no):
    return door_x(room_no) - 24


def start_x(floor):
    """Where a person starts: outside at the Main Gate, or inside the building on other floors."""
    return GATE_X if floor == MAIN_GATE_FLOOR else BLD_X0 + 58


def slot_off(i):
    """Small x offset so several people standing at the same spot do not overlap."""
    return (i % 4) * 9 - 13


# =====================================================================
#  CSP solver (one person)
# =====================================================================
class LiftCSP:
    def __init__(self, start_floor, goal_floor, max_legs=4, allow_down=True,
                 mode=MODES[0], order="Default"):
        self.start = start_floor
        self.goal = goal_floor
        self.limit = max_legs        # user-selected maximum number of legs
        self.max_legs = max_legs     # current depth limit (changes in iterative deepening)
        self.allow_down = allow_down
        self.mode = mode
        self.order = order
        self.legs = []               # list of (lift, from, to)
        self.solutions = []
        self.stop = False
        self.assigns = 0
        self.backtracks = 0

    # ---------- constraint handling ----------
    def current_floor(self):
        return self.legs[-1][2] if self.legs else self.start

    def legal_values(self):
        cur = self.current_floor()
        prev_lift = self.legs[-1][0] if self.legs else None
        visited = {self.start} | {l[2] for l in self.legs}
        vals = []
        for name, info in LIFTS.items():
            if name == prev_lift or cur not in info["stops"]:      # C3, C4
                continue
            for to in info["stops"]:
                if to == cur or to in visited:                     # C5
                    continue
                if not self.allow_down and to < cur:
                    continue
                vals.append((name, cur, to))
        if self.order == "Random":
            random.shuffle(vals)
        elif self.order.startswith("Greedy"):
            vals.sort(key=lambda v: abs(self.goal - v[2]))
        return vals

    # ---------- backtracking search (generator for animation) ----------
    def run(self):
        if self.mode.startswith("Iterative"):
            for k in range(1, self.limit + 1):
                self.max_legs = k
                yield ("deepen", k, None)
                yield from self.search()
                if self.solutions:              # first depth with a solution = fewest legs
                    return
        else:
            self.max_legs = self.limit
            yield from self.search()

    def search(self):
        if self.current_floor() == self.goal:                       # C6
            self.solutions.append(list(self.legs))
            if self.mode.startswith("Stop"):
                self.stop = True
            yield ("solution", len(self.legs), None)
            return
        if len(self.legs) >= self.max_legs:
            return
        for val in self.legal_values():
            self.legs.append(val)
            self.assigns += 1
            idx = len(self.legs)
            yield ("place", idx, val)
            before = len(self.solutions)
            yield from self.search()
            if self.stop:                                           # keep the solution on screen
                return
            self.legs.pop()
            if len(self.solutions) > before:
                yield ("undo", idx, val)                            # not a failure
            else:
                self.backtracks += 1
                yield ("backtrack", idx, val)


# =====================================================================
#  Multi-person schedule with lift occupancy / waiting time
# =====================================================================
def build_schedule(people, per_floor, reposition):
    """
    Turn the planned routes of all people into time lines.

    * A lift serves ONE rider at a time, first come first served.
    * A person who reaches a lift door while it is busy waits there.
    * If `reposition` is True, a free lift first travels (empty) to the floor
      of the caller, which also counts as waiting time.

    Returns (lift_state, total_time, events).
    Each person dict receives: segs (time line), wait (total waiting s), finish (arrival s).
    """
    lifts = {n: {"free": 0.0, "pos": float(DEFAULT_CAB[n]), "segs": []} for n in LIFTS}
    events, heap = [], []

    def walk(p, t0, x0, x1, fl):
        dur = abs(x1 - x0) / WALK_SPEED
        if dur > 1e-9:
            p["segs"].append({"type": "walk", "t0": t0, "t1": t0 + dur, "x0": x0, "x1": x1, "fl": fl})
        return t0 + dur

    def finish(p, t, x):
        t_end = walk(p, t, x, stand_x(p["num"]), p["floor"])
        p["finish"] = t_end
        events.append((t_end, f"{p['name']} reached Room {p['room']}"))

    for i, p in enumerate(people):
        p["segs"], p["wait"], p["finish"] = [], 0.0, None
        legs = p.get("legs")
        if legs is None:                       # no route was found for this person
            continue
        x, t = start_x(p["start"]), float(p["depart"])
        if not legs:                           # already on the right floor
            finish(p, t, x)
            continue
        lx = LIFTS[legs[0][0]]["x"]
        t = walk(p, t, x, lx, p["start"])
        heapq.heappush(heap, (t, i, 0))        # (time person reaches the lift door, person, leg)

    while heap:
        T, i, k = heapq.heappop(heap)
        p = people[i]
        lift, f, to = p["legs"][k]
        L = lifts[lift]
        lx = LIFTS[lift]["x"]

        busy = L["free"] > T + 1e-9            # lift is carrying somebody when requested
        depart = max(T, L["free"])
        reach = depart + (abs(L["pos"] - f) * per_floor if reposition else 0.0)
        L["segs"].append({"type": "come", "t0": depart, "t1": reach,
                          "f0": L["pos"], "f1": float(f)})

        wait = reach - T
        if wait > 1e-9:
            p["segs"].append({"type": "wait", "t0": T, "t1": reach, "x": lx, "fl": f, "busy": busy})
            p["wait"] += wait
            why = "lift occupied" if busy else "lift on its way"
            events.append((T, f"{p['name']} waits {wait:.1f}s at F{f} for {lift} ({why})"))

        end = reach + abs(to - f) * per_floor
        seg = {"type": "ride", "t0": reach, "t1": end, "f0": float(f), "f1": float(to),
               "lift": lift, "who": i}
        L["segs"].append(seg)
        p["segs"].append(seg)
        L["free"], L["pos"] = end, float(to)
        events.append((reach, f"{p['name']} boards {lift}: F{f} -> F{to}"))

        if k + 1 < len(p["legs"]):             # change lift on floor `to`
            nx = LIFTS[p["legs"][k + 1][0]]["x"]
            t2 = walk(p, end, lx, nx, to)
            heapq.heappush(heap, (t2, i, k + 1))
        else:
            finish(p, end, lx)

    events.sort(key=lambda e: e[0])
    total = max((p["finish"] for p in people if p["finish"] is not None), default=0.0)
    return lifts, total, events


def cab_position(segs, t, initial):
    pos = float(initial)
    for s in segs:
        if t < s["t0"]:
            return pos
        if t < s["t1"]:
            u = (t - s["t0"]) / (s["t1"] - s["t0"])
            return s["f0"] + (s["f1"] - s["f0"]) * u
        pos = s["f1"]
    return pos


def lift_occupied(segs, t):
    return any(s["type"] == "ride" and s["t0"] <= t < s["t1"] for s in segs)


def person_state(p, t):
    """Where is the person at simulated time t?"""
    st = {"x": start_x(p["start"]), "fl": float(p["start"]), "lift": None,
          "wait": None, "busy": False, "arrived": False}
    for s in p["segs"]:
        if t < s["t0"]:
            break
        if s["type"] == "walk":
            if t < s["t1"]:
                u = (t - s["t0"]) / (s["t1"] - s["t0"])
                st.update(x=s["x0"] + (s["x1"] - s["x0"]) * u, fl=float(s["fl"]))
                return st
            st.update(x=s["x1"], fl=float(s["fl"]))
        elif s["type"] == "wait":
            st.update(x=s["x"], fl=float(s["fl"]))
            if t < s["t1"]:
                st.update(wait=s["t1"] - t, busy=s["busy"])
                return st
        else:                                   # ride
            lx = LIFTS[s["lift"]]["x"]
            if t < s["t1"]:
                u = (t - s["t0"]) / (s["t1"] - s["t0"])
                st.update(x=lx, fl=s["f0"] + (s["f1"] - s["f0"]) * u, lift=s["lift"])
                return st
            st.update(x=lx, fl=s["f1"])
    st["arrived"] = p.get("finish") is not None and t >= p["finish"] - 1e-9
    return st


# =====================================================================
#  GUI
# =====================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSP Lift Allotment Simulator - multi person")
        self.configure(bg=BG)
        self.people = []
        self.csp = None
        self.gen = None
        self.running = False
        self.flash = None
        self.gold = False
        self.last_kind = None
        self.phase = "plan"              # plan -> sim -> done
        self.plan_idx = 0
        # simulation state
        self.sched = None
        self.sim_t = 0.0
        self.sim_total = 0.0
        self.sim_job = None
        self.events = []
        self.ev_i = 0
        # plan-phase animation state
        self.segs = []
        self.cur_seg = None
        self.busy = False
        self.anim_job = None
        self.cab = dict(DEFAULT_CAB)
        self.px, self.pfl, self.plift = GATE_X, 1.0, None
        self._build_ui()
        self._add_person(MAIN_GATE_FLOOR, 521, 0.0)
        self._refresh_people_list()
        self.reset()

    # ---------- UI ----------
    def _build_ui(self):
        tk.Label(self, text="CSP Lift Allotment Simulator", font=("Segoe UI", 16, "bold"),
                 bg=BG).grid(row=0, column=0, columnspan=2, pady=(8, 0))
        tk.Label(self, text="Each person: own start floor & destination room  •  a lift carries one "
                            "rider at a time  •  others wait when it is occupied",
                 font=("Segoe UI", 9), bg=BG, fg="#555").grid(row=1, column=0, columnspan=2)

        self.canvas = tk.Canvas(self, width=W, height=H, bg="white", highlightthickness=1,
                                highlightbackground="#bbb")
        self.canvas.grid(row=2, column=0, padx=10, pady=6, sticky="n")

        side = tk.Frame(self, bg=BG)
        side.grid(row=2, column=1, padx=(0, 10), pady=6, sticky="n")

        btns = tk.Frame(side, bg=BG)
        btns.pack(pady=(0, 2))
        ttk.Button(btns, text="▶ Run", command=self.run).grid(row=0, column=0, padx=2)
        ttk.Button(btns, text="⏸ Pause", command=self.pause).grid(row=0, column=1, padx=2)
        ttk.Button(btns, text="⏭ Step", command=self.step).grid(row=0, column=2, padx=2)
        ttk.Button(btns, text="⟲ Reset", command=self.reset).grid(row=0, column=3, padx=2)
        tk.Label(side, text="Speed (fast ⇢ slow)", bg=BG).pack(anchor="w")
        self.speed = tk.Scale(side, from_=1, to=1500, orient="horizontal", length=330, bg=BG)
        self.speed.set(500)
        self.speed.pack()

        nb = ttk.Notebook(side)
        nb.pack(fill="both", pady=(4, 0))

        # ---------------- tab: People ----------------
        tab_p = tk.Frame(nb, bg=BG, padx=6, pady=6)
        nb.add(tab_p, text=" People ")
        form = tk.LabelFrame(tab_p, text="Add person(s)", bg=BG, padx=8, pady=4)
        form.pack(fill="x")

        def lab(r, txt):
            tk.Label(form, text=txt, bg=BG).grid(row=r, column=0, sticky="w", pady=1)

        lab(0, "Start floor (1 = Main Gate):")
        self.p_start = tk.StringVar(value="1")
        ttk.Combobox(form, textvariable=self.p_start, state="readonly", width=4,
                     values=[str(f) for f in range(1, FLOORS + 1)]).grid(row=0, column=1, sticky="w", padx=4)
        lab(1, "Destination room:")
        self.p_room = tk.StringVar(value="521")
        ent = tk.Entry(form, textvariable=self.p_room, width=7)
        ent.grid(row=1, column=1, sticky="w", padx=4)
        ent.bind("<Return>", lambda e: self.add_people())
        lab(2, "Leaves at (sec):")
        self.p_depart = tk.StringVar(value="0")
        tk.Spinbox(form, from_=0, to=120, width=5, textvariable=self.p_depart).grid(
            row=2, column=1, sticky="w", padx=4)
        lab(3, "Number of people:")
        self.p_count = tk.StringVar(value="1")
        tk.Spinbox(form, from_=1, to=MAX_PEOPLE, width=5, textvariable=self.p_count).grid(
            row=3, column=1, sticky="w", padx=4)
        ttk.Button(form, text="➕ Add", command=self.add_people).grid(row=3, column=2, padx=6)

        lf = tk.LabelFrame(tab_p, text=f"People in the simulation (max {MAX_PEOPLE})", bg=BG,
                           padx=4, pady=4)
        lf.pack(fill="x", pady=6)
        self.people_lb = tk.Listbox(lf, height=9, width=42, font=("Consolas", 9),
                                   selectmode="extended", exportselection=False)
        lsb = ttk.Scrollbar(lf, orient="vertical", command=self.people_lb.yview)
        self.people_lb.config(yscrollcommand=lsb.set)
        self.people_lb.pack(side="left")
        lsb.pack(side="left", fill="y")
        pb = tk.Frame(tab_p, bg=BG)
        pb.pack()
        ttk.Button(pb, text="Remove selected", command=self.remove_selected).pack(side="left", padx=3)
        ttk.Button(pb, text="Clear all", command=self.clear_people).pack(side="left", padx=3)

        # ---------------- tab: Options ----------------
        tab_o = tk.Frame(nb, bg=BG, padx=6, pady=6)
        nb.add(tab_o, text=" Options ")
        opt = tk.LabelFrame(tab_o, text="Search (changes auto-reset)", bg=BG, padx=8, pady=4)
        opt.pack(fill="x")
        row2 = tk.Frame(opt, bg=BG)
        row2.pack(anchor="w")
        tk.Label(row2, text="Max legs (lift rides):", bg=BG).pack(side="left")
        self.k_var = tk.IntVar(value=4)
        tk.Spinbox(row2, from_=1, to=5, width=3, textvariable=self.k_var,
                   command=self.reset).pack(side="left", padx=4)
        self.down_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt, text="Allow going down", variable=self.down_var, bg=BG,
                       command=self.reset).pack(anchor="w")
        self.anim_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt, text="Animate the search (off = plan instantly)", variable=self.anim_var,
                       bg=BG).pack(anchor="w")
        tk.Label(opt, text="Search mode:", bg=BG).pack(anchor="w")
        self.mode_var = tk.StringVar(value=MODES[0])
        mb = ttk.Combobox(opt, textvariable=self.mode_var, state="readonly", width=32, values=MODES)
        mb.pack(anchor="w")
        mb.bind("<<ComboboxSelected>>", lambda e: self.reset())
        tk.Label(opt, text="Value ordering:", bg=BG).pack(anchor="w")
        self.order_var = tk.StringVar(value="Default")
        cb = ttk.Combobox(opt, textvariable=self.order_var, state="readonly", width=32,
                          values=["Default", "Random", "Greedy (nearest to goal)"])
        cb.pack(anchor="w")
        cb.bind("<<ComboboxSelected>>", lambda e: self.reset())

        wt = tk.LabelFrame(tab_o, text="Lifts & waiting time (changes auto-reset)", bg=BG, padx=8, pady=4)
        wt.pack(fill="x", pady=8)
        row3 = tk.Frame(wt, bg=BG)
        row3.pack(anchor="w")
        tk.Label(row3, text="Ride time per floor (sec):", bg=BG).pack(side="left")
        self.ride_var = tk.StringVar(value="3")
        sp = tk.Spinbox(row3, from_=0.5, to=10, increment=0.5, width=4, textvariable=self.ride_var,
                        command=self.reset)
        sp.pack(side="left", padx=4)
        sp.bind("<Return>", lambda e: self.reset())
        self.repo_var = tk.BooleanVar(value=True)
        tk.Checkbutton(wt, text="Free lift must travel to the caller first", variable=self.repo_var,
                       bg=BG, command=self.reset).pack(anchor="w")
        tk.Label(wt, text="A lift carries one rider at a time (first come, first served).\n"
                          "Anyone arriving while it is occupied waits at its door.",
                 bg=BG, fg="#555", justify="left", font=("Segoe UI", 8)).pack(anchor="w")

        # ---------------- tab: Results ----------------
        tab_r = tk.Frame(nb, bg=BG, padx=6, pady=6)
        nb.add(tab_r, text=" Results ")
        st = tk.LabelFrame(tab_r, text="Statistics", bg=BG, padx=8, pady=4)
        st.pack(fill="x")
        self.stat_lbl = tk.Label(st, text="", justify="left", bg=BG, font=("Consolas", 9))
        self.stat_lbl.pack(anchor="w")

        lg = tk.LabelFrame(tab_r, text="Lifts (stops at floors)", bg=BG, padx=8, pady=2)
        lg.pack(fill="x", pady=4)
        for name, info in LIFTS.items():
            r = tk.Frame(lg, bg=BG)
            r.pack(anchor="w")
            tk.Label(r, text=name, bg=info["color"], fg="white", width=7,
                     font=("Segoe UI", 8, "bold")).pack(side="left")
            tk.Label(r, text="  " + ", ".join(map(str, info["stops"])), bg=BG).pack(side="left")

        sol = tk.LabelFrame(tab_r, text="Valid routes of the person being planned", bg=BG, padx=4, pady=4)
        sol.pack(fill="x")
        self.sol_text = tk.Text(sol, height=5, width=46, font=("Consolas", 8), wrap="word",
                                state="disabled")
        sb = ttk.Scrollbar(sol, orient="vertical", command=self.sol_text.yview)
        self.sol_text.config(yscrollcommand=sb.set)
        self.sol_text.pack(side="left")
        sb.pack(side="left", fill="y")

        rs = tk.LabelFrame(tab_r, text="Journey summary (with waiting time)", bg=BG, padx=4, pady=4)
        rs.pack(fill="x", pady=(4, 0))
        self.res_text = tk.Text(rs, height=7, width=46, font=("Consolas", 8), wrap="word",
                                state="disabled")
        sb2 = ttk.Scrollbar(rs, orient="vertical", command=self.res_text.yview)
        self.res_text.config(yscrollcommand=sb2.set)
        self.res_text.pack(side="left")
        sb2.pack(side="left", fill="y")

        self.log = tk.Text(self, height=5, width=140, font=("Consolas", 9), state="disabled")
        self.log.grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 4))
        self.status = tk.Label(self, text="", bg=BG, font=("Segoe UI", 10, "bold"))
        self.status.grid(row=4, column=0, columnspan=2, pady=(0, 6))

    # ---------- people management ----------
    @staticmethod
    def parse_room(text):
        try:
            room = int(str(text).strip())
            floor, num = divmod(room, 100)
            if 1 <= floor <= FLOORS and 1 <= num <= ROOMS_PER_FLOOR:
                return room, floor, num
        except ValueError:
            pass
        return None

    def _renumber(self):
        for i, p in enumerate(self.people):
            p["name"] = f"P{i + 1}"
            p["color"] = PERSON_COLORS[i % len(PERSON_COLORS)]

    def _add_person(self, start, room, depart):
        floor, num = divmod(room, 100)
        self.people.append({"name": "", "color": "", "start": start, "room": room, "floor": floor,
                            "num": num, "depart": depart, "legs": None, "segs": [],
                            "wait": 0.0, "finish": None})
        self._renumber()

    def _refresh_people_list(self):
        self.people_lb.delete(0, "end")
        for i, p in enumerate(self.people):
            where = "Main Gate" if p["start"] == MAIN_GATE_FLOOR else f"Floor {p['start']}"
            self.people_lb.insert("end", f"{p['name']}  {where:<9} → Room {p['room']}  "
                                        f"(leaves at {p['depart']:g}s)")
            self.people_lb.itemconfig(i, foreground=p["color"])

    def add_people(self):
        parsed = self.parse_room(self.p_room.get())
        if parsed is None:
            self.status.config(text="Invalid room. Use 101-525 (floor 1-5, room 01-25).", fg="#c1121f")
            return
        room = parsed[0]
        try:
            start = int(self.p_start.get())
            depart = max(0.0, float(self.p_depart.get()))
            count = max(1, int(self.p_count.get()))
        except ValueError:
            self.status.config(text="Start floor, leave time and count must be numbers.", fg="#c1121f")
            return
        free = MAX_PEOPLE - len(self.people)
        if free <= 0:
            self.status.config(text=f"Maximum of {MAX_PEOPLE} people reached.", fg="#c1121f")
            return
        for _ in range(min(count, free)):
            self._add_person(start, room, depart)
        self._refresh_people_list()
        note = None
        if count > free:
            note = f"Only {free} added - maximum of {MAX_PEOPLE} people."
        self.reset(note, err=bool(note))

    def remove_selected(self):
        idx = set(self.people_lb.curselection())
        if not idx:
            return
        self.people = [p for i, p in enumerate(self.people) if i not in idx]
        self._renumber()
        self._refresh_people_list()
        self.reset()

    def clear_people(self):
        self.people = []
        self._refresh_people_list()
        self.reset()

    # ---------- helpers ----------
    def cur_person(self):
        return self.people[min(self.plan_idx, len(self.people) - 1)]

    def describe(self, legs, p):
        parts = ["Main Gate" if p["start"] == MAIN_GATE_FLOOR else f"Floor {p['start']}"]
        parts += [f"{l} (F{f}→F{t})" for l, f, t in legs]
        parts.append(f"walk to Room {p['room']}")
        return " → ".join(parts)

    def write(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _set_text(self, widget, text):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.config(state="disabled")

    def set_solutions_text(self, best=None):
        text = ""
        if self.csp is not None and self.people:
            p = self.cur_person()
            for n, legs in enumerate(self.csp.solutions, 1):
                star = "★ " if best is not None and legs is best else "  "
                text += f"{star}#{n} [{len(legs)} leg(s)] {self.describe(legs, p)}\n\n"
        self._set_text(self.sol_text, text)

    def _fill_results(self):
        lines = []
        for p in self.people:
            if p["legs"] is None:
                lines.append(f"{p['name']} F{p['start']}→Room {p['room']}: NO ROUTE FOUND\n")
                continue
            route = ", ".join(f"{l} F{f}→F{t}" for l, f, t in p["legs"]) or "no lift needed"
            lines.append(f"{p['name']} F{p['start']}→Room {p['room']}\n"
                         f"   {route}\n"
                         f"   waiting {p['wait']:.1f}s | arrives at t={p['finish']:.1f}s\n")
        tw = sum(p["wait"] for p in self.people)
        lines.append(f"TOTAL waiting = {tw:.1f}s   |   all arrived at t={self.sim_total:.1f}s")
        self._set_text(self.res_text, "\n".join(lines))

    def _opts(self):
        try:
            k = max(1, min(5, int(self.k_var.get())))
        except (ValueError, tk.TclError):
            k = 4
        try:
            per_floor = max(0.1, float(self.ride_var.get()))
        except ValueError:
            per_floor = 3.0
        return k, per_floor

    # ---------- drawing: the pictorial scene ----------
    @staticmethod
    def _cloud(c, x, y, s=1.0):
        for ox, oy, r in ((0, 0, 14), (16, -7, 18), (34, 0, 14), (17, 6, 14)):
            c.create_oval(x + (ox - r) * s, y + (oy - r) * s, x + (ox + r) * s, y + (oy + r) * s,
                          fill="white", outline="")

    @staticmethod
    def _person(c, x, fy, s=1.0, color="#e63946"):
        c.create_rectangle(x + 5 * s, fy - 24 * s, x + 10 * s, fy - 12 * s, fill="#264653", outline="")
        c.create_line(x - 3 * s, fy - 11 * s, x - 3 * s, fy, width=3, fill="#264653")
        c.create_line(x + 3 * s, fy - 11 * s, x + 3 * s, fy, width=3, fill="#264653")
        c.create_rectangle(x - 7 * s, fy - 25 * s, x + 7 * s, fy - 10 * s, fill=color, outline="#333")
        c.create_line(x - 7 * s, fy - 23 * s, x - 10 * s, fy - 13 * s, width=3, fill=color)
        c.create_line(x + 7 * s, fy - 23 * s, x + 10 * s, fy - 13 * s, width=3, fill=color)
        c.create_oval(x - 6 * s, fy - 37 * s, x + 6 * s, fy - 25 * s, fill="#ffcc99", outline="#b5835a")
        c.create_arc(x - 6 * s, fy - 38 * s, x + 6 * s, fy - 26 * s, start=0, extent=180,
                     fill="#3d2c2e", outline="")

    def get_actors(self):
        """One dict per person describing where he/she is right now."""
        out = []
        for i, p in enumerate(self.people):
            if self.phase in ("sim", "done"):
                out.append(person_state(p, self.sim_t))
            elif i == self.plan_idx and self.csp is not None:
                arrived = (self.plift is None and abs(self.px - stand_x(p["num"])) < 1
                           and abs(self.pfl - p["floor"]) < 0.02)
                out.append({"x": self.px, "fl": self.pfl, "lift": self.plift, "wait": None,
                            "busy": False, "arrived": arrived})
            else:
                out.append({"x": start_x(p["start"]), "fl": float(p["start"]), "lift": None,
                            "wait": None, "busy": False, "arrived": False})
        return out

    def cab_state(self):
        if self.phase in ("sim", "done") and self.sched:
            pos = {n: cab_position(self.sched[n]["segs"], self.sim_t, DEFAULT_CAB[n]) for n in LIFTS}
            occ = {n: lift_occupied(self.sched[n]["segs"], self.sim_t) for n in LIFTS}
            return pos, occ
        return dict(self.cab), {n: self.plift == n for n in LIFTS}

    def draw(self):
        c = self.canvas
        c.delete("all")
        actors = self.get_actors()
        goal_floors = {p["floor"] for p in self.people}
        start_floors = {p["start"] for p in self.people}
        rooms = {}
        for a, p in zip(actors, self.people):
            key = (p["floor"], p["num"])
            rooms[key] = rooms.get(key, False) or a["arrived"]
        doors_on = {}
        for fl, num in rooms:
            doors_on.setdefault(fl, []).append(door_x(num))

        # sky, sun, clouds, ground
        c.create_rectangle(0, 0, W, H, fill="#dff1ff", outline="")
        c.create_oval(760, 8, 806, 54, fill="#ffd166", outline="#ffb703", width=2)
        self._cloud(c, 70, 34, 0.9)
        self._cloud(c, 560, 22, 0.7)
        c.create_rectangle(0, GROUND_Y, W, H, fill="#8ecf8b", outline="")
        c.create_rectangle(0, GROUND_Y - 6, BLD_X0, GROUND_Y + 6, fill="#d9c8a9", outline="")

        # building: roof + walls
        c.create_polygon(BLD_X0 - 14, TOP, BLD_X1 + 14, TOP, BLD_X1 - 12, TOP - 28, BLD_X0 + 12, TOP - 28,
                         fill="#8d99ae", outline="#4a4e69", width=2)
        c.create_text((BLD_X0 + BLD_X1) / 2, TOP - 14, text="BUILDING", fill="white",
                      font=("Segoe UI", 10, "bold"))
        for f in range(1, FLOORS + 1):
            y0 = TOP + (FLOORS - f) * FLOOR_H
            c.create_rectangle(BLD_X0, y0, BLD_X1, y0 + FLOOR_H,
                               fill="#fff3bf" if f in goal_floors else "#fbf3e4", outline="#c9c2b0")
            c.create_rectangle(BLD_X0 - 4, y0 + FLOOR_H - 6, BLD_X1 + 4, y0 + FLOOR_H,
                               fill="#8d99ae", outline="#4a4e69")
            c.create_rectangle(BLD_X0 + 6, y0 + 6, BLD_X0 + 36, y0 + 22, fill="#457b9d", outline="")
            c.create_text(BLD_X0 + 21, y0 + 14, text=f"F{f}", fill="white", font=("Segoe UI", 8, "bold"))
            for i in range(4):                                   # decorative windows only
                wx = 500 + i * 70
                if any(abs(wx + 17 - d) < 36 for d in doors_on.get(f, [])):
                    continue
                c.create_rectangle(wx, y0 + 12, wx + 34, y0 + 42, fill="#bde0fe", outline="#457b9d", width=2)
                c.create_line(wx + 17, y0 + 12, wx + 17, y0 + 42, fill="#457b9d")
                c.create_line(wx, y0 + 27, wx + 34, y0 + 27, fill="#457b9d")
            c.create_text(BLD_X1 - 8, y0 + FLOOR_H - 14, anchor="e", text="corridor →",
                          fill="#9a9484", font=("Segoe UI", 7, "italic"))

        # lift shafts (doors, lamps, "no stop" panels)
        for name, info in LIFTS.items():
            x, col = info["x"], info["color"]
            c.create_rectangle(x - 27, TOP, x + 27, GROUND_Y - 6, fill=info["tint"], outline=col, width=2)
            c.create_rectangle(x - 24, TOP + 2, x + 24, TOP + 16, fill=col, outline="")
            c.create_text(x, TOP + 9, text=name.upper(), fill="white", font=("Segoe UI", 7, "bold"))
            for f in range(1, FLOORS + 1):
                fy = yfl(f)
                if f in info["stops"]:
                    c.create_rectangle(x - 21, fy - 48, x + 21, fy, fill="#f8fafc", outline=col, width=2)
                    c.create_line(x, fy - 48, x, fy, fill=col)
                    c.create_oval(x - 3, fy - 57, x + 3, fy - 51, fill=col, outline="")
                    c.create_text(x + 12, fy - 54, text=str(f), fill=col, font=("Segoe UI", 7, "bold"))
                else:
                    c.create_rectangle(x - 21, fy - 48, x + 21, fy, fill="#e9ecef", outline="#adb5bd")
                    c.create_line(x - 21, fy - 48, x + 21, fy, fill="#ced4da")
                    c.create_line(x + 21, fy - 48, x - 21, fy, fill="#ced4da")

        # route trails (behind the cabins)
        if self.phase == "plan":
            if self.csp is not None and self.people:
                p = self.cur_person()
                sx, sfl = start_x(p["start"]), p["start"]
                legs = self.csp.legs
                if legs:
                    x_first = LIFTS[legs[0][0]]["x"]
                    c.create_line(sx, yfl(sfl) - 3, x_first, yfl(sfl) - 3, fill="#343a40", width=3,
                                  dash=(6, 3))
                for i, (lift, f, t) in enumerate(legs):
                    x = LIFTS[lift]["x"]
                    if self.gold:
                        c.create_line(x, yfl(f) - 25, x, yfl(t) - 25, fill="#ffd60a", width=16)
                    c.create_line(x, yfl(f) - 25, x, yfl(t) - 25, fill=LIFTS[lift]["color"], width=7,
                                  arrow="last", arrowshape=(12, 14, 6))
                    if i > 0:
                        c.create_line(LIFTS[legs[i - 1][0]]["x"], yfl(f) - 3, x, yfl(f) - 3,
                                      fill="#555", width=3, dash=(5, 3))
                if legs and legs[-1][2] == self.csp.goal:
                    c.create_line(LIFTS[legs[-1][0]]["x"], yfl(self.csp.goal) - 3, stand_x(p["num"]),
                                  yfl(self.csp.goal) - 3, fill="#2d6a4f", width=4, dash=(8, 4),
                                  arrow="last")
                if self.flash:
                    lift, f, t = self.flash
                    x = LIFTS[lift]["x"]
                    c.create_line(x, yfl(f) - 25, x, yfl(t) - 25, fill="#e63946", width=7, dash=(6, 4),
                                  arrow="last", arrowshape=(12, 14, 6))
        else:
            for i, p in enumerate(self.people):
                legs = p["legs"]
                if not legs:
                    continue
                off = (i % 4 - 1.5) * 5
                sx, sfl = start_x(p["start"]), p["start"]
                c.create_line(sx, yfl(sfl) - 3, LIFTS[legs[0][0]]["x"] + off, yfl(sfl) - 3,
                              fill=p["color"], width=2, dash=(6, 3))
                for j, (lift, f, t) in enumerate(legs):
                    x = LIFTS[lift]["x"] + off
                    c.create_line(x, yfl(f) - 25, x, yfl(t) - 25, fill=p["color"], width=3,
                                  arrow="last", arrowshape=(8, 10, 4))
                    if j > 0:
                        c.create_line(LIFTS[legs[j - 1][0]]["x"] + off, yfl(f) - 3, x, yfl(f) - 3,
                                      fill=p["color"], width=2, dash=(5, 3))
                lx = LIFTS[legs[-1][0]]["x"] + off
                c.create_line(lx, yfl(p["floor"]) - 3, stand_x(p["num"]), yfl(p["floor"]) - 3,
                              fill=p["color"], width=2, dash=(8, 4), arrow="last")

        # destination doors (only the requested rooms are drawn)
        for (fl, num), arrived in rooms.items():
            dx, fy = door_x(num), yfl(fl)
            room = fl * 100 + num
            c.create_rectangle(dx - 9, fy - 46, dx + 9, fy, fill="#ffe8a3" if arrived else "#9c6644",
                               outline="#5c3d2e", width=2)
            c.create_oval(dx + 3, fy - 24, dx + 7, fy - 20, fill="#ffd60a", outline="#5c3d2e")
            c.create_rectangle(dx - 18, fy - 62, dx + 18, fy - 48, fill="white", outline="#d62828", width=2)
            c.create_text(dx, fy - 55, text=str(room), fill="#d62828", font=("Segoe UI", 8, "bold"))
            c.create_polygon(dx - 5, fy - 70, dx + 5, fy - 70, dx, fy - 64, fill="#d62828", outline="")
            if arrived:                                          # confetti
                for ox, oy, col in ((-30, -66, "#e63946"), (26, -68, "#3a86ff"), (34, -40, "#2a9d8f"),
                                    (-38, -44, "#fb8500"), (0, -78, "#8338ec"), (18, -76, "#ffd60a")):
                    c.create_oval(dx + ox - 3, fy + oy - 3, dx + ox + 3, fy + oy + 3, fill=col, outline="")

        # start flags on the upper floors (floor 1 uses the Main Gate)
        for f in sorted(start_floors):
            if f == MAIN_GATE_FLOOR:
                continue
            fy = yfl(f)
            c.create_line(BLD_X0 + 16, fy, BLD_X0 + 16, fy - 46, fill="#555", width=2)
            c.create_polygon(BLD_X0 + 16, fy - 46, BLD_X0 + 48, fy - 39, BLD_X0 + 16, fy - 32,
                             fill="#2a9d8f", outline="")
            c.create_text(BLD_X0 + 30, fy - 39, text="START", fill="white", font=("Segoe UI", 6, "bold"))

        # main gate
        gy = GROUND_Y - 6
        c.create_rectangle(34, gy - 66, 44, gy, fill="#6c757d", outline="")
        c.create_rectangle(100, gy - 66, 110, gy, fill="#6c757d", outline="")
        c.create_rectangle(30, gy - 80, 114, gy - 64, fill="#343a40", outline="")
        c.create_text(72, gy - 72, text="MAIN GATE", fill="white", font=("Segoe UI", 8, "bold"))
        if MAIN_GATE_FLOOR in start_floors:
            c.create_line(72, gy - 80, 72, gy - 100, fill="#555", width=2)
            c.create_polygon(72, gy - 104, 108, gy - 95, 72, gy - 86, fill="#2a9d8f", outline="")
            c.create_text(87, gy - 95, text="START", fill="white", font=("Segoe UI", 6, "bold"))

        # lift cabins (with cables, floor display, occupied lamp and the rider inside)
        cabs, occ = self.cab_state()
        for name, info in LIFTS.items():
            x, col = info["x"], info["color"]
            p = cabs[name]
            cy = yfl(p)
            c.create_line(x, TOP + 16, x, cy - 50, fill="#666", width=2)
            c.create_rectangle(x - 19, cy - 50, x + 19, cy, fill=col, outline="#222", width=2)
            c.create_rectangle(x - 15, cy - 39, x + 15, cy - 3, fill="#f8fafc", outline="")
            c.create_line(x, cy - 39, x, cy - 3, fill="#adb5bd")
            c.create_rectangle(x - 10, cy - 48, x + 10, cy - 41, fill="#111", outline="")
            c.create_text(x, cy - 44.5, text=str(round(p)), fill="#39ff14", font=("Segoe UI", 7, "bold"))
            c.create_oval(x + 12, cy - 47, x + 17, cy - 42, fill="#e63946" if occ[name] else "#39d353",
                          outline="#222")
            for a, pe in zip(actors, self.people):
                if a["lift"] == name:
                    self._person(c, x, cy - 4, 0.75, pe["color"])

        # people who are not inside a lift
        for i, (a, pe) in enumerate(zip(actors, self.people)):
            if a["lift"] is not None:
                continue
            x, fy = a["x"] + slot_off(i), yfl(a["fl"])
            self._person(c, x, fy, 1.0, pe["color"])
            c.create_text(x, fy - 46, text=pe["name"], fill=pe["color"], font=("Segoe UI", 8, "bold"))
            if a["wait"] is not None:
                c.create_text(x, fy - 58, text=f"wait {a['wait']:.1f}s",
                              fill="#c1121f" if a["busy"] else "#b26a00", font=("Segoe UI", 7, "bold"))

        if self.phase in ("sim", "done"):
            c.create_text(W - 12, H - 14, anchor="e", text=f"t = {self.sim_t:.1f} s",
                          font=("Consolas", 11, "bold"), fill="#264653")

    def refresh_stats(self):
        c = self.csp
        n = len(self.people)
        if c is None or n == 0:
            self.stat_lbl.config(text="No people added yet.\nUse the People tab.")
            return
        best = min((len(s) for s in c.solutions), default=None)
        who = self.cur_person()["name"]
        step = min(self.plan_idx + 1, n)
        text = (f"Person        : {who} ({step}/{n})\n"
                f"Legs assigned : {len(c.legs)}/{c.max_legs} (limit)\n"
                f"Assignments   : {c.assigns}\n"
                f"Backtracks    : {c.backtracks}\n"
                f"Solutions     : {len(c.solutions)}\n"
                f"Best (legs)   : {best if best is not None else '-'}")
        if self.phase in ("sim", "done"):
            text += (f"\nSim time      : {self.sim_t:.1f}/{self.sim_total:.1f}s\n"
                     f"Total waiting : {sum(p['wait'] for p in self.people):.1f}s")
        self.stat_lbl.config(text=text)

    # ---------- plan-phase animation engine (walk / ride segments) ----------
    def _ride_ms(self):
        return int(max(80, min(1200, self.speed.get() * 0.9)))

    def _walk_ms(self):
        return int(max(50, min(700, self.speed.get() * 0.5)))

    def _apply(self, seg, u):
        if seg["type"] == "walk":
            self.px = seg["x0"] + (seg["x1"] - seg["x0"]) * u
            self.pfl, self.plift = float(seg["fl"]), None
        else:
            lift = seg["lift"]
            self.cab[lift] = seg["f"] + (seg["t"] - seg["f"]) * u
            self.pfl = self.cab[lift]
            self.px = LIFTS[lift]["x"]
            self.plift = lift if u < 1 else None

    def _queue_walk(self, x0, x1, fl):
        if abs(x1 - x0) > 1:
            self.segs.append({"type": "walk", "x0": x0, "x1": x1, "fl": fl, "ms": self._walk_ms()})

    def _queue_ride(self, lift, f, t):
        self.segs.append({"type": "ride", "lift": lift, "f": f, "t": t, "ms": self._ride_ms()})

    def _start_anim(self):
        if not self.busy:
            self._next_seg()

    def _next_seg(self):
        if not self.segs:
            self.busy = False
            self.cur_seg = None
            self.draw()
            return
        self.busy = True
        self.cur_seg = self.segs.pop(0)
        n = max(1, self.cur_seg["ms"] // FRAME_MS)
        self._frame(self.cur_seg, 0, n)

    def _frame(self, seg, i, n):
        self._apply(seg, i / n)
        self.draw()
        if i < n:
            self.anim_job = self.after(FRAME_MS, lambda: self._frame(seg, i + 1, n))
        else:
            self.anim_job = None
            self._next_seg()

    def _cancel_anim(self):
        if self.anim_job is not None:
            self.after_cancel(self.anim_job)
            self.anim_job = None
        self.segs.clear()
        self.cur_seg = None
        self.busy = False

    def _finish_anim(self):
        """Jump to the end of every queued animation."""
        if self.anim_job is not None:
            self.after_cancel(self.anim_job)
            self.anim_job = None
        if self.cur_seg:
            self._apply(self.cur_seg, 1.0)
        for s in self.segs:
            self._apply(s, 0.0)
            self._apply(s, 1.0)
        self.segs.clear()
        self.cur_seg = None
        self.busy = False
        self.draw()

    def _reset_actors(self, p=None):
        self.cab = dict(DEFAULT_CAB)
        if p is None:
            self.px, self.pfl, self.plift = GATE_X, 1.0, None
        else:
            self.px, self.pfl, self.plift = start_x(p["start"]), float(p["start"]), None

    # ---------- simulation clock ----------
    def _sim_rate(self):
        """Simulated seconds per real second (slider: fast ⇢ slow)."""
        return max(0.5, min(20.0, 750.0 / max(1, self.speed.get())))

    def _cancel_sim(self):
        if self.sim_job is not None:
            self.after_cancel(self.sim_job)
            self.sim_job = None
        self.sched = None
        self.sim_t = self.sim_total = 0.0
        self.events, self.ev_i = [], 0

    def _emit_events(self):
        while self.ev_i < len(self.events) and self.events[self.ev_i][0] <= self.sim_t + 1e-9:
            t, msg = self.events[self.ev_i]
            self.write(f"[t={t:5.1f}s] {msg}")
            self.ev_i += 1

    def _start_sim_loop(self):
        if self.sim_job is None:
            self.sim_job = self.after(FRAME_MS, self._sim_loop)

    def _sim_loop(self):
        self.sim_job = None
        if not self.running or self.phase != "sim":
            return
        self.sim_t = min(self.sim_total, self.sim_t + self._sim_rate() * FRAME_MS / 1000.0)
        self._emit_events()
        self.draw()
        self.refresh_stats()
        if self.sim_t >= self.sim_total - 1e-9:
            self._sim_done()
            return
        self.sim_job = self.after(FRAME_MS, self._sim_loop)

    def _sim_step(self):
        self.sim_t = min(self.sim_total, self.sim_t + 1.0)
        self._emit_events()
        self.draw()
        self.refresh_stats()
        if self.sim_t >= self.sim_total - 1e-9:
            self._sim_done()

    def _sim_done(self):
        self.phase = "done"
        self.running = False
        self.sim_t = self.sim_total
        self._emit_events()
        failed = [p["name"] for p in self.people if p["legs"] is None]
        tw = sum(p["wait"] for p in self.people)
        msg = f"✔ Everyone arrived by t={self.sim_total:.1f}s • total waiting {tw:.1f}s"
        if failed:
            msg = f"⚠ No route for {', '.join(failed)} • others arrived by t={self.sim_total:.1f}s " \
                  f"(waiting {tw:.1f}s)"
        self.status.config(text=msg, fg="#2d6a4f" if not failed else "#b26a00")
        self.write("SIMULATION COMPLETE - " + msg[2:])
        self.draw()
        self.refresh_stats()

    def _begin_sim(self):
        """All routes are planned -> start the multi-person simulation."""
        _, per_floor = self._opts()
        if all(p["legs"] is None for p in self.people):
            self.phase = "done"
            self.running = False
            self.status.config(text="✘ No valid route for anybody within the allowed legs.", fg="#c1121f")
            self.write("SEARCH COMPLETE - no route found")
            self._fill_results()
            self.draw()
            self.refresh_stats()
            return False
        self.sched, self.sim_total, self.events = build_schedule(self.people, per_floor,
                                                                 self.repo_var.get())
        self.ev_i, self.sim_t, self.phase = 0, 0.0, "sim"
        self._fill_results()
        self.write("=== SIMULATION: everybody moves at once - a lift serves one rider at a time ===")
        self.status.config(text="Simulating movement (waiting shown in red/orange)…", fg="black")
        self._emit_events()
        self.draw()
        self.refresh_stats()
        if self.sim_total <= 0:
            self._sim_done()
        elif self.running:
            self._start_sim_loop()
        return False

    # ---------- planning ----------
    def _start_planning(self, i):
        p = self.people[i]
        k, _ = self._opts()
        self.csp = LiftCSP(p["start"], p["floor"], k, self.down_var.get(), self.mode_var.get(),
                           self.order_var.get())
        self.gen = self.csp.run()
        self.flash, self.gold, self.last_kind = None, False, None
        self._cancel_anim()
        self._reset_actors(p)
        self.set_solutions_text()
        self.write(f"=== Planning {p['name']}: F{p['start']} → Room {p['room']} ===")

    def _person_planned(self):
        """Search for the current person is over. Returns True if planning continues."""
        p = self.people[self.plan_idx]
        sols = self.csp.solutions
        if sols:
            best = min(sols, key=lambda s: (len(s), sum(abs(t - f) for _, f, t in s)))
            p["legs"] = best
            self.set_solutions_text(best)
            self.write(f"{p['name']}: {len(sols)} solution(s), best = {self.describe(best, p)}")
        else:
            p["legs"] = None
            self.write(f"{p['name']}: NO valid route within {self.csp.limit} leg(s)")
        self.plan_idx += 1
        if self.plan_idx < len(self.people):
            self._start_planning(self.plan_idx)
            return True
        self.gen = None
        self._begin_sim()
        return False

    def _plan_all_instantly(self):
        self._cancel_anim()
        while self.phase == "plan" and self.gen is not None:
            for _ in self.gen:
                pass
            if not self._person_planned():
                break

    # ---------- control ----------
    def reset(self, msg=None, err=False):
        self.running = False
        self._cancel_anim()
        self._cancel_sim()
        self.phase = "plan"
        self.plan_idx = 0
        self.flash, self.gold, self.last_kind = None, False, None
        for p in self.people:
            p.update(legs=None, segs=[], wait=0.0, finish=None)
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")
        self._set_text(self.res_text, "")
        if not self.people:
            self.csp = self.gen = None
            self._reset_actors()
            self.set_solutions_text()
            self.status.config(text="Add at least one person (People tab).", fg="#c1121f")
        else:
            self._start_planning(0)
            self.status.config(text=msg or "Ready. Press Run or Step.", fg="#c1121f" if err else "black")
        self.draw()
        self.refresh_stats()

    def run(self):
        if self.phase == "plan" and self.gen is not None:
            self.running = True
            if not self.anim_var.get():
                self._plan_all_instantly()
            else:
                self._tick()
        elif self.phase == "sim":
            self.running = True
            self._start_sim_loop()

    def pause(self):
        self.running = False

    def _tick(self):
        if not self.running or self.phase != "plan":
            return
        if self.busy:
            self.after(30, self._tick)
            return
        if self.step():
            gap = max(20, int(self.speed.get() * 0.3)) * (4 if self.last_kind == "solution" else 1)
            self.after(gap, self._tick)

    def step(self):
        """Plan phase: advance one solver event (animated). Sim phase: advance the clock by 1 s.
        Returns False when there is nothing more to do in the planning phase."""
        if self.phase == "sim":
            self._sim_step()
            return False
        if self.phase != "plan" or self.gen is None:
            return False
        if self.busy:
            self._finish_anim()
        try:
            kind, a, b = next(self.gen)
        except StopIteration:
            return self._person_planned()

        p = self.cur_person()
        self.last_kind = kind
        self.flash = None
        self.gold = False
        if kind == "deepen":
            self.write(f"--- depth limit = {a} leg(s) ---")
        elif kind == "place":
            lift, f, t = b
            self.write(f"ASSIGN    Leg{a} = ({lift}, F{f} → F{t})")
            self._queue_walk(self.px, LIFTS[lift]["x"], f)
            self._queue_ride(lift, f, t)
        elif kind in ("backtrack", "undo"):
            lift, f, t = b
            if kind == "backtrack":
                self.flash = b
                self.write(f"BACKTRACK Leg{a} x ({lift}, F{f} → F{t})")
            else:
                self.write(f"UNDO      Leg{a} ({lift}, F{f} → F{t})  [after a solution]")
            lx = LIFTS[lift]["x"]
            self._queue_walk(self.px, lx, t)
            self._queue_ride(lift, t, f)
            prev_x = LIFTS[self.csp.legs[-1][0]]["x"] if self.csp.legs else start_x(p["start"])
            self._queue_walk(lx, prev_x, f)
        elif kind == "solution":
            self.gold = True
            self.write(f"SOLUTION #{len(self.csp.solutions)} ✔ {self.describe(self.csp.legs, p)}")
            self.set_solutions_text()
            self._queue_walk(self.px, stand_x(p["num"]), self.csp.goal)
        self.draw()
        self.refresh_stats()
        self._start_anim()
        return True


if __name__ == "__main__":
    App().mainloop()