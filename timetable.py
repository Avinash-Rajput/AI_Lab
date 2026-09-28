"""
CSP Timetable Simulator (Python + Tkinter)
------------------------------------------
Variables : 20 slots  (5 days x 4 periods)
Domain    : {DBMS, CN, SQA, OS, DA}
Constraints:
  1. Every subject must get >= 4 periods per week
  2. Computer Network (CN) cannot be in Period 1 (first hour)
  (optional) 3. A subject may appear at most 2 times in one day

Solver    : Backtracking search with
            - MRV heuristic (Minimum Remaining Values)   [optional]
            - Forward-checking style feasibility pruning
The solver is a generator, so the GUI can animate every assign / backtrack step.
"""

import random
import tkinter as tk
from tkinter import ttk

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
PERIODS = 4
SUBJECTS = {
    "DBMS": ("Database Mgmt Systems", "#8ecae6"),
    "CN":   ("Computer Networks",     "#ffb703"),
    "SQA":  ("Software Quality Assr.", "#b7e4c7"),
    "OS":   ("Operating Systems",     "#f4a6c1"),
    "DA":   ("Data Analytics",        "#cdb4db"),
}
MIN_HOURS = 4
NO_FIRST_PERIOD = {"CN"}          # subjects that cannot be in period 1
MAX_PER_DAY = 2                   # used only if the optional constraint is ON


class TimetableCSP:
    def __init__(self, use_mrv=True, randomize=True, limit_per_day=False):
        self.vars = [(d, p) for d in range(len(DAYS)) for p in range(PERIODS)]
        self.subjects = list(SUBJECTS)
        self.use_mrv = use_mrv
        self.randomize = randomize
        self.limit_per_day = limit_per_day
        self.assignment = {}
        self.count = {s: 0 for s in self.subjects}
        self.day_count = [{s: 0 for s in self.subjects} for _ in DAYS]
        # With 20 slots and 5 subjects needing >=4 each, the max any subject can get:
        self.max_count = len(self.vars) - MIN_HOURS * (len(self.subjects) - 1)
        self.assigns = 0
        self.backtracks = 0
        self.solved = False

    # ---------- constraint checks ----------
    def consistent(self, var, val):
        d, p = var
        if val in NO_FIRST_PERIOD and p == 0:
            return False
        if self.count[val] >= self.max_count:
            return False
        if self.limit_per_day and self.day_count[d][val] >= MAX_PER_DAY:
            return False
        return True

    def legal_values(self, var):
        return [v for v in self.subjects if self.consistent(var, v)]

    def feasible(self):
        """Prune: can the remaining free slots still satisfy every minimum?"""
        free = len(self.vars) - len(self.assignment)
        deficit = sum(max(0, MIN_HOURS - self.count[s]) for s in self.subjects)
        if deficit > free:
            return False
        # forward check: every unassigned slot must still have a legal value
        for v in self.vars:
            if v not in self.assignment and not self.legal_values(v):
                return False
        return True

    # ---------- assignment helpers ----------
    def assign(self, var, val):
        self.assignment[var] = val
        self.count[val] += 1
        self.day_count[var[0]][val] += 1
        self.assigns += 1

    def unassign(self, var):
        val = self.assignment.pop(var)
        self.count[val] -= 1
        self.day_count[var[0]][val] -= 1
        self.backtracks += 1

    def select_var(self):
        free = [v for v in self.vars if v not in self.assignment]
        if not free:
            return None
        if self.use_mrv:
            return min(free, key=lambda v: len(self.legal_values(v)))
        return free[0]

    # ---------- backtracking search (generator for animation) ----------
    def search(self):
        var = self.select_var()
        if var is None:
            if all(self.count[s] >= MIN_HOURS for s in self.subjects):
                self.solved = True
                yield ("solution", None, None)
            return
        values = self.legal_values(var)
        if self.randomize:
            random.shuffle(values)
        for val in values:
            self.assign(var, val)
            yield ("assign", var, val)
            if self.feasible():
                yield from self.search()
                if self.solved:
                    return
            self.unassign(var)
            yield ("backtrack", var, val)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSP Timetable Simulator")
        self.configure(bg="#f5f5f7")
        self.csp = None
        self.gen = None
        self.running = False
        self.cells = {}
        self._build_ui()
        self.reset()

    # ---------- UI ----------
    def _build_ui(self):
        tk.Label(self, text="CSP Timetable Simulator", font=("Segoe UI", 16, "bold"),
                 bg="#f5f5f7").grid(row=0, column=0, columnspan=2, pady=(10, 0))
        tk.Label(self, text="5 days • 4 periods/day • each subject ≥ 4 hrs/week • CN not in Period 1",
                 font=("Segoe UI", 9), bg="#f5f5f7", fg="#555").grid(row=1, column=0, columnspan=2)

        # timetable grid
        grid = tk.Frame(self, bg="#f5f5f7")
        grid.grid(row=2, column=0, padx=12, pady=10, sticky="n")
        tk.Label(grid, text="", bg="#f5f5f7").grid(row=0, column=0)
        for d, name in enumerate(DAYS):
            tk.Label(grid, text=name, font=("Segoe UI", 10, "bold"), bg="#f5f5f7",
                     width=11).grid(row=0, column=d + 1, padx=2, pady=2)
        for p in range(PERIODS):
            tk.Label(grid, text=f"P{p + 1}", font=("Segoe UI", 10, "bold"),
                     bg="#f5f5f7").grid(row=p + 1, column=0, padx=4)
            for d in range(len(DAYS)):
                c = tk.Label(grid, text="", width=11, height=3, relief="ridge", bd=2,
                             font=("Segoe UI", 11, "bold"), bg="white")
                c.grid(row=p + 1, column=d + 1, padx=2, pady=2)
                self.cells[(d, p)] = c

        # right panel
        side = tk.Frame(self, bg="#f5f5f7")
        side.grid(row=2, column=1, padx=(0, 12), pady=10, sticky="n")

        opt = tk.LabelFrame(side, text="Options", bg="#f5f5f7", padx=8, pady=4)
        opt.pack(fill="x")
        self.mrv_var = tk.BooleanVar(value=True)
        self.rand_var = tk.BooleanVar(value=True)
        self.day_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opt, text="MRV heuristic", variable=self.mrv_var, bg="#f5f5f7").pack(anchor="w")
        tk.Checkbutton(opt, text="Randomize value order", variable=self.rand_var, bg="#f5f5f7").pack(anchor="w")
        tk.Checkbutton(opt, text=f"Max {MAX_PER_DAY} of a subject per day",
                       variable=self.day_var, bg="#f5f5f7").pack(anchor="w")
        tk.Label(opt, text="Speed (fast ⇢ slow)", bg="#f5f5f7").pack(anchor="w")
        self.speed = tk.Scale(opt, from_=1, to=500, orient="horizontal", length=180, bg="#f5f5f7")
        self.speed.set(60)
        self.speed.pack()

        btns = tk.Frame(side, bg="#f5f5f7")
        btns.pack(pady=6)
        ttk.Button(btns, text="▶ Run", command=self.run).grid(row=0, column=0, padx=2)
        ttk.Button(btns, text="⏸ Pause", command=self.pause).grid(row=0, column=1, padx=2)
        ttk.Button(btns, text="⏭ Step", command=self.step).grid(row=0, column=2, padx=2)
        ttk.Button(btns, text="⟲ Reset", command=self.reset).grid(row=0, column=3, padx=2)

        # stats
        st = tk.LabelFrame(side, text="Statistics", bg="#f5f5f7", padx=8, pady=4)
        st.pack(fill="x")
        self.stat_lbl = tk.Label(st, text="", justify="left", bg="#f5f5f7", font=("Consolas", 10))
        self.stat_lbl.pack(anchor="w")

        # legend + hour counts
        lg = tk.LabelFrame(side, text="Subjects (hours/week)", bg="#f5f5f7", padx=8, pady=4)
        lg.pack(fill="x", pady=6)
        self.count_lbls = {}
        for i, (code, (full, col)) in enumerate(SUBJECTS.items()):
            tk.Label(lg, text=code, bg=col, width=6, relief="ridge").grid(row=i, column=0, pady=1)
            tk.Label(lg, text=full, bg="#f5f5f7", anchor="w", width=22).grid(row=i, column=1)
            l = tk.Label(lg, text="0", bg="#f5f5f7", width=3, font=("Segoe UI", 10, "bold"))
            l.grid(row=i, column=2)
            self.count_lbls[code] = l

        # log
        self.log = tk.Text(self, height=7, width=100, font=("Consolas", 9), state="disabled")
        self.log.grid(row=3, column=0, columnspan=2, padx=12, pady=(0, 10))
        self.status = tk.Label(self, text="", bg="#f5f5f7", font=("Segoe UI", 10, "bold"))
        self.status.grid(row=4, column=0, columnspan=2, pady=(0, 8))

    # ---------- control ----------
    def reset(self):
        self.running = False
        self.csp = TimetableCSP(self.mrv_var.get(), self.rand_var.get(), self.day_var.get())
        self.gen = self.csp.search()
        for c in self.cells.values():
            c.config(text="", bg="white", relief="ridge")
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")
        self.status.config(text="Ready. Press Run or Step.", fg="black")
        self.refresh_stats()

    def run(self):
        if self.csp.solved or self.gen is None:
            return
        self.running = True
        self._tick()

    def pause(self):
        self.running = False

    def _tick(self):
        if not self.running:
            return
        if self.step():
            self.after(self.speed.get(), self._tick)

    def step(self):
        """Advance one solver event. Returns False when the search has finished."""
        if self.gen is None:
            return False
        try:
            kind, var, val = next(self.gen)
        except StopIteration:
            self.gen = None
            self.running = False
            if not self.csp.solved:
                self.status.config(text="✘ No valid timetable exists.", fg="#c1121f")
            return False

        if kind == "assign":
            cell = self.cells[var]
            cell.config(text=val, bg=SUBJECTS[val][1], relief="solid")
            self.write(f"ASSIGN    {DAYS[var[0]][:3]} P{var[1] + 1} = {val}")
        elif kind == "backtrack":
            cell = self.cells[var]
            cell.config(text="", bg="#ffd6d6", relief="ridge")
            self.write(f"BACKTRACK {DAYS[var[0]][:3]} P{var[1] + 1} ✗ {val}")
        elif kind == "solution":
            self.write("SOLUTION FOUND ✔")
            self.status.config(text="✔ Valid timetable found!", fg="#2d6a4f")
            self.running = False
            self.gen = None
            self.refresh_stats()
            return False
        self.refresh_stats()
        return True

    def refresh_stats(self):
        c = self.csp
        self.stat_lbl.config(text=f"Assignments : {c.assigns}\n"
                                  f"Backtracks  : {c.backtracks}\n"
                                  f"Filled slots: {len(c.assignment)}/{len(c.vars)}")
        for s, l in self.count_lbls.items():
            l.config(text=str(c.count[s]), fg="#2d6a4f" if c.count[s] >= MIN_HOURS else "#c1121f")

    def write(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")


if __name__ == "__main__":
    App().mainloop()