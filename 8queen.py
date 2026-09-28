"""
CSP 8-Queens Simulator (Python + Tkinter)
-----------------------------------------
Variables  : 8 columns  (one queen per column)
Domain     : rows 0..7  (row where the queen of that column is placed)
Constraints: no two queens share a row or a diagonal
             (columns are different by construction)

Solver     : Backtracking search with
             - MRV heuristic (Minimum Remaining Values)   [optional]
             - Forward checking (dead-end detection)      [optional]
             - Random value ordering                      [optional]
The solver is a generator, so the GUI can animate every place / backtrack step.
After a solution is found, press Run/Step again to continue to the next one
(there are 92 solutions in total).
"""

import random
import tkinter as tk
from tkinter import ttk

N = 8
LIGHT, DARK = "#f0d9b5", "#b58863"
ATT_LIGHT, ATT_DARK = "#f7c6c6", "#d99a9a"     # attacked squares
QUEEN_BG = "#95d5b2"
FLASH_BG = "#ff6b6b"


class QueensCSP:
    def __init__(self, use_mrv=True, use_fc=True, randomize=False):
        self.use_mrv = use_mrv
        self.use_fc = use_fc
        self.randomize = randomize
        self.assignment = {}          # column -> row
        self.assigns = 0
        self.backtracks = 0
        self.solutions = 0

    # ---------- constraint check ----------
    def consistent(self, col, row):
        for c, r in self.assignment.items():
            if r == row or abs(r - row) == abs(c - col):
                return False
        return True

    def legal_values(self, col):
        return [r for r in range(N) if self.consistent(col, r)]

    def free_cols(self):
        return [c for c in range(N) if c not in self.assignment]

    def select_var(self):
        free = self.free_cols()
        if not free:
            return None
        if self.use_mrv:
            return min(free, key=lambda c: len(self.legal_values(c)))
        return free[0]

    def forward_check_ok(self):
        """Every unassigned column must still have at least one legal row."""
        return all(self.legal_values(c) for c in self.free_cols())

    # ---------- backtracking search (generator for animation) ----------
    def search(self):
        col = self.select_var()
        if col is None:
            self.solutions += 1
            yield ("solution", None, None)
            return
        values = self.legal_values(col)
        if self.randomize:
            random.shuffle(values)
        for row in values:
            self.assignment[col] = row
            self.assigns += 1
            yield ("place", col, row)
            if not self.use_fc or self.forward_check_ok():
                yield from self.search()
            del self.assignment[col]
            self.backtracks += 1
            yield ("backtrack", col, row)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSP 8-Queens Simulator")
        self.configure(bg="#f5f5f7")
        self.csp = None
        self.gen = None
        self.running = False
        self.flash = None
        self.cells = {}
        self._build_ui()
        self.reset()

    # ---------- UI ----------
    def _build_ui(self):
        tk.Label(self, text="CSP 8-Queens Simulator", font=("Segoe UI", 16, "bold"),
                 bg="#f5f5f7").grid(row=0, column=0, columnspan=2, pady=(10, 0))
        tk.Label(self, text="1 queen per column • no two queens in the same row or diagonal",
                 font=("Segoe UI", 9), bg="#f5f5f7", fg="#555").grid(row=1, column=0, columnspan=2)

        board = tk.Frame(self, bg="#333", bd=2)
        board.grid(row=2, column=0, padx=12, pady=10, sticky="n")
        for r in range(N):
            for c in range(N):
                lbl = tk.Label(board, text="", width=4, height=2, font=("Segoe UI Symbol", 22, "bold"))
                lbl.grid(row=r, column=c)
                self.cells[(c, r)] = lbl

        side = tk.Frame(self, bg="#f5f5f7")
        side.grid(row=2, column=1, padx=(0, 12), pady=10, sticky="n")

        opt = tk.LabelFrame(side, text="Options", bg="#f5f5f7", padx=8, pady=4)
        opt.pack(fill="x")
        self.mrv_var = tk.BooleanVar(value=True)
        self.fc_var = tk.BooleanVar(value=True)
        self.rand_var = tk.BooleanVar(value=False)
        self.att_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt, text="MRV heuristic", variable=self.mrv_var, bg="#f5f5f7").pack(anchor="w")
        tk.Checkbutton(opt, text="Forward checking", variable=self.fc_var, bg="#f5f5f7").pack(anchor="w")
        tk.Checkbutton(opt, text="Randomize value order", variable=self.rand_var, bg="#f5f5f7").pack(anchor="w")
        tk.Checkbutton(opt, text="Show attacked squares", variable=self.att_var, bg="#f5f5f7",
                       command=self.draw).pack(anchor="w")
        tk.Label(opt, text="Speed (fast ⇢ slow)", bg="#f5f5f7").pack(anchor="w")
        self.speed = tk.Scale(opt, from_=1, to=800, orient="horizontal", length=180, bg="#f5f5f7")
        self.speed.set(150)
        self.speed.pack()

        btns = tk.Frame(side, bg="#f5f5f7")
        btns.pack(pady=6)
        ttk.Button(btns, text="▶ Run", command=self.run).grid(row=0, column=0, padx=2)
        ttk.Button(btns, text="⏸ Pause", command=self.pause).grid(row=0, column=1, padx=2)
        ttk.Button(btns, text="⏭ Step", command=self.step).grid(row=0, column=2, padx=2)
        ttk.Button(btns, text="⟲ Reset", command=self.reset).grid(row=0, column=3, padx=2)

        st = tk.LabelFrame(side, text="Statistics", bg="#f5f5f7", padx=8, pady=4)
        st.pack(fill="x")
        self.stat_lbl = tk.Label(st, text="", justify="left", bg="#f5f5f7", font=("Consolas", 10))
        self.stat_lbl.pack(anchor="w")

        self.log = tk.Text(self, height=8, width=90, font=("Consolas", 9), state="disabled")
        self.log.grid(row=3, column=0, columnspan=2, padx=12, pady=(0, 6))
        self.status = tk.Label(self, text="", bg="#f5f5f7", font=("Segoe UI", 10, "bold"))
        self.status.grid(row=4, column=0, columnspan=2, pady=(0, 8))

    # ---------- drawing ----------
    def attacked(self):
        att = set()
        for c, r in self.csp.assignment.items():
            for cc in range(N):
                for rr in range(N):
                    if cc == c or rr == r or abs(cc - c) == abs(rr - r):
                        att.add((cc, rr))
        return att

    def draw(self):
        if self.csp is None:
            return
        att = self.attacked() if self.att_var.get() else set()
        for (c, r), lbl in self.cells.items():
            dark = (c + r) % 2 == 1
            bg = (ATT_DARK if dark else ATT_LIGHT) if (c, r) in att else (DARK if dark else LIGHT)
            text = ""
            if self.csp.assignment.get(c) == r:
                bg, text = QUEEN_BG, "♛"
            if self.flash == (c, r):
                bg = FLASH_BG
            lbl.config(bg=bg, text=text, fg="#1b4332")

    def refresh_stats(self):
        c = self.csp
        self.stat_lbl.config(text=f"Queens placed : {len(c.assignment)}/{N}\n"
                                  f"Placements    : {c.assigns}\n"
                                  f"Backtracks    : {c.backtracks}\n"
                                  f"Solutions     : {c.solutions}")

    def write(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    # ---------- control ----------
    def reset(self):
        self.running = False
        self.flash = None
        self.csp = QueensCSP(self.mrv_var.get(), self.fc_var.get(), self.rand_var.get())
        self.gen = self.csp.search()
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")
        self.status.config(text="Ready. Press Run or Step.", fg="black")
        self.draw()
        self.refresh_stats()

    def run(self):
        if self.gen is None:
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
        """Advance one solver event. Returns False if the animation should stop."""
        if self.gen is None:
            return False
        try:
            kind, col, row = next(self.gen)
        except StopIteration:
            self.gen = None
            self.running = False
            self.flash = None
            self.status.config(text=f"Search complete — {self.csp.solutions} solution(s) found.",
                               fg="#1d3557")
            self.write(f"SEARCH COMPLETE: {self.csp.solutions} solutions")
            self.draw()
            self.refresh_stats()
            return False

        self.flash = None
        if kind == "place":
            self.write(f"PLACE     col {col + 1} -> row {row + 1}")
        elif kind == "backtrack":
            self.flash = (col, row)
            self.write(f"BACKTRACK col {col + 1} x row {row + 1}")
        elif kind == "solution":
            self.write(f"SOLUTION #{self.csp.solutions} FOUND ✔")
            self.status.config(text=f"✔ Solution #{self.csp.solutions} found! "
                                    f"Press Run/Step for the next one.", fg="#2d6a4f")
            self.running = False
            self.draw()
            self.refresh_stats()
            return False
        self.draw()
        self.refresh_stats()
        return True


if __name__ == "__main__":
    App().mainloop()