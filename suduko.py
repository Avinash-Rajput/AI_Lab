"""
CSP Sudoku Simulator (Python + Tkinter)
---------------------------------------
Variables  : the 81 cells (given cells are already fixed)
Domain     : digits 1..9
Constraints: All-Different on every row, every column and every 3x3 box

Solver     : Backtracking search with
             - MRV heuristic (Minimum Remaining Values)   [optional]
             - Forward checking (dead-end detection)      [optional]
             - Random value ordering                      [optional]
The solver is a generator, so the GUI can animate every place / backtrack step.
Edit the PUZZLES dictionary below to add your own puzzles
(81 characters, use 0 or . for empty cells).
"""

import random
import tkinter as tk
from tkinter import ttk

PUZZLES = {
    "Easy": "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
    "Medium": "003020600900305001001806400008102900700000008006708200002609500800203009005010300",
    "Hard (17 clues)": "..............3.85..1.2.......5.7.....4...1...9.......5......73..2.1........4...9",
}

# ---------- precompute peers (cells sharing a row, column or box) ----------
PEERS = []
for i in range(81):
    r, c = divmod(i, 9)
    br, bc = 3 * (r // 3), 3 * (c // 3)
    s = set()
    for k in range(9):
        s.add(r * 9 + k)
        s.add(k * 9 + c)
    for rr in range(br, br + 3):
        for cc in range(bc, bc + 3):
            s.add(rr * 9 + cc)
    s.discard(i)
    PEERS.append(sorted(s))

GIVEN_BG, EMPTY_BG = "#e5e5ea", "white"
BOX_ALT_BG = "#f4f6fb"
PLACE_FG, GIVEN_FG = "#1d4ed8", "#111111"
CUR_BG, FLASH_BG, DONE_BG = "#ffe066", "#ff8787", "#b7e4c7"


class SudokuCSP:
    def __init__(self, puzzle, use_mrv=True, use_fc=True, randomize=False):
        self.grid = [int(ch) if ch in "123456789" else 0 for ch in puzzle]
        self.given = {i for i, v in enumerate(self.grid) if v}
        self.use_mrv = use_mrv
        self.use_fc = use_fc
        self.randomize = randomize
        self.assigns = 0
        self.backtracks = 0
        self.solved = False

    # ---------- constraint handling ----------
    def legal_values(self, i):
        used = {self.grid[p] for p in PEERS[i]}
        return [v for v in range(1, 10) if v not in used]

    def select_var(self):
        empties = [i for i in range(81) if self.grid[i] == 0]
        if not empties:
            return None
        if not self.use_mrv:
            return empties[0]
        best, best_n = None, 10
        for i in empties:
            n = len(self.legal_values(i))
            if n < best_n:
                best, best_n = i, n
                if n <= 1:
                    break
        return best

    def forward_check_ok(self, i):
        """Every empty peer of the changed cell must still have a legal digit."""
        return all(self.legal_values(p) for p in PEERS[i] if self.grid[p] == 0)

    def empty_count(self):
        return sum(1 for v in self.grid if v == 0)

    # ---------- backtracking search (generator for animation) ----------
    def search(self):
        i = self.select_var()
        if i is None:
            self.solved = True
            yield ("solution", None, None)
            return
        values = self.legal_values(i)
        if self.randomize:
            random.shuffle(values)
        for v in values:
            self.grid[i] = v
            self.assigns += 1
            yield ("place", i, v)
            if not self.use_fc or self.forward_check_ok(i):
                yield from self.search()
                if self.solved:
                    return
            self.grid[i] = 0
            self.backtracks += 1
            yield ("backtrack", i, v)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSP Sudoku Simulator")
        self.configure(bg="#f5f5f7")
        self.csp = None
        self.gen = None
        self.running = False
        self.fast = False
        self.cur = None       # cell currently highlighted
        self.flash = None     # cell that just backtracked
        self.cells = []
        self._build_ui()
        self.reset()

    # ---------- UI ----------
    def _build_ui(self):
        tk.Label(self, text="CSP Sudoku Simulator", font=("Segoe UI", 16, "bold"),
                 bg="#f5f5f7").grid(row=0, column=0, columnspan=2, pady=(10, 0))
        tk.Label(self, text="Digits 1-9 • All-Different in every row, column and 3×3 box",
                 font=("Segoe UI", 9), bg="#f5f5f7", fg="#555").grid(row=1, column=0, columnspan=2)

        board = tk.Frame(self, bg="#222", bd=2)
        board.grid(row=2, column=0, padx=12, pady=10, sticky="n")
        for i in range(81):
            r, c = divmod(i, 9)
            padx = (3 if c % 3 == 0 and c else 1, 1)
            pady = (3 if r % 3 == 0 and r else 1, 1)
            lbl = tk.Label(board, text="", width=3, height=1, font=("Segoe UI", 18, "bold"))
            lbl.grid(row=r, column=c, padx=padx, pady=pady)
            self.cells.append(lbl)

        side = tk.Frame(self, bg="#f5f5f7")
        side.grid(row=2, column=1, padx=(0, 12), pady=10, sticky="n")

        pz = tk.LabelFrame(side, text="Puzzle", bg="#f5f5f7", padx=8, pady=4)
        pz.pack(fill="x")
        self.puzzle_var = tk.StringVar(value="Easy")
        cb = ttk.Combobox(pz, textvariable=self.puzzle_var, values=list(PUZZLES),
                          state="readonly", width=20)
        cb.pack()
        cb.bind("<<ComboboxSelected>>", lambda e: self.reset())

        opt = tk.LabelFrame(side, text="Options", bg="#f5f5f7", padx=8, pady=4)
        opt.pack(fill="x", pady=6)
        self.mrv_var = tk.BooleanVar(value=True)
        self.fc_var = tk.BooleanVar(value=True)
        self.rand_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opt, text="MRV heuristic", variable=self.mrv_var, bg="#f5f5f7").pack(anchor="w")
        tk.Checkbutton(opt, text="Forward checking", variable=self.fc_var, bg="#f5f5f7").pack(anchor="w")
        tk.Checkbutton(opt, text="Randomize value order", variable=self.rand_var, bg="#f5f5f7").pack(anchor="w")
        tk.Label(opt, text="Speed (fast ⇢ slow)", bg="#f5f5f7").pack(anchor="w")
        self.speed = tk.Scale(opt, from_=1, to=800, orient="horizontal", length=180, bg="#f5f5f7")
        self.speed.set(40)
        self.speed.pack()

        btns = tk.Frame(side, bg="#f5f5f7")
        btns.pack(pady=4)
        ttk.Button(btns, text="▶ Run", command=self.run).grid(row=0, column=0, padx=2)
        ttk.Button(btns, text="⏸ Pause", command=self.pause).grid(row=0, column=1, padx=2)
        ttk.Button(btns, text="⏭ Step", command=self.step).grid(row=0, column=2, padx=2)
        ttk.Button(btns, text="⟲ Reset", command=self.reset).grid(row=0, column=3, padx=2)
        ttk.Button(side, text="⚡ Solve Fast (no animation)", command=self.run_fast).pack(fill="x", pady=2)

        st = tk.LabelFrame(side, text="Statistics", bg="#f5f5f7", padx=8, pady=4)
        st.pack(fill="x", pady=6)
        self.stat_lbl = tk.Label(st, text="", justify="left", bg="#f5f5f7", font=("Consolas", 10))
        self.stat_lbl.pack(anchor="w")

        lg = tk.Label(side, text="Legend:  gray = given   blue = placed by solver\n"
                                 "yellow = current   red = backtracked",
                      bg="#f5f5f7", font=("Segoe UI", 8), justify="left", fg="#555")
        lg.pack(anchor="w")

        self.log = tk.Text(self, height=8, width=90, font=("Consolas", 9), state="disabled")
        self.log.grid(row=3, column=0, columnspan=2, padx=12, pady=(0, 6))
        self.status = tk.Label(self, text="", bg="#f5f5f7", font=("Segoe UI", 10, "bold"))
        self.status.grid(row=4, column=0, columnspan=2, pady=(0, 8))

    # ---------- drawing ----------
    def draw(self):
        g = self.csp.grid
        for i, lbl in enumerate(self.cells):
            r, c = divmod(i, 9)
            box_alt = ((r // 3) + (c // 3)) % 2 == 1
            if i in self.csp.given:
                bg, fg = GIVEN_BG, GIVEN_FG
            else:
                bg, fg = (BOX_ALT_BG if box_alt else EMPTY_BG), PLACE_FG
            if self.csp.solved and i not in self.csp.given:
                bg = DONE_BG
            if i == self.cur:
                bg = CUR_BG
            if i == self.flash:
                bg = FLASH_BG
            lbl.config(text=str(g[i]) if g[i] else "", bg=bg, fg=fg)

    def refresh_stats(self):
        c = self.csp
        self.stat_lbl.config(text=f"Empty cells : {c.empty_count()}\n"
                                  f"Placements  : {c.assigns}\n"
                                  f"Backtracks  : {c.backtracks}")

    def write(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 300:
            self.log.delete("1.0", "100.0")
        self.log.see("end")
        self.log.config(state="disabled")

    # ---------- control ----------
    def reset(self):
        self.running = False
        self.fast = False
        self.cur = self.flash = None
        self.csp = SudokuCSP(PUZZLES[self.puzzle_var.get()], self.mrv_var.get(),
                             self.fc_var.get(), self.rand_var.get())
        self.gen = self.csp.search()
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")
        self.status.config(text="Ready. Press Run, Step or Solve Fast.", fg="black")
        self.draw()
        self.refresh_stats()

    def run(self):
        if self.gen is None:
            return
        self.fast = False
        self.running = True
        self._tick()

    def run_fast(self):
        if self.gen is None:
            return
        self.fast = True
        self.running = True
        self.status.config(text="Solving…", fg="black")
        self._fast_tick()

    def pause(self):
        self.running = False

    def _tick(self):
        if not self.running or self.fast:
            return
        if self.step():
            self.after(self.speed.get(), self._tick)

    def _fast_tick(self):
        if not self.running or not self.fast:
            return
        for _ in range(3000):
            if not self._advance(log=False):
                self.draw()
                self.refresh_stats()
                return
        self.draw()
        self.refresh_stats()
        self.after(1, self._fast_tick)

    def step(self):
        ok = self._advance(log=True)
        self.draw()
        self.refresh_stats()
        return ok

    def _advance(self, log):
        """Process one solver event. Returns False when the run should stop."""
        if self.gen is None:
            return False
        try:
            kind, i, v = next(self.gen)
        except StopIteration:
            self.gen = None
            self.running = False
            self.status.config(text="✘ No solution exists for this puzzle.", fg="#c1121f")
            return False

        self.flash = None
        if kind == "place":
            self.cur = i
            if log:
                self.write(f"PLACE     R{i // 9 + 1}C{i % 9 + 1} = {v}")
        elif kind == "backtrack":
            self.cur = None
            self.flash = i
            if log:
                self.write(f"BACKTRACK R{i // 9 + 1}C{i % 9 + 1} x {v}")
        elif kind == "solution":
            self.cur = None
            self.running = False
            self.gen = None
            self.status.config(text="✔ Sudoku solved!", fg="#2d6a4f")
            self.write(f"SOLVED ✔  ({self.csp.assigns} placements, {self.csp.backtracks} backtracks)")
            return False
        return True


if __name__ == "__main__":
    App().mainloop()