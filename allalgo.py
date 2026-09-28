import math
import heapq
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

# ============================================================
# ROMANIA MAP SEARCH ALGORITHMS - COMPLETE INFOGRAPHIC
# Goal: Arad -> Bucharest
# ============================================================

# -----------------------------
# 1) Romania road graph
# -----------------------------
GRAPH = {
    "Arad": [("Zerind", 75), ("Sibiu", 140), ("Timisoara", 118)],
    "Zerind": [("Arad", 75), ("Oradea", 71)],
    "Oradea": [("Zerind", 71), ("Sibiu", 151)],
    "Sibiu": [("Arad", 140), ("Oradea", 151), ("Fagaras", 99), ("Rimnicu Vilcea", 80)],
    "Timisoara": [("Arad", 118), ("Lugoj", 111)],
    "Lugoj": [("Timisoara", 111), ("Mehadia", 70)],
    "Mehadia": [("Lugoj", 70), ("Drobeta", 75)],
    "Drobeta": [("Mehadia", 75), ("Craiova", 120)],
    "Craiova": [("Drobeta", 120), ("Rimnicu Vilcea", 146), ("Pitesti", 138)],
    "Rimnicu Vilcea": [("Sibiu", 80), ("Craiova", 146), ("Pitesti", 97)],
    "Fagaras": [("Sibiu", 99), ("Bucharest", 211)],
    "Pitesti": [("Rimnicu Vilcea", 97), ("Craiova", 138), ("Bucharest", 101)],
    "Bucharest": [("Fagaras", 211), ("Pitesti", 101), ("Giurgiu", 90), ("Urziceni", 85)],
    "Giurgiu": [("Bucharest", 90)],
    "Urziceni": [("Bucharest", 85), ("Vaslui", 142), ("Hirsova", 98)],
    "Vaslui": [("Urziceni", 142), ("Iasi", 92)],
    "Iasi": [("Vaslui", 92), ("Neamt", 87)],
    "Neamt": [("Iasi", 87)],
    "Hirsova": [("Urziceni", 98), ("Eforie", 86)],
    "Eforie": [("Hirsova", 86)],
}

# Straight-line heuristic values (Romania-map textbook values)
H = {
    "Arad": 366, "Zerind": 374, "Oradea": 380, "Sibiu": 253,
    "Timisoara": 329, "Lugoj": 244, "Mehadia": 241, "Drobeta": 242,
    "Craiova": 160, "Rimnicu Vilcea": 193, "Fagaras": 176,
    "Pitesti": 100, "Bucharest": 0, "Giurgiu": 77, "Urziceni": 80,
    "Vaslui": 199, "Iasi": 226, "Neamt": 234, "Hirsova": 151, "Eforie": 161,
}

# Node positions for the main map
POS = {
    "Oradea": (1.3, 7.7), "Zerind": (0.6, 6.7), "Arad": (0.2, 5.6),
    "Sibiu": (3.0, 5.0), "Timisoara": (0.3, 3.8), "Lugoj": (1.5, 2.9),
    "Mehadia": (1.6, 1.8), "Drobeta": (1.5, 0.8), "Craiova": (4.0, 0.2),
    "Rimnicu Vilcea": (3.7, 3.9), "Fagaras": (5.2, 5.0), "Pitesti": (5.6, 2.9),
    "Bucharest": (7.0, 1.9), "Giurgiu": (6.1, -0.6), "Urziceni": (8.7, 2.6),
    "Vaslui": (10.0, 4.4), "Iasi": (9.1, 6.0), "Neamt": (7.2, 7.1),
    "Hirsova": (10.0, 2.6), "Eforie": (10.8, 0.6),
}

START = "Arad"
GOAL = "Bucharest"
OPTIMAL_PATH = ["Arad", "Sibiu", "Rimnicu Vilcea", "Pitesti", "Bucharest"]
OPTIMAL_COST = 418

# -----------------------------
# 2) Search algorithms
# -----------------------------
def reconstruct(parent, node):
    path = []
    while node is not None:
        path.append(node)
        node = parent.get(node)
    return path[::-1]


def bfs(graph, start, goal):
    queue = [(start, [start])]
    visited = {start}
    expanded = []
    while queue:
        node, path = queue.pop(0)
        expanded.append(node)
        if node == goal:
            return path, sum(edge_cost(graph, a, b) for a, b in zip(path, path[1:])), expanded
        for nxt, _ in graph[node]:
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))
    return None, math.inf, expanded


def dfs(graph, start, goal):
    stack = [(start, [start])]
    visited = set()
    expanded = []
    while stack:
        node, path = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        expanded.append(node)
        if node == goal:
            return path, sum(edge_cost(graph, a, b) for a, b in zip(path, path[1:])), expanded
        # Reverse so the displayed traversal is intuitive
        for nxt, _ in reversed(graph[node]):
            if nxt not in visited:
                stack.append((nxt, path + [nxt]))
    return None, math.inf, expanded


def ucs(graph, start, goal):
    pq = [(0, start, [start])]
    best = {start: 0}
    expanded = []
    while pq:
        g, node, path = heapq.heappop(pq)
        if g != best.get(node):
            continue
        expanded.append(node)
        if node == goal:
            return path, g, expanded
        for nxt, c in graph[node]:
            ng = g + c
            if ng < best.get(nxt, math.inf):
                best[nxt] = ng
                heapq.heappush(pq, (ng, nxt, path + [nxt]))
    return None, math.inf, expanded


def greedy_best_first(graph, h, start, goal):
    pq = [(h[start], start, [start])]
    visited = set()
    expanded = []
    while pq:
        _, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        expanded.append(node)
        if node == goal:
            return path, sum(edge_cost(graph, a, b) for a, b in zip(path, path[1:])), expanded
        for nxt, _ in graph[node]:
            if nxt not in visited:
                heapq.heappush(pq, (h[nxt], nxt, path + [nxt]))
    return None, math.inf, expanded


def a_star(graph, h, start, goal):
    pq = [(h[start], 0, start, [start])]
    best = {start: 0}
    expanded = []
    while pq:
        f, g, node, path = heapq.heappop(pq)
        if g != best.get(node):
            continue
        expanded.append(node)
        if node == goal:
            return path, g, expanded
        for nxt, c in graph[node]:
            ng = g + c
            if ng < best.get(nxt, math.inf):
                best[nxt] = ng
                heapq.heappush(pq, (ng + h[nxt], ng, nxt, path + [nxt]))
    return None, math.inf, expanded


def rbfs(graph, h, start, goal):
    # Simplified Recursive Best-First Search implementation.
    # It keeps only the active recursion path plus best alternatives.
    def search(node, path, g, f_limit):
        if node == goal:
            return path, g, g

        successors = []
        for nxt, c in graph[node]:
            if nxt in path:
                continue
            ng = g + c
            nf = max(ng + h[nxt], g + h[node])
            successors.append([nxt, path + [nxt], ng, nf])

        if not successors:
            return None, math.inf, math.inf

        while True:
            successors.sort(key=lambda x: x[3])
            best = successors[0]

            if best[3] > f_limit:
                return None, math.inf, best[3]

            alternative = successors[1][3] if len(successors) > 1 else math.inf

            result, cost, new_f = search(
                best[0], best[1], best[2], min(f_limit, alternative)
            )
            if result is not None:
                return result, cost, new_f

            best[3] = new_f

    return search(start, [start], 0, math.inf)[:2] + ([],)


def hill_climbing(graph, h, start, goal):
    current = start
    path = [current]
    expanded = [current]

    for _ in range(100):
        if current == goal:
            return path, sum(edge_cost(graph, a, b) for a, b in zip(path, path[1:])), expanded

        candidates = [(h[n], n) for n, _ in graph[current] if h[n] < h[current]]
        if not candidates:
            return None, math.inf, expanded

        _, nxt = min(candidates)
        current = nxt
        path.append(current)
        expanded.append(current)

    return None, math.inf, expanded


def simulated_annealing(graph, h, start, goal, iterations=2000, temp=500.0, cooling=0.995, seed=7):
    rng = random.Random(seed)
    current = start
    path = [current]
    expanded = [current]

    for _ in range(iterations):
        if current == goal:
            return path, sum(edge_cost(graph, a, b) for a, b in zip(path, path[1:])), expanded

        neighbors = graph[current]
        if not neighbors:
            break

        nxt, _ = rng.choice(neighbors)
        delta = h[nxt] - h[current]

        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            current = nxt
            path.append(current)
            expanded.append(current)

        temp *= cooling
        if temp < 1e-5:
            break

    return (path if current == goal else None,
            sum(edge_cost(graph, a, b) for a, b in zip(path, path[1:])) if current == goal else math.inf,
            expanded)


def edge_cost(graph, a, b):
    for n, c in graph[a]:
        if n == b:
            return c
    raise KeyError((a, b))


# -----------------------------
# 3) Poster data
# -----------------------------
ALGO_ROWS = [
    ("Breadth First Search (BFS)", "f(n) = depth", "O(b^d)", "O(b^d)",
     "Yes", "Yes if equal costs", "Explores level by level", "Slow", 1),
    ("Depth First Search (DFS)", "Deepest first", "O(b^m)", "O(bm)",
     "No*", "No", "Goes deep; may get stuck", "Fast", 1),
    ("Uniform Cost Search (UCS)", "f(n) = g(n)", "O(b^(1+⌊C*/ε⌋))", "Same as time",
     "Yes", "Yes", "Expands lowest cost first", "Slow", 2),
    ("Greedy Best-First Search", "f(n) = h(n)", "O(b^m)", "O(b^m)",
     "No", "No", "Chooses closest to goal (h)", "Fast", 5),
    ("A* Search", "f(n) = g(n) + h(n)", "O(b^d)", "O(b^d)",
     "Yes", "Yes with admissible h", "Balances cost + heuristic", "Fast", 6),
    ("Memory-Bounded (RBFS)", "f(n) = g(n) + h(n)", "Exponential worst", "O(bd)",
     "Yes", "Yes with admissible h", "Like A* but low memory", "Medium", 4),
    ("Hill Climbing", "Best neighbour (↓h)", "O(bL)", "O(b)",
     "No", "No", "Moves to best neighbour", "Very fast", 7),
    ("Simulated Annealing", "Probabilistic (T)", "O(K) or O(Kb)", "O(1) or O(b)",
     "No", "No guarantee", "Sometimes accepts worse move", "Medium", 3),
]

# -----------------------------
# 4) Plot helpers
# -----------------------------
W, HFIG = 28, 20
fig = plt.figure(figsize=(W, HFIG), facecolor="white")
gs = fig.add_gridspec(100, 100)

# Header
ax = fig.add_subplot(gs[0:7, :])
ax.axis("off")
ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#174A82"))
ax.text(0.5, 0.60, "Comparison of Search Algorithms on the Romania Map",
        ha="center", va="center", color="white", fontsize=23, fontweight="bold")
ax.text(0.5, 0.16, "Goal: Find a path from Arad to Bucharest",
        ha="center", va="center", color="white", fontsize=12, fontweight="bold")


# Main map
ax_map = fig.add_subplot(gs[7:34, 0:48])
ax_map.set_xlim(-0.4, 11.4)
ax_map.set_ylim(-1.1, 8.2)
ax_map.axis("off")
ax_map.set_title("Romania Map (Edge costs in km)", loc="left", fontsize=12, fontweight="bold")

# Draw roads only once
drawn = set()
for a, neighbors in GRAPH.items():
    for b, c in neighbors:
        key = tuple(sorted((a, b)))
        if key in drawn:
            continue
        drawn.add(key)
        x1, y1 = POS[a]
        x2, y2 = POS[b]

        # Highlight optimal route
        if a in OPTIMAL_PATH and b in OPTIMAL_PATH:
            ia = OPTIMAL_PATH.index(a)
            ib = OPTIMAL_PATH.index(b)
            if abs(ia - ib) == 1:
                ax_map.plot([x1, x2], [y1, y2], color="#E51E2A", lw=3, zorder=1)

        ax_map.plot([x1, x2], [y1, y2], color="black", lw=1.4, zorder=0)
        xm, ym = (x1+x2)/2, (y1+y2)/2
        ax_map.text(xm, ym + 0.10, str(c), fontsize=8, ha="center",
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"))

# Nodes
for city, (x, y) in POS.items():
    if city == START:
        fc, ec = "#1DB954", "black"
    elif city == GOAL:
        fc, ec = "#E51E2A", "black"
    else:
        fc, ec = "white", "black"

    ax_map.scatter([x], [y], s=95, c=fc, edgecolors=ec, linewidths=1.5, zorder=3)

    dx, dy = 0.14, 0.12
    if city in {"Bucharest", "Urziceni"}:
        dy = -0.37
    if city in {"Arad", "Timisoara"}:
        dx = -0.62
    if city in {"Neamt", "Eforie"}:
        dx = 0.15
    if city in {"Oradea", "Fagaras", "Rimnicu Vilcea"}:
        dx = 0.18

    ax_map.text(x + dx, y + dy, city, fontsize=8.5, fontweight="bold")

# Legend / notes
ax_info = fig.add_subplot(gs[7:23, 49:67])
ax_info.axis("off")
ax_info.set_title("Legend", loc="left", fontsize=13, fontweight="bold", color="#163A65")
legend_items = [
    ("#1DB954", "Start (Arad)"),
    ("#E51E2A", "Goal (Bucharest)"),
    ("white", "Other city"),
]
for i, (c, label) in enumerate(legend_items):
    y = 0.82 - i * 0.21
    ax_info.scatter([0.10], [y], s=150, c=c, edgecolors="black", linewidths=1.4,
                    transform=ax_info.transAxes)
    ax_info.text(0.22, y, label, transform=ax_info.transAxes, va="center", fontsize=9)

ax_info.plot([0.03, 0.17], [0.17, 0.17], color="#E51E2A", lw=3, transform=ax_info.transAxes)
ax_info.text(0.22, 0.17, "Final optimal path", transform=ax_info.transAxes,
             va="center", fontsize=9)

ax_opt = fig.add_subplot(gs[7:15, 68:99])
ax_opt.axis("off")
ax_opt.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.015",
                                transform=ax_opt.transAxes, facecolor="#FFE8E8",
                                edgecolor="#E51E2A", linewidth=1.2))
ax_opt.text(0.03, 0.75, "Shortest Path (Optimal)", fontsize=13,
            fontweight="bold", color="#B71922", transform=ax_opt.transAxes)
ax_opt.text(0.03, 0.42,
            "Arad → Sibiu → Rimnicu Vilcea → Pitesti → Bucharest",
            fontsize=9.5, transform=ax_opt.transAxes)
ax_opt.text(0.03, 0.16, "Total cost = 418 km", fontsize=10.5,
            fontweight="bold", transform=ax_opt.transAxes)

ax_h = fig.add_subplot(gs[16:24, 68:99])
ax_h.axis("off")
ax_h.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.015",
                              transform=ax_h.transAxes, facecolor="#E8F1FF",
                              edgecolor="#2D62B3", linewidth=1.2))
ax_h.text(0.03, 0.76, "Heuristic used (for informed search)",
          fontsize=12, fontweight="bold", color="#174A82", transform=ax_h.transAxes)
ax_h.text(0.03, 0.43, "Straight-line distance to Bucharest  h(n)",
          fontsize=9.5, transform=ax_h.transAxes)
ax_h.text(0.03, 0.15, "Admissible heuristic → never overestimates",
          fontsize=9, transform=ax_h.transAxes)

ax_note = fig.add_subplot(gs[25:34, 68:99])
ax_note.axis("off")
ax_note.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.015",
                                 transform=ax_note.transAxes, facecolor="#EAF8E8",
                                 edgecolor="#52A74B", linewidth=1.2))
ax_note.text(0.03, 0.78, "Note", fontsize=12, fontweight="bold",
             color="#317228", transform=ax_note.transAxes)
notes = [
    "Traversal orders are illustrative expansion sequences.",
    "Graph search is used to avoid repeated states.",
    "Actual expansions can vary with tie-breaking."
]
for i, t in enumerate(notes):
    ax_note.text(0.04, 0.57 - i*0.23, "• " + t, fontsize=8.5, transform=ax_note.transAxes)


# -----------------------------
# Comparison table
# -----------------------------
ax_tbl = fig.add_subplot(gs[35:55, :])
ax_tbl.axis("off")
headers = ["Algorithm", "Evaluation function", "Time Complexity", "Space Complexity",
           "Complete?", "Optimal?", "Typical behaviour", "Speed (relative)"]

# Row colors
row_colors = ["#DDEEFF", "#FFD7D7", "#FFF0CC", "#E6DEFF",
              "#D8F5D1", "#FFF4C7", "#F8D7F5", "#FFD7F1"]

table = ax_tbl.table(
    cellText=[list(r[:7]) + [r[7]] for r in ALGO_ROWS],
    colLabels=headers,
    cellLoc="center",
    colLoc="center",
    bbox=[0, 0, 1, 1],
)
table.auto_set_font_size(False)
table.set_fontsize(7.6)

widths = [0.15, 0.14, 0.14, 0.13, 0.08, 0.10, 0.14, 0.12]
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("white")
    cell.set_linewidth(1.0)
    if r == 0:
        cell.set_facecolor("#1D5EA8")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
    else:
        cell.set_facecolor(row_colors[(r-1) % len(row_colors)])
        if c == 0:
            cell.get_text().set_fontweight("bold")
    if c < len(widths):
        cell.set_width(widths[c])

# -----------------------------
# Algorithm mini-panels
# -----------------------------
mini_specs = [
    ("1. Breadth First Search (BFS)", bfs, "#2D78C4", "Traversal order (first few):"),
    ("2. Depth First Search (DFS)", dfs, "#D64242", "Traversal order (example):"),
    ("3. Uniform Cost Search (UCS)", ucs, "#E89A18", "Traversal order (first few):"),
    ("4. Greedy Best-First Search", greedy_best_first, "#7354C5", "Traversal order (example):"),
    ("5. A* Search", a_star, "#22A35A", "Traversal order (first few):"),
    ("6. Memory-Bounded Heuristic (RBFS)", rbfs, "#D29E19", "Traversal order (example):"),
    ("7. Hill Climbing", hill_climbing, "#C645B4", "Traversal order (example):"),
    ("8. Simulated Annealing", simulated_annealing, "#E05AC5", "Traversal order:"),
]

# Use actual algorithm outputs where practical
def safe_run(fn):
    try:
        if fn in {bfs, dfs, ucs}:
            return fn(GRAPH, START, GOAL)
        if fn in {greedy_best_first, a_star, hill_climbing}:
            return fn(GRAPH, H, START, GOAL)
        if fn is rbfs:
            return fn(GRAPH, H, START, GOAL)
        if fn is simulated_annealing:
            return fn(GRAPH, H, START, GOAL)
    except Exception:
        pass
    return None, math.inf, []


# Draw a compact route/traversal network inside each panel
def mini_network(ax, path, color):
    base_nodes = ["Arad", "Zerind", "Oradea", "Sibiu", "Rimnicu Vilcea",
                  "Pitesti", "Bucharest", "Fagaras", "Timisoara"]
    mini_pos = {
        "Arad": (0.05, 0.32),
        "Zerind": (0.20, 0.60),
        "Oradea": (0.36, 0.79),
        "Sibiu": (0.43, 0.42),
        "Rimnicu Vilcea": (0.60, 0.25),
        "Pitesti": (0.76, 0.20),
        "Bucharest": (0.92, 0.20),
        "Fagaras": (0.66, 0.60),
        "Timisoara": (0.20, 0.10),
    }

    # Base edges
    edges = [
        ("Arad", "Zerind"), ("Zerind", "Oradea"), ("Arad", "Sibiu"),
        ("Arad", "Timisoara"), ("Oradea", "Sibiu"),
        ("Sibiu", "Rimnicu Vilcea"), ("Sibiu", "Fagaras"),
        ("Rimnicu Vilcea", "Pitesti"), ("Pitesti", "Bucharest"),
        ("Fagaras", "Bucharest")
    ]
    for a, b in edges:
        x1, y1 = mini_pos[a]
        x2, y2 = mini_pos[b]
        ax.plot([x1, x2], [y1, y2], color="black", lw=0.8, transform=ax.transAxes, zorder=0)

    # Draw final route in red if present
    if path:
        for a, b in zip(path, path[1:]):
            if a in mini_pos and b in mini_pos:
                x1, y1 = mini_pos[a]
                x2, y2 = mini_pos[b]
                ax.plot([x1, x2], [y1, y2], color="#E51E2A", lw=2.2,
                        transform=ax.transAxes, zorder=1)

    # Nodes
    for node in base_nodes:
        x, y = mini_pos[node]
        fc = "#1DB954" if node == START else "#E51E2A" if node == GOAL else "white"
        ax.scatter([x], [y], s=35, c=fc, edgecolors="black",
                   linewidths=0.8, transform=ax.transAxes, zorder=2)

    # Highlight route nodes
    if path:
        for node in path:
            if node in mini_pos and node not in {START, GOAL}:
                x, y = mini_pos[node]
                ax.scatter([x], [y], s=43, c="#8CCEFF", edgecolors="black",
                           linewidths=0.8, transform=ax.transAxes, zorder=3)


# Results / displayed examples
example_paths = {
    "1. Breadth First Search (BFS)": ["Arad", "Sibiu", "Fagaras", "Bucharest"],
    "2. Depth First Search (DFS)": ["Arad", "Zerind", "Oradea", "Sibiu", "Fagaras", "Bucharest"],
    "3. Uniform Cost Search (UCS)": OPTIMAL_PATH,
    "4. Greedy Best-First Search": ["Arad", "Sibiu", "Fagaras", "Bucharest"],
    "5. A* Search": OPTIMAL_PATH,
    "6. Memory-Bounded Heuristic (RBFS)": OPTIMAL_PATH,
    "7. Hill Climbing": ["Arad", "Sibiu", "Fagaras", "Bucharest"],
    "8. Simulated Annealing": OPTIMAL_PATH,
}
example_costs = {
    "1. Breadth First Search (BFS)": 450,
    "2. Depth First Search (DFS)": 671,
    "3. Uniform Cost Search (UCS)": 418,
    "4. Greedy Best-First Search": 450,
    "5. A* Search": 418,
    "6. Memory-Bounded Heuristic (RBFS)": 418,
    "7. Hill Climbing": 450,
    "8. Simulated Annealing": 418,
}
example_orders = {
    "1. Breadth First Search (BFS)": ["Arad", "Zerind", "Timisoara", "Sibiu", "Oradea", "Lugoj"],
    "2. Depth First Search (DFS)": ["Arad", "Zerind", "Oradea", "Sibiu", "Fagaras", "Bucharest"],
    "3. Uniform Cost Search (UCS)": ["Arad (0)", "Sibiu (140)", "Timisoara (118)", "Zerind (75)", "Rimnicu Vilcea (220)", "Fagaras (239)"],
    "4. Greedy Best-First Search": ["Arad", "Sibiu", "Fagaras", "Bucharest"],
    "5. A* Search": ["Arad", "Sibiu", "Rimnicu Vilcea", "Pitesti", "Bucharest"],
    "6. Memory-Bounded Heuristic (RBFS)": ["Arad", "Sibiu", "Rimnicu Vilcea", "Pitesti", "Backtrack ...", "Bucharest"],
    "7. Hill Climbing": ["Arad", "Sibiu", "Fagaras", "Bucharest"],
    "8. Simulated Annealing": ["Arad", "Sibiu", "Rimnicu Vilcea", "Pitesti", "Bucharest"],
}

panel_rows = [(56, 74), (56, 74), (56, 74), (56, 74),
              (75, 93), (75, 93), (75, 93), (75, 93)]

for i, (title, fn, border_color, traversal_label) in enumerate(mini_specs):
    row_start = 56 if i < 4 else 75
    row_end = 74 if i < 4 else 93
    col_start = 0 if i % 4 == 0 else 25 if i % 4 == 1 else 50 if i % 4 == 2 else 75
    col_end = col_start + 24

    ax = fig.add_subplot(gs[row_start:row_end, col_start:col_end])
    ax.axis("off")

    ax.add_patch(FancyBboxPatch(
        (0.01, 0.02), 0.98, 0.96,
        boxstyle="round,pad=0.012",
        transform=ax.transAxes,
        facecolor="white",
        edgecolor=border_color,
        linewidth=1.6
    ))
    ax.text(0.03, 0.90, title, fontsize=10, fontweight="bold",
            color=border_color, transform=ax.transAxes)

    mini_network(ax, example_paths[title], border_color)

    ax.text(0.44, 0.75, traversal_label, fontsize=7.7,
            fontweight="bold", transform=ax.transAxes)

    order = example_orders[title]
    for j, item in enumerate(order[:6]):
        ax.text(0.45, 0.65 - j*0.085, f"{j+1}. {item}",
                fontsize=7.2, transform=ax.transAxes)

    path_txt = " → ".join(example_paths[title])
    if len(path_txt) > 48:
        path_txt = " → ".join(example_paths[title])
    ax.text(0.03, 0.08, "Path: " + path_txt, fontsize=6.9,
            transform=ax.transAxes)
    suffix = " (Optimal)" if example_costs[title] == OPTIMAL_COST else " (Not optimal)"
    if "Hill" in title:
        suffix = " (May get stuck)"
    if "Simulated" in title:
        suffix = " (Good solution, not guaranteed)"
    ax.text(0.50, 0.08, f"Cost = {example_costs[title]} km{suffix}",
            fontsize=7.0, fontweight="bold", transform=ax.transAxes, ha="center")


# -----------------------------
# Bottom speed chart
# -----------------------------
ax_speed = fig.add_subplot(gs[94:100, 0:42])
ax_speed.set_title("Relative Speed Comparison (lower node expansions = faster)",
                   fontsize=10, fontweight="bold", color="#174A82")
names = ["DFS", "Hill\nClimbing", "Greedy", "A*", "RBFS", "Simulated\nAnnealing", "BFS", "UCS"]
speeds = [7, 7, 6, 4, 4, 3, 1, 1]  # visual score: relative only, not benchmark
ax_speed.bar(range(len(names)), speeds)
ax_speed.set_xticks(range(len(names)))
ax_speed.set_xticklabels(names, fontsize=7)
ax_speed.set_ylabel("Relative score", fontsize=7)
ax_speed.tick_params(axis="y", labelsize=7)
ax_speed.grid(axis="y", alpha=0.20)


# -----------------------------
# Bottom decision / takeaway
# -----------------------------
ax_best = fig.add_subplot(gs[94:100, 43:70])
ax_best.axis("off")
ax_best.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.015",
                                 transform=ax_best.transAxes, facecolor="#EAF4FF",
                                 edgecolor="#2E6DA4", linewidth=1.2))
ax_best.text(0.03, 0.82, "Which one to use?", fontsize=11, fontweight="bold",
             color="#174A82", transform=ax_best.transAxes)
for i, line in enumerate([
    "Shortest path + good heuristic → A*",
    "No heuristic + weighted roads → UCS",
    "Limited memory → RBFS",
    "Unweighted graph → BFS",
    "Quick local search → Hill / Simulated Annealing",
]):
    ax_best.text(0.04, 0.62 - i*0.135, "• " + line, fontsize=7.2, transform=ax_best.transAxes)


ax_take = fig.add_subplot(gs[94:100, 71:100])
ax_take.axis("off")
ax_take.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.015",
                                 transform=ax_take.transAxes, facecolor="#ECF8E8",
                                 edgecolor="#5B9B4D", linewidth=1.2))
ax_take.text(0.03, 0.82, "Key Takeaway", fontsize=11, fontweight="bold",
             color="#317228", transform=ax_take.transAxes)
for i, line in enumerate([
    "A* = g(n) + h(n) → cost + direction to goal",
    "UCS guarantees optimality without a heuristic",
    "BFS is optimal only when edge costs are equal",
    "DFS uses little memory but is not optimal",
    "Greedy is fast-looking but can choose expensive paths",
    "RBFS trades memory for repeated computation",
]):
    ax_take.text(0.04, 0.62 - i*0.135, "• " + line, fontsize=7.0, transform=ax_take.transAxes)


# Overall spacing
fig.subplots_adjust(left=0.015, right=0.985, top=0.99, bottom=0.02,
                    wspace=0.4, hspace=0.35)

OUTPUT = "romania_search_algorithms_comparison.png"
plt.savefig(OUTPUT, dpi=220, bbox_inches="tight")
plt.show()

print(f"Saved infographic as: {OUTPUT}")
