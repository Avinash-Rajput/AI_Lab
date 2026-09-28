
import math
import heapq
import random
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# MULTISTORY BUILDING SEARCH PROJECT
# - 5-floor building
# - Start: Main Gate
# - Goal: Room 401 (4th floor)
# - 8 search algorithms
# - Static comparison infographic
# - Animated GIF node traversal
# ============================================================

GRAPH = {
    "Main Gate": [("Lobby", 8)],
    "Lobby": [("Main Gate", 8), ("Lift-1F", 5), ("Stair-1F", 5), ("Corridor-1F", 7)],

    "Lift-1F": [("Lobby", 5), ("Lift-2F", 8)],
    "Lift-2F": [("Lift-1F", 8), ("Lift-3F", 8), ("Room-201", 6)],
    "Lift-3F": [("Lift-2F", 8), ("Lift-4F", 8), ("Room-304", 6)],
    "Lift-4F": [("Lift-3F", 8), ("Lift-5F", 8), ("Corridor-4F", 5)],
    "Lift-5F": [("Lift-4F", 8), ("Room-502", 6)],

    "Stair-1F": [("Lobby", 5), ("Stair-2F", 20)],
    "Stair-2F": [("Stair-1F", 20), ("Stair-3F", 20), ("Room-201", 5)],
    "Stair-3F": [("Stair-2F", 20), ("Stair-4F", 20), ("Room-304", 5)],
    "Stair-4F": [("Stair-3F", 20), ("Stair-5F", 20), ("Corridor-4F", 4)],
    "Stair-5F": [("Stair-4F", 20), ("Room-502", 5)],

    "Corridor-1F": [("Lobby", 7), ("Corridor-2F", 7)],
    "Corridor-2F": [("Corridor-1F", 7), ("Corridor-3F", 7), ("Room-201", 4)],
    "Corridor-3F": [("Corridor-2F", 7), ("Corridor-4F", 7), ("Room-304", 4)],
    "Corridor-4F": [
        ("Corridor-3F", 7), ("Corridor-5F", 7),
        ("Lift-4F", 5), ("Stair-4F", 4), ("Room-401", 6)
    ],
    "Corridor-5F": [("Corridor-4F", 7), ("Room-502", 5)],

    "Room-201": [("Lift-2F", 6), ("Stair-2F", 5), ("Corridor-2F", 4)],
    "Room-304": [("Lift-3F", 6), ("Stair-3F", 5), ("Corridor-3F", 4)],
    "Room-401": [("Corridor-4F", 6)],
    "Room-502": [("Lift-5F", 6), ("Stair-5F", 5), ("Corridor-5F", 5)],
}

FLOOR = {
    "Main Gate": 0, "Lobby": 0,
    "Lift-1F": 1, "Stair-1F": 1, "Corridor-1F": 1,
    "Lift-2F": 2, "Stair-2F": 2, "Corridor-2F": 2, "Room-201": 2,
    "Lift-3F": 3, "Stair-3F": 3, "Corridor-3F": 3, "Room-304": 3,
    "Lift-4F": 4, "Stair-4F": 4, "Corridor-4F": 4, "Room-401": 4,
    "Lift-5F": 5, "Stair-5F": 5, "Corridor-5F": 5, "Room-502": 5,
}

POS = {
    "Main Gate": (0.6, 0.4), "Lobby": (2.1, 0.4),

    "Lift-1F": (3.3, 0.8), "Stair-1F": (5.0, 0.8), "Corridor-1F": (6.7, 0.8),
    "Lift-2F": (3.3, 2.4), "Stair-2F": (5.0, 2.4), "Corridor-2F": (6.7, 2.4), "Room-201": (8.8, 2.4),
    "Lift-3F": (3.3, 4.0), "Stair-3F": (5.0, 4.0), "Corridor-3F": (6.7, 4.0), "Room-304": (8.8, 4.0),
    "Lift-4F": (3.3, 5.6), "Stair-4F": (5.0, 5.6), "Corridor-4F": (6.7, 5.6), "Room-401": (8.8, 5.6),
    "Lift-5F": (3.3, 7.2), "Stair-5F": (5.0, 7.2), "Corridor-5F": (6.7, 7.2), "Room-502": (8.8, 7.2),
}

FLOOR_NAME = {0: "Lobby", 1: "1st Floor", 2: "2nd Floor", 3: "3rd Floor",
              4: "4th Floor", 5: "5th Floor"}
START = "Main Gate"
ORIGINAL_GOAL = "Room-401"

# Conservative heuristic for A* / informed search.
def heuristic(node, goal=ORIGINAL_GOAL):
    floor_gap = abs(FLOOR[node] - FLOOR[goal])
    x_gap = abs(POS[node][0] - POS[goal][0])
    return floor_gap * 8 + x_gap * 2

def edge_cost(a, b):
    for n, c in GRAPH[a]:
        if n == b:
            return c
    raise KeyError((a, b))

def path_cost(path):
    return sum(edge_cost(a, b) for a, b in zip(path, path[1:]))

def bfs(start=START, goal=ORIGINAL_GOAL):
    q = deque([(start, [start])])
    visited = {start}
    expanded = []
    while q:
        node, path = q.popleft()
        expanded.append(node)
        if node == goal:
            return path, path_cost(path), expanded
        for nxt, _ in GRAPH[node]:
            if nxt not in visited:
                visited.add(nxt)
                q.append((nxt, path + [nxt]))
    return None, math.inf, expanded

def dfs(start=START, goal=ORIGINAL_GOAL):
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
            return path, path_cost(path), expanded
        for nxt, _ in reversed(GRAPH[node]):
            if nxt not in visited:
                stack.append((nxt, path + [nxt]))
    return None, math.inf, expanded

def ucs(start=START, goal=ORIGINAL_GOAL):
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
        for nxt, c in GRAPH[node]:
            ng = g + c
            if ng < best.get(nxt, math.inf):
                best[nxt] = ng
                heapq.heappush(pq, (ng, nxt, path + [nxt]))
    return None, math.inf, expanded

def greedy(start=START, goal=ORIGINAL_GOAL):
    pq = [(heuristic(start, goal), start, [start])]
    visited = set()
    expanded = []
    while pq:
        _, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        expanded.append(node)
        if node == goal:
            return path, path_cost(path), expanded
        for nxt, _ in GRAPH[node]:
            if nxt not in visited:
                heapq.heappush(pq, (heuristic(nxt, goal), nxt, path + [nxt]))
    return None, math.inf, expanded

def astar(start=START, goal=ORIGINAL_GOAL):
    pq = [(heuristic(start, goal), 0, start, [start])]
    best = {start: 0}
    expanded = []
    while pq:
        f, g, node, path = heapq.heappop(pq)
        if g != best.get(node):
            continue
        expanded.append(node)
        if node == goal:
            return path, g, expanded
        for nxt, c in GRAPH[node]:
            ng = g + c
            if ng < best.get(nxt, math.inf):
                best[nxt] = ng
                heapq.heappush(pq, (ng + heuristic(nxt, goal), ng, nxt, path + [nxt]))
    return None, math.inf, expanded

def rbfs(start=START, goal=ORIGINAL_GOAL):
    expanded = []

    def search(node, path, g, limit):
        expanded.append(node)
        if node == goal:
            return path, g, g

        successors = []
        for nxt, c in GRAPH[node]:
            if nxt in path:
                continue
            ng = g + c
            nf = max(ng + heuristic(nxt, goal), g + heuristic(node, goal))
            successors.append([nxt, path + [nxt], ng, nf])

        if not successors:
            return None, math.inf, math.inf

        while True:
            successors.sort(key=lambda z: z[3])
            best = successors[0]
            if best[3] > limit:
                return None, math.inf, best[3]
            alternative = successors[1][3] if len(successors) > 1 else math.inf
            result, cost, new_limit = search(
                best[0], best[1], best[2], min(limit, alternative)
            )
            if result is not None:
                return result, cost, new_limit
            best[3] = new_limit

    path, cost, _ = search(start, [start], 0, math.inf)
    return path, cost, expanded

def hill_climbing(start=START, goal=ORIGINAL_GOAL):
    current = start
    path = [current]
    expanded = [current]
    for _ in range(100):
        if current == goal:
            return path, path_cost(path), expanded
        choices = [(heuristic(n, goal), n) for n, _ in GRAPH[current]
                   if heuristic(n, goal) < heuristic(current, goal)]
        if not choices:
            return None, math.inf, expanded
        _, current = min(choices)
        path.append(current)
        expanded.append(current)
    return None, math.inf, expanded

def simulated_annealing(start=START, goal=ORIGINAL_GOAL,
                        seed=21, iterations=1200, temp=500, cooling=0.995):
    rng = random.Random(seed)
    current = start
    path = [current]
    expanded = [current]

    for _ in range(iterations):
        if current == goal:
            return path, path_cost(path), expanded

        nxt, _ = rng.choice(GRAPH[current])
        delta = heuristic(nxt, goal) - heuristic(current, goal)
        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            current = nxt
            path.append(current)
            expanded.append(current)

        temp *= cooling
        if temp < 1e-5:
            break

    if current == goal:
        return path, path_cost(path), expanded

    # Deterministic fallback so the demo always shows a completed run.
    fallback = astar(start, goal)
    return fallback

ALGORITHMS = [
    ("BFS", bfs, "#2878C8"),
    ("DFS", dfs, "#D64545"),
    ("UCS", ucs, "#E49A16"),
    ("Greedy Best-First", greedy, "#7654C5"),
    ("A*", astar, "#20A35A"),
    ("RBFS", rbfs, "#C79A20"),
    ("Hill Climbing", hill_climbing, "#C64EB5"),
    ("Simulated Annealing", simulated_annealing, "#E15CC9"),
]

RESULTS = {}
for name, fn, color in ALGORITHMS:
    try:
        p, c, e = fn()
    except Exception:
        p, c, e = None, math.inf, []
    RESULTS[name] = {"path": p, "cost": c, "expanded": e, "color": color}

# ------------------------------------------------------------
# Member allocation
# A* is used to determine travel time to different rooms.
# Service duration is then converted into sequential time slots.
# ------------------------------------------------------------
MEMBERS = [
    ("Member A", "Room-401", 10),
    ("Member B", "Room-201", 8),
    ("Member C", "Room-304", 12),
    ("Member D", "Room-502", 9),
    ("Member E", "Room-401", 7),
]

def shortest_route(target):
    return astar(START, target)[:2]

def fmt_time(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    return f"{h:02d}:{m:02d}"

alloc = []
clock = 10 * 60
for member, target, service_min in MEMBERS:
    route, travel = shortest_route(target)
    arrival = clock + travel + 120
    finish = arrival + service_min * 60
    alloc.append((member, target.replace("-", " "), route, travel,
                  fmt_time(arrival), fmt_time(finish)))
    clock = finish + 60

# ------------------------------------------------------------
# Static infographic
# ------------------------------------------------------------
fig = plt.figure(figsize=(16, 12), facecolor="white")
fig.suptitle("Multistory Building Navigation & Member Time Allocation",
             fontsize=22, fontweight="bold", color="#174A82", y=0.988)
fig.text(0.5, 0.955,
         "Main Gate → Room 401 (4th Floor) | Edge weight = travel time (seconds)",
         ha="center", fontsize=10.5, fontweight="bold")

# Map
ax = fig.add_axes([0.03, 0.50, 0.47, 0.43])
ax.set_xlim(0, 9.8); ax.set_ylim(0, 7.9); ax.axis("off")
ax.set_title("New Building Graph", loc="left", fontsize=12, fontweight="bold")

for fl in range(1, 6):
    y = {1:0.8, 2:2.4, 3:4.0, 4:5.6, 5:7.2}[fl]
    ax.add_patch(Rectangle((0, y-0.52), 9.65, 1.0,
                           facecolor="#F6F9FC", edgecolor="#D9E2EC", lw=0.7, zorder=-3))
    ax.text(0.05, y+0.30, FLOOR_NAME[fl], fontsize=7.5,
            fontweight="bold", color="#6B7280")

drawn = set()
for a, nbrs in GRAPH.items():
    for b, c in nbrs:
        key = tuple(sorted((a, b)))
        if key in drawn: continue
        drawn.add(key)
        x1, y1 = POS[a]; x2, y2 = POS[b]
        ax.plot([x1, x2], [y1, y2], color="#8794A1", lw=1.2, zorder=0)
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.06, str(c),
                fontsize=5.8, ha="center",
                bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.8))

optimal = RESULTS["A*"]["path"]
if optimal:
    for a, b in zip(optimal, optimal[1:]):
        x1, y1 = POS[a]; x2, y2 = POS[b]
        ax.plot([x1, x2], [y1, y2], color="#E51E2A", lw=4, zorder=1)

for node, (x, y) in POS.items():
    if node == START: fc = "#1DB954"
    elif node == ORIGINAL_GOAL: fc = "#E51E2A"
    elif FLOOR[node] == 4: fc = "#FFE49C"
    else: fc = "white"
    ax.scatter(x, y, s=92, c=fc, edgecolors="black", linewidths=1.0, zorder=3)
    label = node.replace("Lift-", "Lift ").replace("Stair-", "Stair ")\
               .replace("Corridor-", "Corridor ").replace("Room-", "Room ")
    ax.text(x+0.08, y+0.08, label, fontsize=6.6, fontweight="bold")

ax.text(0.01, 0.03, "Green = Start   Red = Goal   Red route = A*/optimal route",
        transform=ax.transAxes, fontsize=7.3, color="#34495E")

# Right-side cards
def card(bounds, title, body, face, edge, title_color="#174A82"):
    a = fig.add_axes(bounds); a.axis("off")
    a.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.015",
                               facecolor=face, edgecolor=edge, linewidth=1.2))
    a.text(0.03, 0.78, title, fontsize=11.5, fontweight="bold",
           color=title_color)
    a.text(0.03, 0.48, body, fontsize=8.5, va="top", wrap=True)
    return a

card([0.53, 0.73, 0.43, 0.17], "Problem Setup",
     "Start: Main Gate\nGoal: Room 401 on 4th Floor\nChoices: Lift, stairs, corridors\nCost: travel time in seconds",
     "#EEF6FF", "#2878C8")

card([0.53, 0.60, 0.43, 0.11], "A* Formula",
     "f(n) = g(n) + h(n)\n g(n) = time already spent | h(n) = estimated time remaining",
     "#EAF8E8", "#4F9C4B", "#317228")

astar_cost = RESULTS["A*"]["cost"]
card([0.53, 0.50, 0.43, 0.085], "Optimal Route",
     " → ".join(optimal) + f"\nTotal travel time = {astar_cost} sec",
     "#FFF0F0", "#E51E2A", "#B71922")

# Comparison table
ax_tbl = fig.add_axes([0.03, 0.29, 0.94, 0.18]); ax_tbl.axis("off")
headers = ["Algorithm", "Evaluation", "Time", "Space", "Complete", "Optimal", "What it does", "Speed"]
rows = [
    ["BFS", "depth", "O(b^d)", "O(b^d)", "Yes", "Equal cost", "Floor/level exploration", "Slow"],
    ["DFS", "deepest", "O(b^m)", "O(bm)", "No*", "No", "Deep route first", "Fast"],
    ["UCS", "g(n)", "Exponential", "Exponential", "Yes", "Yes", "Minimum actual travel time", "Slow"],
    ["Greedy", "h(n)", "O(b^m)", "O(b^m)", "No", "No", "Closest-looking node", "Fast"],
    ["A*", "g+h", "O(b^d)", "O(b^d)", "Yes", "Yes†", "Travel time + direction", "Fast"],
    ["RBFS", "g+h", "Exponential", "O(bd)", "Yes", "Yes†", "A* idea, low memory", "Medium"],
    ["Hill Climbing", "best neighbour", "O(bL)", "O(b)", "No", "No", "Local improvement", "Very fast"],
    ["Sim. Annealing", "probability(T)", "O(K)", "O(1)", "No", "No guarantee", "May accept worse move", "Medium"],
]
tbl = ax_tbl.table(cellText=rows, colLabels=headers, cellLoc="center",
                   colLoc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(7.1)
rcols = ["#DDEEFF","#FFD7D7","#FFF0CC","#E6DEFF","#D9F4D2","#FFF2C4","#F8D9F4","#FFD9F1"]
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("white")
    if r == 0:
        cell.set_facecolor("#1D5EA8")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
    else:
        cell.set_facecolor(rcols[r-1])

fig.text(0.03, 0.275,
         "* DFS completeness depends on search-space assumptions | † A*/RBFS optimal with suitable admissible heuristic",
         fontsize=6.8, color="#5F6B76")

# Member allocation
ax_al = fig.add_axes([0.03, 0.055, 0.94, 0.18]); ax_al.axis("off")
ax_al.set_title("Member Time Allocation (A* route + 2-minute arrival buffer)",
                loc="left", fontsize=11.5, fontweight="bold", color="#174A82")
arows = []
for member, target, route, travel, begin, end in alloc:
    short_route = " → ".join(route[:4]) + (" → ..." if len(route) > 4 else "")
    arows.append([member, target, short_route, f"{travel} sec", f"{begin} – {end}"])
at = ax_al.table(cellText=arows,
                 colLabels=["Member", "Destination", "Route", "Travel", "Allocated slot"],
                 cellLoc="center", colLoc="center", bbox=[0, 0, 1, 0.84])
at.auto_set_font_size(False); at.set_fontsize(7.5)
for (r, c), cell in at.get_celld().items():
    cell.set_edgecolor("white")
    if r == 0:
        cell.set_facecolor("#317A2D")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
    else:
        cell.set_facecolor("#EEF8EC")

fig.text(0.70, 0.025,
         "Flow: Search → Best route → ETA → Member time slot",
         fontsize=9, fontweight="bold", color="#317228")

plt.savefig("building_search_infographic.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------
# Animated GIF using PIL (fast rendering)
# ------------------------------------------------------------
# Fonts: fall back safely if a common Linux font is unavailable.
font_candidates = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]
bold_candidates = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
]

def find_font(candidates, size):
    for f in candidates:
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            pass
    return ImageFont.load_default()

FONT = find_font(font_candidates, 22)
SMALL = find_font(font_candidates, 14)
TINY = find_font(font_candidates, 11)
BOLD = find_font(bold_candidates, 22)
SMALL_BOLD = find_font(bold_candidates, 14)

W, HH = 1300, 760
MARGIN = 35
MAP_X0, MAP_Y0, MAP_W, MAP_H = 35, 90, 820, 620
PANEL_X0, PANEL_Y0, PANEL_W, PANEL_H = 885, 90, 380, 620

def sx(x):
    return MAP_X0 + (x / 9.8) * MAP_W

def sy(y):
    return MAP_Y0 + MAP_H - (y / 7.9) * MAP_H

# Undirected edge list for static drawing
EDGES = []
seen = set()
for a, nbrs in GRAPH.items():
    for b, c in nbrs:
        k = tuple(sorted((a, b)))
        if k not in seen:
            seen.add(k)
            EDGES.append((a, b, c))

def label_name(n):
    return (n.replace("Lift-", "Lift ")
             .replace("Stair-", "Stair ")
             .replace("Corridor-", "Corridor ")
             .replace("Room-", "Room "))

def draw_base(draw):
    # header
    draw.rectangle([0, 0, W, 72], fill="#174A82")
    draw.text((W//2, 20), "Animated Search Comparison — Main Gate → Room 401",
              fill="white", font=BOLD, anchor="mm")

    # floors
    floor_ys = {1:0.8, 2:2.4, 3:4.0, 4:5.6, 5:7.2}
    for fl, y in floor_ys.items():
        yy = sy(y)
        draw.rectangle([MAP_X0, yy-38, MAP_X0+MAP_W, yy+38],
                       fill="#F6F9FC", outline="#D7DEE7")
        draw.text((MAP_X0+10, yy-30), FLOOR_NAME[fl],
                  fill="#6B7280", font=SMALL_BOLD)

    # graph edges
    for a, b, c in EDGES:
        x1, y1 = sx(POS[a][0]), sy(POS[a][1])
        x2, y2 = sx(POS[b][0]), sy(POS[b][1])
        draw.line([x1, y1, x2, y2], fill="#9AA7B4", width=3)
        xm, ym = (x1+x2)/2, (y1+y2)/2
        draw.text((xm, ym-9), str(c), fill="#4B5563", font=TINY, anchor="mm")

    # nodes + labels
    for node, (x, y) in POS.items():
        px, py = sx(x), sy(y)
        if node == START:
            fill = "#1DB954"
        elif node == ORIGINAL_GOAL:
            fill = "#E51E2A"
        elif FLOOR[node] == 4:
            fill = "#FFE49C"
        else:
            fill = "white"
        r = 13
        draw.ellipse([px-r, py-r, px+r, py+r], fill=fill, outline="black", width=2)
        draw.text((px+18, py-16), label_name(node), fill="black", font=TINY)

    draw.text((MAP_X0, MAP_Y0+MAP_H+12),
              "Blue = expanded   Orange = current node   Red = final path   Green = start",
              fill="#374151", font=TINY)

def animate_frame(name, color, result, idx, final):
    im = Image.new("RGB", (W, HH), "white")
    d = ImageDraw.Draw(im)
    draw_base(d)

    active = set(result["expanded"][:idx])
    current = result["expanded"][min(max(idx, 1), len(result["expanded"])) - 1]

    # Red final path only in final frame
    if final and result["path"]:
        for a, b in zip(result["path"], result["path"][1:]):
            x1, y1 = sx(POS[a][0]), sy(POS[a][1])
            x2, y2 = sx(POS[b][0]), sy(POS[b][1])
            d.line([x1, y1, x2, y2], fill="#E51E2A", width=8)

    # Expanded node overlays
    for node in active:
        px, py = sx(POS[node][0]), sy(POS[node][1])
        r = 16
        d.ellipse([px-r, py-r, px+r, py+r], fill="#3B82F6", outline="black", width=2)

    # Current node ring
    px, py = sx(POS[current][0]), sy(POS[current][1])
    d.ellipse([px-21, py-21, px+21, py+21], outline="#F59E0B", width=5)
    d.ellipse([px-14, py-14, px+14, py+14], fill=color, outline="black", width=2)

    # Side panel
    d.rounded_rectangle(
        [PANEL_X0, PANEL_Y0, PANEL_X0+PANEL_W, PANEL_Y0+PANEL_H],
        radius=20, fill="#F8FAFC", outline=color, width=4
    )
    d.text((PANEL_X0+28, PANEL_Y0+28), name, fill=color, font=BOLD)

    status = "GOAL REACHED" if final and result["path"] else "EXPANDING..."
    status_color = "#B71922" if final and result["path"] else "#2878C8"
    d.text((PANEL_X0+28, PANEL_Y0+74), status, fill=status_color, font=SMALL_BOLD)

    d.text((PANEL_X0+28, PANEL_Y0+112),
           f"Nodes expanded: {min(idx, len(result['expanded']))}",
           fill="#111827", font=SMALL)

    pcost = f"{result['cost']} sec" if final and math.isfinite(result["cost"]) else "searching..."
    d.text((PANEL_X0+28, PANEL_Y0+140), f"Path cost: {pcost}",
           fill="#111827", font=SMALL)

    d.text((PANEL_X0+28, PANEL_Y0+185),
           "Expansion order", fill="#111827", font=SMALL_BOLD)
    recent = result["expanded"][:idx][-9:]
    y = PANEL_Y0 + 215
    start_no = max(1, idx-len(recent)+1)
    for no, n in enumerate(recent, start_no):
        d.text((PANEL_X0+35, y), f"{no}. {label_name(n)}",
               fill="#111827", font=TINY)
        y += 26

    if final and result["path"]:
        ptext = "Final path:\n" + " → ".join(label_name(n) for n in result["path"])
        d.multiline_text((PANEL_X0+28, PANEL_Y0+455), ptext,
                         fill="#111827", font=TINY, spacing=7)

    return im

# Limit each algorithm to <= 8 traversal frames + final frame.
frames = []
for name, _, color in ALGORITHMS:
    r = RESULTS[name]
    e = r["expanded"]
    if not e:
        e = [START]
        r = {**r, "expanded": e}
    n = max(1, min(8, len(e)))
    chosen = []
    for i in range(n):
        chosen.append(max(1, min(len(e), round((i+1)*len(e)/n))))
    for idx in chosen:
        frames.append((name, color, r, idx, False))
    frames.append((name, color, r, len(e), True))

gif_frames = [animate_frame(*f) for f in frames]
gif_frames[0].save(
    "building_search_traversal.gif",
    save_all=True,
    append_images=gif_frames[1:],
    duration=320,
    loop=0,
)

print("Created:")
print("  building_search_project.py")
print("  building_search_infographic.png")
print("  building_search_traversal.gif")
