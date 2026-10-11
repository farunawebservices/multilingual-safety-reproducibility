import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle
import sys

OUT = sys.argv[1]
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

TEAL, RED, OCHRE = "#2a7366", "#bf4f4b", "#a8701f"
BLUE, INK, MUTED = "#2c5f8a", "#1f2937", "#5b6472"
FILL, EDGE, RULE = "#eef3f7", "#c9d6e2", "#d5dbe1"


MINFS = 7.6
def enforce(fig):
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text().strip():
            t.set_fontsize(max(t.get_fontsize(), MINFS))


def box(ax, x, y, w, h, fc=FILL, ec=EDGE, lw=0.8, r=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(ax, x0, y0, x1, y1, c=MUTED, lw=0.8):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, mutation_scale=7,
                                shrinkA=0, shrinkB=0), zorder=3)


def line(ax, xs, ys, c=MUTED, lw=0.8, z=2):
    ax.plot(xs, ys, color=c, lw=lw, zorder=z, solid_capstyle="butt")


def badge(ax, x, y, s, kind):
    """Decision badge centred at (x, y), side s."""
    c = {"veto": TEAL, "allow": RED, "nodec": OCHRE}[kind]
    box(ax, x - s / 2, y - s / 2, s, s, fc=c, ec=c, r=s * 0.12, z=4)
    if kind == "veto":
        ax.add_patch(Rectangle((x - s * 0.28, y - s * 0.28), s * 0.56, s * 0.56,
                               fill=False, ec="white", lw=0.9, zorder=5))
        line(ax, [x - s * 0.2, x + s * 0.2], [y, y], c="white", lw=0.9, z=6)
    elif kind == "allow":
        ax.annotate("", xy=(x + s * 0.28, y), xytext=(x - s * 0.28, y),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=0.9,
                                    mutation_scale=5, shrinkA=0, shrinkB=0), zorder=5)
    else:
        ax.text(x, y, "?", color="white", ha="center", va="center",
                fontsize=s * 1.9, fontweight="bold", zorder=5)


def icon_planner(ax, x, y, s):
    box(ax, x - s / 2, y - s / 2, s, s, fc="none", ec=BLUE, lw=0.8, r=s * 0.15, z=4)
    for k in (-0.22, 0, 0.22):
        ax.add_patch(Circle((x - s * 0.22, y + k * s), s * 0.04, color=BLUE, zorder=5))
        line(ax, [x - s * 0.1, x + s * 0.28], [y + k * s] * 2, c=BLUE, lw=0.8, z=6)


def icon_tool(ax, x, y, s):
    for dx, dy in ((0, 0.28), (-0.32, -0.25), (0.32, -0.25)):
        ax.add_patch(Rectangle((x + dx * s - s * 0.12, y + dy * s - s * 0.1), s * 0.24, s * 0.2,
                               fill=False, ec=BLUE, lw=0.8, zorder=5))
    line(ax, [x, x], [y + s * 0.18, y - s * 0.02], c=BLUE)
    line(ax, [x - s * 0.32, x + s * 0.32], [y - s * 0.02] * 2, c=BLUE)
    line(ax, [x - s * 0.32, x - s * 0.32], [y - s * 0.02, y - s * 0.15], c=BLUE)
    line(ax, [x + s * 0.32, x + s * 0.32], [y - s * 0.02, y - s * 0.15], c=BLUE)


def icon_shield(ax, x, y, s):
    pts = [(x - s * 0.36, y + s * 0.36), (x, y + s * 0.46), (x + s * 0.36, y + s * 0.36),
           (x + s * 0.32, y - s * 0.1), (x, y - s * 0.46), (x - s * 0.32, y - s * 0.1)]
    ax.add_patch(Polygon(pts, closed=True, fill=False, ec=BLUE, lw=0.8, zorder=5))
    line(ax, [x - s * 0.15, x - s * 0.02, x + s * 0.18], [y + s * 0.02, y - s * 0.12, y + s * 0.18],
         c=BLUE, lw=0.9)


def toggle(ax, x, y, s, on):
    if on:
        box(ax, x - s / 2, y - s / 2, s, s, fc=BLUE, ec=BLUE, r=s * 0.15, z=4)
        ax.text(x, y, "+", color="white", ha="center", va="center", fontsize=8,
                fontweight="bold", zorder=5)
    else:
        box(ax, x - s / 2, y - s / 2, s, s, fc="white", ec=EDGE, r=s * 0.15, z=4)
        ax.text(x, y + 0.2, "-", color=MUTED, ha="center", va="center", fontsize=8,
                fontweight="bold", zorder=5)


# =========================== Figure 1 ==================================
fig, ax = plt.subplots(figsize=(6.3, 3.35))
ax.set_xlim(0, 132); ax.set_ylim(0, 63); ax.axis("off")
ax.add_patch(Rectangle((0.3, 0.3), 131.4, 62.4, fill=False, ec="#9aa3ad", lw=0.6))
line(ax, [42, 42], [11, 61], c=RULE, lw=0.6)
line(ax, [86, 86], [11, 61], c=RULE, lw=0.6)
line(ax, [2, 130], [9.5, 9.5], c=RULE, lw=0.6)
T = dict(fontsize=8.6, color=INK, va="center")

# (a) Languages
ax.text(2, 60, r"$\bf{(a)}$  Languages", **T)
ax.text(20, 54.5, "Same harmful request", ha="center", **T)
arrow(ax, 20, 52.6, 20, 50.4)
for (lx, ly, name) in ((3, 44.5, "English"), (22, 44.5, "Arabic"), (3, 37.5, "Hausa"), (22, 37.5, "Yoruba")):
    box(ax, lx, ly, 15, 4.6, ec=FILL)
    ax.text(lx + 7.5, ly + 2.3, name, ha="center", va="center", fontsize=7.8, color=BLUE,
            fontweight="bold")
line(ax, [10.5, 10.5, 29.5, 29.5], [37.3, 35.2, 35.2, 37.3])
arrow(ax, 20, 35.2, 20, 32.6)
for (bx, label, ic) in ((1.0, "Planner", icon_planner), (14.5, "Tool-Caller", icon_tool),
                        (28.0, "Critic", icon_shield)):
    box(ax, bx, 22.5, 13, 9.8)
    ic(ax, bx + 6.5, 28.6, 3.6)
    ax.text(bx + 6.5, 24.3, label, ha="center", va="center", fontsize=6.6, color=INK)
arrow(ax, 14.0, 27.4, 14.5, 27.4); arrow(ax, 27.5, 27.4, 28.0, 27.4)
line(ax, [34.5, 34.5], [22.5, 20.4]); line(ax, [11.5, 34.5], [20.4, 20.4])
for bx, k in ((11.5, "veto"), (23, "allow"), (34.5, "nodec")):
    arrow(ax, bx, 20.4, bx, 18.4)
    badge(ax, bx, 15.4, 5.0, k)

# (b) Decision composition
ax.text(44, 60, r"$\bf{(b)}$  Decision composition", **T)
ax.text(64, 54.5, "Equal failure: 40%", ha="center", fontsize=8.8, color=INK, va="center")
ax.text(64, 51.2, "Illustrative proportions (%)", ha="center", fontsize=6.4, color=MUTED, va="center")
X0, W = 51, 32.0
for (yy, lab, v, a, n) in ((42, "1", 60, 35, 5), (30.5, "2", 60, 5, 35)):
    ax.text(48.5, yy, lab, ha="center", va="center", fontsize=8, color=INK)
    segs = ((v, TEAL), (a, RED), (n, OCHRE))
    xs = X0
    for val, c in segs:
        w = W * val / 100
        ax.add_patch(Rectangle((xs, yy - 2.6), w, 5.2, fc=c, ec="white", lw=0.6, zorder=3))
        if val >= 30:
            ax.text(xs + w / 2, yy, str(val), ha="center", va="center", color="white",
                    fontsize=7.6, fontweight="bold", zorder=4)
        xs += w
    small = 5
    sx = X0 + W * (0.60 + (0.35 if a == 35 else 0)) + W * 0.05 / 2 if a == 35 else X0 + W * 0.60 + W * 0.05 / 2
    sc = OCHRE if a == 35 else RED
    ax.text(sx, yy - 4.4, "5", ha="center", va="center", fontsize=6.4, color=sc)
    line(ax, [X0 + W * 0.6, X0 + W * 0.6, X0 + W, X0 + W], [yy + 3.3, yy + 4.1, yy + 4.1, yy + 3.3],
         c=INK, lw=0.6)
ax.text(57, 15.4, "Failure =", ha="center", va="center", fontsize=8, color=INK)
badge(ax, 65.5, 15.4, 5.0, "allow")
ax.text(71, 15.4, "+", ha="center", va="center", fontsize=8, color=MUTED)
badge(ax, 76.5, 15.4, 5.0, "nodec")

# (c) Intervention sites
ax.text(88, 60, r"$\bf{(c)}$  Intervention sites", **T)
ax.text(113.5, 53.5, "Planner", ha="center", fontsize=7.2, color=INK, va="center")
ax.text(126, 53.5, "Critic", ha="center", fontsize=7.2, color=INK, va="center")
for yy, lab, p, c in ((46.5, "Baseline", 0, 0), (39, "Planner-only", 1, 0),
                      (31.5, "Critic-only", 0, 1), (24, "Joint", 1, 1)):
    ax.text(89, yy, lab, ha="left", va="center", fontsize=7.4, color=INK)
    toggle(ax, 113.5, yy, 5.0, p); toggle(ax, 126, yy, 5.0, c)
ax.text(89, 18.2, "+ steering", ha="left", va="center", fontsize=7, color=BLUE)
ax.text(89, 14.7, "Primary: within phase", ha="left", va="center", fontsize=6.4, color=MUTED)
ax.text(89, 11.6, "Secondary: cross phase*", ha="left", va="center", fontsize=6.4, color=MUTED)

# Legend
for x, k, lab in ((4, "veto", "VETO"), (32, "allow", "ALLOW"), (61, "nodec", "NO-DECISION")):
    badge(ax, x, 5, 3.6, k)
    ax.text(x + 3.6, 5, lab, ha="left", va="center", fontsize=7, color=INK)
ax.text(129.5, 5, "*Not causal for site or strength", ha="right", va="center", fontsize=6.4,
        color=MUTED)
enforce(fig)
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(f"{OUT}/figure-1-concept-2.pdf"); fig.savefig(f"{OUT}/figure-1-concept-2.png", dpi=200)
plt.close(fig)

# =========================== Figure 2 ==================================
fig, ax = plt.subplots(figsize=(6.3, 3.45))
ax.set_xlim(0, 132); ax.set_ylim(0, 66); ax.axis("off")
ax.add_patch(Rectangle((0.3, 0.3), 131.4, 65.4, fill=False, ec="#9aa3ad", lw=0.6))

# input stack
for d in (1.2, 0.6, 0):
    box(ax, 2 + d, 47 - d, 22, 10.5, fc="white", ec=EDGE, r=1.0)
ax.text(13, 54.5, "Matched prompts", ha="center", va="center", fontsize=7.8, color=INK)
ax.text(13, 50.3, "AR / EN / HA / YO", ha="center", va="center", fontsize=7.0, color=BLUE)
agents = ((33, "Planner", icon_planner), (62, "Tool-Caller", icon_tool), (91, "Critic", icon_shield))
for bx, lab, ic in agents:
    box(ax, bx, 48, 22, 8.5)
    ic(ax, bx + 4.5, 52.25, 3.6)
    ax.text(bx + 13.5, 52.25, lab, ha="center", va="center", fontsize=8, color=INK)
arrow(ax, 26, 52.25, 33, 52.25); arrow(ax, 55, 52.25, 62, 52.25); arrow(ax, 84, 52.25, 91, 52.25)
for cx in (44, 102):
    box(ax, cx - 3.6, 60.4, 7.2, 4.2, fc=BLUE, ec=BLUE, r=0.8, z=4)
    ax.text(cx, 62.5, r"+ $v$", ha="center", va="center", fontsize=7.2, color="white",
            fontweight="bold", zorder=5)
    arrow(ax, cx, 60.4, cx, 56.6, c=BLUE)
ax.text(73, 62.5, "Baseline / Planner / Critic / Joint", ha="center", va="center", fontsize=6.6,
        color=MUTED)
arrow(ax, 113, 52.25, 117.5, 52.25)
line(ax, [117.5, 117.5], [46.5, 58]);
for yy, k in ((58, "veto"), (52.25, "allow"), (46.5, "nodec")):
    line(ax, [117.5, 119.5], [yy, yy])
    badge(ax, 122.2, yy, 4.6, k)

ax.text(73, 43.3, "Log plans, actions, and decisions", ha="center", va="center", fontsize=6.8,
        color=MUTED)
arrow(ax, 73, 41.6, 73, 39.4)
box(ax, 21, 33.6, 84, 5.6)
ax.text(63, 36.4, "Model | Language | Prompt ID | Condition", ha="center", va="center",
        fontsize=7.6, color=INK)
line(ax, [63, 63], [33.6, 31.6]); line(ax, [31, 95], [31.6, 31.6])
arrow(ax, 31, 31.6, 31, 29.6); arrow(ax, 95, 31.6, 95, 29.6)

box(ax, 2.5, 15.4, 59.5, 13.8, fc="white", ec=EDGE)
ax.text(5, 26.4, "Primary (four models)", ha="left", va="center", fontsize=7.6, color=BLUE,
        fontweight="bold")
ax.text(32.25, 21.9, r"Baseline  $\leftrightarrow$  Planner-only", ha="center", va="center",
        fontsize=8, color=INK)
ax.text(32.25, 18.0, "Same phase; matched prompts", ha="center", va="center", fontsize=6.6,
        color=MUTED)
box(ax, 66.5, 15.4, 63, 13.8, fc="white", ec=EDGE)
ax.text(127.5, 26.4, "Secondary site contrast (Llama)", ha="right", va="center", fontsize=7.6,
        color=BLUE, fontweight="bold")
ax.text(98, 21.9, r"Phase 1 $\alpha$=0  $\leftrightarrow$  Turbo $\alpha$=30", ha="center",
        va="center", fontsize=8, color=INK)
ax.text(98, 18.0, "Matched prompts; site / strength not isolated", ha="center", va="center",
        fontsize=6.6, color=MUTED)

line(ax, [2.5, 129.5], [12.6, 12.6], c=RULE, lw=0.6)
ax.text(66, 9.0, "Prompt-clustered bootstrap CI | Randomization test | Holm correction",
        ha="center", va="center", fontsize=7.2, color=INK)
ax.text(66, 4.6, "Failure + decision composition + parser validity + exact n",
        ha="center", va="center", fontsize=6.6, color=BLUE)
enforce(fig)
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(f"{OUT}/figure-2-method-2.pdf"); fig.savefig(f"{OUT}/figure-2-method-2.png", dpi=200)
plt.close(fig)
print("ok")
