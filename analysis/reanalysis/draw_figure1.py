"""Figure 1 drawn at native single-column size (7.7 cm = 3.03 in wide).
Axis units are inches, so fontsize values are the printed point sizes."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle
from matplotlib import font_manager

OUT = sys.argv[1]
plt.rcParams.update({"font.family": "Liberation Sans", "pdf.fonttype": 42})

TEAL, RED, OCHRE = "#2a7366", "#bf4f4b", "#a8701f"
BLUE, INK, MUTED = "#2c5f8a", "#1f2937", "#4b5563"
FILL, EDGE, RULE = "#eef3f7", "#c9d6e2", "#d5dbe1"
FS, FS_S = 8.0, 7.5          # body and small label sizes (pt)

W, H = 3.03, 3.02
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, fc=FILL, ec=EDGE, lw=0.6, r=0.04, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def line(xs, ys, c=MUTED, lw=0.6, z=2):
    ax.plot(xs, ys, color=c, lw=lw, zorder=z, solid_capstyle="butt")


def arrow(x0, y0, x1, y1, c=MUTED, lw=0.6):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, mutation_scale=5,
                                shrinkA=0, shrinkB=0), zorder=3)


def text(x, y, s, size=FS, color=INK, ha="center", weight="normal"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center", fontweight=weight, zorder=6)


def badge(x, y, s, kind):
    c = {"veto": TEAL, "allow": RED, "nodec": OCHRE}[kind]
    box(x - s / 2, y - s / 2, s, s, fc=c, ec=c, r=s * 0.15, z=4)
    if kind == "veto":
        ax.add_patch(Rectangle((x - s * .28, y - s * .28), s * .56, s * .56, fill=False,
                               ec="white", lw=0.7, zorder=5))
        line([x - s * .2, x + s * .2], [y, y], c="white", lw=0.7, z=6)
    elif kind == "allow":
        ax.annotate("", xy=(x + s * .3, y), xytext=(x - s * .3, y),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=0.7, mutation_scale=4,
                                    shrinkA=0, shrinkB=0), zorder=6)
    else:
        text(x, y, "?", size=FS, color="white", weight="bold")


def toggle(x, y, s, on):
    if on:
        box(x - s / 2, y - s / 2, s, s, fc=BLUE, ec=BLUE, r=s * .18, z=4)
        text(x, y, "+", size=FS, color="white", weight="bold")
    else:
        box(x - s / 2, y - s / 2, s, s, fc="white", ec=EDGE, r=s * .18, z=4)
        text(x, y + .01, "–", size=FS_S, color=MUTED)


ax.add_patch(Rectangle((0.004, 0.004), W - 0.008, H - 0.008, fill=False, ec="#9aa3ad", lw=0.5))

# ---------------- (a) languages -> pipeline -> decisions ----------------
y_top = H - 0.14
text(0.08, y_top, r"$\bf{(a)}$ Same harmful request in four languages", size=FS, ha="left")
langs = ["English", "Arabic", "Hausa", "Yoruba"]
cw, gap = 0.64, 0.07
x0 = (W - (4 * cw + 3 * gap)) / 2
yl = y_top - 0.25
for i, L in enumerate(langs):
    bx = x0 + i * (cw + gap)
    box(bx, yl - 0.09, cw, 0.18, ec=FILL)
    text(bx + cw / 2, yl, L, size=FS, color=BLUE, weight="bold")
centers = [x0 + i * (cw + gap) + cw / 2 for i in range(4)]
line([centers[0], centers[-1]], [yl - 0.16, yl - 0.16])
for c in centers:
    line([c, c], [yl - 0.09, yl - 0.16])
arrow(W / 2, yl - 0.16, W / 2, yl - 0.24)

ya = yl - 0.40
aw, agap = 0.86, 0.13
ax0 = (W - (3 * aw + 2 * agap)) / 2
for i, lab in enumerate(["Planner", "Tool-Caller", "Critic"]):
    bx = ax0 + i * (aw + agap)
    box(bx, ya - 0.14, aw, 0.28)
    text(bx + aw / 2, ya, lab, size=FS)
    if i < 2:
        arrow(bx + aw, ya, bx + aw + agap, ya)
cx = ax0 + 2 * (aw + agap) + aw / 2
yb = ya - 0.38
line([cx, cx], [ya - 0.14, yb + 0.16])
outs = [("veto", "VETO"), ("allow", "ALLOW"), ("nodec", "NO-DECISION")]
bxs = [0.42, 1.27, 2.12]
line([bxs[0], cx], [yb + 0.16, yb + 0.16])
for bx, (k, lab) in zip(bxs, outs):
    arrow(bx, yb + 0.16, bx, yb + 0.09)
    badge(bx, yb, 0.16, k)
    text(bx + 0.12, yb, lab, size=FS_S, ha="left")

line([0.08, W - 0.08], [yb - 0.17, yb - 0.17], c=RULE, lw=0.5)

# ---------------- (b) decision composition  |  (c) intervention sites ----
yt = yb - 0.32
text(0.08, yt, r"$\bf{(b)}$ Composition", size=FS, ha="left")
text(1.66, yt, r"$\bf{(c)}$ Intervention sites", size=FS, ha="left")
line([1.58, 1.58], [0.06, yt + 0.08], c=RULE, lw=0.5)

bx0, bw = 0.18, 1.30
for yy, lab, segs in ((yt - 0.30, "1", (60, 35, 5)), (yt - 0.62, "2", (60, 5, 35))):
    text(0.10, yy, lab, size=FS_S)
    xs = bx0
    for val, c in zip(segs, (TEAL, RED, OCHRE)):
        w = bw * val / 100
        ax.add_patch(Rectangle((xs, yy - 0.09), w, 0.18, fc=c, ec="white", lw=0.4, zorder=3))
        if val >= 30:
            text(xs + w / 2, yy, str(val), size=FS_S, color="white", weight="bold")
        xs += w
    line([bx0 + bw * .6, bx0 + bw * .6, bx0 + bw, bx0 + bw], [yy + .11, yy + .14, yy + .14, yy + .11],
         c=INK, lw=0.5)
text(bx0 + bw * .8, yt - 0.11, "40% failure", size=FS_S, color=MUTED)
yf = yt - 0.92
text(0.18, yf, "Failure =", size=FS_S, ha="left")
badge(0.80, yf, 0.16, "allow")
text(0.95, yf, "+", size=FS_S, color=MUTED)
badge(1.10, yf, 0.16, "nodec")
text(0.18, yf - 0.22, "Same failure,", size=FS_S, color=MUTED, ha="left")
text(0.18, yf - 0.37, "different mechanism", size=FS_S, color=MUTED, ha="left")

colP, colC = 2.47, 2.83
text(colP, yt - 0.21, "Plan.", size=FS_S, color=MUTED)
text(colC, yt - 0.21, "Critic", size=FS_S, color=MUTED)
rows = (("Baseline", 0, 0), ("Planner-only", 1, 0), ("Critic-only", 0, 1), ("Joint", 1, 1))
for k, (lab, p, c) in enumerate(rows):
    yy = yt - 0.42 - k * 0.21
    text(1.66, yy, lab, size=FS_S, ha="left")
    toggle(colP, yy, 0.15, p); toggle(colC, yy, 0.15, c)
text(1.66, yt - 1.25, "+ = steered", size=FS_S, color=BLUE, ha="left")

fig.savefig(f"{OUT}/figure-1-concept-3.pdf")
fig.savefig(f"{OUT}/figure-1-concept-3.png", dpi=300)
print("ok")
