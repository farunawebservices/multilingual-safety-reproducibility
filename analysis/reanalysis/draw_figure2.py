"""Figure 2 at native single-column size (3.03 in wide); units are inches."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = sys.argv[1]
plt.rcParams.update({"font.family": "Liberation Sans", "pdf.fonttype": 42})
TEAL, RED, OCHRE = "#2a7366", "#bf4f4b", "#a8701f"
BLUE, INK, MUTED = "#2c5f8a", "#1f2937", "#4b5563"
FILL, EDGE, RULE = "#eef3f7", "#c9d6e2", "#d5dbe1"
FS, FS_S = 8.0, 7.5
W, H = 3.03, 2.76
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


ax.add_patch(Rectangle((0.004, 0.004), W - 0.008, H - 0.008, fill=False, ec="#9aa3ad", lw=0.5))

# input
y = H - 0.18
box(0.10, y - 0.11, 2.05, 0.22, fc="white")
text(0.10 + 2.05 / 2, y, "Matched prompts: AR / EN / HA / YO", size=FS)

# steering markers + pipeline
yp = y - 0.48
aw, agap = 0.86, 0.13
ax0 = (W - (3 * aw + 2 * agap)) / 2
centers = []
for i, lab in enumerate(["Planner", "Tool-Caller", "Critic"]):
    bx = ax0 + i * (aw + agap)
    box(bx, yp - 0.14, aw, 0.28)
    text(bx + aw / 2, yp, lab)
    centers.append(bx + aw / 2)
    if i < 2:
        arrow(bx + aw, yp, bx + aw + agap, yp)
arrow(centers[0], y - 0.11, centers[0], yp + 0.14)
for cx in (centers[0], centers[2]):
    box(cx - 0.17, yp - 0.38, 0.34, 0.16, fc=BLUE, ec=BLUE, r=0.03, z=4)
    text(cx, yp - 0.30, r"+ $v$", size=FS_S, color="white", weight="bold")
    arrow(cx, yp - 0.22, cx, yp - 0.14, c=BLUE)

# log record
yr = yp - 0.58
arrow(W / 2, yp - 0.14, W / 2, yr + 0.11)
box(0.25, yr - 0.11, W - 0.50, 0.22)
text(W / 2, yr, "Log: model | language | prompt ID | decision", size=FS_S)

# two comparison boxes
yc = yr - 0.55
bw = (W - 0.30) / 2
line([W / 2, W / 2], [yr - 0.11, yr - 0.20]); line([0.10 + bw / 2, W - 0.10 - bw / 2], [yr - 0.20, yr - 0.20])
for k, (title, l1, l2) in enumerate((("Primary (4 models)", "Baseline vs.", "Planner-only, same phase"),
                                     ("Secondary (Llama)", r"Phase 1 $\alpha$=0 vs.", r"Turbo $\alpha$=30, by site"))):
    bx = 0.10 + k * (bw + 0.10)
    arrow(bx + bw / 2, yr - 0.20, bx + bw / 2, yc + 0.27)
    box(bx, yc - 0.27, bw, 0.54, fc="white")
    text(bx + bw / 2, yc + 0.15, title, size=FS, color=BLUE, weight="bold")
    text(bx + bw / 2, yc - 0.02, l1, size=FS_S)
    text(bx + bw / 2, yc - 0.17, l2, size=FS_S)

# inference footer
line([0.10, W - 0.10], [yc - 0.37, yc - 0.37], c=RULE, lw=0.5)
text(W / 2, yc - 0.50, "Prompt-clustered CI | randomization test | Holm", size=FS_S)
text(W / 2, yc - 0.66, "Failure + decision composition + exact n", size=FS_S, color=BLUE)

fig.savefig(f"{OUT}/figure-2-method-3.pdf")
fig.savefig(f"{OUT}/figure-2-method-3.png", dpi=300)
print("ok")
