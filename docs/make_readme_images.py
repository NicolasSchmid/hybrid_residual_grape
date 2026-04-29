"""Generate raster README diagrams with ImageMagick.

The diagrams are intentionally generated as PNGs, not SVGs, because they are
meant to be dropped into GitHub README pages and slides as bitmap teaching
figures. The drawing code is deterministic so labels stay exact.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"

MAGICK = os.environ.get("MAGICK", "magick")
CHALK = "/System/Library/Fonts/Supplemental/ChalkboardSE.ttc"
TITLE_FONT = "/System/Library/Fonts/Supplemental/Chalkduster.ttf"
CODE_FONT = "/System/Library/Fonts/Menlo.ttc"

BG = "#080c10"
PANEL = "#111820"
WHITE = "#f2f2ed"
MUTED = "#b8c0cc"
BLUE = "#42b7ff"
YELLOW = "#ffd54a"
GREEN = "#78e26f"
ORANGE = "#ff9d3b"
RED = "#ff6268"
PURPLE = "#c77dff"
GRAY = "#d8dde5"


class Diagram:
    def __init__(self, width: int, height: int, out_name: str):
        self.width = width
        self.height = height
        self.out_path = OUT / out_name
        self.cmd: list[str] = [
            MAGICK,
            "-size",
            f"{width}x{height}",
            f"xc:{BG}",
            "-alpha",
            "set",
            "-antialias",
        ]

    def draw(self, spec: str) -> None:
        self.cmd += ["-draw", spec]

    def rect(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: str,
        *,
        fill: str = PANEL,
        width: int = 5,
        radius: int = 24,
        dash: bool = False,
    ) -> None:
        dash_part = "stroke-dasharray 12 12 " if dash else ""
        self.draw(
            f"fill '{fill}' stroke '{color}' stroke-width {width} "
            f"{dash_part}roundrectangle {x1},{y1} {x2},{y2} {radius},{radius}"
        )

    def line(self, x1: int, y1: int, x2: int, y2: int, color: str, width: int = 5) -> None:
        self.draw(f"stroke '{color}' stroke-width {width} line {x1},{y1} {x2},{y2}")

    def arrow(self, x1: int, y1: int, x2: int, y2: int, color: str, width: int = 5) -> None:
        self.line(x1, y1, x2, y2, color, width)
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) >= abs(dy):
            sign = 1 if dx >= 0 else -1
            pts = [
                (x2, y2),
                (x2 - sign * 26, y2 - 15),
                (x2 - sign * 26, y2 + 15),
            ]
        else:
            sign = 1 if dy >= 0 else -1
            pts = [
                (x2, y2),
                (x2 - 15, y2 - sign * 26),
                (x2 + 15, y2 - sign * 26),
            ]
        self.draw(
            "fill '{0}' stroke '{0}' stroke-width 1 polygon {1}".format(
                color,
                " ".join(f"{x},{y}" for x, y in pts),
            )
        )

    def text(
        self,
        x: int,
        y: int,
        text: str,
        *,
        color: str = WHITE,
        size: int = 34,
        font: str = CHALK,
    ) -> None:
        self.cmd += [
            "-font",
            font,
            "-pointsize",
            str(size),
            "-fill",
            color,
            "-annotate",
            f"+{x}+{y}",
            text,
        ]

    def multiline(
        self,
        x: int,
        y: int,
        lines: list[str] | str,
        *,
        color: str = WHITE,
        size: int = 31,
        font: str = CHALK,
        leading: float = 1.32,
        wrap_chars: int | None = None,
    ) -> int:
        if isinstance(lines, str):
            raw_lines = lines.splitlines()
        else:
            raw_lines = lines
        rendered: list[str] = []
        for line in raw_lines:
            if wrap_chars:
                rendered.extend(textwrap.wrap(line, width=wrap_chars) or [""])
            else:
                rendered.append(line)
        dy = int(size * leading)
        for i, line in enumerate(rendered):
            self.text(x, y + i * dy, line, color=color, size=size, font=font)
        return y + len(rendered) * dy

    def title(self, text: str) -> None:
        self.text(70, 90, text, color=WHITE, size=54, font=TITLE_FONT)

    def save(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        self.cmd += [str(self.out_path)]
        env = os.environ.copy()
        env["XDG_CACHE_HOME"] = "/private/tmp/fontconfig-cache"
        subprocess.run(self.cmd, check=True, env=env)
        print(self.out_path)


def closed_loop_overview() -> None:
    d = Diagram(2200, 1300, "closed_loop_overview.png")
    d.title("Closed-loop Fock-state preparation")
    d.multiline(
        80,
        150,
        "Goal: choose one 80-parameter pulse u that prepares |n> photons in the cavity.",
        color=MUTED,
        size=32,
    )

    boxes = [
        (90, 275, 430, 510, BLUE, "Pulse coefficients", ["u in R^80", "4 real channels", "20 quadratic B-splines"]),
        (565, 275, 905, 510, YELLOW, "Nominal physics", ["FockPhysicsModel", "unitary Hamiltonian", "JAX differentiable"]),
        (1040, 275, 1380, 510, PURPLE, "Hybrid surrogate", ["logit P_phys", "+ support * f_RBF", "optimized by GRAPE"]),
        (1515, 275, 1855, 510, GREEN, "AD-GRAPE", ["L-BFGS on u", "jax.grad", "through model", "warm start"]),
    ]
    for x1, y1, x2, y2, color, title, body in boxes:
        d.rect(x1, y1, x2, y2, color)
        d.text(x1 + 35, y1 + 65, title, color=color, size=37)
        d.multiline(x1 + 35, y1 + 120, body, color=WHITE, size=27)

    for x1, y1, x2, y2 in [(430, 392, 565, 392), (905, 392, 1040, 392), (1380, 392, 1515, 392)]:
        d.arrow(x1, y1, x2, y2, WHITE, 5)

    d.rect(1515, 710, 1855, 975, ORANGE, fill="#14191f")
    d.text(1550, 775, "Experiment", color=ORANGE, size=39)
    d.multiline(
        1550,
        835,
        ["play proposed pulse", "selective photon test", "successes / shots"],
        color=WHITE,
        size=29,
    )
    d.arrow(1685, 510, 1685, 710, ORANGE, 5)

    d.rect(1040, 710, 1380, 975, RED, fill="#14191f", dash=True)
    d.text(1075, 775, "Hidden truth", color=RED, size=39)
    d.multiline(
        1075,
        835,
        ["only in notebook", "detuning + Kerr", "T1/T2 decay"],
        color=WHITE,
        size=29,
    )
    d.arrow(1515, 842, 1380, 842, RED, 5)

    d.rect(565, 710, 905, 975, BLUE, fill="#14191f")
    d.text(600, 775, "Dataset D", color=BLUE, size=39)
    d.multiline(
        600,
        835,
        ["(u_i, s_i, N_i)", "cache P_phys(u_i)", "append every round"],
        color=WHITE,
        size=29,
    )
    d.line(1515, 945, 1515, 1030, BLUE, 5)
    d.line(1515, 1030, 905, 1030, BLUE, 5)
    d.arrow(905, 1030, 905, 945, BLUE, 5)
    d.arrow(735, 710, 735, 510, BLUE, 5)

    d.rect(90, 710, 430, 975, GREEN, fill="#14191f")
    d.text(125, 775, "Fit residual", color=GREEN, size=39)
    d.multiline(
        125,
        835,
        ["RBF ridge model", "learns local mismatch", "regularized + gated"],
        color=WHITE,
        size=29,
    )
    d.arrow(565, 842, 430, 842, GREEN, 5)
    d.arrow(260, 710, 1040, 510, GREEN, 5)

    d.rect(90, 1080, 1855, 1205, GRAY, fill="#0e141b", width=3, radius=18)
    d.multiline(
        130,
        1133,
        "The optimizer never sees the hidden true Hamiltonian. It sees f_phys plus a residual fitted from binary measurements.",
        color=WHITE,
        size=34,
        wrap_chars=92,
    )
    d.save()


def hybrid_model_math() -> None:
    d = Diagram(2200, 1300, "hybrid_model_math.png")
    d.title("What the RBF residual learns")

    d.rect(80, 210, 620, 470, YELLOW)
    d.text(125, 285, "1. Nominal physics", color=YELLOW, size=42)
    d.multiline(
        125,
        350,
        ["P_phys(u)", "= Pr(cavity has n photons)", "from differentiable JAX evolution"],
        color=WHITE,
        size=28,
        wrap_chars=40,
    )

    d.rect(830, 210, 1370, 470, BLUE)
    d.text(875, 285, "2. Measurements", color=BLUE, size=42)
    d.multiline(
        875,
        350,
        ["y_i = s_i / N_i", "s_i ~ Binomial(N_i, P_true(u_i))"],
        color=WHITE,
        size=28,
        wrap_chars=39,
    )

    d.rect(1580, 210, 2120, 470, GREEN)
    d.text(1625, 285, "3. Residual target", color=GREEN, size=42)
    d.multiline(
        1625,
        350,
        ["r_i = logit(y_i)", "      - logit(P_phys(u_i))", "fit this, not P_n from scratch"],
        color=WHITE,
        size=27,
        font=CODE_FONT,
        wrap_chars=39,
    )

    d.arrow(620, 340, 830, 340, WHITE)
    d.arrow(1370, 340, 1580, 340, WHITE)

    d.rect(150, 650, 1040, 940, PURPLE)
    d.text(195, 725, "RBF ridge model", color=PURPLE, size=45)
    d.multiline(
        195,
        800,
        [
            "f_RBF(u) = b + sum_k w_k phi_k(u)",
            "phi_k(u) = exp(-||u-c_k||^2 / 2 l^2)",
            "centers c_k are measured pulses",
            "ridge penalty keeps correction smooth",
        ],
        color=WHITE,
        size=26,
        font=CODE_FONT,
        wrap_chars=55,
    )

    d.rect(1170, 650, 2070, 940, ORANGE)
    d.text(1215, 725, "Support-gated hybrid prediction", color=ORANGE, size=39)
    d.multiline(
        1215,
        800,
        [
            "logit(P_hybrid) = logit(P_phys)",
            "                + support(u) * clip(f_RBF)",
            "far from data: support -> 0",
            "so the model falls back to physics",
        ],
        color=WHITE,
        size=25,
        font=CODE_FONT,
        wrap_chars=55,
    )
    d.arrow(1040, 795, 1170, 795, WHITE)

    # Small bump sketch.
    base_y = 1130
    d.line(230, base_y, 1930, base_y, GRAY, 3)
    d.line(230, base_y, 230, 1010, GRAY, 3)
    for cx, amp, col in [(520, 95, BLUE), (870, 65, GREEN), (1210, 105, PURPLE), (1580, 55, ORANGE)]:
        d.draw(
            f"fill none stroke '{col}' stroke-width 5 "
            f"path 'M {cx-180},{base_y} C {cx-115},{base_y} {cx-90},{base_y-amp} {cx},{base_y-amp} "
            f"C {cx+90},{base_y-amp} {cx+115},{base_y} {cx+180},{base_y}'"
        )
        d.draw(f"fill '{col}' stroke none circle {cx},{base_y} {cx+8},{base_y}")
    d.text(250, 1015, "local RBF bumps over pulse space", color=MUTED, size=30)
    d.text(1590, 1195, "u: 80-dimensional pulse", color=MUTED, size=30)

    d.save()


def code_map() -> None:
    d = Diagram(2200, 1500, "code_map.png")
    d.title("Code map: where each idea lives")

    modules = [
        (80, 220, 560, 455, BLUE, "configuration + config.py", ["load chi, Kerr, T1/T2", "unit conversions", "hardware numbers"]),
        (80, 560, 560, 820, YELLOW, "physics.py", ["SimulationConfig", "PhysicsParams", "FockPhysicsModel", "B-splines -> Hamiltonian -> P_n"]),
        (710, 220, 1190, 455, GREEN, "residual.py", ["RBFResidualConfig", "fit_rbf_residual", "support-gated correction"]),
        (710, 560, 1190, 820, PURPLE, "grape.py", ["HybridGrapeConfig", "bounded controls", "Optax L-BFGS", "jax.grad objective"]),
        (1340, 220, 1820, 455, ORANGE, "calibration.py", ["binomial likelihood", "fit physical params", "adaptive shots + LCB"]),
        (1340, 560, 1820, 820, RED, "experiment.py", ["sample binomial shots", "append dataset", "local noisy pulse batch"]),
        (80, 950, 960, 1155, RED, "hybrid_residual_grape.ipynb", ["RBF residual closed loop", "diagnostics + best pulse"]),
        (1120, 950, 2100, 1155, ORANGE, "adaptive_calibrated_grape.ipynb", ["physical calibration loop", "oracle true-GRAPE", "adaptive shot allocation"]),
    ]
    for x1, y1, x2, y2, color, title, body in modules:
        d.rect(x1, y1, x2, y2, color)
        d.text(x1 + 35, y1 + 62, title, color=color, size=34)
        d.multiline(x1 + 35, y1 + 118, body, color=WHITE, size=24, wrap_chars=38)

    d.arrow(560, 338, 710, 338, WHITE)
    d.arrow(560, 690, 710, 690, WHITE)
    d.arrow(1190, 338, 1340, 338, WHITE)
    d.arrow(1190, 690, 1340, 690, WHITE)
    d.arrow(1580, 455, 1580, 560, WHITE)
    d.arrow(950, 820, 600, 950, PURPLE)
    d.arrow(1580, 820, 1580, 950, ORANGE)

    d.rect(225, 1250, 1975, 1455, GRAY, fill="#0e141b", width=4, radius=18)
    d.text(275, 1310, "Mental model", color=WHITE, size=38)
    d.multiline(
        275,
        1368,
        [
            "RBF notebook: learn a local residual over pulses. Calibrated notebook: fit physical nuisance parameters, then run GRAPE.",
        ],
        color=MUTED,
        size=24,
        wrap_chars=86,
    )

    d.save()


def plot_diagnostics() -> None:
    d = Diagram(2200, 1300, "reading_the_diagnostics.png")
    d.title("How to tell if the correction matters")

    panels = [
        (95, 225, 665, 575, YELLOW, "Physics-only baseline", ["Run full GRAPE on nominal f_phys.", "Then evaluate that pulse on hidden truth.", "This is the fair reference line."]),
        (815, 225, 1385, 575, GREEN, "Hybrid progress", ["Track best measured P_n.", "Also plot -log10(1-P_n).", "Small fidelity changes become visible."]),
        (1535, 225, 2105, 575, PURPLE, "Residual significance", ["Compare true - physics", "against hybrid - physics.", "They should align if RBF helps."]),
    ]
    for x1, y1, x2, y2, color, title, body in panels:
        d.rect(x1, y1, x2, y2, color)
        d.text(x1 + 35, y1 + 70, title, color=color, size=39)
        d.multiline(x1 + 35, y1 + 135, body, color=WHITE, size=27, wrap_chars=36)

    # Stylized diagnostic plots.
    d.rect(180, 740, 640, 1110, BLUE, fill="#0b1118", width=3, radius=10)
    d.line(245, 1040, 585, 1040, GRAY, 3)
    d.line(245, 1040, 245, 800, GRAY, 3)
    d.draw(f"fill none stroke '{YELLOW}' stroke-width 6 path 'M 260,1000 C 330,910 410,850 570,825'")
    d.draw(f"fill none stroke '{GREEN}' stroke-width 6 path 'M 260,1010 C 335,900 420,810 570,790'")
    d.text(270, 785, "-log10(1 - P_n)", color=MUTED, size=24)
    d.text(365, 1095, "measurements", color=MUTED, size=24)

    d.rect(870, 740, 1330, 1110, GREEN, fill="#0b1118", width=3, radius=10)
    d.line(930, 1040, 1270, 1040, GRAY, 3)
    d.line(930, 1040, 930, 800, GRAY, 3)
    d.line(955, 1015, 1245, 825, WHITE, 3)
    for px, py, col in [(980, 990, BLUE), (1035, 950, GREEN), (1090, 925, GREEN), (1155, 880, ORANGE), (1215, 850, PURPLE)]:
        d.draw(f"fill '{col}' stroke none circle {px},{py} {px+9},{py}")
    d.text(955, 785, "hybrid - physics", color=MUTED, size=24)
    d.text(1010, 1095, "true - physics", color=MUTED, size=24)

    d.rect(1560, 740, 2020, 1110, RED, fill="#0b1118", width=3, radius=10)
    d.line(1620, 1040, 1960, 1040, GRAY, 3)
    d.line(1620, 1040, 1620, 800, GRAY, 3)
    d.draw(f"fill none stroke '{RED}' stroke-width 6 path 'M 1640,860 C 1720,960 1820,980 1950,900'")
    d.draw(f"fill none stroke '{BLUE}' stroke-width 6 path 'M 1640,830 C 1720,840 1830,860 1950,850'")
    d.text(1640, 785, "bad sign: RBF pulls away", color=MUTED, size=24)
    d.text(1725, 1095, "pulse path", color=MUTED, size=24)

    d.multiline(
        160,
        1210,
        "Good correction: hybrid prediction moves toward measured/true data and improves held-out likelihood. Bad correction: physics and truth agree, but the residual invents structure.",
        color=WHITE,
        size=31,
        wrap_chars=120,
    )
    d.save()


if __name__ == "__main__":
    closed_loop_overview()
    hybrid_model_math()
    code_map()
    plot_diagnostics()
