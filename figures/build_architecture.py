"""Generate the editable SVG architecture from the corrected revision model.

The SVG is the manuscript source; make_diagrams.py exports matching PDF/PNG.
No model fitting is performed. All expressions follow revision_models.py.
"""

from html import escape
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
INK = "#183243"
MUTED = "#607582"
BLUE = "#0072B2"
TEAL = "#008878"
ORANGE = "#D55E00"
BORDER = "#D9E3E8"
W, H = 1600, 1160
parts = []


def add(value):
    parts.append(value)


def text(x, y, value, size=23, color=INK, weight=400, anchor="start", math=False):
    """Sub/superscript notation becomes editable native SVG tspans."""
    spans = []
    for token in re.split(r"([_^]\{[^}]+\})", value):
        if token.startswith(("_{", "^{")):
            shift = "sub" if token[0] == "_" else "super"
            spans.append(f'<tspan baseline-shift="{shift}" font-size="70%">{escape(token[2:-1])}</tspan>')
        else:
            spans.append(escape(token))
    font = "Cambria, Times New Roman, serif" if math else "Arial, Helvetica, sans-serif"
    add(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{"".join(spans)}</text>')


def rect(x, y, w, h, fill="white", stroke=BORDER, radius=10, sw=1.6):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def path(d, color=MUTED, width=2, arrow=False, dash=None):
    attrs = ' marker-end="url(#arrow)"' if arrow else ""
    attrs += f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{attrs}/>')


def section(letter, x, label):
    text(x, 151, letter, 25, BLUE, 700)
    text(x + 36, 151, label, 23, INK, 700)


def build():
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="183mm" height="{183*H/W:.3f}mm" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title description">')
    add('<title id="title">EnergyPhasePINN: context-conditioned energy and full force differentiation</title>')
    add('<desc id="description">Standardized grain size, temperature and indentation velocity determine constrained energy and gate parameters. A single indentation-depth variable enters all energy branches, gates and a residual energy network. The gated scalar loading potential is differentiated including gate derivatives. Force-data fitting and physics penalties train the mean model; a separate frozen-mean uncertainty stage is shown below.</desc>')
    add(f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 1L9 5L0 9Z" fill="{MUTED}"/></marker></defs>')
    rect(0, 0, W, H, stroke="none", radius=0)

    text(40, 54, "EnergyPhasePINN", 39, INK, 700)
    text(40, 90, "A context-conditioned loading potential with force obtained by differentiation", 24, MUTED)
    text(1560, 53, "AlCuNiTiZr · nanoindentation", 23, MUTED, anchor="end")
    path("M40 111H1560", BORDER, 1.6)

    section("a", 40, "Inputs & conditioning")
    section("b", 510, "Energy construction")
    section("c", 1165, "Force prediction")

    # The parameter branch is independent of indentation depth.
    rect(40, 181, 370, 219, "#F0F6FA")
    text(62, 216, "Material & loading context", 25, INK, 700)
    text(62, 252, "(g, T, v)", 30, math=True)
    text(62, 287, "Grain size · temperature · velocity", 20.5, MUTED)
    text(62, 325, "Training-set standardization → x", 22, MUTED)
    path("M62 343H388", BORDER, 1.3)
    text(62, 379, "MLP  3 → 128 → 128 → 12", 25, BLUE, 700)
    path("M225 400V435", arrow=True)

    rect(40, 436, 370, 207)
    text(62, 472, "Constrained parameters", 25, INK, 700)
    text(62, 514, "k, F_{y}, a, b, c > 0", 27, math=True)
    text(62, 553, "δ_{h} > δ_{y} > 0;   1 < m < 4", 27, math=True)
    text(62, 592, "w_{1}, w_{2}, b_{1}, b_{2} > 0", 27, math=True)
    text(62, 624, "Softplus / bounded sigmoid transforms", 19.5, MUTED)

    # Parameters feed both the analytical energies and their transition gates.
    path("M410 470H449V221H509", arrow=True)
    path("M449 470V568H509", arrow=True)
    text(437, 428, "θ(x)", 24, BLUE, anchor="end", math=True)

    rect(40, 683, 370, 94, "#F6F8F9")
    text(62, 722, "Indentation depth  δ", 29, INK, 700)
    text(62, 754, "One shared differentiation variable", 20, MUTED)
    # Depth reaches all three depth-dependent modules. The bus crosses no
    # parameter node; its junction dots identify the intended connections.
    path("M479 581Q493 568 479 555", "white", 8)
    path("M410 737H479V581Q493 568 479 555V485H509", arrow=True)
    path("M479 641H509", arrow=True)
    path("M479 737V776H509", arrow=True)
    for cy in (641, 737):
        add(f'<circle cx="479" cy="{cy}" r="3.5" fill="{MUTED}"/>')

    # Three analytical branches, with an explicitly regularized depth.
    rect(510, 181, 580, 320)
    text(533, 216, "Analytical energy branches", 25, INK, 700)
    for y, color, fill, label, eq in [
        (232, BLUE, "#F0F6FA", "Elastic", "U_{e} = (2/5) k d^{5/2}"),
        (299, TEAL, "#EDF7F3", "Plastic", "U_{p} = F_{y}δ + (a/2) Δ^{2} + (b/3) Δ^{3}"),
        (366, ORANGE, "#FDF3EC", "Hardening", "U_{h} = F_{y}δ + c Δ^{m+1}/(m + 1)"),
    ]:
        rect(529, y, 542, 58, fill, "none", 5)
        path(f"M530 {y+7}V{y+51}", color, 4)
        text(546, y + 36, label, 21, color, 700)
        text(677, y + 37, eq, 26, math=True)
    text(534, 455, "Δ = softplus_{10}(δ − δ_{y}) + ε", 25, math=True)
    text(534, 486, "d = max(δ, ε);  ε = 10^{−6}", 23, MUTED, math=True)

    rect(510, 529, 580, 156, "#F6F8F9")
    text(533, 565, "Depth-dependent sigmoid gates", 25, INK, 700)
    text(543, 611, "π_{1} = sigmoid[(δ − δ_{y})/w_{1} − b_{1}]", 28, math=True)
    text(543, 654, "π_{2} = sigmoid[(δ − δ_{h})/w_{2} − b_{2}]", 28, math=True)

    rect(510, 715, 580, 91, "#F0F6FA")
    text(533, 748, "Learned residual energy  U_{res}(δ, x)", 25, INK, 700)
    text(533, 785, "MLP  [d, d^{3/2}, x] → 128 → 128 → 1", 25, BLUE)

    # All three contributions enter the one scalar potential.
    path("M1090 335H1110V285H1164", arrow=True)
    path("M1090 606H1130V386H1164", arrow=True)
    path("M1090 761H1150V474H1164", arrow=True)

    rect(1165, 234, 395, 273, "#F0F6FA", "#B8CFDC", 10, 2)
    text(1188, 270, "Gated loading potential", 25, INK, 700)
    text(1362, 321, "U = Σ_{j} w_{j}U_{j} + U_{res}", 34, anchor="middle", math=True)
    path("M1189 344H1536", "#C7DAE5", 1.4)
    text(1190, 379, "w_{e} = 1 − π_{1}", 28, BLUE, math=True)
    text(1190, 418, "w_{p} = π_{1}(1 − π_{2})", 28, TEAL, math=True)
    text(1190, 457, "w_{h} = π_{1}π_{2}", 28, ORANGE, math=True)
    text(1535, 489, "Σ_{j} w_{j} = 1", 24, MUTED, anchor="end", math=True)
    path("M1362 507V563", arrow=True)
    text(1379, 547, "Full autograd", 21, MUTED)

    rect(1165, 566, 395, 132, "#EDF7F3", "#8DBAAE", 10, 2)
    text(1362, 617, "F̂ = ∂U / ∂δ", 43, TEAL, anchor="middle", math=True)
    text(1362, 657, "Differentiate branches, gates", 22, INK, anchor="middle")
    text(1362, 684, "and residual energy", 22, INK, anchor="middle")
    path("M1362 698V741", arrow=True)
    rect(1165, 744, 395, 62, "white")
    text(1362, 784, "Predicted force–depth response", 23, INK, 700, "middle")

    # The product-rule term is deliberately made visible, not buried in a caption.
    path("M40 842H1560", BORDER, 1.6)
    text(40, 879, "Full derivative", 24, INK, 700)
    text(40, 912, "j ∈ {e, p, h}", 22, MUTED, math=True)
    text(376, 901, "F̂ = Σ_{j} w_{j}U′_{j}", 36, INK, math=True)
    text(745, 901, "+ Σ_{j} w′_{j}U_{j}", 36, TEAL, math=True)
    text(1110, 901, "+ U′_{res}", 36, INK, math=True)
    text(455, 939, "Branch derivatives", 22, MUTED, anchor="middle")
    text(879, 939, "Gate derivatives", 22, TEAL, 700, "middle")
    text(1241, 939, "Residual derivative", 22, MUTED, anchor="middle")
    text(1560, 972, "′ = ∂/∂δ at fixed context", 20, MUTED, anchor="end", math=True)

    path("M40 991H1560", BORDER, 1.6)
    text(40, 1027, "01  Train the mean model", 23, INK, 700)
    text(40, 1064, "Force MSE + physics penalties + residual L² regularization", 22)
    text(40, 1096, "Hertzian response · monotonicity · force positivity · gate timing/order", 20, MUTED)
    text(40, 1125, "SiLU hidden activations in both MLPs; context also enters the residual.", 19.5, MUTED)
    path("M891 1013V1127", BORDER, 1.4)
    text(922, 1027, "02  Fit uncertainty after freezing the mean", 23, INK, 700)
    text(922, 1064, "[d, d^{3/2}, x] → SigmaHead → σ > 0", 24)
    text(922, 1096, "Residual Gaussian NLL + scale regularization", 22, MUTED)
    text(922, 1125, "Separate MLP: 5 → 96 → 96 → 1", 21, MUTED)

    add("</svg>")
    destination = HERE / "architecture.svg"
    destination.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(destination.name)


if __name__ == "__main__":
    build()
