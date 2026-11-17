import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Design a spiral object

    A `Spiral` object is a class that can be used to progressively build spiral-shaped objects amenable to DNA origami designs. It starts empty, and different segments from pre-made templates can be progressively added, with desired characteristics such as lengths or radii. The following notebook shows how this works.
    """)
    return


@app.cell
def _():
    from autorigami.spiral_generation import Spiral, load_curves
    from autorigami.viz import show_curves, show_spiral

    return Spiral, load_curves, show_curves, show_spiral


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Straight segments

    The first segment needs a length and two radii. One turn is 2.65 nm high, which keeps the turns a small distance above the 2.6 nm limit.
    """)
    return


@app.cell
def _(Spiral, show_spiral):
    spiral = Spiral()
    stem = spiral.add_straight_segment(
        40.0, x_radius=16.0, y_radius=16.0, rise_per_turn=2.65
    )
    show_spiral(spiral)
    return spiral, stem


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `validate` applies a validation algorithm against both of our constraints: the turning angle between consecutive edges (3.25° maximum) and the distance between two parts of the helix (2.6 nm minimum).
    """)
    return


@app.cell
def _(spiral):
    print(spiral.validate())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Y segments

    A Y segment divides a round tube into two arms. It has a middle part, where the tube becomes wide, and a top part, where the two arms become round. Each part has its own opening angle: the angle between the vertical axis and the axis of an arm.

    The segment starts at the terminus of the stem.
    """)
    return


@app.cell
def _(show_spiral, spiral, stem):
    y = stem.termini[0].add_y_segment(
        middle_length=40.0,
        top_length=40.0,
        middle_opening_angle_deg=38.66,
        top_opening_angle_deg=23.71,
    )
    show_spiral(spiral, highlight=y)
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Segments on the arms

    A Y segment has two termini, one for each arm. A straight segment on a terminus keeps the radii of that terminus.
    """)
    return


@app.cell
def _(show_spiral, spiral, y):
    arm = y.termini[0].add_straight_segment(25.0)
    show_spiral(spiral, highlight=arm)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Funnel segments

    A funnel segment changes the radii linearly, from the radii of the terminus to the end radii that you give.
    """)
    return


@app.cell
def _(show_spiral, spiral, y):
    funnel = y.termini[1].add_funnel_segment(25.0, end_x_radius=10.0, end_y_radius=10.0)
    show_spiral(spiral, highlight=funnel)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `spiral.segments` method lists all segments, so `spiral.segments[1].termini[0]` is the same terminus as `y.termini[0]`.
    """)
    return


@app.cell
def _(spiral):
    [type(segment).__name__ for segment in spiral.segments]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Curves

    One DNA helix cannot follow two arms. Thus the segment on `y.termini[1]` starts a second curve. The unique curves that compose a design can be visualized with the `spiral.curves` method.
    """)
    return


@app.cell
def _(show_curves, spiral):
    curves = spiral.curves()
    show_curves(curves)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Validate

    `validate` on a `spiral` object with multiple curves now examines each curve, and also the distance between each pair of curves.
    """)
    return


@app.cell
def _(spiral):
    result = spiral.validate()
    print(result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This spiral is not valid, and that is the expected result at this stage. The middle part of a Y segment has sharp corners at its waist, and its turns are nearer than 2.6 nm. Gravity packing will repair these faults: see the next notebook.

    ## Save

    `save` writes the curves to one file in numpy format, and `load_curves` reads them again.
    """)
    return


@app.cell
def _(load_curves, spiral):
    spiral.save("../outputs/spiral_1.npz")
    [curve.shape for curve in load_curves("../outputs/spiral_1.npz")]
    return


if __name__ == "__main__":
    app.run()
