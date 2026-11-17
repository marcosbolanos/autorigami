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
    # 2. Pack a spiral under gravity

    This notebook reads the spiral that notebook 1 saved, packs it along z, and validates the result.

    All lengths are in nm. Each 3D view is interactive: drag to rotate, scroll to zoom.
    """)
    return


@app.cell
def _():
    from autorigami.packing import pack_curves_under_gravity
    from autorigami.spiral_generation import load_curves, save_curves, validate_curves
    from autorigami.viz import show_curves

    return (
        load_curves,
        pack_curves_under_gravity,
        save_curves,
        show_curves,
        validate_curves,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load

    The spiral has two curves, because one DNA helix cannot follow the two arms of the Y segment.
    """)
    return


@app.cell
def _(load_curves, show_curves):
    curves = load_curves("../outputs/spiral_1.npz")
    show_curves(curves)
    return (curves,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Validate

    The spiral is not valid as designed. The Y segment has sharp corners at its waist, and its turns are nearer than 2.6 nm.
    """)
    return


@app.cell
def _(curves, validate_curves):
    print(validate_curves(curves))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pack

    `pack_curves_under_gravity` moves the vertices along z only. The first turn of the first curve stays where it is, and the function does three steps:
    1. Repair horizontal curvature, using gradient descent.
    2. Compact the spiral vertically, using PhysX gravity simulations.
    3. Repair the vertical curvature, using a simple linear method.

    This cell runs in approximately two minutes on a high-end CPU.
    """)
    return


@app.cell
def _(curves, pack_curves_under_gravity):
    packed = pack_curves_under_gravity(curves)
    print(f"height: {packed.initial_height:.1f} nm -> {packed.final_height:.1f} nm")
    return (packed,)


@app.cell
def _(packed, show_curves):
    show_curves(packed.curves)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Validate again

    The packed spiral obeys our two specified limits: no turning angle is larger than 3.25°, and no two parts of the curves are nearer than 2.6 nm.
    """)
    return


@app.cell
def _(packed, validate_curves):
    print(validate_curves(packed.curves))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save
    """)
    return


@app.cell
def _(packed, save_curves):
    save_curves("../outputs/spiral_2_packed.npz", packed.curves)
    return


if __name__ == "__main__":
    app.run()
