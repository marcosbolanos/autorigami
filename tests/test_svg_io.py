import src.curvepack.svg_io as svg_io


class TestSvgIO:
    _svg_path = "../svg/blob1.svg"

    def test_one(self) -> None:
        assert hasattr(svg_io, "load_single_path_polygon")
