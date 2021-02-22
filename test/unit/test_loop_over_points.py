import pytest


@pytest.mark.skip(reason="Obsolete: data representation changed")
class TestLoops:
    def test_loop_over_no_points(self, empty_cobj):
        point_iterator = empty_cobj.loop_over_points()
        with pytest.raises(StopIteration):
            next(point_iterator)

    def test_loop_over_points_of_circles(self, random_circles_cobj):
        point_iterator = random_circles_cobj.loop_over_points()
        with pytest.raises(StopIteration):
            for i in iter(int, 1):
                # print(f"Points: {i}", end="\r")
                next(point_iterator)
