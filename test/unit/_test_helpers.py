import time

from cnnclustering import cnn


class TestTimedDecorator:
    def test_decorate_none_returning_function(self):
        def some_function():
            time.sleep(0.1)

        decorated = cnn.timed(some_function)
        decorated_result = decorated()
        assert isinstance(decorated_result, tuple)
        assert decorated_result[0] is None
        assert isinstance(decorated_result[1], float)

    def test_decorate_record_returning_function(self, capsys):
        def some_function(v=False):
            return cnn.CNNRecord(10, 1.0, 1, 1, 2, 2, 0.9, 0.05, None)

        decorated = cnn.timed(some_function)
        decorated_result = decorated()
        assert isinstance(decorated_result, tuple)
        assert isinstance(decorated_result[0], cnn.CNNRecord)
        assert isinstance(decorated_result[1], float)

        decorated_result = decorated(v=True)
        out, err = capsys.readouterr()
        assert out.split(":")[0] == "Execution time for call of some_function"
        assert err == ""
