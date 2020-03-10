import pytest

import core.cnn
import core.cmsm


@pytest.fixture
def empty_cobj():
    return core.cnn.CNN(alias="empty")