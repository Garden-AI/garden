import pytest  # noqa


def test_hello_world():
    assert "Hello" != "World"


@pytest.mark.integration
def test_integration_mark():
    """This test should be automatically skipped unless --intgration or -m 'integration' is given.

    Sanity check test
    """
    assert True


@pytest.mark.cli
def test_cli_mark():
    """This test should run when -m 'cli' is given

    Sanity check test
    """
    assert True


@pytest.mark.integration
@pytest.mark.cli
def test_cli_and_integration_marks():
    """This test should run when both intgration and cli, but not just cli

    Sanity check test
    """
    assert True
