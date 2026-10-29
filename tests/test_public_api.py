import pt_neural_memory as package


def test_public_api_names_are_importable():
    missing = [name for name in package.__all__ if not hasattr(package, name)]
    assert not missing


def test_version_is_a_nonempty_string():
    assert isinstance(package.__version__, str)
    assert package.__version__
