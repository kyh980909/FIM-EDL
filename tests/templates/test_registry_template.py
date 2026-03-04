"""Template: copy and replace keys for new plugin registry tests."""

from src.registry.core import Registry


def test_registry_duplicate_key_fails() -> None:
    reg = Registry(name="demo")

    @reg.register("x")
    def f():
        return 1

    try:
        @reg.register("x")
        def g():
            return 2
        assert False, "duplicate registration should fail"
    except ValueError:
        assert True
