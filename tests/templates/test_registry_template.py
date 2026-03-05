"""Registry and schema smoke checks."""

from src.registry.core import Registry
from src.contracts.schemas import MetricRecord, RESULTS_SCHEMA_VERSION


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


def test_metric_record_new_fields_exist() -> None:
    row = MetricRecord(
        results_schema_version=RESULTS_SCHEMA_VERSION,
        method="info_edl",
        seed=0,
        dataset="svhn",
        split="eval",
        metrics={"auroc": 0.9},
        config_hash="abc",
        git_commit="def",
        method_variant="info_edl",
        score_type="vacuity",
        calibration_type="none",
    )
    assert row.method_variant == "info_edl"
    assert row.score_type == "vacuity"
    assert row.calibration_type == "none"
