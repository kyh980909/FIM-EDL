"""Registry and schema smoke checks."""

from src.registry.core import Registry
from src.contracts.schemas import MetricRecord, RESULTS_SCHEMA_VERSION
import src.models.backbones.convnet  # noqa: F401
import src.models.backbones.vgg16  # noqa: F401
import src.models.backbones.wrn28_10  # noqa: F401
from src.registry.backbones import BACKBONE_REGISTRY


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


def test_paper_backbones_registered() -> None:
    keys = set(BACKBONE_REGISTRY.keys())
    assert "convnet" in keys
    assert "vgg16" in keys
    assert "wrn28_10" in keys
