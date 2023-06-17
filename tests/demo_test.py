"""Demo version test."""

from pathlib import Path

import yaml
from linkml_runtime.loaders import yaml_loader

from boomer_gpt.datamodel.configuration import MainConfiguration
from boomer_gpt.main import BoomerGPT

THIS_DIR = Path(__file__).parent
INPUT_DIR = THIS_DIR / "input"
CASES_DIR = INPUT_DIR / "cases"
DEVSTAGES_DIR = CASES_DIR / "devstages"
DEVSTAGES_CONF_PATH = DEVSTAGES_DIR / "config.yaml"
ANAT_DIR = CASES_DIR / "anatomy"
ANAT_CONF_PATH = ANAT_DIR / "config.yaml"


def test_run():
    """Test run."""
    engine = BoomerGPT()
    engine.load_configuration(DEVSTAGES_CONF_PATH)
    assert len(engine.ontology_sources()) == 3
    # assert "mmusdv.owl" in engine.ontology_sources()
    engine.run()


def test_anat():
    """Test run."""
    engine = BoomerGPT()
    engine.load_configuration(ANAT_CONF_PATH)
    assert len(engine.ontology_sources()) == 3
    # assert "mmusdv.owl" in engine.ontology_sources()
    engine.run()
