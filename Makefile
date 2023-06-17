RUN = poetry run
DM = src/boomer_gpt/datamodel

test: data-test unit-test

unit-test:
	$(RUN) pytest

data-test: tests/input/test_config.yaml.validate


$(DM)/%.py: $(DM)/%.yaml
	$(RUN) gen-pydantic $< > $@


%.validate: %
	$(RUN) linkml-validate -s $(DM)/configuration.yaml $<
