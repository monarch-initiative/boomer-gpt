"""Main python file."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import funrun
import yaml
from funrun.product import Workspace

from boomer_gpt.datamodel.configuration import MainConfiguration
from boomer_gpt.mapping_interpreter import load_sssom, assign_all_probabilities, save_sssom, calculate_ptable, \
    remove_mapping_megacliques
from boomer_gpt.runners import lexmatch, create_ptable, robot_merge, OWLOutput, PTableOutput, run_boomer, fetch_ontology


@dataclass
class BoomerGPT:
    """Main class."""

    configuration: MainConfiguration = None
    runner: funrun.Runner = field(default_factory=lambda: funrun.Runner(use_cached=True))

    def load_configuration(self, path: Union[str, Path]):
        with open(path) as file:
            obj = yaml.safe_load(file)
            self.configuration = MainConfiguration(**obj)
            self.configuration.source_directory = str(Path(path).parent)

    def dump_prefixes(self) -> Path:
        wd = self.runner.workspace.directory
        path = wd / "prefixes.yaml"
        prefixmap = {k: ont.prefix_expansion for k, ont in self.configuration.ontologies.items()}
        with open(path, "w") as file:
            yaml.dump(prefixmap, file)
        return path

    def run(self):
        """Run the program."""
        runner = self.runner
        workspace = Workspace(directory=Path.cwd() / ".funrun")
        self.runner.workspace = workspace
        prefixes_path = self.dump_prefixes()
        merged_ontology_output = self.run_merge()
        mapping_output = self.run_lexmatch()
        boomer_output = runner.make(run_boomer(merged_ontology_output, mapping_output, prefixes_path), verbose=True)

    def run_merge(self) -> OWLOutput:
        runner = self.runner
        sources = self.ontology_sources()
        return runner.make(robot_merge(sources))

    def run_lexmatch(self) -> PTableOutput:
        """Run the program."""
        runner = self.runner
        wd = runner.workspace.directory
        sources = self.ontology_sources()
        confdir = Path(self.configuration.source_directory)
        lexmatch_rules_path = confdir / self.configuration.lexmatch.lexmatch_rules_path
        sources_as_sqlite = ["sqlite:" + s for s in sources]
        exclude = [k for k, ont in self.configuration.ontologies.items() if ont.bridging]
        cmd = lexmatch(sources_as_sqlite, rules_path=str(lexmatch_rules_path), prefixes=self.ontology_prefixes(), exclude=exclude)
        lexmatch_output = runner.make(cmd, verbose=True)
        print(lexmatch_output.content()[0:300])
        lexmatch_output.dump(str(Path(wd / "tmp.sssom.tsv")))
        msd = load_sssom(lexmatch_output.output)
        msd = assign_all_probabilities(msd, self.configuration)
        processed_sssom_file = str(Path(wd / "processed.sssom.tsv"))
        save_sssom(msd, processed_sssom_file)
        remove_mapping_megacliques(msd, self.configuration)
        reduced_sssom_file = str(Path(wd / "reduced.sssom.tsv"))
        save_sssom(msd, reduced_sssom_file)
        ptable_rows = calculate_ptable(msd, self.configuration)
        ptable_path = str(Path(wd / "ptable.tsv"))
        with open(ptable_path, "w") as file:
            for row in ptable_rows:
                file.write("\t".join([str(x) for x in row]) + "\n")
        #ptable_output = runner.make(create_ptable(processed_sssom_file), verbose=True)
        #print(ptable_output.content())
        return ptable_path



    def ontology_sources(self):
        """Get ontology sources."""
        return [self.ontology_source(k) for k in self.configuration.ontologies]

    def ontology_prefixes(self):
        """Get ontology prefixes."""
        return [k for k in self.configuration.ontologies]

    def ontology_source(self, key):
        """Get ontology source."""
        sources = self.configuration.ontologies[key].sources
        if sources:
            return sources[0].format(pwd=self.configuration.source_directory)
        else:
            fetch_output = self.runner.make(fetch_ontology(key.lower()))
            return fetch_output.output


def demo():
    """Define API."""
    print("Hello, World!")
