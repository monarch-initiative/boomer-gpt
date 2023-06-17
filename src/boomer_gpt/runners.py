from typing import List

from funrun import action, Product


class LexMatchOutput(Product):
    pass


class OWLOutput(Product):
    pass

class PTableOutput(Product):
    pass


class BoomerOutput(Product):
    pass

@action
def lexmatch(sources: List[str], rules_path: str, prefixes: List[str], exclude: List[str] = None) -> LexMatchOutput:
    """Run lexmatch."""
    if exclude is None:
        exclude = []
    def _inputs(sources):
        return f"-i {sources[0]} " + " ".join([f"-a {s}" for s in sources[1:]])
    def _queries(prefixes):
        return " ".join([f"i^{pfx}:" for pfx in prefixes if pfx not in exclude])
    return LexMatchOutput(lambda: f"runoak {_inputs(sources)} lexmatch -R {rules_path} {_queries(prefixes)} -o {{output}}")


@action
def fetch_ontology(ontology_id: str) -> OWLOutput:
    """Fetch ontology."""
    return OWLOutput(lambda: f"robot merge -I http://purl.obolibrary.org/obo/{ontology_id}.owl convert -f owl -o {{output}}",
                     output=f"{ontology_id}.owl")

@action
def create_ptable(mapping_output: LexMatchOutput) -> PTableOutput:
    """Create ptable."""
    return PTableOutput(lambda: f"sssom ptable {mapping_output}")


@action
def robot_merge(sources: List[str]) -> OWLOutput:
    """Run robot merge."""
    inputs = " ".join([f"-i {s}" for s in sources])
    return OWLOutput(lambda: f"robot merge {inputs} convert -f owl -o {{output}}")

@action
def run_boomer(ontology: OWLOutput, ptable: PTableOutput, prefixes_path: str) -> BoomerOutput:
    return BoomerOutput(lambda: f"boomer --ptable {ptable} --prefixes {prefixes_path} --ontology {ontology} -e 10 -w 3 -r 3 --subsequent-solutions 3 --output {{output}}")