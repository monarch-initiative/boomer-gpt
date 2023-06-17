import math
from collections import defaultdict
from copy import deepcopy
from functools import reduce
from typing import List, Tuple, Dict, Iterator

import networkx as nx
import numpy as np
from oaklib.datamodels.vocabulary import SKOS_EXACT_MATCH, SKOS_BROAD_MATCH, SKOS_RELATED_MATCH, SKOS_CLOSE_MATCH, SKOS_NARROW_MATCH
from sssom.constants import SEMAPV
from sssom.parsers import parse_sssom_table, to_mapping_set_document
from sssom.sssom_document import MappingSetDocument
from sssom.util import to_mapping_set_dataframe
from sssom.writers import to_dataframe
from sssom_schema import Mapping, EntityReference
import sssom.writers as sssom_writers

from boomer_gpt.datamodel.configuration import MainConfiguration


def load_sssom(path: str) -> MappingSetDocument:
    """Load an SSSOM file into a MappingSetDocument."""
    msdf = parse_sssom_table(path)
    return to_mapping_set_document(msdf)


def save_sssom(msd: MappingSetDocument, path: str):
    """Save a MappingSetDocument to an SSSOM file."""
    msdf = to_mapping_set_dataframe(msd)
    with open(path, "w", encoding="utf-8") as file:
        sssom_writers.write_table(msdf, file)


def assign_sources(mapping: Mapping):
    if not mapping.subject_source:
        mapping.subject_source = mapping.subject_id.split(":")[0]
    if not mapping.object_source:
        mapping.object_source = mapping.object_id.split(":")[0]

def group_mappings(msd: MappingSetDocument, configuration: MainConfiguration) -> Dict[Tuple[str, str, str], List[Mapping]]:
    """Group mappings in a MappingSetDocument."""
    triples = defaultdict(list)
    for m in msd.mapping_set.mappings:
        assign_sources(m)
        if m.subject_source == m.object_source:
            continue
        pred = m.predicate_id
        if configuration.lexmatch:
            for pm in configuration.lexmatch.predicate_mappings:
                if pm.source_predicate == pred:
                    pred = pm.target_predicate
                    break
        t = m.subject_id, pred, m.object_id
        triples[t].append(m)
    return triples


def assign_all_probabilities(msd: MappingSetDocument, configuration: MainConfiguration) -> MappingSetDocument:
    """Assign probabilities to mappings in a MappingSetDocument."""
    triples = group_mappings(msd, configuration)
    mappings = []
    for t, ms in triples.items():
        #for pm in ms:
        #    pm.probability = 1.0
        mapping = assign_probability(t, ms, configuration)
        mappings.append(mapping)
    new_msd = deepcopy(msd)
    new_msd.mapping_set.mappings = mappings
    return new_msd


def assign_probability(triple: Tuple[str, str, str], ms: List[Mapping], configuration: MainConfiguration) -> Mapping:
    """Assign probabilities to mappings in a triple."""
    prior_probability_independent = 1.0
    prob_multiplier = 0.5
    mappings = sorted(ms, key=lambda m: -m.confidence)
    evidences = set()
    neg_probs = []
    match_strings = []
    for i, m in enumerate(mappings):
        evidence = m.match_string
        if evidences.intersection(evidence):
            # only consider independent evidence
            continue
        match_strings.append(";".join(m.match_string))
        neg_prob = (1.0 - m.confidence) * prior_probability_independent + (1.0 - prior_probability_independent)
        prior_probability_independent *= prob_multiplier ** i
        neg_probs.append(neg_prob)
        evidences.update(evidence)
    # Pr = 1-(1-np1)(1-np2)...(1-npN)
    pr = 1 - reduce(lambda x, y: x*y, neg_probs)
    mapping = Mapping(
        subject_id=triple[0],
        predicate_id=triple[1],
        object_id=triple[2],
        mapping_justification=EntityReference(SEMAPV.CompositeMatching.value),
        confidence=pr,
        match_string=match_strings,
    )
    return mapping


def logit(p: float) -> float:
    """Calculate the logit of a probability."""
    max_p = 0.9999
    if p > max_p:
        p = max_p
    return math.log(p / (1 - p))


def softmax(x: List[float]):
    """
    Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def calculate_ptable(msd: MappingSetDocument, configuration: MainConfiguration) -> List[Tuple]:
    """Calculate a ptable from a MappingSetDocument."""
    pairs = {}
    pmap = {
        SKOS_BROAD_MATCH: 0,
        SKOS_NARROW_MATCH: 1,
        SKOS_EXACT_MATCH: 2,
        SKOS_RELATED_MATCH: 3,
        SKOS_CLOSE_MATCH: (2, 3),
    }
    # defaults = [-0.5, -0.5, 0.0, 0.5]
    # defaults = [-1.0, -1.0, -0.5, -0.5]
    defaults = [0, 0, 0, 0]
    for p, pw in configuration.predicate_weights.items():
        pos = pmap[p]
        defaults[pos] = pw.weight
    rows = []
    for m in msd.mapping_set.mappings:
        pos = pmap[m.predicate_id]
        pair = m.subject_id, m.object_id
        if pair not in pairs:
            pairs[pair] = list(defaults)
        if isinstance(pos, tuple):
            for p in pos:
                pairs[pair][p] = logit(m.confidence) / len(pos)
        else:
            pairs[pair][pos] = logit(m.confidence)
    for pair in pairs:
        probs = softmax(pairs[pair])
        print(f"SOFTMAX {pair} // {pairs[pair]} == {probs}")
        rows.append(tuple(list(pair) + list(probs)))
    return rows


def mappings_to_digraph_complex(msd: MappingSetDocument) -> nx.DiGraph:
    """Convert a MappingSetDocument to a networkx DiGraph."""
    dirmap = {
        SKOS_BROAD_MATCH: (1,),
        SKOS_NARROW_MATCH: (-1,),
        SKOS_EXACT_MATCH: (1, -1),
        SKOS_RELATED_MATCH: (1, -1),
        SKOS_CLOSE_MATCH: (1, -1),
    }
    G = nx.DiGraph()
    for m in msd.mapping_set.mappings:
        dirns = dirmap[m.predicate_id]
        for dirn in dirns:
            if dirn == 1:
                G.add_edge(m.subject_id, m.object_id, predicate=m.predicate_id)
            else:
                G.add_edge(m.object_id, m.subject_id, predicate=m.predicate_id)
    return G


def mappings_to_graph(msd: MappingSetDocument) -> nx.Graph:
    """Convert a MappingSetDocument to a networkx Graph."""
    G = nx.Graph()
    for m in msd.mapping_set.mappings:
        kwargs = {"predicate": m.predicate_id, "confidence": m.confidence}
        # check if edge exists:
        if G.has_edge(m.subject_id, m.object_id):
            # if so, update confidence
            if G.edges[m.subject_id, m.object_id]["confidence"] < m.confidence:
                G.edges[m.subject_id, m.object_id]["confidence"] = m.confidence
                G.edges[m.subject_id, m.object_id]["predicate"] = m.predicate_id
        else:
            G.add_edge(m.subject_id, m.object_id, **kwargs)
    for u, v, d in list(G.edges(data=True)):
        G.add_edge(v, u, **d, inverted=True)
    return G


def remove_mapping_megacliques(msd: MappingSetDocument, config: MainConfiguration):
    graph = mappings_to_graph(msd)
    removed = break_clique(graph, max_clique_size=config.declique.max_clique_size)
    print(f"REMOVED: {len(removed)}")
    remove_edges(msd, removed)


def break_clique(graph: nx.Graph, clique: List[str] = None, max_clique_size: int = 10) -> List[Tuple[str, str, Dict]]:
    """Break cliques larger than N in size."""
    removed = []
    if max_clique_size < 1:
        raise ValueError(f"max_clique_size must be > 0; for {max_clique_size}")
    if clique is not None and not isinstance(clique, list):
        raise ValueError(f"clique must be a list; for {clique}")
    if clique:
        if len(clique) <= max_clique_size:
            return []
        subgraph = graph.subgraph(clique)
    else:
        cliques = list(nx.connected_components(graph))
        for clique in cliques:
            # print(f"INITIAL CLIQUE: {clique}")
            if len(clique) > max_clique_size:
                removed.extend(break_clique(graph, list(clique), max_clique_size=max_clique_size))
        return removed
    # sort edges by confidence attribute
    edges = sorted(subgraph.edges(data=True), key=lambda e: -e[2]["confidence"])
    subgraph = nx.Graph(subgraph)  # unfreeze
    while edges:
        edge = edges.pop()
        removed.append(edge)
        subgraph.remove_edge(*edge[:2])
        # cliques = list(nx.find_cliques(subgraph))
        cliques = list(nx.connected_components(subgraph))
        if len(cliques) > 1:
            for clique in cliques:
                removed.extend(break_clique(subgraph, list(clique), max_clique_size=max_clique_size))
            break
    if not edges:
        print(f"All edges removed to force max = {max_clique_size} for clique: {clique}")
        # raise AssertionError("Could not break clique")
    return removed


def remove_edges(msd: MappingSetDocument, edges: List[Tuple[str, str, Dict]]):
    """Remove edges from a MappingSetDocument."""
    mappings = []
    edge_index = {(e[0], e[1]): e for e in edges}
    for m in msd.mapping_set.mappings:
        pair = m.subject_id, m.object_id
        if pair not in edge_index:
            mappings.append(m)
    msd.mapping_set.mappings = mappings
    return msd


