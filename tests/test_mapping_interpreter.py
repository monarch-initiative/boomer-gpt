"""Demo version test."""
import math
import random
from typing import Optional, Tuple

import networkx as nx
import pytest
from oaklib.datamodels.vocabulary import SKOS_EXACT_MATCH, SKOS_RELATED_MATCH, SKOS_CLOSE_MATCH
from sssom.sssom_document import MappingSetDocument
from sssom_schema import MappingSet, Mapping

from boomer_gpt.datamodel.configuration import MainConfiguration
from boomer_gpt.mapping_interpreter import calculate_ptable, break_clique, load_sssom, \
    mappings_to_graph, remove_edges, group_mappings
from tests.demo_test import INPUT_DIR

A1 = "a:1"
B1 = "b:1"

FAKE_LICENSE = "http://example.org"

config = MainConfiguration()

def _mapping(s, p, o, confidence) -> Mapping:
    return Mapping(subject_id=s, predicate_id=p, object_id=o, confidence=confidence, mapping_justification="test")

def _compare(row: tuple, expected: tuple):
    for i, v in enumerate(row):
        diff = abs(v - expected[i])
        assert diff < 0.05

@pytest.mark.parametrize("triples,expected",
                         [
                             (
                                     [(A1, SKOS_EXACT_MATCH, B1, 1.0)],
                                     {(A1, B1): (0, 0, 1, 0)}
                              ),
                             (
                                     [(A1, SKOS_RELATED_MATCH, B1, 1.0)],
                                     {(A1, B1): (0, 0, 0, 1)}
                            ),
                            (
                                     [(A1, SKOS_CLOSE_MATCH, B1, 1.0)],
                                     {(A1, B1): (0, 0, 0.5, 0.5)}
                            ),
                            (
                                     [(A1, SKOS_EXACT_MATCH, B1, 0.5)],
                                     {(A1, B1): (0.25, 0.25, 0.25, 0.25)}
                              ),
                            (
                                     [(A1, SKOS_EXACT_MATCH, B1, 1.0),
                                      (A1, SKOS_RELATED_MATCH, B1, 1.0)],
                                     {(A1, B1): (0, 0, 0.5, 0.5)}
                              ),
                         ])
def test_mapping_interpreter(triples, expected):
    mset = MappingSet("test", license=FAKE_LICENSE, mappings=[_mapping(*t) for t in triples])
    msd = MappingSetDocument(mapping_set=mset, prefix_map={})
    ptable = calculate_ptable(msd, config)
    print(ptable)
    for row in ptable:
        pair = (row[0], row[1])
        _compare(row[2:], expected[pair])


@pytest.mark.parametrize("size,edge_prob,max_clique_size,min_max,name", [
    (100, 0.04, 10, None, "normal"),
    (1000, 0.002, 25, None, "normal, larger"),
    (100, 0.5, 2, None, "megaclique"),
    (10, 1.0, 1, (25, 100), "edge case, force singletons"),
    (10, 1.0, 10, (0, 0), "edge case, already within max, do not perturb"),
    (10, 0.0, 1, (0, 0), "edge case, already singletons, do not perturb"),
])
def test_break_clique(size, edge_prob, max_clique_size, min_max: Optional[Tuple[int, int]], name):
    params = size, edge_prob, max_clique_size, name
    # create random networkx graph
    graph = nx.gnp_random_graph(size, edge_prob, directed=False)
    for u, v, d in graph.edges(data=True):
        d["confidence"] = random.random()
    cliques = list(nx.connected_components(graph))
    singletons = []
    print(f"\nPARAM={params}")
    for i, clique in enumerate(cliques):
        clique = list(clique)
        if len(clique) == 1:
            singletons.append(clique.pop())
        else:
            print(f"INPUT {i} clique [{len(clique)}]: {clique}")
    print(f"Singletons: {len(singletons)}")
    removed = break_clique(graph, max_clique_size=max_clique_size)
    print(f"Removed {len(removed)} edges")
    # find edge with maximum confidence
    if removed:
        max_confidence_edge = max(removed, key=lambda x: x[2]["confidence"])
        print(f"Max confidence edge removed: {max_confidence_edge}")
    if min_max is not None:
        min_, max_ = min_max
        assert len(removed) >= min_
        assert len(removed) <= max_
    reduced_graph = nx.Graph(graph)
    for e in removed:
        # remove edge plus attributes from reduced graph
        reduced_graph.remove_edge(*e[:2])
    cliques = list(nx.connected_components(reduced_graph))
    new_singletons = []
    for i, clique in enumerate(cliques):
        assert len(clique) <= max_clique_size
        if len(clique) == 1:
            mbr = clique.pop()
            if mbr in singletons:
                pass
            else:
                new_singletons.append(mbr)
                # print(f"New singleton: {mbr}")
        else:
            print(f"OUTPUT {i} clique [{len(clique)}]: {clique}")
    print(f"New singletons: {len(new_singletons)}")


def test_sssom_to_graph():
    msd = load_sssom(INPUT_DIR / "test.sssom.tsv")
    graph = mappings_to_graph(msd)
    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)
    n_mappings = len(msd.mapping_set.mappings)
    print(n_edges)
    # for u, v, d in graph.edges(data=True):
    #    print(u, v, d)
    removed = break_clique(graph, max_clique_size=3)
    print(f"Removed {len(removed)} edges")
    remove_edges(msd, removed)
    assert len(msd.mapping_set.mappings) < n_mappings


def test_group_mappings():
    msd = load_sssom(INPUT_DIR / "test.sssom.tsv")
    grp = group_mappings(msd, config)
    for t in grp:
        print(t)



