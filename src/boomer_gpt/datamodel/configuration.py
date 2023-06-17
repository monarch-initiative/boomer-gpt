from __future__ import annotations
from datetime import datetime, date
from enum import Enum
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel as BaseModel, Field
from linkml_runtime.linkml_model import Decimal
import sys
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


metamodel_version = "None"
version = "None"

class WeakRefShimBaseModel(BaseModel):
   __slots__ = '__weakref__'

class ConfiguredBaseModel(WeakRefShimBaseModel,
                validate_assignment = True,
                validate_all = True,
                underscore_attrs_are_private = True,
                extra = 'forbid',
                arbitrary_types_allowed = True,
                use_enum_values = True):
    pass


class PrefixExpansion(ConfiguredBaseModel):
    
    prefix: Optional[str] = Field(None)
    prefix_expansion: Optional[str] = Field(None)
    


class Ontology(ConfiguredBaseModel):
    
    id: Optional[str] = Field(None)
    prefixmaps: Optional[Dict[str, PrefixExpansion]] = Field(default_factory=dict)
    sources: Optional[List[str]] = Field(default_factory=list)
    term_query: Optional[str] = Field(None)
    bridging: Optional[bool] = Field(None, description="""For ontologies that serve to bridge and are not to be aligned""")
    prefix_expansion: Optional[str] = Field(None)
    


class PredicateMapping(ConfiguredBaseModel):
    
    source_predicate: Optional[str] = Field(None)
    target_predicate: Optional[str] = Field(None)
    confidence: Optional[float] = Field(None)
    


class LexMatchConfiguration(ConfiguredBaseModel):
    
    working_directory: Optional[str] = Field(None)
    lexical_index_path: Optional[str] = Field(None)
    lexmatch_rules_path: Optional[str] = Field(None)
    weighting_rules: Optional[Any] = Field(None)
    synonymizer_rules: Optional[Any] = Field(None)
    probability_decay: Optional[float] = Field(None, description="""Rate at which the probability of independence decays with each piece of evidence""")
    predicate_mappings: Optional[List[PredicateMapping]] = Field(default_factory=list)
    


class LogMapConfiguration(ConfiguredBaseModel):
    
    working_directory: Optional[str] = Field(None)
    configuration_files: Optional[List[str]] = Field(default_factory=list)
    


class DecliqueConfiguration(ConfiguredBaseModel):
    
    max_clique_size: Optional[int] = Field(None)
    


class BoomerConfiguration(ConfiguredBaseModel):
    
    working_directory: Optional[str] = Field(None)
    window_count: Optional[int] = Field(None)
    runs: Optional[int] = Field(None)
    exhaustive_search_limit: Optional[int] = Field(None)
    subsequent_solutions: Optional[int] = Field(None)
    


class OntoGPTConfiguration(ConfiguredBaseModel):
    
    weight: Optional[float] = Field(None)
    max_invocations: Optional[int] = Field(None)
    invocation_threshold: Optional[float] = Field(None)
    models: Optional[List[str]] = Field(default_factory=list)
    


class PredicateWeight(ConfiguredBaseModel):
    
    predicate: Optional[str] = Field(None)
    weight: Optional[float] = Field(None)
    


class MainConfiguration(ConfiguredBaseModel):
    
    name: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    source_directory: Optional[str] = Field(None)
    prefixmaps: Optional[Dict[str, PrefixExpansion]] = Field(default_factory=dict)
    predicate_weights: Optional[Dict[str, PredicateWeight]] = Field(default_factory=dict)
    ontologies: Optional[Dict[str, Ontology]] = Field(default_factory=dict)
    lexmatch: Optional[LexMatchConfiguration] = Field(None)
    logmap: Optional[LogMapConfiguration] = Field(None)
    declique: Optional[DecliqueConfiguration] = Field(None)
    boomer: Optional[BoomerConfiguration] = Field(None)
    ontogpt: Optional[OntoGPTConfiguration] = Field(None)
    



# Update forward refs
# see https://pydantic-docs.helpmanual.io/usage/postponed_annotations/
PrefixExpansion.update_forward_refs()
Ontology.update_forward_refs()
PredicateMapping.update_forward_refs()
LexMatchConfiguration.update_forward_refs()
LogMapConfiguration.update_forward_refs()
DecliqueConfiguration.update_forward_refs()
BoomerConfiguration.update_forward_refs()
OntoGPTConfiguration.update_forward_refs()
PredicateWeight.update_forward_refs()
MainConfiguration.update_forward_refs()

