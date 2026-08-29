"""
A module that contains the RoughGranulation class, which is used to provide the necessary
granulation operations required for rough set theory. This class is inherited by more specialized
classes, such as RoughApproximation, to provide the basic granulation operations.

Architecture and design history (2026-08-29)
=============================================
This package is organized as a 4-level inheritance chain, each level adding a set of
functions that build toward the next higher-level operation:

    RoughGranulation      - owns the igraph.Graph + attribute_table + O(1) item index.
                            Generic vertex/edge/tag plumbing: set_granules(), select_by_tags(),
                            add_parent_relation(), export_visual(), equivalence classes (/).
      -> RoughApproximation  - Pawlak lower/upper approximation, positive/negative/boundary
                                regions, accuracy/roughness - built on RoughGranulation's
                                tags/equivalence-class machinery.
        -> RoughOperations     - reducts, core, (in)dependence/dispensability - built on
                                  RoughApproximation's indiscernibility()/approximation methods.
          -> RoughDecisions       - discernibility matrices, decision-table decompose/simplify -
                                    built on RoughOperations' reduct/core-finding.

An audit of every consumer of this chain across the wider codebase (KnowledgeBase in the
separate fuzzy-theory package, Regime in the separate regime package, fuzzy_ml's fyd/
summary.py/temporal.py, experiments/pipeline.py) found a decisive fact: NOBODY outside
this package's own test suite ever calls a RoughApproximation/RoughOperations/
RoughDecisions method. Every real consumer only ever touches RoughGranulation-level
surface (.graph, .attribute_table, .select_by_tags(), .set_granules(), __getitem__).
Both KnowledgeBase and Regime independently inherited the FULL chain (`class
KnowledgeBase(RoughDecisions, FuzzySystem)`, `class Regime(RoughDecisions)`) just to get
that graph/tagging plumbing, dragging in ~40 reduct/decision-table/approximation methods
neither ever calls - a real cost to anyone reading or using those classes, and (per the
user directly) a contributing reason the analysis capabilities go unused: it's awkward to
reach for something bolted onto a class that also does a hundred other things.

The deeper reason the analysis layers go unused, though, isn't really the API shape - it's
that there's rarely a graph worth analyzing that's actually kept current. A KnowledgeBase's
graph is mutated in place over its lifetime; a CO-FIS-style neuro-fuzzy network (see
PySoft's own PLANNED_FEATURES.md item #2) doesn't build a graph like this AT ALL today.
Fixing the inheritance shape doesn't fix that by itself - but it's a necessary enabler:
once ANY RoughGranulation-shaped graph exists (a KnowledgeBase's, a Regime's, or a future
CO-FIS one), the fix below makes it trivial to attach analysis to it on demand, exactly
when the graph is actually current, without that graph's owning class needing to inherit
this hierarchy at all.

Two designs were considered:
  1. Full composition rewrite (every level holds a reference to the one below instead of
     inheriting it, all the way down) - REJECTED for now. Nothing outside this package's
     own tests calls RoughApproximation/RoughOperations/RoughDecisions methods
     individually, so there's no live usage feedback to design a good composed API
     against - reorganizing well-tested, 100%-covered code with no real consumer input
     would be guessing at a shape nobody has asked for yet.
  2. Make RoughGranulation.__init__ optionally wrap an EXISTING graph/attribute_table
     (alias, not copy) instead of always building fresh ones - CHOSEN. Since none of
     RoughApproximation/RoughOperations/RoughDecisions override __init__, this
     propagates to all three for free, with zero changes needed in
     approximation.py/operations.py/decisions.py. KnowledgeBase and Regime were then
     changed to inherit ONLY RoughGranulation (what they actually use) instead of the
     full chain - see their own module's docstrings for that half of the change.

The resulting pattern for attaching analysis to any RoughGranulation-shaped graph:

    analysis = RoughDecisions(graph=obj.graph, attribute_table=obj.attribute_table)
    reducts = analysis.find_reducts(...)

This constructs a FRESH, cheap wrapper aliasing the same graph object - not a snapshot,
not a long-lived object to keep in sync. Reconstruct it every time you actually want to
analyze, right before you need it. This deliberately mirrors the "rebuild fresh rather
than incrementally patch a stale structure" convention this same session adopted for the
still-not-yet-built CO-FIS igraph work (see PySoft's PLANNED_FEATURES.md item #2/#3) -
same philosophy, so a future CO-FIS graph can plug into this exact pattern too.

See also: fuzzy-theory's fuzzy/logic/knowledge_base.py (KnowledgeBase's own docstring
explains why it dropped RoughDecisions in favor of RoughGranulation) and regime's
regime/flow/impl.py (Regime's class comment, same change).
"""

from collections import Counter
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Set, Union

import graphviz
import igraph as ig


class RoughGranulation:
    """
    A class that represents the necessary granulation operations required for rough set theory. This
    class is inherited by more specialized classes, such as RoughApproximation, to provide the basic
    granulation operations.

    RoughApproximation/RoughOperations/RoughDecisions all subclass this without overriding
    __init__, so passing graph/attribute_table works identically all the way up that chain -
    e.g. RoughDecisions(graph=knowledge_base.graph, attribute_table=knowledge_base.attribute_table)
    wraps an EXISTING graph-bearing object (a KnowledgeBase, a Regime, or any other
    RoughGranulation-shaped graph) in a fresh analysis instance, aliasing its graph/table rather
    than copying them. This is the intended way to attach rough-set analysis (reducts, decision
    tables, approximations) to a graph you don't otherwise inherit this hierarchy for: construct
    the wrapper fresh whenever you actually want to analyze, rather than keeping one long-lived
    analysis object that could drift out of sync with a graph that keeps changing underneath it.
    """

    def __init__(
        self,
        graph: Optional[ig.Graph] = None,
        attribute_table: Optional[dict] = None,
    ):
        self.graph = graph if graph is not None else ig.Graph(directed=True)
        # keys: hashed frozenset or attribute name (if given) mapped to
        # attribute values
        self.attribute_table = attribute_table if attribute_table is not None else {}
        # maps a vertex's "item" attribute value to the indices of every vertex
        # with that value - igraph has no index for arbitrary-Python-object
        # vertex attributes, so `self.graph.vs.find(item_eq=x)`/`.select(item_eq=x)`
        # are each an O(V) linear scan; this turns every such lookup into O(1)
        # (amortized) by maintaining the mapping incrementally as vertices are
        # added (see _register_item(), called from set_granules() and
        # add_parent_relation() - the only two places vertices are ever added to
        # self.graph). Kept as a private implementation detail: _find_by_item()/
        # _select_by_item() below are the only intended way to read it.
        self._item_index: Dict[Any, List[int]] = {}
        if graph is not None:
            # a caller-supplied graph may already have vertices this instance
            # didn't create (e.g. wrapping an existing KnowledgeBase/Regime) -
            # populate the index from what's actually there now, rather than
            # starting empty and immediately disagreeing with reality.
            self._rebuild_item_index()

    def _register_item(self, item: Any, vertex_index: int) -> None:
        """
        Record that the vertex at `vertex_index` has "item" attribute `item`,
        so later _find_by_item()/_select_by_item() calls can find it in O(1)
        instead of scanning every vertex in the graph.

        Args:
            item: The vertex's "item" attribute value.
            vertex_index: The vertex's index in self.graph.

        Returns:
            None
        """
        self._item_index.setdefault(item, []).append(vertex_index)

    def _rebuild_item_index(self) -> None:
        """
        Rebuild self._item_index from scratch by scanning every vertex's
        current "item" attribute.

        Needed because code outside this class (e.g. fuzzy-theory's
        KnowledgeBase, a subclass of a subclass of this one) sometimes
        reassigns an existing vertex's "item" attribute directly
        (`vertex["item"] = ...`) instead of going through _register_item(),
        and igraph re-numbers every vertex's index after any deletion - both
        silently desync self._item_index from the real graph. _resolve_indices()
        calls this once, only when its cheap verification against the live
        graph fails, so the common (already-consistent) case stays O(1).

        Returns:
            None
        """
        self._item_index = {}
        for index, item in enumerate(self.graph.vs["item"]):
            self._register_item(item, index)

    def _resolve_indices(self, item: Any) -> List[int]:
        """
        Return every vertex index whose CURRENT "item" attribute actually
        equals `item` right now - not just whatever self._item_index last
        recorded, which _rebuild_item_index()'s docstring explains can go
        stale. Vertex indices are also range-checked, since a deletion
        elsewhere can shrink the graph out from under a cached index.

        On a miss, this always rebuilds and retries once before giving up -
        it is tempting to skip the rebuild when `item` was never a key in
        self._item_index at all (reasoning "it was never registered, so it
        can't be here"), but that reasoning is unsound: an external
        reassignment (see _rebuild_item_index()'s docstring) can give a
        vertex an "item" value that was never indexed under that value
        before at all, not just a stale one. A genuine miss costs one O(V)
        rebuild either way - the same cost this class had for every lookup
        before this index existed - so this only matters for the
        already-exceptional not-found path, not the hot one.

        Args:
            item: The "item" attribute value to search for.

        Returns:
            Every currently-valid vertex index with a matching "item"
            attribute (empty if none).
        """
        vertex_count = self.graph.vcount()
        matches = [
            index
            for index in self._item_index.get(item, [])
            if index < vertex_count and self.graph.vs[index]["item"] == item
        ]
        if matches:
            return matches
        self._rebuild_item_index()
        return [
            index
            for index in self._item_index.get(item, [])
            if self.graph.vs[index]["item"] == item
        ]

    def _find_by_item(self, item: Any) -> ig.Vertex:
        """
        Equivalent to self.graph.vs.find(item_eq=item), but O(1) (amortized)
        via self._item_index instead of an O(V) linear scan. Returns the
        first vertex registered with this item, matching igraph's own find()
        semantics when multiple vertices share the same "item" value.

        Args:
            item: The "item" attribute value to search for.

        Returns:
            The first vertex with a matching "item" attribute.

        Raises:
            ValueError: If no vertex has this "item" attribute.
        """
        matches = self._resolve_indices(item)
        if not matches:
            raise ValueError("no such vertex")
        return self.graph.vs[matches[0]]

    def _select_by_item(self, item: Any) -> ig.VertexSeq:
        """
        Equivalent to self.graph.vs.select(item_eq=item), but O(1) (amortized)
        via self._item_index instead of an O(V) linear scan.

        Args:
            item: The "item" attribute value to search for.

        Returns:
            Every vertex with a matching "item" attribute (empty if none).
        """
        return self.graph.vs[self._resolve_indices(item)]

    def __getitem__(self, item: Union[str, int]) -> Dict[str, list]:
        vertex = self._find_by_item(item)
        neighbor_vertices = self.graph.vs[self.graph.neighbors(vertex)]

        # get any vertices from vertex's neighbors that actively apply a
        # relation upon 'vertex'
        relations = [
            vertex["item"] for vertex in neighbor_vertices if vertex["item"] is not None
        ]

        results = {}
        for relation in relations:
            if relation not in results:
                results[relation] = []
            results[relation].extend(
                [
                    category
                    for category in self / relation
                    if item in category and category not in results[relation]
                ]
            )
            if len(results[relation]) == 1:
                # remove the list if unnecessary; rough set expects no list
                results[relation] = results[relation][0]
        return results

    def __div_helper(self, other) -> frozenset:
        categories = []
        # the neighbors of this vertex are the equivalence classes
        equivalence_vertices = self._select_by_item(other)
        for category in equivalence_vertices:
            nodes = self.graph.predecessors(category.index)
            vertices = self.graph.vs.select(nodes)
            elements = [vertex["item"] for vertex in vertices]
            categories.append(frozenset(elements))
        return frozenset(categories)

    def __truediv__(
        self, other: Union[str, Iterable]
    ) -> Union[Dict[str, Iterable], frozenset]:
        """
        Given a relation, obtain the equivalence classes.
        Args:
            other:

        Returns:
            A frozenset of equivalence classes, where each class is a frozenset.
        """
        if isinstance(other, Iterable) and not isinstance(
            other, str
        ):  # if is a list of relations
            results = {}
            for relation in other:
                results[relation] = self.__div_helper(relation)
            return results

        # else, when only given a single relation
        return self.__div_helper(other)

    def select_by_tags(self, tags: Union[str, Set[str]]) -> ig.VertexSeq:
        """
        Get the vertices that have the given tag(s).

        Returns:
            The vertices that have the given tag(s).
        """
        if isinstance(tags, str):
            tags: Set[str] = {tags}
        return self.graph.vs.select(
            lambda vertex: vertex["tags"] is not None and tags.issubset(vertex["tags"])
        )  # must first check that the tags are not None

    def set_granules(
        self,
        items: List[Any],
        tags: Union[None, str, Set[str], List[Set[str]]],
        **kwargs,
    ) -> None:
        """
        Adds the given items to the graph's layer where each item is tagged with the given tag(s).
        For example, if the items are ['a', 'b', 'c'] and the tags are [{'x', 'y'}, {'y'}, {'z'}],
        then the graph will have 3 new vertices, where the first vertex will have the item 'a' and
        the tags 'x' and 'y', the second vertex will have the item 'b' and the tag 'y', and the
        third vertex will have the item 'c' and the tag 'z'. Tagging allows easy access to the
        vertices that have the same tag(s). Provide 'None' if no tags are to be associated with the
        items. If a single (set of) string(s) is given for 'tags', then it is applied to all
        'items'. Additional keyword arguments can be provided as well.

        Args:
            items: The items to be added.
            tags: The tag(s) to be associated with the items. Can be 'None' if no tags are to be
            associated with the items.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        if isinstance(tags, str):
            # a fresh set per vertex - `[{tags}] * len(items)` would instead
            # give every vertex the SAME mutable set object, so mutating one
            # vertex's tags in place (e.g. via `.add()`) would silently
            # affect every other vertex added in this same call too
            tags = [{tags} for _ in items]
        elif isinstance(tags, set):
            tags = [set(tags) for _ in items]
        elif tags is not None:  # if tags is a list of sets
            assert len(items) == len(
                tags
            ), "The number of tags must match the number of items."
            assert all(
                isinstance(tag, set) for tag in tags
            ), "Items in 'tags' must be sets."
        start_index = self.graph.vcount()
        self.graph.add_vertices(
            len(items),
            attributes={
                "item": items,
                "tags": tags,
                **kwargs,
            },
        )
        for offset, item in enumerate(items):
            self._register_item(item, start_index + offset)

    def create_compound_edges(self, args, target_vertices) -> list:
        """
        A helper method to self.add_parent_relation

        Args:
            args: A collection of lists, where each element in the collection (a list),
            stores the 'item' of the vertex in the graph.
            target_vertices: The vertices that have been created that represent
            the parent relationships.

        Returns:
            A list of edges to be added for the given relation (type) with respect to the given
            items (args).
        """
        edges = []
        for compound, target_vertex in zip(args, target_vertices):
            try:
                for node_id in compound:
                    self.create_compound_edges_helper(node_id, edges, target_vertex)
            except TypeError:  # the "compound" is already a source vertex
                self.create_compound_edges_helper(compound, edges, target_vertex)
        return edges

    def create_compound_edges_helper(self, source, edges, targets):
        """
        A helper method for the create_compound_edges method that creates edges between the
        'source' and 'targets', and appends them to the list 'edges'. If the given 'source' is not
        an igraph.Vertex, attempt to look up the vertex, and then iterate over the 'targets'. If a
        'target' at any moment is not an igraph.Vertex, attempt to look up the vertex.

        Args:
            source:
            edges:
            targets:

        Returns:
            A list of edges, where each edge is a 2-tuple in the form of (source, target)
        """
        if not isinstance(source, ig.Vertex):  # if the source is not a vertex
            try:
                source_vertex = self._find_by_item(source)  # try to find its vertex
            except ValueError as exception:  # no such vertex;
                raise ValueError(
                    f"A vertex could not be found in the graph: {source}."
                ) from exception
        else:
            source_vertex = source
        try:
            for target in targets:
                if not isinstance(target, ig.Vertex):  # if the source is not a vertex
                    target_vertex = self._find_by_item(target)  # try to find its vertex
                else:
                    target_vertex = target

                edges.append((source_vertex.index, target_vertex.index))
        except TypeError:  # the "targets" is already a target vertex
            edges.append((source_vertex.index, targets.index))

    def add_parent_relation(self, attr_type, args) -> list:
        """
        Add a relation (attr_type) that references the provided items (args).

        Args:
            attr_type: The type of the relation, this can be a callable function as well
            (e.g., AlgebraicProduct).
            args: A collection of lists, where each element in the collection (a list),
            stores the 'type' of the vertex in the graph.

        Returns:
            A list of vertices that represent the parent relationships.
        """
        if isinstance(attr_type, str) and str.isdigit(attr_type):
            attr_type = int(attr_type)  # for saving/loading purposes

        vertices = []
        for _ in range(len(args)):
            new_vertex = self.graph.add_vertex(item=attr_type, tags={"relation"})
            self._register_item(attr_type, new_vertex.index)
            vertices.append(new_vertex)

        edges = self.create_compound_edges(args, vertices)
        self.add_weighted_edges(edges)
        return vertices

    def add_weighted_edges(self, edges) -> None:
        """
        Add edges to the RoughGranulation.graph, with a weight that is equal to its frequency (in
        the argument, 'edges').

        Args:
            edges: A list of edges to be added.

        Returns:
            None
        """
        unique_edges_and_frequencies = Counter(edges)  # keep only the unique edges
        unique_edges, frequencies = (
            unique_edges_and_frequencies.keys(),
            unique_edges_and_frequencies.values(),
        )
        self.graph.add_edges(es=unique_edges, attributes={"weight": list(frequencies)})

    def export_visual(self, filename, file_format="png", engine="dot") -> None:
        """
        Creates and exports a visual of the graph using the format (file_format)
        and the layout (engine), as specified.

        Args:
            filename: The name of the file to save.
            file_format: The format of the file to save (e.g., 'png', 'svg').
            engine: The method to use when creating the layout of the graph (e.g., 'twopi', 'sfdp').

        Returns:
            None
        """
        self.graph.write(f=f"{filename}.dot")
        # for SelfOrganize
        # ig.plot(self.graph, target="llm.png",
        #         vertex_label=[repr(v["function"]) for v in self.graph.vs])
        file_formats = ["png", "svg", "svgz", "pdf"]
        if file_format in file_formats:  # https://graphviz.org/docs/outputs/
            render_engines = ["twopi", "sfdp", "dot"]
            if engine in render_engines:  # https://graphviz.org/docs/layouts/
                graphviz.render(  # pragma: no cover
                    # requires a real graphviz install (the dot/twopi/sfdp
                    # system binaries), not guaranteed to be present in a
                    # test environment - the two branches above/below that
                    # validate file_format/engine before reaching this line
                    # are exercised directly instead (see tests)
                    format=file_format,
                    filepath=f"{filename}",
                    engine=engine,
                )
            else:
                raise UserWarning(
                    f"Exporting graph visual was unsuccessful. "
                    f"Please select a permitted format: {render_engines}."
                )
        else:
            raise UserWarning(
                f"Exporting graph visual was unsuccessful. "
                f"Please select a permitted format: {file_formats}."
            )

    def family_intersection(self, relative_to: set) -> frozenset:
        """
        Get the intersection of a family of sets/categories 'relative_to' some set of relations.

        Args:
            relative_to: A selection of relations.

        Returns:
            The family intersection.

        Raises:
            ValueError: If 'relative_to' is empty. Unlike an empty union (unambiguously
                the empty set), the intersection of an empty family of sets is
                mathematically undefined without a fixed universe to intersect within,
                so this cannot silently return a value.
        """

        categories = [
            next(iter(category)) for category in (self / relative_to).values()
        ]
        if not categories:
            raise ValueError(
                "Cannot compute the family intersection of an empty family "
                f"'relative_to'={relative_to!r}: the intersection of zero sets is "
                "undefined."
            )
        return frozenset.intersection(*categories)

    def family_union(self, relative_to: set) -> frozenset:
        """
        Get the union of a family of sets/categories 'relative_to' some set of relations.

        Args:
            relative_to: A selection of relations.

        Returns:
            The family union. The union of an empty family is unambiguously the
            empty set (frozenset()).
        """
        categories = [
            next(iter(category)) for category in (self / relative_to).values()
        ]
        if not categories:
            return frozenset()
        return frozenset.union(*categories)

    def edges(self, relation: str) -> Set[frozenset]:
        """
        Get the 'name' (vertex attribute) of each vertex that interacts with the given relation.

        Args:
            relation: A relation.

        Returns:
            A set of frozensets, where each frozenset contains the 'name' of vertices that are
            neighbors which interact with .
        """
        # get the vertices that interact w/ relation
        vertices = self._select_by_item(relation)

        return {
            frozenset(
                self.graph.vs[node]["item"] for node in self.graph.neighbors(rule_node)
            )
            for rule_node in vertices
        }
