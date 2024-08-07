"""
A module that contains the RoughGranulation class, which is used to provide the necessary
granulation operations required for rough set theory. This class is inherited by more specialized
classes, such as RoughApproximation, to provide the basic granulation operations.
"""

from typing import Union, Dict, Set
from collections.abc import Iterable
from collections import Counter

import graphviz
import igraph as ig


class RoughGranulation:
    """
    A class that represents the necessary granulation operations required for rough set theory. This
    class is inherited by more specialized classes, such as RoughApproximation, to provide the basic
    granulation operations.
    """

    def __init__(self):
        self.graph = ig.Graph(directed=True)
        self.index: int = 0
        # keys: hashed frozenset or attribute name (if given) mapped to attribute values
        self.attribute_table = {}

    def __getitem__(self, item: Union[str, int]) -> Dict[str, list]:
        vertex = self.graph.vs.find(type_eq=item)
        neighbor_vertices = self.graph.vs[self.graph.neighbors(vertex)]

        # get any vertices from vertex's neighbors that actively apply a relation upon 'vertex'
        relations = [
            vertex["type"] for vertex in neighbor_vertices if vertex["type"] is not None
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
        equivalence_vertices = self.graph.vs.select(type_eq=other)
        for category in equivalence_vertices:
            nodes = self.graph.predecessors(category.index)
            vertices = self.graph.vs.select(nodes)
            elements = [vertex["type"] for vertex in vertices]
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

    @property
    def universe(self) -> frozenset:
        """
        Get the universe of discourse.

        Returns:
            A frozenset of elements in the universe of discourse.
        """
        return frozenset(self.graph.vs.select(layer_ne=None)["type"])

    def set_granules(
        self, nodes, source=None, is_input: Union[None, bool] = None
    ) -> None:
        """
        Adds the given nodes to the graph to a layer, and increments the saved indexed layer.

        Args:
            nodes: The nodes to be added.
            source: The source of the nodes.
            is_input: Whether this vertex should be considered 'input' (e.g., input granules or
            output granules). If None, then the 'input' attribute is not set. If True, then the
            'input' attribute is set to True. If False, then the 'input' attribute is set to False.
            Default is None.

        Returns:
            None
        """
        if nodes is not None and isinstance(nodes, Iterable):
            self.graph.add_vertices(
                len(nodes),
                attributes={
                    "type": list(nodes),
                    "layer": self.index,
                    "source": source,
                    "input": is_input,
                },
            )
            self.index += 1

    def create_compound_edges(self, args, target_vertices) -> list:
        """
        A helper method to self.add_parent_relation

        Args:
            args: A collection of lists, where each element in the collection (a list),
            stores the 'type' of the vertex in the graph.
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
                source_vertex = self.graph.vs.find(
                    type_eq=source
                )  # try to find its vertex
            except ValueError as exception:  # no such vertex;
                raise ValueError(
                    f"A vertex could not be found in the graph: {source}."
                ) from exception
        else:
            source_vertex = source
        try:
            for target in targets:
                if not isinstance(target, ig.Vertex):  # if the source is not a vertex
                    target_vertex = self.graph.vs.find(
                        type_eq=target
                    )  # try to find its vertex
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
            vertices.append(self.graph.add_vertex(type=attr_type))

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
                graphviz.render(
                    format=file_format, filepath=f"{filename}", engine=engine
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
        """

        categories = [
            next(iter(category)) for category in (self / relative_to).values()
        ]
        return frozenset.intersection(*categories)

    def family_union(self, relative_to: set) -> frozenset:
        """
        Get the union of a family of sets/categories 'relative_to' some set of relations.

        Args:
            relative_to: A selection of relations.

        Returns:
            The family union.
        """
        categories = [
            next(iter(category)) for category in (self / relative_to).values()
        ]
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
        vertices = self.graph.vs.select(type_eq=relation)

        return {
            frozenset(
                self.graph.vs[node]["type"] for node in self.graph.neighbors(rule_node)
            )
            for rule_node in vertices
        }
