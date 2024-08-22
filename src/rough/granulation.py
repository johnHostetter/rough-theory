"""
A module that contains the RoughGranulation class, which is used to provide the necessary
granulation operations required for rough set theory. This class is inherited by more specialized
classes, such as RoughApproximation, to provide the basic granulation operations.
"""

from typing import Union, Dict, Set, Any, List
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
        # keys: hashed frozenset or attribute name (if given) mapped to attribute values
        self.attribute_table = {}

    def __getitem__(self, item: Union[str, int]) -> Dict[str, list]:
        vertex = self.graph.vs.find(item_eq=item)
        neighbor_vertices = self.graph.vs[self.graph.neighbors(vertex)]

        # get any vertices from vertex's neighbors that actively apply a relation upon 'vertex'
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
        equivalence_vertices = self.graph.vs.select(item_eq=other)
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
            tags = [{tags}] * len(items)
        elif isinstance(tags, set):
            tags = [tags] * len(items)
        elif tags is not None:  # if tags is a list of sets
            assert len(items) == len(
                tags
            ), "The number of tags must match the number of items."
            assert all(
                isinstance(tag, set) for tag in tags
            ), "Items in 'tags' must be sets."
        self.graph.add_vertices(
            len(items),
            attributes={
                "item": items,
                "tags": tags,
                **kwargs,
            },
        )

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
                source_vertex = self.graph.vs.find(
                    item_eq=source
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
                        item_eq=target
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
            vertices.append(self.graph.add_vertex(item=attr_type, tags={"relation"}))

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
        vertices = self.graph.vs.select(item_eq=relation)

        return {
            frozenset(
                self.graph.vs[node]["item"] for node in self.graph.neighbors(rule_node)
            )
            for rule_node in vertices
        }
