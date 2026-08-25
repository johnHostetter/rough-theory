"""
White-box tests for RoughGranulation's item index (_item_index, _register_item,
_find_by_item, _select_by_item).

igraph has no index for arbitrary-Python-object vertex attributes, so
self.graph.vs.find(item_eq=x)/.select(item_eq=x) are each an O(V) linear scan
over every vertex in the graph. _item_index turns these into O(1) (amortized)
lookups by maintaining a dict mapping each "item" attribute value to the
indices of every vertex with that value, updated incrementally wherever a
vertex is added (set_granules()'s bulk insert, add_parent_relation()'s
one-at-a-time insert - the only two places vertices are ever added to
self.graph).

These tests directly exercise the private _register_item/_find_by_item/
_select_by_item methods and cross-check their results against igraph's own
item_eq scan, to prove the index is behaviorally equivalent - not merely "does
something" - rather than only exercising it incidentally through higher-level
methods.
"""

import unittest

from rough.granulation import RoughGranulation


class TestItemIndex(unittest.TestCase):
    """
    Covers _register_item/_find_by_item/_select_by_item directly, and their
    maintenance by set_granules()/add_parent_relation().
    """

    def test_register_and_find_single_item(self) -> None:
        """A freshly registered item is found by _find_by_item()."""
        granulation = RoughGranulation()
        granulation.graph.add_vertices(1, attributes={"item": ["x1"]})
        granulation._register_item("x1", 0)  # pylint: disable=protected-access
        found = granulation._find_by_item("x1")  # pylint: disable=protected-access
        self.assertEqual(found.index, 0)

    def test_find_by_item_matches_igraph_scan(self) -> None:
        """
        _find_by_item() must return the same vertex igraph's own
        vs.find(item_eq=...) linear scan would - proving it is a faster
        implementation of the same lookup, not a different one.
        """
        granulation = RoughGranulation()
        granulation.set_granules(["x1", "x2", "x3"], tags="element")

        expected = granulation.graph.vs.find(item_eq="x2")
        actual = granulation._find_by_item("x2")  # pylint: disable=protected-access
        self.assertEqual(actual.index, expected.index)

    def test_find_by_item_raises_for_missing_item(self) -> None:
        """
        _find_by_item() raises ValueError for an item that was never
        registered, matching igraph's vs.find(item_eq=...) behavior for no
        match.
        """
        granulation = RoughGranulation()
        granulation.set_granules(["x1"], tags="element")
        with self.assertRaises(ValueError):
            granulation._find_by_item(  # pylint: disable=protected-access
                "never_registered"
            )

    def test_select_by_item_returns_all_matching_vertices(self) -> None:
        """
        Multiple vertices can share the same "item" value (e.g. relation
        vertices from add_parent_relation()) - _select_by_item() must return
        every one of them, matching igraph's own vs.select(item_eq=...) scan.
        """
        granulation = RoughGranulation()
        granulation.set_granules(["x1", "x2", "x3"], tags="element")
        granulation.add_parent_relation("R", ({"x1"}, {"x2", "x3"}))

        expected = {v.index for v in granulation.graph.vs.select(item_eq="R")}
        matches = granulation._select_by_item("R")  # pylint: disable=protected-access
        actual = {v.index for v in matches}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 2)  # one vertex per compound in args

    def test_select_by_item_returns_empty_for_missing_item(self) -> None:
        """
        _select_by_item() returns an empty result (not an error) for an item
        that was never registered, matching igraph's own
        vs.select(item_eq=...) behavior for no match.
        """
        granulation = RoughGranulation()
        granulation.set_granules(["x1"], tags="element")
        result = granulation._select_by_item(  # pylint: disable=protected-access
            "never_registered"
        )
        self.assertEqual(len(result), 0)

    def test_set_granules_registers_every_item_at_its_true_vertex_index(
        self,
    ) -> None:
        """
        set_granules() is a single bulk add_vertices() call - the index must
        record each item's *actual* vertex index (start_index + offset), not
        just "some" index, including when items are added in more than one
        set_granules() call (indices continue to accumulate correctly).
        """
        granulation = RoughGranulation()
        granulation.set_granules(["x1", "x2"], tags="element")
        granulation.set_granules(["x3"], tags="element")  # a second batch

        self.assertEqual(
            granulation._find_by_item("x1").index,  # pylint: disable=protected-access
            0,
        )
        self.assertEqual(
            granulation._find_by_item("x2").index,  # pylint: disable=protected-access
            1,
        )
        self.assertEqual(
            granulation._find_by_item("x3").index,  # pylint: disable=protected-access
            2,
        )

    def test_add_parent_relation_registers_each_relation_vertex(self) -> None:
        """
        add_parent_relation() adds vertices one at a time (unlike
        set_granules()'s bulk insert) - each must still be individually
        registered in the index.
        """
        granulation = RoughGranulation()
        granulation.set_granules(["x1", "x2", "x3"], tags="element")
        granulation.add_parent_relation("R", ({"x1"}, {"x2"}, {"x3"}))

        matches = granulation._select_by_item("R")  # pylint: disable=protected-access
        self.assertEqual(len(matches), 3)


class TestCreateCompoundEdgesHelperItemLookup(unittest.TestCase):
    """
    Covers create_compound_edges_helper()'s two _find_by_item() call sites
    (source and target), including the not-found and multi-target-vertex
    branches that are not otherwise exercised by add_parent_relation()'s
    normal single-target call pattern.
    """

    def test_missing_source_item_raises_wrapped_value_error(self) -> None:
        """
        add_parent_relation() references a universe element that was never
        added via set_granules() - the underlying _find_by_item() ValueError
        must be caught and re-raised with a message naming the missing item.
        """
        granulation = RoughGranulation()
        granulation.set_granules(["x1", "x2"], tags="element")
        with self.assertRaisesRegex(ValueError, "never_added"):
            granulation.add_parent_relation("R", ({"x1", "never_added"},))

    def test_helper_accepts_an_actual_iterable_of_target_items(self) -> None:
        """
        create_compound_edges_helper()'s `targets` parameter is normally a
        single already-resolved vertex (add_parent_relation() only ever calls
        it that way - the surrounding try/except TypeError exists to detect
        exactly that non-iterable case). Calling it directly with a genuine
        iterable of target items instead exercises the for-loop branch that
        resolves each target via _find_by_item(), one edge per target.
        """
        granulation = RoughGranulation()
        granulation.set_granules(["x1", "x2", "x3"], tags="element")
        source_vertex = granulation._find_by_item(  # pylint: disable=protected-access
            "x1"
        )

        edges = []
        granulation.create_compound_edges_helper(source_vertex, edges, ["x2", "x3"])

        expected_targets = {
            granulation._find_by_item("x2").index,  # pylint: disable=protected-access
            granulation._find_by_item("x3").index,  # pylint: disable=protected-access
        }
        actual_targets = {target_index for _, target_index in edges}
        self.assertEqual(actual_targets, expected_targets)
        self.assertTrue(all(src == source_vertex.index for src, _ in edges))

    def test_helper_accepts_an_iterable_of_already_resolved_target_vertices(
        self,
    ) -> None:
        """
        Companion to the above: when the iterable's elements are already
        igraph.Vertex objects (not raw item values), each is used directly
        without a redundant _find_by_item() lookup.
        """
        granulation = RoughGranulation()
        granulation.set_granules(["x1", "x2", "x3"], tags="element")
        source_vertex = granulation._find_by_item(  # pylint: disable=protected-access
            "x1"
        )
        target_vertices = [
            granulation._find_by_item("x2"),  # pylint: disable=protected-access
            granulation._find_by_item("x3"),  # pylint: disable=protected-access
        ]

        edges = []
        granulation.create_compound_edges_helper(source_vertex, edges, target_vertices)

        expected_targets = {v.index for v in target_vertices}
        actual_targets = {target_index for _, target_index in edges}
        self.assertEqual(actual_targets, expected_targets)


if __name__ == "__main__":
    unittest.main()
