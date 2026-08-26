"""
Test that discernibility is calculated correctly, such as building the discernibility matrix.
"""

import unittest

from rough.decisions import RoughDecisions


class TestDiscernibility(unittest.TestCase):
    """
    Test that discernibility is calculated correctly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = list(range(1, 6))
        self.knowledge_base = RoughDecisions()
        self.knowledge_base.set_granules(self.universe, tags="element")
        self.knowledge_base.add_parent_relation("a", ({1}, {2, 3, 5}, {4}))
        self.knowledge_base.add_parent_relation("b", ({3}, {1, 4, 5}, {2}))
        self.knowledge_base.add_parent_relation("c", ({2, 4, 5}, {3}, {1}))
        self.knowledge_base.add_parent_relation("d", ({1, 3}, {4}, {2, 5}))

    def test_discernibility_matrix(self) -> None:
        """
        Test the discernibility matrix is correctly built.

        Returns:
            None
        """
        relations = {"a", "b", "c", "d"}

        assert self.knowledge_base.discernibility_matrix(relations) == {
            frozenset({1, 2}): {"c", "b", "a", "d"},
            frozenset({1, 3}): {"c", "b", "a"},
            frozenset({1, 4}): {"c", "a", "d"},
            frozenset({1, 5}): {"c", "a", "d"},
            frozenset({2, 3}): {"c", "b", "d"},
            frozenset({2, 4}): {"b", "a", "d"},
            frozenset({2, 5}): {"b"},
            frozenset({3, 4}): {"c", "b", "a", "d"},
            frozenset({3, 5}): {"c", "b", "d"},
            frozenset({4, 5}): {"a", "d"},
        }

        assert self.knowledge_base.minimum_discernibility_matrix(relations) == {
            frozenset({1, 2}): frozenset({"b"}),
            frozenset({1, 3}): frozenset({"b"}),
            frozenset({1, 4}): frozenset({"a", "d"}),
            frozenset({1, 5}): frozenset({"a", "d"}),
            frozenset({2, 3}): frozenset({"b"}),
            frozenset({2, 4}): frozenset({"b"}),
            frozenset({2, 5}): frozenset({"b"}),
            frozenset({3, 4}): frozenset({"b"}),
            frozenset({3, 5}): frozenset({"b"}),
            frozenset({4, 5}): frozenset({"a", "d"}),
        }

    def test_discernibility_matrix_on_decision_attribute(self) -> None:
        """
        Test the discernibility matrix is correctly built when there is a decision attribute.

        Returns:
            None
        """
        relations = {"a", "b", "c"}

        assert self.knowledge_base.discernibility_matrix(relations, "d") == {
            frozenset({1, 2}): {"c", "b", "a"},
            frozenset({1, 4}): {"c", "a"},
            frozenset({1, 5}): {"c", "a"},
            frozenset({2, 3}): {"c", "b"},
            frozenset({2, 4}): {"b", "a"},
            frozenset({3, 4}): {"c", "b", "a"},
            frozenset({3, 5}): {"c", "b"},
            frozenset({4, 5}): {"a"},
        }

        assert self.knowledge_base.minimum_discernibility_matrix(
            relations, decision_attributes="d"
        ) == {
            frozenset({1, 2}): {"a"},
            frozenset({1, 4}): {"a"},
            frozenset({1, 5}): {"a"},
            frozenset({2, 3}): {"c", "b"},
            frozenset({2, 4}): {"a"},
            frozenset({3, 4}): {"a"},
            frozenset({3, 5}): {"c", "b"},
            frozenset({4, 5}): {"a"},
        }


class TestFindCoreByMatrixEmptyCore(unittest.TestCase):
    """
    Regression test for find_core_by_matrix()'s empty-CORE guard: the CORE is the
    union of every singleton matrix entry - with zero singleton entries, that used
    to crash with a bare TypeError (frozenset.union() with no arguments), even
    though an empty CORE is itself a perfectly normal, valid outcome (unlike an
    empty *intersection*, which is undefined - see find_core()'s own guard).
    """

    def test_returns_empty_frozenset_when_no_matrix_entry_is_a_singleton(
        self,
    ) -> None:
        """
        Two fully redundant relations ("P" and "Q" identical partitions) make
        every non-empty discernibility-matrix entry contain BOTH of them (never
        just one alone), so there are no singleton entries at all.
        """
        knowledge_base = RoughDecisions()
        knowledge_base.set_granules(["x1", "x2", "x3", "x4"], tags="element")
        knowledge_base.add_parent_relation("P", ({"x1", "x2"}, {"x3", "x4"}))
        knowledge_base.add_parent_relation("Q", ({"x1", "x2"}, {"x3", "x4"}))

        matrix = knowledge_base.discernibility_matrix({"P", "Q"})
        self.assertTrue(
            all(len(value) != 1 for value in matrix.values()),
            "test premise violated: matrix must have no singleton entries",
        )
        self.assertEqual(knowledge_base.find_core_by_matrix(matrix), frozenset())
