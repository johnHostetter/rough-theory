"""
Test functions that relation to creating a decision table, simplifying a decision table,
whether rules are consistent, etc.
"""

import unittest

from rough.decisions import RoughDecisions
from tests.test_knowledge_representation_system import make_example


class TestDecisionTable(unittest.TestCase):
    """
    Test rule consistency or decision table decomposition.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe, self.knowledge_base = make_example(class_to_test=RoughDecisions)
        self.set_c, self.set_d = {"a", "b", "c"}, {"d", "e"}

    def test_rule_consistency(self) -> None:
        """
        Test the consistency of rules returns the expected results.

        Returns:
            None
        """
        # a decision table is consistent iff set_c ==> set_d
        assert not self.knowledge_base.depends_on(self.set_c, self.set_d)

        equivalence_classes = self.knowledge_base.indiscernibility(self.set_c)
        inconsistent_rules = [
            indiscernible_rules
            for indiscernible_rules in equivalence_classes
            if len(indiscernible_rules) > 1
        ]

        # there should be 2 equivalent groups of rules, each containing 2 rules
        assert len(inconsistent_rules) == 2

    def test_table_decompose(self) -> None:
        """
        Test the decision table decomposition into two separate sets: rules that are consistent
        and rules that are inconsistent.

        Returns:
            None
        """
        (
            consistent_rules,
            inconsistent_rules,
        ) = self.knowledge_base.decompose_decision_table(self.set_c, self.set_d)
        assert consistent_rules == frozenset({3, 4, 6, 7})
        assert inconsistent_rules == frozenset({1, 2, 5, 8})


class TestSimplificationOfDecisionTable(unittest.TestCase):
    """
    Test the simplification of the decision table, such as whether an attribute is dispensable,
    and check that condition classes are correctly calculated.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = list(range(1, 8))
        self.knowledge_base = RoughDecisions()
        self.knowledge_base.set_granules(self.universe, tags="element")
        self.knowledge_base.add_parent_relation("a", ({3}, {1, 2, 4, 5}, {6, 7}))
        self.knowledge_base.add_parent_relation("b", ({1, 2, 3}, {4, 5, 6}, {7}))
        self.knowledge_base.add_parent_relation("c", ({1, 2, 3, 4, 5, 6}, {7}))
        self.knowledge_base.add_parent_relation("d", ({2, 3}, {1, 4}, {5, 6, 7}))
        self.knowledge_base.add_parent_relation("e", ({3, 4}, {1, 2}, {5, 6, 7}))
        self.set_c, self.set_d = {"a", "b", "c", "d"}, {"e"}

    def test_c_is_dispensable(self) -> None:
        """
        Test whether attribute 'c' is dispensable.

        Returns:
            None
        """
        assert self.knowledge_base.dispensable(
            self.set_c, "c", mode=self.knowledge_base.indiscernibility
        )

        # pick the first relative reduct
        (subset_of_set_c,) = self.knowledge_base.find_reducts(
            self.set_c, relative_to=self.set_d
        )
        assert subset_of_set_c == frozenset({"b", "a", "d"})
        assert self.knowledge_base.remove_redundant_attributes(
            self.set_c, self.set_d
        ) == frozenset({"b", "a", "d"})

    def test_condition_classes(self) -> None:
        """
        Test that condition classes are correctly calculated or stored.

        Returns:
            None
        """
        partition_in_each_attribute = self.knowledge_base[1]

        # pick the first relative reduct
        (subset_of_set_c,) = self.knowledge_base.find_reducts(
            self.set_c, relative_to=self.set_d
        )
        family_of_sets = {
            key: value
            for key, value in partition_in_each_attribute.items()
            if key in subset_of_set_c
        }
        assert frozenset.intersection(*family_of_sets.values()) == frozenset({1})

    def test_simplify_decision_table(self) -> None:
        """
        Test that decision tables are simplified as expected.

        Returns:
            None
        """
        (
            core_attributes,
            reduct_attributes,
        ) = self.knowledge_base.simplify_decision_table(self.set_c, self.set_d)
        assert core_attributes == {
            1: {"b"},
            2: {"a"},
            3: {"a"},
            4: {"b", "d"},
            5: {"d"},
        }
        assert reduct_attributes == {
            1: {frozenset({"b", "d"}), frozenset({"b", "a"})},
            2: {frozenset({"d", "a"}), frozenset({"b", "a"})},
            3: {frozenset({"a"})},
            4: {frozenset({"b", "d"})},
            5: {frozenset({"d"})},
            6: {frozenset({"a"}), frozenset({"d"})},
            7: {frozenset({"a"}), frozenset({"d"}), frozenset({"b"})},
        }

    def test_find_attribute_cores_skips_attributes_undefined_for_an_element(
        self,
    ) -> None:
        """
        Regression test: if none of a selected-attribute combination's members
        are actually defined for a given element (indiscernibility()'s own
        docstring notes "some elements might not be defined for all
        relations"), family_of_condition_attributes ends up empty and
        `frozenset.intersection(*[])` used to crash with a bare TypeError. Such
        a combination must be skipped instead - here "never_a_relation" is
        included in minimal_condition_attributes but was never added to the
        knowledge base at all, so it's undefined for every element.
        """
        core_attributes = self.knowledge_base.find_attribute_cores(
            frozenset({"a", "never_a_relation"})
        )
        # must not raise, and "a" alone (the only real relation) is used for
        # every element's evaluation
        self.assertIsInstance(core_attributes, dict)

    def test_find_attribute_reducts_skips_attributes_undefined_for_an_element(
        self,
    ) -> None:
        """
        Same regression as above, for find_attribute_reducts()'s matching guard.
        """
        reduct_attributes = self.knowledge_base.find_attribute_reducts(
            frozenset({"a", "never_a_relation"})
        )
        self.assertIsInstance(reduct_attributes, dict)
        # "a" alone must still be found as a valid reduct for every element,
        # proving the real combination wasn't skipped too, just the bogus one
        self.assertTrue(
            all(frozenset({"a"}) in reducts for reducts in reduct_attributes.values())
        )


class TestDecomposeDecisionTableWithNoBoundaryRegions(unittest.TestCase):
    """
    Regression test for decompose_decision_table()'s empty-union guard: with no
    boundary regions at all, `frozenset.union(*[])` used to crash with a bare
    TypeError, even though "nothing is inconsistent" (the empty set) is the
    mathematically correct, unambiguous answer.
    """

    def test_returns_empty_inconsistent_table_when_relations_fully_determine_decisions(
        self,
    ) -> None:
        """
        When condition_attributes fully determine decision_attributes (every
        boundary_region is empty), the union of zero non-empty boundary regions
        must be frozenset(), not a crash.
        """
        knowledge_base = RoughDecisions()
        knowledge_base.set_granules(["x1", "x2", "x3", "x4"], tags="element")
        knowledge_base.add_parent_relation("a", ({"x1", "x2"}, {"x3", "x4"}))
        knowledge_base.add_parent_relation("d", ({"x1", "x2"}, {"x3", "x4"}))

        consistent, inconsistent = knowledge_base.decompose_decision_table({"a"}, {"d"})
        self.assertEqual(inconsistent, frozenset())
        self.assertEqual(consistent, frozenset({"x1", "x2", "x3", "x4"}))
