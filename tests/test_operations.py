"""
White-box tests for rough.operations: the dispensable()/_as_relation_set() fix for
multi-character relation names, find_core()'s empty-reducts guard, and powerset().
"""

import unittest

from rough.operations import RoughOperations, _as_relation_set, powerset


class TestAsRelationSet(unittest.TestCase):
    """
    _as_relation_set() must treat a string relation name as ONE element, not as an
    iterable of its characters - the exact bug dispensable() used to have.
    """

    def test_wraps_a_multi_character_string_as_a_single_element(self) -> None:
        """`frozenset(relation)`/`set(relation)` would decompose a string into
        characters; _as_relation_set() must not."""
        self.assertEqual(_as_relation_set("temperature"), frozenset({"temperature"}))

    def test_wraps_a_single_character_string_as_a_single_element_too(self) -> None:
        """The single-character case (every existing example in this repo) must
        still behave the same as before the fix."""
        self.assertEqual(_as_relation_set("P"), frozenset({"P"}))

    def test_passes_through_an_already_a_set(self) -> None:
        """A set/frozenset form of `relation` (e.g. a singleton set, as used by
        dispensable()'s relative_to != None branch) is used as-is, not re-wrapped."""
        self.assertEqual(_as_relation_set({"P"}), frozenset({"P"}))
        self.assertEqual(_as_relation_set(frozenset({"P", "Q"})), frozenset({"P", "Q"}))


class TestDispensableWithMultiCharacterRelationNames(unittest.TestCase):
    """
    Regression test for the dispensable() bug: every existing test in this repo uses
    single-character relation names ('P', 'Q', 'R', ...), which accidentally worked
    even with the old, broken `frozenset(relation)` call (since frozenset("P") ==
    frozenset({"P"}) for a one-character string). A real, multi-character relation
    name is required to actually exercise the bug.
    """

    def setUp(self) -> None:
        self.knowledge_base = RoughOperations()
        self.knowledge_base.set_granules(["x1", "x2", "x3", "x4"], tags="element")
        # "temperature" is NOT redundant with "humidity": IND({temperature, humidity})
        # != IND({humidity}) alone (see test below), so temperature is indispensable.
        self.knowledge_base.add_parent_relation(
            "temperature", ({"x1", "x2"}, {"x3", "x4"})
        )
        self.knowledge_base.add_parent_relation(
            "humidity", ({"x1"}, {"x2", "x3", "x4"})
        )

    def test_indiscernibility_partitions_actually_differ(self) -> None:
        """
        Sanity check establishing the premise: dropping "temperature" really does
        change the indiscernibility partition, so "temperature" must be reported as
        indispensable (NOT dispensable).
        """
        self.assertNotEqual(
            self.knowledge_base.indiscernibility({"temperature", "humidity"}),
            self.knowledge_base.indiscernibility({"humidity"}),
        )

    def test_multi_character_relation_is_correctly_indispensable(self) -> None:
        """
        Before the fix: `frozenset("temperature")` decomposed into 7 unique
        characters, none of which equal the string "temperature" or "humidity", so
        `relations - frozenset(relation)` was a silent no-op and dispensable()
        always compared IND({temperature, humidity}) to itself - trivially equal,
        incorrectly reporting "temperature" as dispensable no matter what.
        """
        self.assertFalse(
            self.knowledge_base.dispensable(
                {"temperature", "humidity"},
                "temperature",
                mode=self.knowledge_base.indiscernibility,
            )
        )

    def test_family_is_correctly_reported_as_independent(self) -> None:
        """
        independent() calls dispensable() for every relation in the family - with
        the bug, "temperature" would incorrectly appear dispensable, making
        independent() incorrectly return False. Both relations are genuinely
        needed here, so the family must be reported as independent (True).
        """
        self.assertTrue(self.knowledge_base.independent({"temperature", "humidity"}))

    def test_relative_dispensable_branch_also_handles_multi_character_names(
        self,
    ) -> None:
        """
        dispensable()'s relative_to != None branch also builds
        `relations - _as_relation_set(relation)` (previously `relations -
        set(relation)`, the same character-decomposition bug) - exercise it
        directly with a multi-character relation name passed as a plain string
        (not pre-wrapped in a set, unlike every existing test of this branch).
        """
        self.knowledge_base.add_parent_relation(
            "occupancy", ({"x1", "x2", "x3"}, {"x4"})
        )
        # with relative_to={"occupancy"}, dropping "temperature" changes the
        # relative positive region (confirmed indirectly: the two relations
        # produce different relative positive regions), so "temperature" must
        # be indispensable here too.
        with_both = self.knowledge_base.find_relative_positive_region(
            {"temperature", "humidity"}, {"occupancy"}
        )
        without_temperature = self.knowledge_base.find_relative_positive_region(
            {"humidity"}, {"occupancy"}
        )
        self.assertNotEqual(with_both, without_temperature)
        self.assertFalse(
            self.knowledge_base.dispensable(
                {"temperature", "humidity"},
                "temperature",
                relative_to={"occupancy"},
                mode=self.knowledge_base.find_relative_positive_region,
            )
        )


class TestFindCoreEmptyReducts(unittest.TestCase):
    """
    find_core() computes the CORE as the intersection of ALL reducts - with zero
    reducts, that intersection is mathematically undefined (unlike an empty union,
    which is unambiguously the empty set), so it must raise a clear, specific error
    instead of a bare TypeError from `frozenset.intersection()` with no arguments.
    """

    def test_raises_a_clear_error_when_there_are_no_reducts(self) -> None:
        """
        Two fully redundant relations (identical partitions) make both relations
        dispensable simultaneously, so find_reducts() legitimately returns an empty
        frozenset (no candidate reduct keeps at least one indispensable relation).
        """
        knowledge_base = RoughOperations()
        knowledge_base.set_granules(["x1", "x2", "x3", "x4"], tags="element")
        knowledge_base.add_parent_relation("A", ({"x1", "x2"}, {"x3", "x4"}))
        knowledge_base.add_parent_relation("B", ({"x1", "x2"}, {"x3", "x4"}))

        self.assertEqual(knowledge_base.find_reducts({"A", "B"}), frozenset())
        with self.assertRaisesRegex(ValueError, "no reducts were found"):
            knowledge_base.find_core({"A", "B"})


class TestYDispensableRaises(unittest.TestCase):
    """
    Coverage/regression tests for y_dispensable()'s two raise branches -
    previously untested (test_knowledge.py only exercises the normal,
    non-raising path).
    """

    def setUp(self) -> None:
        self.knowledge_base = RoughOperations()
        self.knowledge_base.set_granules(
            ["x1", "x2", "x3", "x4", "x5", "x6"], tags="element"
        )
        self.knowledge_base.add_parent_relation("X", {frozenset({"x1", "x2", "x3"})})
        self.knowledge_base.add_parent_relation("T", {frozenset({"x1", "x3"})})

    def test_raises_when_category_is_not_an_element_of_categories(self) -> None:
        """The `category` argument must be a member of `categories`."""
        with self.assertRaisesRegex(
            ValueError, "category must be an element of categories"
        ):
            self.knowledge_base.y_dispensable({"X"}, "T", "not_a_category")

    def test_raises_when_family_intersection_is_not_a_subset_of_the_relation(
        self,
    ) -> None:
        """
        family_intersection({"X"}) == {x1, x2, x3}, which is NOT a subset of
        edges("T") == {x1, x3} (x2 is extra) - this must raise rather than
        silently proceed with an invalid premise.
        """
        with self.assertRaisesRegex(ValueError, "must be a subset of the given"):
            self.knowledge_base.y_dispensable({"X"}, "T", "X")


class TestIsYReductOfFRaises(unittest.TestCase):
    """
    Coverage/regression test for is_y_reduct_of_f()'s raise branch.
    """

    def test_raises_when_categories_h_is_not_a_subset_of_categories_f(self) -> None:
        knowledge_base = RoughOperations()
        knowledge_base.set_granules(["x1", "x2", "x3"], tags="element")
        knowledge_base.add_parent_relation("X", {frozenset({"x1", "x2"})})
        knowledge_base.add_parent_relation("T", {frozenset({"x1"})})

        with self.assertRaisesRegex(ValueError, "must be a subset of the family set"):
            knowledge_base.is_y_reduct_of_f({"X"}, {"X", "Y"}, "T")


class TestDispensableRelativeToRaises(unittest.TestCase):
    """
    Coverage/regression test for dispensable()'s relative_to != None branch's
    raise: `relation` must actually be an element of `relations`.
    """

    def test_raises_when_relation_is_not_in_relations(self) -> None:
        knowledge_base = RoughOperations()
        knowledge_base.set_granules(["x1", "x2", "x3"], tags="element")
        knowledge_base.add_parent_relation("X", {frozenset({"x1", "x2"})})
        knowledge_base.add_parent_relation("T", {frozenset({"x1"})})

        with self.assertRaisesRegex(ValueError, "must be an element of 'relations'"):
            knowledge_base.dispensable(
                {"X"},
                "not_in_relations",
                relative_to={"T"},
                mode=knowledge_base.find_relative_positive_region,
            )


class TestPartialDependsOnWithCategory(unittest.TestCase):
    """
    Coverage/regression test for partial_depends_on()'s `category is not None`
    branch (the "alternative definition" described in its own docstring) -
    previously flagged in-repo with a "# TODO: check this is reached by code
    coverage" comment, now confirmed reachable and covered; the TODO comment
    has been removed accordingly.
    """

    def test_alternative_definition_with_an_explicit_category(self) -> None:
        knowledge_base = RoughOperations()
        knowledge_base.set_granules(["x1", "x2", "x3", "x4"], tags="element")
        knowledge_base.add_parent_relation("X", {frozenset({"x1", "x2", "x3"})})

        category = frozenset({"x1", "x2", "x3"})
        result = knowledge_base.partial_depends_on(
            "X", other_relations=None, category=category
        )
        expected = len(knowledge_base.lower_approximation("X", category)) / len(
            category
        )
        self.assertEqual(result, expected)
        self.assertEqual(result, 1.0)


class TestPowerset(unittest.TestCase):
    """
    powerset() (re-exported from rough.utils - see TestPowersetInUtils for the
    canonical, more thorough tests) must still work when imported from
    rough.operations, since callers may import it from either module.
    """

    def test_powerset_of_a_set(self) -> None:
        """Basic sanity check that the re-export behaves identically."""
        result = list(powerset({"a", "b", "c"}, min_items=2))
        self.assertEqual(
            {frozenset(item) for item in result},
            {
                frozenset({"a", "b"}),
                frozenset({"a", "c"}),
                frozenset({"b", "c"}),
                frozenset({"a", "b", "c"}),
            },
        )


if __name__ == "__main__":
    unittest.main()
