"""
Test the definability (e.g., roughly definable, totally undefinable) of sets
given a family of relations.
"""

import unittest

from rough.approximation import RoughApproximation


class TestDefinable(unittest.TestCase):
    """
    Test the various forms of definability return the expected results.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = [f"x{i}" for i in range(0, 11)]
        self.knowledge_base = RoughApproximation()
        self.knowledge_base.set_granules(self.universe, tags="element")
        self.set_e_1 = {"x0", "x1"}
        self.set_e_2 = {"x2", "x6", "x9"}
        self.set_e_3 = {"x3", "x5"}
        self.set_e_4 = {"x4", "x8"}
        self.set_e_5 = {"x7", "x10"}
        self.knowledge_base.add_parent_relation(
            "R", (self.set_e_1, self.set_e_2, self.set_e_3, self.set_e_4, self.set_e_5)
        )

    def test_definable(self) -> None:
        """
        Test if the sets are defineable, with respect to the
        given family of relations 'R'.

        Returns:
            None
        """
        set_x_1 = frozenset({"x0", "x1", "x4", "x8"})
        set_y_1 = frozenset({"x3", "x4", "x5", "x8"})
        set_z_1 = frozenset({"x2", "x3", "x5", "x6", "x9"})

        assert type(self.knowledge_base.definable("R", set_x_1)).__name__ == "Definable"
        assert type(self.knowledge_base.definable("R", set_y_1)).__name__ == "Definable"
        assert type(self.knowledge_base.definable("R", set_z_1)).__name__ == "Definable"

    def test_roughly_definable(self) -> None:
        """
        Test if the sets are roughly definable, with respect to the
        given family of relations 'R'.

        Returns:
            None
        """
        set_x_2 = frozenset({"x0", "x3", "x4", "x5", "x8", "x10"})
        set_y_2 = frozenset({"x1", "x7", "x8", "x10"})
        set_z_2 = frozenset({"x2", "x3", "x4", "x8"})

        assert (
            type(self.knowledge_base.definable("R", set_x_2)).__name__
            == "RoughlyDefinable"
        )
        assert (
            type(self.knowledge_base.definable("R", set_y_2)).__name__
            == "RoughlyDefinable"
        )
        assert (
            type(self.knowledge_base.definable("R", set_z_2)).__name__
            == "RoughlyDefinable"
        )

        # the approximations

        assert self.knowledge_base.lower_approximation("R", set_x_2) == frozenset(
            self.set_e_3
        ).union(self.set_e_4)
        assert self.knowledge_base.upper_approximation("R", set_x_2) == frozenset(
            self.set_e_1
        ).union(self.set_e_3).union(self.set_e_4).union(self.set_e_5)

        assert self.knowledge_base.lower_approximation("R", set_y_2) == frozenset(
            self.set_e_5
        )
        assert self.knowledge_base.upper_approximation("R", set_y_2) == frozenset(
            self.set_e_1
        ).union(self.set_e_4).union(self.set_e_5)

        assert self.knowledge_base.lower_approximation("R", set_z_2) == frozenset(
            self.set_e_4
        )
        assert self.knowledge_base.upper_approximation("R", set_z_2) == frozenset(
            self.set_e_2
        ).union(self.set_e_3).union(self.set_e_4)

        # the boundaries

        assert self.knowledge_base.boundary_region("R", set_x_2) == frozenset(
            self.set_e_1
        ).union(self.set_e_5)
        assert self.knowledge_base.boundary_region("R", set_y_2) == frozenset(
            self.set_e_1
        ).union(self.set_e_4)
        assert self.knowledge_base.boundary_region("R", set_z_2) == frozenset(
            self.set_e_2
        ).union(self.set_e_3)

        # the accuracies

        assert self.knowledge_base.accuracy("R", set_x_2) == 1 / 2
        assert self.knowledge_base.accuracy("R", set_y_2) == 1 / 3
        assert self.knowledge_base.accuracy("R", set_z_2) == 2 / 7

    def test_externally_undefinable(self) -> None:
        """
        Test if the sets are externally undefineable, with respect to the
        given family of relations 'R'.

        Returns:
            None
        """
        set_x_3 = frozenset({"x0", "x1", "x2", "x3", "x4", "x7"})
        set_y_3 = frozenset({"x1", "x2", "x3", "x6", "x8", "x9", "x10"})
        set_z_3 = frozenset({"x0", "x2", "x3", "x4", "x8", "x10"})

        assert (
            type(self.knowledge_base.definable("R", set_x_3)).__name__
            == "ExternallyUndefinable"
        )
        assert (
            type(self.knowledge_base.definable("R", set_y_3)).__name__
            == "ExternallyUndefinable"
        )
        assert (
            type(self.knowledge_base.definable("R", set_z_3)).__name__
            == "ExternallyUndefinable"
        )

        # the approximations

        assert self.knowledge_base.lower_approximation("R", set_x_3) == frozenset(
            self.set_e_1
        )
        assert self.knowledge_base.upper_approximation("R", set_x_3) == frozenset(
            self.universe
        )

        assert self.knowledge_base.lower_approximation("R", set_y_3) == frozenset(
            self.set_e_2
        )
        assert self.knowledge_base.upper_approximation("R", set_y_3) == frozenset(
            self.universe
        )

        assert self.knowledge_base.lower_approximation("R", set_z_3) == frozenset(
            self.set_e_4
        )
        assert self.knowledge_base.upper_approximation("R", set_z_3) == frozenset(
            self.universe
        )

        # the boundaries

        assert self.knowledge_base.boundary_region("R", set_x_3) == frozenset(
            self.set_e_2
        ).union(self.set_e_3).union(self.set_e_4).union(self.set_e_5)
        assert self.knowledge_base.boundary_region("R", set_y_3) == frozenset(
            self.set_e_1
        ).union(self.set_e_3).union(self.set_e_4).union(self.set_e_5)
        assert self.knowledge_base.boundary_region("R", set_z_3) == frozenset(
            self.set_e_1
        ).union(self.set_e_2).union(self.set_e_3).union(self.set_e_5)

        # the accuracies

        assert self.knowledge_base.accuracy("R", set_x_3) == 2 / 11
        assert self.knowledge_base.accuracy("R", set_y_3) == 3 / 11
        assert self.knowledge_base.accuracy("R", set_z_3) == 2 / 11

    def test_internally_undefinable(self) -> None:
        """
        Test if the sets are internally undefineable, with respect to the
        given family of relations 'R'.

        Returns:
            None
        """
        set_x_4 = frozenset({"x0", "x2", "x3"})
        set_y_4 = frozenset({"x1", "x2", "x4", "x7"})
        set_z_4 = frozenset({"x2", "x3", "x4"})

        assert (
            type(self.knowledge_base.definable("R", set_x_4)).__name__
            == "InternallyUndefinable"
        )
        assert (
            type(self.knowledge_base.definable("R", set_y_4)).__name__
            == "InternallyUndefinable"
        )
        assert (
            type(self.knowledge_base.definable("R", set_z_4)).__name__
            == "InternallyUndefinable"
        )

        # the approximations

        assert self.knowledge_base.upper_approximation("R", set_x_4) == frozenset(
            self.set_e_1
        ).union(self.set_e_2).union(self.set_e_3)
        assert self.knowledge_base.upper_approximation("R", set_y_4) == frozenset(
            self.set_e_1
        ).union(self.set_e_2).union(self.set_e_4).union(self.set_e_5)
        assert self.knowledge_base.upper_approximation("R", set_z_4) == frozenset(
            self.set_e_2
        ).union(self.set_e_3).union(self.set_e_4)

    def test_totally_undefinable(self) -> None:
        """
        Test if the sets are totally undefineable, with respect to the
        given family of relations 'R'.

        Returns:
            None
        """
        set_x_5 = frozenset({"x0", "x2", "x3", "x4", "x7"})
        set_y_5 = frozenset({"x1", "x5", "x6", "x8", "x10"})
        set_z_5 = frozenset({"x0", "x2", "x4", "x5", "x7"})

        assert (
            type(self.knowledge_base.definable("R", set_x_5)).__name__
            == "TotallyUndefinable"
        )
        assert (
            type(self.knowledge_base.definable("R", set_y_5)).__name__
            == "TotallyUndefinable"
        )
        assert (
            type(self.knowledge_base.definable("R", set_z_5)).__name__
            == "TotallyUndefinable"
        )

    def test_family_of_categories_matches_individual_calls(self) -> None:
        """
        Coverage/regression test: approximation()'s list-of-categories branch
        (used by lower_approximation()/upper_approximation() when passed a
        list rather than a single frozenset) was previously untested -
        confirm it returns the set of each individual category's own
        approximation, not something else.

        Returns:
            None
        """
        set_x_2 = frozenset({"x0", "x3", "x4", "x5", "x8", "x10"})
        set_y_2 = frozenset({"x1", "x7", "x8", "x10"})
        set_z_2 = frozenset({"x2", "x3", "x4", "x8"})
        categories = [set_x_2, set_y_2, set_z_2]

        expected_lowers = {
            self.knowledge_base.lower_approximation("R", category)
            for category in categories
        }
        expected_uppers = {
            self.knowledge_base.upper_approximation("R", category)
            for category in categories
        }
        self.assertEqual(
            self.knowledge_base.lower_approximation("R", categories), expected_lowers
        )
        self.assertEqual(
            self.knowledge_base.upper_approximation("R", categories), expected_uppers
        )

    def test_quality_of_approximation_over_a_family_of_categories(self) -> None:
        """
        Coverage/regression test: quality_of_approximation()'s family-of-
        categories branch was previously untested. Expected value derived
        independently: sum of each category's own lower_approximation size
        (4 + 2 + 2 = 8) over the universe size (11).

        Returns:
            None
        """
        set_x_2 = frozenset({"x0", "x3", "x4", "x5", "x8", "x10"})
        set_y_2 = frozenset({"x1", "x7", "x8", "x10"})
        set_z_2 = frozenset({"x2", "x3", "x4", "x8"})

        self.assertEqual(
            self.knowledge_base.quality_of_approximation(
                "R", [set_x_2, set_y_2, set_z_2]
            ),
            8 / 11,
        )

    def test_accuracy_over_a_family_of_categories(self) -> None:
        """
        Coverage/regression test: accuracy()'s family-of-categories branch
        was previously untested. Expected value derived independently: sum
        of each category's own lower_approximation size (4 + 2 + 2 = 8) over
        the sum of each category's own upper_approximation size
        (8 + 6 + 7 = 21) - each individual accuracy already established in
        test_roughly_definable (1/2, 1/3, 2/7) is a per-category ratio, not
        the same thing as this pooled family ratio.

        Returns:
            None
        """
        set_x_2 = frozenset({"x0", "x3", "x4", "x5", "x8", "x10"})
        set_y_2 = frozenset({"x1", "x7", "x8", "x10"})
        set_z_2 = frozenset({"x2", "x3", "x4", "x8"})

        self.assertEqual(
            self.knowledge_base.accuracy("R", [set_x_2, set_y_2, set_z_2]),
            8 / 21,
        )

    def test_raises_for_an_unsupported_categories_type(self) -> None:
        """
        Coverage/regression test: approximation()'s final else-branch raise,
        for a `categories` argument that has a length but is neither a set,
        frozenset, nor list (e.g. a tuple) - previously untested.

        Returns:
            None
        """
        set_x_2 = frozenset({"x0", "x3", "x4", "x5", "x8", "x10"})
        with self.assertRaisesRegex(
            ValueError, "must be a set, a frozenset, or a list"
        ):
            self.knowledge_base.lower_approximation("R", (set_x_2,))

    def test_raises_when_a_category_in_the_list_is_empty(self) -> None:
        """
        Coverage/regression test: approximation()'s list-of-categories branch
        must reject an empty element within the list, not just an empty
        `categories` argument overall - previously untested.

        Returns:
            None
        """
        set_x_2 = frozenset({"x0", "x3", "x4", "x5", "x8", "x10"})
        with self.assertRaisesRegex(
            ValueError, "may not have an element with a length of zero"
        ):
            self.knowledge_base.lower_approximation("R", [set_x_2, frozenset()])

    def test_quality_of_approximation_raises_for_an_empty_categories_argument(
        self,
    ) -> None:
        """
        Coverage/regression test: quality_of_approximation()'s own
        len(categories) == 0 guard - previously untested (the sibling guards
        on lower/upper_approximation/boundary_region/accuracy/definable were
        already covered, this one wasn't).

        Returns:
            None
        """
        with self.assertRaisesRegex(ValueError, "may not have a length of zero"):
            self.knowledge_base.quality_of_approximation("R", frozenset())

    def test_quality_of_approximation_single_category(self) -> None:
        """
        Coverage/regression test: quality_of_approximation()'s non-list
        (single-category) branch was previously untested - every existing
        test either used a family (list) or didn't call this method at all.

        Returns:
            None
        """
        set_x_2 = frozenset({"x0", "x3", "x4", "x5", "x8", "x10"})
        self.assertEqual(
            self.knowledge_base.quality_of_approximation("R", set_x_2),
            len(self.knowledge_base.lower_approximation("R", set_x_2)) / 11,
        )

    def test_quality_of_approximation_raises_when_a_category_is_empty(self) -> None:
        """
        Coverage/regression test: quality_of_approximation()'s family-of-
        categories branch must reject an empty element - previously
        untested.

        Returns:
            None
        """
        set_x_2 = frozenset({"x0", "x3", "x4", "x5", "x8", "x10"})
        with self.assertRaisesRegex(
            ValueError, "may not have an element with a length of zero"
        ):
            self.knowledge_base.quality_of_approximation("R", [set_x_2, frozenset()])

    def test_accuracy_raises_when_a_category_is_empty(self) -> None:
        """
        Coverage/regression test: accuracy()'s family-of-categories branch
        must reject an empty element - previously untested.

        Returns:
            None
        """
        set_x_2 = frozenset({"x0", "x3", "x4", "x5", "x8", "x10"})
        with self.assertRaisesRegex(
            ValueError, "may not have an element with a length of zero"
        ):
            self.knowledge_base.accuracy("R", [set_x_2, frozenset()])

    def test_invalid_argument_for_lower_approximation(self) -> None:
        """
        Test the argument 'categories' may not have a length of zero.

        Returns:
            None
        """
        self.assertRaises(
            ValueError,
            self.knowledge_base.lower_approximation,
            relations="R",
            categories=frozenset({}),
        )

    def test_invalid_argument_for_upper_approximation(self) -> None:
        """
        Test the argument 'categories' may not have a length of zero.

        Returns:
            None
        """
        self.assertRaises(
            ValueError,
            self.knowledge_base.upper_approximation,
            relations="R",
            categories=frozenset({}),
        )

    def test_invalid_argument_for_boundary_region(self) -> None:
        """
        Test the argument 'categories' may not have a length of zero.

        Returns:
            None
        """
        self.assertRaises(
            ValueError,
            self.knowledge_base.boundary_region,
            relations="R",
            categories=frozenset({}),
        )

    def test_invalid_argument_for_accuracy(self) -> None:
        """
        Test the argument 'categories' may not have a length of zero.

        Returns:
            None
        """
        self.assertRaises(
            ValueError,
            self.knowledge_base.accuracy,
            relations="R",
            category=frozenset({}),
        )

    def test_invalid_argument_for_definable(self) -> None:
        """
        Test the argument 'categories' may not have a length of zero.

        Returns:
            None
        """
        self.assertRaises(
            ValueError,
            self.knowledge_base.definable,
            relations="R",
            category=frozenset({}),
        )
