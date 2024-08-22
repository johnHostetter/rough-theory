"""
Test the equivalence relation, indiscernibility, rough equality, and rough inclusion of sets.
"""

import unittest
from typing import List

from rough.approximation import RoughApproximation


def example_knowledge_base(
    knowledge_base: RoughApproximation, universe: List[str]
) -> None:
    """
    Apply granules and relations in-place to an example RoughApproximation object. Page 4 in book.

    Args:
        knowledge_base: The RoughApproximation object to apply granules and relations to.
        universe: The universe of discourse.

    Returns:
        None
    """
    knowledge_base.set_granules(universe, tags="element")
    knowledge_base.add_parent_relation(
        "R1", ({"x1", "x3", "x7"}, {"x2", "x4"}, {"x5", "x6", "x8"})
    )
    knowledge_base.add_parent_relation(
        "R2", ({"x1", "x5"}, {"x2", "x6"}, {"x3", "x4", "x7", "x8"})
    )
    knowledge_base.add_parent_relation(
        "R3", ({"x2", "x7", "x8"}, {"x1", "x3", "x4", "x5", "x6"})
    )


class TestEquivalenceRelation(unittest.TestCase):
    """
    Test the equivalence relation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.universe = [f"x{i}" for i in range(1, 9)]

    def test_get_category(self) -> None:
        """
        Test that categories are correctly stored by equivalence relation(s).

        Returns:
            None
        """
        knowledge_base = RoughApproximation()
        example_knowledge_base(knowledge_base, self.universe)

        # test that the equivalence relation is correctly stored.
        assert knowledge_base / "R1" == frozenset(
            {
                frozenset({"x8", "x6", "x5"}),
                frozenset({"x3", "x7", "x1"}),
                frozenset({"x2", "x4"}),
            }
        )
        # test that the equivalence relation is correctly stored.
        assert knowledge_base / "R2" == frozenset(
            {
                frozenset({"x2", "x6"}),
                frozenset({"x1", "x5"}),
                frozenset({"x3", "x8", "x4", "x7"}),
            }
        )
        # test that the equivalence relation is correctly stored.
        assert knowledge_base / "R3" == frozenset(
            {frozenset({"x2", "x8", "x7"}), frozenset({"x1", "x4", "x5", "x3", "x6"})}
        )

        expected_indexing_result = {
            "R1": frozenset({"x3", "x1", "x7"}),
            "R2": frozenset({"x1", "x5"}),
            "R3": frozenset({"x3", "x1", "x4", "x5", "x6"}),
        }

        # test that the element "x1" belongs to the correct equivalence classes.
        assert knowledge_base["x1"] == expected_indexing_result

        assert knowledge_base["x1"]["R1"].intersection(
            knowledge_base["x3"]["R2"]
        ) == frozenset({"x3", "x7"})
        assert knowledge_base["x2"]["R1"].intersection(
            knowledge_base["x2"]["R2"]
        ) == frozenset({"x2"})
        assert knowledge_base["x5"]["R1"].intersection(
            knowledge_base["x3"]["R2"]
        ) == frozenset({"x8"})

        assert knowledge_base["x1"]["R1"].intersection(
            knowledge_base["x3"]["R2"]
        ).intersection(knowledge_base["x2"]["R3"]) == frozenset({"x7"})
        assert knowledge_base["x2"]["R1"].intersection(
            knowledge_base["x2"]["R2"]
        ).intersection(knowledge_base["x2"]["R3"]) == frozenset({"x2"})
        assert knowledge_base["x5"]["R1"].intersection(
            knowledge_base["x3"]["R2"]
        ).intersection(knowledge_base["x2"]["R3"]) == frozenset({"x8"})

        assert knowledge_base["x1"]["R1"].union(
            knowledge_base["x2"]["R1"]
        ) == frozenset({"x1", "x2", "x3", "x4", "x7"})
        assert knowledge_base["x2"]["R1"].union(
            knowledge_base["x5"]["R1"]
        ) == frozenset({"x2", "x4", "x5", "x6", "x8"})
        assert knowledge_base["x1"]["R1"].union(
            knowledge_base["x5"]["R1"]
        ) == frozenset({"x1", "x3", "x5", "x6", "x7", "x8"})

        assert knowledge_base["x2"]["R1"] == frozenset(("x2", "x4"))
        assert knowledge_base["x1"]["R2"] == frozenset(("x1", "x5"))
        assert (
            knowledge_base["x2"]["R1"].intersection(knowledge_base["x1"]["R2"])
            == frozenset()
        )

        assert knowledge_base["x1"]["R1"] == frozenset(("x1", "x3", "x7"))
        assert knowledge_base["x2"]["R2"] == frozenset(("x2", "x6"))
        assert (
            knowledge_base["x1"]["R1"].intersection(knowledge_base["x2"]["R2"])
            == frozenset()
        )


class TestIndiscernibilityRelation(unittest.TestCase):
    """
    Test the indiscernibility relation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = [f"x{i}" for i in range(1, 10)]

    def test_indiscernibility(self) -> None:
        """
        Test the indiscernibility relation calculates as expected on page 5 of the book.

        Returns:
            None
        """
        knowledge_base = RoughApproximation()
        example_knowledge_base(knowledge_base, self.universe)

        # the given equivalence relation argument must have at least one element
        self.assertRaises(ValueError, knowledge_base.indiscernibility, [])
        # test that the indiscernibility relation is calculated correctly
        assert knowledge_base.indiscernibility(["R1", "R2"]) == {
            frozenset({"x2"}),
            frozenset({"x5"}),
            frozenset({"x6"}),
            frozenset({"x4"}),
            frozenset({"x1"}),
            frozenset({"x3", "x7"}),
            frozenset({"x8"}),
        }


class TestRoughEqualityOfSets(unittest.TestCase):
    """
    Test the rough equality of sets.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = [f"x{i}" for i in range(1, 9)]
        self.knowledge_base = RoughApproximation()
        self.knowledge_base.set_granules(self.universe, tags="element")
        self.set_e_1 = {"x2", "x3"}
        self.set_e_2 = {"x1", "x4", "x5"}
        self.set_e_3 = {"x6"}
        self.set_e_4 = {"x7", "x8"}
        self.knowledge_base.add_parent_relation(
            "R", (self.set_e_1, self.set_e_2, self.set_e_3, self.set_e_4)
        )

    def test_bottom_rough_equal(self) -> None:
        """
        Test the rough bottom equality of sets is correctly calculated.

        Returns:
            None
        """
        set_x_1 = frozenset({"x1", "x2", "x3"})
        set_x_2 = frozenset({"x2", "x3", "x7"})
        # test that the lower approximation is calculated correctly for the given set
        assert self.knowledge_base.lower_approximation("R", set_x_1) == frozenset(
            self.set_e_1
        )
        # test that the lower approximation is calculated correctly for the given set
        assert self.knowledge_base.lower_approximation("R", set_x_2) == frozenset(
            self.set_e_1
        )
        # test that the rough bottom equality is calculated correctly for the given sets,
        # since the lower approximation of both sets is the same, they are bottom-roughly equal
        assert self.knowledge_base.roughly_equal("R", set_x_1, set_x_2, mode="bottom")

    def test_top_rough_equal(self) -> None:
        """
        Test the rough top equality of sets is correctly calculated.

        Returns:
            None
        """
        set_y_1 = frozenset({"x1", "x2", "x7"})
        set_y_2 = frozenset({"x2", "x3", "x4", "x8"})
        # test that the upper approximation is calculated correctly for the given set
        assert self.knowledge_base.upper_approximation("R", set_y_1) == frozenset(
            self.set_e_1
        ).union(self.set_e_2).union(self.set_e_4)
        # test that the upper approximation is calculated correctly for the given set
        assert self.knowledge_base.upper_approximation("R", set_y_2) == frozenset(
            self.set_e_1
        ).union(self.set_e_2).union(self.set_e_4)
        # test that the rough top equality is calculated correctly, since the upper approximation
        # of the first set is equal to the upper approximation of the second set, then the sets
        # are top-roughly equal
        assert self.knowledge_base.roughly_equal("R", set_y_1, set_y_2, mode="top")

    def test_rough_equal(self) -> None:
        """
        Test the rough equality of sets is correctly calculated.

        Returns:
            None
        """
        set_z_1 = frozenset({"x1", "x2", "x6"})
        set_z_2 = frozenset({"x3", "x4", "x6"})
        # test that the lower approximation is calculated correctly for the given set
        assert self.knowledge_base.lower_approximation("R", set_z_1) == frozenset(
            self.set_e_3
        )
        # test that the lower approximation is calculated correctly for the given set
        assert self.knowledge_base.lower_approximation("R", set_z_2) == frozenset(
            self.set_e_3
        )
        # test that the upper approximation is calculated correctly for the given set
        assert self.knowledge_base.upper_approximation("R", set_z_1) == frozenset(
            self.set_e_1
        ).union(self.set_e_2).union(self.set_e_3)
        # test that the upper approximation is calculated correctly for the given set
        assert self.knowledge_base.upper_approximation("R", set_z_2) == frozenset(
            self.set_e_1
        ).union(self.set_e_2).union(self.set_e_3)
        # test that the rough equality is calculated correctly, since the lower and upper
        # approximations are equal, the sets are roughly equal
        assert self.knowledge_base.roughly_equal("R", set_z_1, set_z_2)

    def test_invalid_mode_argument(self) -> None:
        """
        Test that an invalid mode argument raises a ValueError.

        Returns:
            None
        """
        set_z_1 = frozenset({"x1", "x2", "x6"})
        set_z_2 = frozenset({"x3", "x4", "x6"})
        self.assertRaises(
            ValueError,
            self.knowledge_base.roughly_equal,
            "R",
            set_z_1,
            set_z_2,
            "invalid",
        )


class TestRoughInclusionOfSets(unittest.TestCase):
    """
    Test the rough inclusion of sets.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = [f"x{i}" for i in range(1, 9)]
        self.knowledge_base = RoughApproximation()
        self.knowledge_base.set_granules(self.universe, tags="element")
        self.set_e_1 = {"x2", "x3"}
        self.set_e_2 = {"x1", "x4", "x5"}
        self.set_e_3 = {"x6"}
        self.set_e_4 = {"x7", "x8"}
        self.knowledge_base.add_parent_relation(
            "R", (self.set_e_1, self.set_e_2, self.set_e_3, self.set_e_4)
        )

    def test_bottom_rough_included(self) -> None:
        """
        Test the rough bottom inclusion of sets is correctly calculated.

        Returns:
            None
        """
        set_x_1 = frozenset({"x2", "x4", "x6", "x7"})
        set_x_2 = frozenset({"x2", "x3", "x4", "x6"})
        # test that the lower approximation is calculated correctly for the given set
        assert self.knowledge_base.lower_approximation("R", set_x_1) == frozenset(
            self.set_e_3
        )
        # test that the lower approximation is calculated correctly for the given set
        assert self.knowledge_base.lower_approximation("R", set_x_2) == frozenset(
            self.set_e_1
        ).union(self.set_e_3)
        # test that the rough bottom inclusion is calculated correctly, since the lower
        # approximation of the first set is roughly included in the lower approximation of
        # the second set, then the first set is bottom-roughly included in the second set
        assert self.knowledge_base.roughly_included(
            "R", set_x_1, set_x_2, mode="bottom"
        )

    def test_top_rough_included(self) -> None:
        """
        Test the rough top inclusion of sets is correctly calculated.

        Returns:
            None
        """
        set_y_1 = frozenset({"x2", "x3", "x7"})
        set_y_2 = frozenset({"x1", "x2", "x7"})
        # test that the upper approximation is calculated correctly for the given set
        assert self.knowledge_base.upper_approximation("R", set_y_1) == frozenset(
            self.set_e_1
        ).union(self.set_e_4)
        # test that the upper approximation is calculated correctly for the given set
        assert self.knowledge_base.upper_approximation("R", set_y_2) == frozenset(
            self.set_e_1
        ).union(self.set_e_2).union(self.set_e_4)
        # test that the rough top inclusion is calculated correctly, since the upper
        # approximation of the first set is roughly included in the upper approximation of
        # the second set, then the first set is top-roughly included in the second set
        assert self.knowledge_base.roughly_included("R", set_y_1, set_y_2, mode="top")

    def test_rough_included(self) -> None:
        """
        Test the rough inclusion of sets is correctly calculated.

        Returns:
            None
        """
        set_z_1 = frozenset({"x2", "x3"})
        set_z_2 = frozenset({"x1", "x2", "x3", "x7"})
        # test that the lower approximation is calculated correctly for the given set
        assert self.knowledge_base.lower_approximation("R", set_z_1) == frozenset(
            self.set_e_1
        )
        # test that the lower approximation is calculated correctly for the given set
        assert self.knowledge_base.lower_approximation("R", set_z_2) == frozenset(
            self.set_e_1
        )
        # test that the upper approximation is calculated correctly for the given set
        assert self.knowledge_base.upper_approximation("R", set_z_1) == frozenset(
            self.set_e_1
        )
        # test that the upper approximation is calculated correctly for the given set
        assert self.knowledge_base.upper_approximation("R", set_z_2) == frozenset(
            self.set_e_1
        ).union(self.set_e_2).union(self.set_e_4)
        # test that the rough inclusion is calculated correctly, since the lower
        # approximation of the first set is roughly included in the lower approximation of
        # the second set and the upper approximation of the first set is roughly included
        # in the upper approximation of the second set, then the first set is roughly
        # included in the second set
        assert self.knowledge_base.roughly_included("R", set_z_1, set_z_2)

    def test_invalid_mode_argument(self) -> None:
        """
        Test the rough inclusion throws a ValueError if the mode argument is invalid.

        Returns:
            None
        """
        set_z_1 = frozenset({"x2", "x3"})
        set_z_2 = frozenset({"x1", "x2", "x3", "x7"})
        with self.assertRaises(ValueError):
            self.knowledge_base.roughly_included("R", set_z_1, set_z_2, mode="invalid")
