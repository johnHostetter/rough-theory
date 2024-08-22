"""
Test the properties of rough sets and equivalence classes.
"""

import unittest

from rough.approximation import RoughApproximation


class TestRoughSets(unittest.TestCase):
    """
    Test the properties of rough sets, equivalence classes, and the like, using the example on
    page 13 of "Rough Sets: Theoretical Aspects of Reasoning About Data".
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = [f"x{i}" for i in range(1, 9)]
        self.knowledge_base = RoughApproximation()
        self.knowledge_base.set_granules(self.universe, tags="element")
        self.set_e_1 = {"x1", "x4", "x8"}
        self.set_e_2 = {"x2", "x5", "x7"}
        self.set_e_3 = {"x3"}
        self.set_e_4 = {"x6"}
        self.knowledge_base.add_parent_relation(
            "R", (self.set_e_1, self.set_e_2, self.set_e_3, self.set_e_4)
        )

    def test_equivalence_classes(self) -> None:
        """
        Test that the "equivalence classes" of a family of relations is correctly calculated.

        Returns:
            None
        """
        assert self.knowledge_base / "R" == frozenset(
            {
                frozenset(self.set_e_1),
                frozenset(self.set_e_2),
                frozenset(self.set_e_3),
                frozenset(self.set_e_4),
            }
        )

    def test_lower_approximation(self) -> None:
        """
        Test that the "lower approximation" (i.e., positive region) of a set is correctly
        calculated.

        Returns:
            None
        """
        set_x_1 = frozenset({"x1", "x4", "x7"})
        set_x_2 = frozenset({"x2", "x8"})

        # test that the lower approximation of a set is the union of the lower approximations
        # of its elements; test the positive region as well (they should be the same)
        assert (
            self.knowledge_base.lower_approximation("R", set_x_1)
            == self.knowledge_base.positive_region("R", set_x_1)
            == frozenset()
        )
        assert (
            self.knowledge_base.lower_approximation("R", set_x_2)
            == self.knowledge_base.positive_region("R", set_x_2)
            == frozenset()
        )
        assert (
            self.knowledge_base.lower_approximation("R", set_x_1.union(set_x_2))
            == self.knowledge_base.positive_region("R", set_x_1.union(set_x_2))
            == frozenset(self.set_e_1)
        )
        assert (
            self.knowledge_base.lower_approximation("R", set_x_1)
            .union(self.knowledge_base.lower_approximation("R", set_x_2))
            .issubset(
                self.knowledge_base.lower_approximation("R", set_x_1.union(set_x_2))
            )
        )
        assert (
            self.knowledge_base.positive_region("R", set_x_1)
            .union(self.knowledge_base.positive_region("R", set_x_2))
            .issubset(self.knowledge_base.positive_region("R", set_x_1.union(set_x_2)))
        )

    def test_upper_approximation(self) -> None:
        """
        Test that the "upper approximation" of a set is correctly calculated.

        Returns:
            None
        """
        set_y_1 = frozenset({"x1", "x3", "x5"})
        set_y_2 = frozenset({"x2", "x3", "x4", "x6"})

        assert self.knowledge_base.upper_approximation(
            "R", set_y_1.intersection(set_y_2)
        ) == frozenset(self.set_e_3)
        assert self.knowledge_base.upper_approximation("R", set_y_1) == frozenset(
            self.set_e_1
        ).union(self.set_e_2).union(self.set_e_3)
        assert self.knowledge_base.upper_approximation("R", set_y_2) == frozenset(
            self.set_e_1
        ).union(self.set_e_2).union(self.set_e_3).union(self.set_e_4)
        assert self.knowledge_base.upper_approximation("R", set_y_2) == frozenset(
            self.universe
        )
        assert self.knowledge_base.upper_approximation(
            "R", set_y_1.intersection(set_y_2)
        ).issubset(
            self.knowledge_base.upper_approximation("R", set_y_1).intersection(
                self.knowledge_base.upper_approximation("R", set_y_2)
            )
        )

    def test_boundary_region(self) -> None:
        """
        Test that the "boundary" of a set is correctly calculated.

        Returns:
            None
        """
        set_x_1 = frozenset({"x1", "x4", "x5"})
        set_x_2 = frozenset({"x3", "x5"})
        set_x_3 = frozenset({"x3", "x6", "x8"})

        assert self.knowledge_base.boundary_region("R", set_x_1) == frozenset(
            self.set_e_1
        ).union(self.set_e_2)
        assert self.knowledge_base.boundary_region("R", set_x_2) == frozenset(
            self.set_e_2
        )
        assert self.knowledge_base.boundary_region("R", set_x_3) == frozenset(
            self.set_e_1
        )

    def test_negative_region(self) -> None:
        """
        Test that the "negative" of a set is correctly calculated.

        Returns:
            None
        """
        set_x_1 = frozenset({"x1", "x4", "x5"})
        set_x_2 = frozenset({"x3", "x5"})
        set_x_3 = frozenset({"x3", "x6", "x8"})

        assert self.knowledge_base.negative_region("R", set_x_1) == frozenset(
            {"x3", "x6"}
        )
        assert self.knowledge_base.negative_region("R", set_x_2) == frozenset(
            self.set_e_1
        ).union(self.set_e_4)
        assert self.knowledge_base.negative_region("R", set_x_3) == frozenset(
            self.set_e_2
        )

    def test_accuracy(self) -> None:
        """
        Test that the "accuracy" of a set is correctly calculated.

        Returns:
            None
        """
        set_x_1 = frozenset({"x1", "x4", "x5"})
        set_x_2 = frozenset({"x3", "x5"})
        set_x_3 = frozenset({"x3", "x6", "x8"})

        assert self.knowledge_base.accuracy("R", set_x_1) == 0.0
        assert self.knowledge_base.accuracy("R", set_x_2) == 0.25
        assert self.knowledge_base.accuracy("R", set_x_3) == 0.4

    def test_roughness(self) -> None:
        """
        Test that the "roughness" of a set is correctly calculated.

        Returns:
            None
        """
        set_x_1 = frozenset({"x1", "x4", "x5"})
        set_x_2 = frozenset({"x3", "x5"})
        set_x_3 = frozenset({"x3", "x6", "x8"})

        # this should be the complement of accuracy
        assert self.knowledge_base.roughness("R", set_x_1) == 1
        assert self.knowledge_base.roughness("R", set_x_2) == 0.75
        assert self.knowledge_base.roughness("R", set_x_3) == 0.6


class TestApproximationOfClassifications(unittest.TestCase):
    """
    Test classifications using an existing equivalence relation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = [f"x{i}" for i in range(1, 9)]
        self.knowledge_base = RoughApproximation()
        self.knowledge_base.set_granules(self.universe, tags="element")
        self.set_x_1 = {"x1", "x3", "x5"}
        self.set_x_2 = {"x2", "x4"}
        self.set_x_3 = {"x6", "x7", "x8"}
        self.knowledge_base.add_parent_relation(
            "R", (self.set_x_1, self.set_x_2, self.set_x_3)
        )

    def test_classifications_1(self) -> None:
        """
        Example 1.

        Returns:
            None
        """
        set_y_1 = frozenset({"x1", "x2", "x4"})
        set_y_2 = frozenset({"x3", "x5", "x8"})
        set_y_3 = frozenset({"x6", "x7"})

        assert self.knowledge_base.lower_approximation("R", set_y_1) == frozenset(
            self.set_x_2
        )
        assert (
            self.knowledge_base.upper_approximation("R", set_y_2)
            == frozenset(self.set_x_1).union(self.set_x_3)
            != frozenset(self.universe)
        )
        assert (
            self.knowledge_base.upper_approximation("R", set_y_3)
            == frozenset(self.set_x_3)
            != frozenset(self.universe)
        )

    def test_classifications_2(self) -> None:
        """
        Example 2.

        Returns:
            None
        """
        set_z_1 = frozenset({"x1", "x2", "x6"})
        set_z_2 = frozenset({"x3", "x4"})
        set_z_3 = frozenset({"x5", "x7", "x8"})

        assert (
            self.knowledge_base.upper_approximation("R", set_z_1)
            == frozenset(self.set_x_1).union(self.set_x_2).union(self.set_x_3)
            == frozenset(self.universe)
        )
        assert self.knowledge_base.lower_approximation("R", set_z_2) == frozenset()
        assert self.knowledge_base.lower_approximation("R", set_z_3) == frozenset()
