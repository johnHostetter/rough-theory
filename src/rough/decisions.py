"""
Implements the methods required to work with rough theory.
"""

import itertools

from rough.operations import RoughOperations


class RoughDecisions(RoughOperations):
    """
    This class implements the necessary methods for handling decisions in rough set theory. For
    example, the discernibility matrix, the CORE, and the reduct. Features include the ability to
    simplify decision tables as well as remove redundant attributes.
    """

    # begin of discernibility methods

    def __delete_indiscernable_comparison(self, matrix, key, relations):
        if (
            any(
                key.issubset(decision_class)
                for decision_class in self.indiscernibility(relations)
            )
            and key in matrix
        ):
            del matrix[key]

    def __discernibility_matrix_helper(self, matrix, relations, func: callable) -> dict:
        for relation in relations:
            equivalence_classes = self / relation
            for equivalence_class in equivalence_classes:
                pairs = itertools.combinations(equivalence_class, r=2)
                for item_1, item_2 in pairs:
                    key = frozenset({item_1, item_2})
                    func(matrix, key, relation)
        return matrix

    def discernibility_matrix(self, relations, decision_attributes=None) -> dict:
        """
        Calculate the discernibility matrix for a set of relations.

        Args:
            relations: The set of relations.
            decision_attributes: The decision attributes.

        Returns:
            The discernibility matrix.
        """

        def delete_dispensable_relation(matrix, key, relations):
            if key in matrix:
                matrix[key] -= frozenset(relations)

        keys = itertools.combinations(self.universe, r=2)
        matrix = {
            frozenset(key): (
                frozenset(relations)
                if len(set(key)) > 1 and key[0] != key[1]
                else set()
            )
            for key in keys
        }

        if decision_attributes is not None:
            # Remove any pair-wise comparisons between objects that appear indiscernable
            # w.r.t. the decision attributes.
            self.__discernibility_matrix_helper(
                matrix, decision_attributes, func=self.__delete_indiscernable_comparison
            )

        return self.__discernibility_matrix_helper(
            matrix, relations, func=delete_dispensable_relation
        )

    def find_core_by_matrix(self, matrix) -> frozenset:
        """
        Find the CORE by the matrix method. The CORE is the set of attributes that are not
        redundant.

        Args:
            matrix: The discernibility matrix.

        Returns:
            The CORE.
        """
        relation_subsets = list(matrix.values())  # all subsets found in the matrix
        relation_subsets.sort(
            key=len
        )  # smaller sets start at 0, larger sets near end of list
        # each matrix entry that is a singleton set belongs to the CORE set
        return frozenset.union(
            *[relations for relations in relation_subsets if len(relations) == 1]
        )

    def __find_reduct_by_matrix(self, matrix):
        relation_subsets = list(matrix.values())  # all subsets found in the matrix
        relation_subsets.sort(
            key=len
        )  # smaller sets start at 0, larger sets near end of list

        core = self.find_core_by_matrix(matrix)

        # any matrix entry that has cardinality > 1 can possibly be used as a reduct
        possible_reducts = [
            relations for relations in relation_subsets if len(relations) > 1
        ]

        # now apply the CORE, wherever it is possible to choose CORE to
        # discern an object from another, do so

        for key, values in matrix.items():
            new_value = values.intersection(core)
            if len(new_value) > 0:  # check if the CORE may be used here
                matrix[key] = new_value  # replace the value with a CORE attribute
            else:  # find the next smallest subset and
                # update the matrix pair-wise comparison with it
                for possible_reduct in possible_reducts:
                    if possible_reduct.issubset(matrix[key]):
                        matrix[key] = matrix[key].intersection(possible_reduct)

        return matrix

    def minimum_discernibility_matrix(
        self, relations, matrix=None, decision_attributes=None
    ):
        """
        A. Skowron has proposed to represent knowledge in this form. It is a n x n matrix,
        where 'n' is the number of items in the universe of discourse. Each entry, c_{ij},
        in the matrix follows:

        $(c_{ij}) = { a in A : a(x_{i}) != a(x_{j}) } for i, j = 1, 2, ..., n$

        Thus entry c_{ij} is the set of all attributes which discern objects x_{i} and x_{j}.

        Args:
            relations:
            matrix:
            decision_attributes:

        Returns:

        """
        if matrix is None:  # if the matrix has not been calculated yet
            matrix = self.discernibility_matrix(
                relations, decision_attributes
            )  # then calculate it

        return self.__find_reduct_by_matrix(matrix)

    # end of discernibility methods

    # begin of decision table methods

    def decompose_decision_table(self, condition_attributes, decision_attributes):
        """
        Given the condition attributes (condition_attributes) and the decision attributes
        (decision_attributes), decompose the decision table into a consistent decision
        table and an inconsistent decision table.

        Args:
            condition_attributes: The condition attributes.
            decision_attributes: The decision attributes.

        Returns:
            A consistent decision table (frozenset), an inconsistent decision table (frozenset)
        """
        consistent_table = self.find_relative_positive_region(
            condition_attributes, decision_attributes
        )
        inconsistent_table = frozenset.union(
            *[
                self.boundary_region(condition_attributes, X)
                for X in self.indiscernibility(decision_attributes)
            ]
        )
        return consistent_table, inconsistent_table

    def find_attribute_cores(self, minimal_condition_attributes):
        """
        Find the core attributes.

        Args:
            minimal_condition_attributes:

        Returns:

        """
        core_attributes = {}

        for rule_idx in self.universe:
            attr_partitions = self[rule_idx]
            num_of_condition_attributes = len(minimal_condition_attributes) - 1
            condition_attributes_combinations = itertools.combinations(
                minimal_condition_attributes, num_of_condition_attributes
            )

            for selected_condition_attributes in condition_attributes_combinations:
                selected_condition_attributes = frozenset(selected_condition_attributes)
                family_of_condition_attributes = {
                    key: value
                    for key, value in attr_partitions.items()
                    if key in selected_condition_attributes
                }
                decision_category = frozenset.intersection(
                    *family_of_condition_attributes.values()
                )
                if not decision_category.issubset(attr_partitions["e"]):
                    # by only using the selected_condition_attributes,
                    # the decision_category has changed;
                    # this means that whatever we got rid of was actually important
                    if rule_idx not in core_attributes:
                        core_attributes[rule_idx] = set()
                    missing_category = (
                        minimal_condition_attributes - selected_condition_attributes
                    )
                    core_attributes[rule_idx] = core_attributes[rule_idx].union(
                        missing_category
                    )
        return core_attributes

    def find_attribute_reducts(self, minimal_condition_attributes):
        """
        Find the reduct attributes.

        Args:
            minimal_condition_attributes:

        Returns:

        """
        reduct_attributes = {}

        for rule_idx in self.universe:
            attr_partitions = self[rule_idx]

            for num_of_condition_attributes in range(
                1, len(minimal_condition_attributes)
            ):
                condition_attributes_combinations = itertools.combinations(
                    minimal_condition_attributes, num_of_condition_attributes
                )

                for selected_condition_attributes in condition_attributes_combinations:
                    selected_condition_attributes = frozenset(
                        selected_condition_attributes
                    )
                    family_of_condition_attributes = {
                        key: value
                        for key, value in attr_partitions.items()
                        if key in selected_condition_attributes
                    }
                    decision_category = frozenset.intersection(
                        *family_of_condition_attributes.values()
                    )

                    if decision_category.issubset(attr_partitions["e"]):
                        if rule_idx not in reduct_attributes:
                            reduct_attributes[rule_idx] = set()
                        # elif rule_idx not in reduct_done or not reduct_done[rule_idx]:
                        reduct_attributes[rule_idx].add(selected_condition_attributes)

                if rule_idx in reduct_attributes:  # we only want the smallest reducts
                    break
        return reduct_attributes

    def remove_redundant_attributes(self, condition_attributes, decision_attributes):
        """
        Remove any attributes identified as redundant.

        Args:
            condition_attributes: The condition attributes.
            decision_attributes: The decision attributes.

        Returns:

        """
        (result,) = self.find_reducts(
            condition_attributes, relative_to=decision_attributes
        )
        return result

    def simplify_decision_table(self, condition_attributes, decision_attributes):
        """
        Simplify the decision table, given the condition attributes and the decision attributes.

        Args:
            condition_attributes: The condition attributes.
            decision_attributes: The decision attributes.

        Returns:

        """
        minimal_condition_attributes = self.remove_redundant_attributes(
            condition_attributes, decision_attributes
        )
        return self.find_attribute_cores(
            minimal_condition_attributes
        ), self.find_attribute_reducts(minimal_condition_attributes)

    # end of decision table methods
