"""
A module that contains the RoughApproximation class, which is a class that represents the rough
approximation of a set of objects. The rough approximation is a set of objects that are
indiscernible from each other with respect to a set of relations. The rough approximation is a
generalization of the lower and upper approximation, which are defined by Pawlak on page 10 of his
book "Rough Sets: Theoretical Aspects of Reasoning About Data".
"""

from collections import namedtuple
from collections.abc import Iterable
from typing import List, Union, Set

from rough.granulation import RoughGranulation


class RoughApproximation(RoughGranulation):
    """
    A class that represents the rough approximation of a set of objects. The rough approximation is
    a set of objects that are indiscernible from each other with respect to a set of relations. The
    rough approximation is a generalization of the lower and upper approximation, which are defined
    by Pawlak on page 10 of his book "Rough Sets: Theoretical Aspects of Reasoning About Data".
    """

    def indiscernibility(self, equivalence_relations: Union[str, set, list]):
        """
        The indiscernibility relation over a vector of equivalence relations (referred to as
        "P"). Defined on page 3 of "Rough Sets: Theoretical Aspects of Reasoning About Data".

        Args:
            equivalence_relations: An iterable of equivalence relations that must be a subset
            of the available relations and cannot be empty. If a string is passed, it is assumed to
            be a single equivalence relation. If a set is passed, it is assumed to be a set of
            equivalence relations. If a list is passed, it is assumed to be a list of equivalence
            relations.

        Returns:
            An indiscernibility relation over a collection of equivalence relations (i.e.,
            the family of equivalence classes of the equivalence relation IND(P)) yields the
            family of equivalence classes; denotes knowledge associated with the family of
            equivalence relations P, called P-basic knowledge about U (universe) in K (knowledge
            base).
        """
        if len(equivalence_relations) == 0:
            raise ValueError("The relations' length must be greater than zero.")

        possible_relations = self.graph.vs.select(type_ne=None)
        relation_names = frozenset(
            [relation["type"] for relation in possible_relations]
        )
        if frozenset(equivalence_relations).issubset(relation_names):
            indiscernibility_relation = set()
            for element in self.universe:
                categories = self[element]
                if len(categories) > 0:
                    # it is possible for self[element] to have no equivalence relations
                    new_category = set(self.universe)
                    for relation in equivalence_relations:
                        if relation in categories:
                            # some elements might not be defined for all relations
                            new_category = new_category.intersection(
                                categories[relation]
                            )
                    indiscernibility_relation.add(frozenset(new_category))
            return indiscernibility_relation
        raise ValueError(
            "The relations must be a subset of the existing relations on the graph."
        )

    def lower_approximation(
        self,
        relations: Union[str, set],
        categories: Union[set, frozenset, List[frozenset]],
    ) -> Union[frozenset, Set[frozenset]]:
        """
        Get the lower approximation of a (set of) category(s), as defined by Pawlak on page 10 of
        his book "Rough Sets: Theoretical Aspects of Reasoning About Data".

        Args:
            relations: Either a set of relations or a string that references a specific relation
            in the RoughApproximation.
            categories: The category (frozenset) or a family of categories (list of frozensets).

        Returns:
            Either a frozenset or a set of frozensets that represent the lower approximation.
        """

        return self.approximation(
            relations,
            categories,
            mode=lambda subset, category: subset.issubset(category),
        )

    def upper_approximation(
        self,
        relations: Union[str, set],
        categories: Union[set, frozenset, List[frozenset]],
    ) -> Union[frozenset, Set[frozenset]]:
        """
        Get the upper approximation of a (set of) category(s), as defined by Pawlak on page 10 of
        his book "Rough Sets: Theoretical Aspects of Reasoning About Data".

        Args:
            relations: Either a set of relations or a string that references a specific relation
            in the RoughApproximation.
            categories: The category (frozenset) or a family of categories (list of frozensets).

        Returns:
            Either a frozenset or a set of frozensets that represent the upper approximation.
        """
        return self.approximation(
            relations,
            categories,
            mode=lambda subset, category: len(subset.intersection(category)) > 0,
        )

    def approximation(
        self,
        relations: Union[str, set],
        categories: Union[set, frozenset, List[frozenset]],
        mode: callable,
    ) -> Union[frozenset, Set[frozenset]]:
        """
        Get the approximation of a (set of) category(s), as defined by Pawlak on page 10 of his
        book "Rough Sets: Theoretical Aspects of Reasoning About Data". The approximation is
        determined by the mode argument, which is a function that takes two arguments: a subset
        and the category. The mode function should return True if the subset satisfies the category.

        Args:
            relations: Either a set of relations or a string that references a specific relation
            in the RoughApproximation.
            categories: The category (frozenset) or a family of categories (list of frozensets).
            mode: The mode of approximation, which is a function that takes two arguments: a
            subset and a category. The mode function should return True if the subset satisfies
            the category. In the case of the lower approximation, the mode function should return
            True if the subset is a subset of the category. In the case of the upper
            approximation, the mode function should return True if the subset intersects with the
            category.

        Returns:
            Either a frozenset or a set of frozensets that represent the [mode] approximation.
        """

        def __approximation(
            relations: Union[str, set],
            category: frozenset,
            satisfied_constraint: callable,
        ) -> frozenset:
            """
            Get the approximation of a category.

            Args:
                relations: Either a set of relations or a string that references a specific relation
                in the RoughApproximation.
                categories: The category (frozenset) or a family of categories (list of frozensets).
                satisfied_constraint: The constraint that is to be satisfied or otherwise
                known as the mode of approximation, which is a function that takes two arguments: a
                subset and a category. The mode function should return True if the subset satisfies
                the category. In the case of the lower approximation, the mode function should
                return True if the subset is a subset of the category. In the case of the upper
                approximation, the mode function should return True if the subset intersects with
                the category.

            Returns:
                A frozenset of the approximation.
            """
            indiscernibility_relation = self.indiscernibility(relations)
            result = set()
            for subset in indiscernibility_relation:
                if satisfied_constraint(subset, category):
                    result = result.union(subset)
            return frozenset(result)

        if len(categories) == 0:
            raise ValueError("The argument 'categories' may not have a length of zero.")
        if isinstance(categories, Iterable) and isinstance(categories, list):
            # categories is a family of non-empty sets
            result = set()
            for category in categories:
                if len(category) == 0:
                    raise ValueError(
                        "The argument 'categories' may not have an element with a length of zero."
                    )
                result.add(__approximation(relations, category, mode))
        elif isinstance(categories, (set, frozenset)):
            result = __approximation(relations, categories, mode)
        else:
            raise ValueError(
                "The argument 'categories' must be a set, a frozenset, or a list."
            )

        return frozenset(result)

    def positive_region(
        self,
        relations: Union[str, set],
        categories: Union[set, frozenset, List[frozenset]],
    ) -> Union[frozenset, Set[frozenset]]:
        """
        Analogous to the R-lower approximation.

        Args:
            relations: Either a set of relations or a string that references a specific relation
            in the RoughApproximation.
            categories: The category (frozenset) or a family of categories (list of frozensets).

        Returns:
            Either a frozenset or a set of frozensets that represent the positive region or
            the lower approximation.
        """
        return self.lower_approximation(relations, categories)

    def negative_region(
        self, relations: Union[str, set], categories: Union[frozenset, List[frozenset]]
    ) -> Union[frozenset, Set[frozenset]]:
        """
        The set of objects with which it can be determined without
        any ambiguity, employing knowledge 'relations', that they
        do not belong to the 'categories' (i.e., they belong to the
        complement of 'categories')

        Args:
            relations: Either a set of relations or a string that references a specific relation
            in the RoughApproximation.
            categories: The category (frozenset) or a family of categories (list of frozensets).

        Returns:
            Either a frozenset or a set of frozensets that represent the negative region.
        """
        return self.universe - self.upper_approximation(relations, categories)

    def boundary_region(
        self,
        relations: Union[str, set],
        categories: Union[set, frozenset, List[frozenset]],
    ) -> Union[frozenset, Set[frozenset]]:
        """
        The undecidable area of the universe (i.e., none of the objects
        belonging to the boundary can be classified with certainty into
        categories or not categories, as far as knowledge 'relations' is concerned.

        Args:
            relations: Either a set of relations or a string that references a specific relation
            in the RoughApproximation.
            categories: The category (frozenset) or a family of categories (list of frozensets).

        Returns:
            Either a frozenset or a set of frozensets that represent the boundary region.
        """
        return self.upper_approximation(
            relations, categories
        ) - self.lower_approximation(relations, categories)

    def definable(self, relations: Union[str, set], category: frozenset) -> namedtuple:
        """
        A category X is called:

        R-definable if X is the union of some R-basic categories;
        otherwise X is R-undefinable. Also called R-exact sets. R-undefinable sets are also
        called R-inexact or R-rough sets.

        Roughly definable if we are able to decide whether some elements of the
        universe belong to X or not X.

        Internally undefinable if we are able to decide whether some elements of the
        universe belong to not X, but we are unable to indicate one element of X.

        Externally undefinable if we are able to decide for some elements of the
        universe whether they belong to X, but we are unable to indicate one element of not X.

        Totally undefinable if we are unable to decide for any element of the
        universe whether it belongs to X or not X.

        Args:
            relations: Either a set of relations or a string that references a specific relation
            in the RoughApproximation.
            category: The category (frozenset).

        Returns:
            A namedtuple with the lower and upper approximation of the given category. The name
            of the tuple depends on the type of definability of the category.
        """
        lower_approximation = self.lower_approximation(relations, category)
        upper_approximation = self.upper_approximation(relations, category)

        if lower_approximation == upper_approximation:
            # The set X is called R-definable if X is the union of some R-basic categories;
            # otherwise X is R-undefinable. Also called R-exact sets. R-undefinable sets are also
            # called R-inexact or R-rough sets.
            return namedtuple(
                "Definable", ["lower_approximation", "upper_approximation"]
            )(lower_approximation, upper_approximation)
        if (
            len(self.lower_approximation(relations, category)) > 0
            and self.upper_approximation(relations, category) != self.universe
        ):
            # We are able to decide whether some elements of the universe belong to X or not X.
            return namedtuple(
                "RoughlyDefinable", ["lower_approximation", "upper_approximation"]
            )(lower_approximation, upper_approximation)
        if (
            len(self.lower_approximation(relations, category)) == 0
            and self.upper_approximation(relations, category) != self.universe
        ):
            # We are able to decide whether some elements of the universe belong to not X,
            # but we are unable to indicate one element of X.
            return namedtuple(
                "InternallyUndefinable", ["lower_approximation", "upper_approximation"]
            )(lower_approximation, upper_approximation)
        if (
            len(self.lower_approximation(relations, category)) != 0
            and self.upper_approximation(relations, category) == self.universe
        ):
            # We are able to decide for some elements of the universe whether they belong to X,
            # but we are unable to indicate one element of not X.
            return namedtuple(
                "ExternallyUndefinable", ["lower_approximation", "upper_approximation"]
            )(lower_approximation, upper_approximation)
        if (
            len(self.lower_approximation(relations, category)) == 0
            and self.upper_approximation(relations, category) == self.universe
        ):
            # We are unable to decide for any element of the universe whether
            # it belongs to X or not X.
            return namedtuple(
                "TotallyUndefinable", ["lower_approximation", "upper_approximation"]
            )(lower_approximation, upper_approximation)
        raise ValueError("The given category is not definable or roughly definable.")

    def quality_of_approximation(self, relations, categories):
        """
        The quality of approximation of X, a family of non-empty sets,
        using knowledge 'relations'.

        This metric expresses the percentage of possible correct
        decisions when classifying objects employing the knowledge 'relations'.

        Args:
            relations:
            categories:

        Returns:

        """
        if len(categories) == 0:
            raise ValueError("The argument 'categories' may not have a length of zero.")
        if isinstance(categories, Iterable) and isinstance(categories, list):
            # categories is a family of non-empty sets
            numerator = 0.0
            for category in categories:
                if len(category) == 0:
                    raise ValueError(
                        "The argument 'category' may not have an element with a length of zero."
                    )
                numerator += len(self.lower_approximation(relations, category))
        else:
            numerator = len(self.lower_approximation(relations, categories))
        denominator = len(self.universe)

        return numerator / denominator

    def roughly_equal(
        self,
        relations,
        category,
        other_category,
        mode: Union["bottom", "top", "both"] = "both",
    ):
        """
        Determines if the category and the other category are roughly equal given knowledge about
        "relations". The definition of equality depends on the mode.

        If mode is "bottom", then the lower approximation of category is equal to the lower
        approximation of the other category given "relations"; essentially, the positive examples of
        the sets "category" and the "other category" are the same.

        If mode is "top", then the upper approximation of the given category is equal to the
        upper approximation of the other category given knowledge about "relations"; essentially,
        the negative examples of the sets "category" and the "other category" are the same.

        If mode is "both", then the lower and upper approximation of category is equal to the lower
        and upper approximation of the other category given "relations"; essentially, the positive
        and negative examples of the sets "category" and the "other category" are the same.

        Args:
            relations: The equivalence relations.
            category: The category in question.
            other_category: The other category in question.
            mode: The mode of equality.

        Returns:
            Whether the category and the other category are roughly equal given knowledge about
            "relations".
        """
        if mode == "bottom":
            return self.lower_approximation(
                relations, category
            ) == self.lower_approximation(relations, other_category)
        if mode == "top":
            return self.upper_approximation(
                relations, category
            ) == self.upper_approximation(relations, other_category)
        if mode == "both":
            return self.roughly_equal(
                relations, category, other_category, mode="bottom"
            ) and self.roughly_equal(relations, category, other_category, mode="top")

        raise ValueError("The argument 'mode' is not valid.")

    def roughly_included(
        self,
        relations,
        category,
        other_category,
        mode: Union["bottom", "top", "both"] = "both",
    ):
        """
        Determines whether the category is roughly included in the other category given knowledge
        about "relations". The definition of inclusion depends on the mode.

        If mode is "bottom", then the lower approximation of category is equal to the lower
        approximation of the other category given "relations"; essentially, the positive examples of
        the sets "category" and the "other category" are the same.

        If mode is "top", then the upper approximation of the given category is equal to the
        upper approximation of the other category given knowledge about "relations"; essentially,
        the negative examples of the sets "category" and the "other category" are the same.

        If mode is "both", then the lower and upper approximation of category is equal to the lower
        and upper approximation of the other category given "relations"; essentially, the positive
        and negative examples of the sets "category" and the "other category" are the same.

        Whether the lower and upper approximations of X are subset to the
        lower and upper approximations of Y given 'relations', respectively.

        Args:
            relations: The equivalence relations.
            category: The category in question.
            other_category: The other category in question.
            mode: The mode of inclusion (i.e., subsethood).

        Returns:
            Whether the category is roughly included in (i.e., a subset of) the other category
            given knowledge about "relations".
        """
        if mode == "bottom":
            return self.lower_approximation(relations, category).issubset(
                self.lower_approximation(relations, other_category)
            )
        if mode == "top":
            return self.upper_approximation(relations, category).issubset(
                self.upper_approximation(relations, other_category)
            )
        if mode == "both":
            return self.roughly_included(
                relations, category, other_category, mode="bottom"
            ) and self.roughly_included(relations, category, other_category, mode="top")

        raise ValueError("The argument 'mode' is not valid.")

    def accuracy(self, relations, category):
        """
        Calculate the inexactness of a set (category) due to the
        existence of a borderline region. The greater the borderline region,
        the lower the accuracy is of the set.

        If the accuracy is 1, then the set X is R-definable, otherwise,
        it is R-undefinable (i.e., our knowledge is incomplete).

        Args:
            relations:
            category:

        Returns:

        """
        if len(category) == 0:
            raise ValueError("The argument 'category' may not have a length of zero.")
        if isinstance(category, Iterable) and isinstance(category, list):
            # category is a family of non-empty sets
            numerator, denominator = 0.0, 0.0
            for set_x_i in category:
                if len(set_x_i) == 0:
                    raise ValueError(
                        "The argument 'category' may not have an element with a length of zero."
                    )
                numerator += len(self.lower_approximation(relations, set_x_i))
                denominator += len(self.upper_approximation(relations, set_x_i))
        else:
            numerator = len(self.lower_approximation(relations, category))
            denominator = len(self.upper_approximation(relations, category))
        return numerator / denominator

    def roughness(self, relations, category):
        """
        The complement of a rough set's accuracy or inexactness; represents
        the degree of incompleteness of knowledge 'relations' about the set X.

        Args:
            relations:
            category:

        Returns:

        """
        return 1 - self.accuracy(relations, category)
