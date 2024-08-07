"""
Implements the methods required to work with rough theory.
"""

from collections.abc import Iterable
from typing import Union, Tuple, List
from itertools import chain, combinations

from rough.approximation import RoughApproximation


def powerset(iterable: Iterable, min_items: int):
    """
    Get the powerset of an iterable.

    Args:
        iterable: An iterable collection of elements.
        min_items: The minimum number of items that must be in each subset.

    Returns:
        The powerset of the given iterable.
    """
    # https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    return chain.from_iterable(
        combinations(list(iterable), r)
        for r in range(min_items, len(list(iterable)) + 1)
    )


class RoughOperations(RoughApproximation):
    """
    This class implements several methods that are relevant to working with rough set theory.
    """

    # begin of granulation methods

    def y_dispensable(
        self, categories: Union[set, list], relation: str, category: str
    ) -> bool:
        """
        Determine if the category in categories is (relation)-dispensable in the intersection of
        categories.

        Args:
            categories: A set of categories.
            relation: A relation.
            category: A category.

        Returns:
            Whether the given category is dispensable.
        """
        if category in categories:
            (y_items,) = self.edges(relation)
            if self.family_intersection(categories).issubset(y_items):
                return self.family_intersection(categories - {category}).issubset(
                    y_items
                )
            raise ValueError(
                "The intersection of the family set 'categories' must be a subset of the given "
                "relation."
            )
        raise ValueError("The category must be an element of categories.")

    def y_independent(self, categories: set, category_y: str) -> bool:
        """
        The family of sets, categories, is Y-independent in the intersection of categories if all
        of its components are Y-indispensible in the intersection of categories; otherwise,
        the categories are Y-dependent in the intersection of categories.

        Args:
            categories: A set of categories.
            category_y:

        Returns:

        """
        for category in categories:
            if self.y_dispensable(categories, category_y, category):
                return False
        return True

    def is_y_reduct_of_f(self, categories_f, categories_h, category_y) -> bool:
        """
        The family H is a subset of F and a Y-reduct of the intersection of F,
        if H is Y-independent in the intersection of F, and the intersection of H
        is a subset of Y.

        Args:
            categories_f:
            categories_h:
            category_y:

        Returns:

        """
        if categories_h.issubset(categories_f):
            return self.y_independent(
                categories_h, category_y
            ) and self.family_intersection(categories_h).issubset(category_y)
        raise ValueError(
            "The set 'categories_h' must be a subset of the family set 'categories_f'."
        )

    # end of granulation methods

    # begin of metrics methods

    def dispensable(
        self,
        relations: Union[set, frozenset],
        relation: Union[str, set],
        relative_to: Union[set, frozenset] = None,
        mode: callable = None,
    ) -> bool:
        """
        Checks whether the relation is dispensable from the set "relations".

        Args:
            relations: The set of relations to check for dispensability.
            relation: The relation to check for dispensability.
            relative_to: The set of relations to find the relative positive region. If None, then
                the relative positive region is the positive region of the set "relations".
            mode: The mode to determine dispensability.

        Returns:
            True if dispensable, False otherwise.
        """
        if relative_to is None:
            try:
                return mode(relations, relative_to) == mode(
                    relations - frozenset(relation), relative_to
                )
            except TypeError:  # e.g., indiscernibility
                return mode(relations) == mode(relations - frozenset(relation))
        else:
            if relation in relations or (
                isinstance(relation, set)
                and len(relation) == 1
                and relation.intersection(relations)
            ):
                return self.find_relative_positive_region(
                    relations, relative_to
                ) == self.find_relative_positive_region(
                    relations - set(relation), relative_to
                )
            raise ValueError("The 'relation' must be an element of 'relations'.")

    def independent(
        self,
        relations: Union[set, frozenset],
        relative_to: Union[set, frozenset] = None,
        mode: callable = None,
    ) -> bool:
        """
        Checks each relation within the set "relations" to see if it is dispensable;
        if at least one relation is dispensable, then the set "relations" are dependent.
        Otherwise, the set of "relations" are independent.

        Args:
            relations: The set of relations to check for independence.
            relative_to: The set of relations to find the relative positive region. If None, then
            the relative positive region is the positive region of the set "relations".
            mode: The mode to determine dispensability. If None, then the mode is the
            indiscernibility.

        Returns:
            True if independent, False otherwise.
        """
        if mode is None:
            mode = self.indiscernibility
        for relation in relations:
            if self.dispensable(
                relations, relation, relative_to=relative_to, mode=mode
            ):
                return False
        return True

    # end of metrics methods

    # start of relations methods

    def __find_reducts_helper(
        self, relative_to, func, relative_target=None
    ) -> Tuple[set, list, set]:
        """
        A helper method for find_reducts().

        Args:
            relative_to:
            func:

        Returns:
            The indispensables, possible reducts, and target knowledge.
        """
        indispensables = set()
        dispensable_relations = set()

        for relation in relative_to:
            if self.dispensable(relative_to, relation, relative_target, func):
                dispensable_relations.add(relation)
            else:
                indispensables.add(relation)

        possible_reducts = list(powerset(relative_to, min_items=2))
        if relative_target is not None:
            possible_reducts = [
                frozenset(reduct)
                for reduct in possible_reducts
                if len(frozenset(reduct).intersection(indispensables)) > 0
            ]
        try:
            target_knowledge = func(relative_to, relative_target)
        except TypeError:  # e.g., indiscernibility
            target_knowledge = func(relative_to)
        return indispensables, possible_reducts, target_knowledge

    def find_reducts(
        self, relations: set, relative_to: set = None, mode: callable = None
    ) -> frozenset:
        """
        Find the reducts of the knowledge base given the "relations", relative to the relations
        specified in "relative_to", if applicable.

        Args:
            relations: The set of relations to find the reducts.
            relative_to: The set of relations to find the relative positive region. If None, then
            the relative positive region is the positive region of the set "relations".
            mode: The mode to determine dispensability. If relative_to is None and mode is None,
            then the mode is the indiscernibility.

        Returns:
            The set of reducts.
        """
        if relative_to is None:
            if mode is None:
                mode = self.indiscernibility
            (
                indispensables,
                possible_reducts,
                target_knowledge,
            ) = self.__find_reducts_helper(relations, mode, relative_to)
            # has to be independent and at least 1 indispensable relation
            return frozenset(
                [
                    frozenset(possible_reduct)
                    for possible_reduct in possible_reducts
                    if self.independent(frozenset(possible_reduct), mode=mode)
                    # and self.depends_on(relations, frozenset(possible_reduct))
                    and mode(possible_reduct) == target_knowledge
                    and len(indispensables.intersection(possible_reduct)) > 0
                ]
            )
        # calculate the relative REDUCT
        if mode is None:
            mode = self.find_relative_positive_region
        (
            indispensables,
            possible_reducts,
            target_knowledge,
        ) = self.__find_reducts_helper(relations, mode, relative_to)
        # has to be independent and at least 1 indispensable relation
        return frozenset(
            [
                frozenset(possible_reduct)
                for possible_reduct in possible_reducts
                if self.independent(frozenset(possible_reduct), relative_to=relative_to)
                # and self.depends_on(relative_to, frozenset(possible_reduct))
                and self.find_relative_positive_region(possible_reduct, relative_to)
                == target_knowledge
                and len(indispensables.intersection(possible_reduct)) > 0
            ]
        )

    def find_core(
        self, relations: set, relative_to: set = None, mode: callable = None
    ) -> frozenset:
        """
        Obtains the reducts of the knowledge and identifies the most important part aka the core.

        Args:
            relations:
            relative_to:
            mode: The function to use to calculate the core. Defaults to indiscernibility.

        Returns:

        """
        if relative_to is None or relations == relative_to:  # calculate CORE
            if mode is None:
                mode = self.indiscernibility
            return frozenset.intersection(
                *self.find_reducts(relations, relative_to=None, mode=mode)
            )
        # calculate the relative CORE (i.e., Q-CORE, where "Q" is relative_to)
        results = set()
        for relation in relations:
            if not self.dispensable(relations, relation, relative_to=relative_to):
                results.add(relation)
        return frozenset(results)

    def find_relative_positive_region(
        self, relations: Union[str, set], relative_to: Union[str, set]
    ) -> frozenset:
        """
        The P-positive region (POS) of Q is the set of all objects of the universe, U,
        which can be properly classified to classes of U / Q employing knowledge
        expressed by the classification U / P.

        Args:
            relations: The set of relations to find the relative positive region.
            relative_to:

        Returns:
            The relative positive region.
        """
        # categories = self / relative_to
        categories = self.indiscernibility(relative_to)
        results = set()
        for category in categories:
            results = results.union(self.lower_approximation(relations, category))
        return frozenset(results)

    def is_q_reduct_of_p(self, relations: set, restrictions: set, set_s: set) -> bool:
        """
        The family S that is a subset of P is called a Q-reduct of P,
        if and only if S is the Q-independent subfamily of P and
        POS_{S}(Q) == POS_{P}(Q).

        Args:
            relations:
            restrictions:
            set_s:

        Returns:

        """
        return self.find_relative_positive_region(
            set_s, restrictions
        ) == self.find_relative_positive_region(relations, restrictions)

    # end of relation methods

    # begin of dependency methods

    def depends_on(
        self, relations: Union[str, set], other_relations: Union[str, set]
    ) -> bool:
        """
        Knowledge Q depends on knowledge P (P ==> Q) if and
        only if IND(P) is a subset of IND(Q). More precisely,
        knowledge Q is derivable from knowledge P, if all
        elementary categories of Q can be defined in terms
        of some elementary categories of knowledge P.

        Args:
            relations:
            other_relations:

        Returns:

        """
        return all(
            # the elements of IND(P) must have at least 1 subset match in IND(Q)
            any(
                # the element in IND(P) is a subset of any element in IND(Q)
                P_element.issubset(Q_element)
                for Q_element in self.indiscernibility(other_relations)
            )
            for P_element in self.indiscernibility(relations)
        )

    def equivalent_to(
        self, relations: Union[str, set], other_relations: Union[str, set]
    ) -> bool:
        """
        Knowledge P and Q are equivalent if and only if
        knowledge P derives Q and knowledge Q derives P.

        Args:
            relations:
            other_relations:

        Returns:

        """
        return self.indiscernibility(relations) == self.indiscernibility(
            other_relations
        )

    def independent_of(
        self, relations: Union[str, set], other_relations: Union[str, set]
    ) -> bool:
        """
        Knowledge P and Q are independent, denoted as P != Q,
        if neither P ==> Q nor Q ==> P hold.

        Args:
            relations:
            other_relations:

        Returns:

        """
        # pylint: disable=arguments-out-of-order
        return not self.depends_on(relations, other_relations) and not self.depends_on(
            other_relations, relations
        )

    def partial_depends_on(
        self,
        relations: Union[str, set],
        other_relations: Union[str, set],
        category: Union[set, frozenset, List[frozenset]] = None,
    ) -> float:
        """
        Partial dependency of knowledge: the derivation (dependency)
        can also be partial, which means that only part of the knowledge Q
        is derivable from knowledge P.

        In essence, knowledge Q depends on a degree k (0 <= k <= 1) from
        knowledge P, symbolically P ==>_{k} Q.

        If k = 1, then Q totally depends on P; if 0 < k < 1, then Q roughly
        (partially) depends on P, and if k = 0, then Q is totally independent
        from knowledge P.

        In other words, if k = 1, then all the elements of the universe, U,
        can be classified to elementary categories of U / Q. If k != 1, only
        those elements of the universe which belong to the positive region
        can be classified to categories of knowledge Q, employing knowledge P.
        In particular, if k = 0, then none of the elements of the universe can
        be classified using knowledge P to elementary categories of knowledge Q.

        Disclaimer: The measure k of dependency, $P -->_{k} Q$, does not capture
        how this partial dependency is actually distributed among classes of
        U / Q. For example, some decision classes can be fully characterized by P,
        whereas others may be characterized only partially. Hence, an alternative
        definition is proposed where $gamma(X) = card(P_lower X) / card(X)$ where
        $X in U / Q$ which shows how many elements of each class of U / Q can be
        classified by employing knowledge P.

        To use the 'alternative definition', change the 'category' argument to a set
        that occurs in U / Q. When using the alternative definition, 'Q' can
        be 'None'.

        Args:
            relations:
            other_relations:
            category:

        Returns:
            The partial dependency of knowledge.
        """
        if category is not None:
            # pylint: disable=fixme
            # TODO: check this is reached by code coverage
            return len(self.lower_approximation(relations, category)) / len(category)
        return len(
            self.find_relative_positive_region(relations, other_relations)
        ) / len(self.universe)

    # end of dependency methods
