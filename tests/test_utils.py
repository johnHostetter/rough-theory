"""
White-box tests for rough.utils.powerset() - the canonical implementation (previously
duplicated verbatim in rough.operations, which now just re-imports it).
"""

import unittest

from rough.utils import powerset


class TestPowerset(unittest.TestCase):
    """
    Test powerset()'s subset-generation and min_items filtering.
    """

    def test_all_subsets_with_min_items_zero(self) -> None:
        """min_items=0 includes the empty set among the subsets."""
        result = {frozenset(item) for item in powerset({"a", "b"}, min_items=0)}
        self.assertEqual(
            result,
            {frozenset(), frozenset({"a"}), frozenset({"b"}), frozenset({"a", "b"})},
        )

    def test_min_items_excludes_smaller_subsets(self) -> None:
        """min_items=2 excludes the empty set and every singleton."""
        result = {frozenset(item) for item in powerset({"a", "b", "c"}, min_items=2)}
        self.assertEqual(
            result,
            {
                frozenset({"a", "b"}),
                frozenset({"a", "c"}),
                frozenset({"b", "c"}),
                frozenset({"a", "b", "c"}),
            },
        )

    def test_min_items_greater_than_length_yields_nothing(self) -> None:
        """No subset can satisfy min_items larger than the input's size."""
        result = list(powerset({"a", "b"}, min_items=3))
        self.assertEqual(result, [])

    def test_accepts_a_one_shot_iterator_not_just_a_re_iterable_collection(
        self,
    ) -> None:
        """
        Regression test: powerset() used to call `list(iterable)` twice - once for
        `len(list(iterable))` and again inside the generator body for each `r` - so
        passing a genuine one-shot iterator (as opposed to a set/list, which every
        existing caller happens to use) silently returned nothing at all, since the
        iterator was already exhausted well before the first combinations() call.
        `iter([...])` is exactly such a one-shot iterator: re-iterating it a second
        time yields nothing.
        """
        one_shot_iterator = iter(["a", "b", "c"])
        result = {frozenset(item) for item in powerset(one_shot_iterator, min_items=2)}
        self.assertEqual(
            result,
            {
                frozenset({"a", "b"}),
                frozenset({"a", "c"}),
                frozenset({"b", "c"}),
                frozenset({"a", "b", "c"}),
            },
        )

    def test_empty_iterable_with_min_items_zero_yields_only_the_empty_set(
        self,
    ) -> None:
        """Edge case: an empty input still yields the empty set when allowed."""
        result = list(powerset([], min_items=0))
        self.assertEqual(result, [()])


if __name__ == "__main__":
    unittest.main()
