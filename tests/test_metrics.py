"""Regression tests for list and join-path scoring (no API or graph deps)."""
import unittest

from evaluation.metrics import score_answer


class ListScoringTests(unittest.TestCase):
    def test_empty_answers_match(self):
        self.assertEqual(
            score_answer([], [], 'list'),
            {'exact_match': True, 'f1': 1.0, 'precision': 1.0, 'recall': 1.0},
        )

    def test_extra_predictions_do_not_match_empty_gold(self):
        self.assertEqual(
            score_answer(['unexpected_table'], [], 'list'),
            {'exact_match': False, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
        )

    def test_missing_predictions_do_not_match_nonempty_gold(self):
        self.assertFalse(score_answer([], ['table_a'], 'list')['exact_match'])
        self.assertEqual(score_answer([], ['table_a'], 'list')['f1'], 0.0)

    def test_unordered_lists_still_ignore_order_and_duplicates(self):
        scores = score_answer(['b', 'a', 'a'], ['a', 'b'], 'list')
        self.assertTrue(scores['exact_match'])
        self.assertEqual(scores['f1'], 1.0)


class JoinPathScoringTests(unittest.TestCase):
    def setUp(self):
        self.gold = ['a', 'b', 'c', 'd']
        self.names = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
        self.adj = {0: {1, 4}, 1: {2}, 2: {3}, 4: {2}}

    def score(self, predicted, with_graph=True):
        kwargs = {'fk_adj': self.adj, 'name_to_idx': self.names} if with_graph else {}
        return score_answer(predicted, self.gold, 'ordered_list', **kwargs)

    def test_identical_path_matches(self):
        scores = self.score(self.gold)
        self.assertTrue(scores['exact_match'])
        self.assertTrue(scores['order_correct'])
        self.assertFalse(scores['valid_alt_path'])

    def test_invalid_permutation_is_not_an_exact_match(self):
        for with_graph in (False, True):
            with self.subTest(with_graph=with_graph):
                scores = self.score(['a', 'c', 'b', 'd'], with_graph)
                self.assertFalse(scores['exact_match'])
                self.assertFalse(scores['order_correct'])
                self.assertFalse(scores['valid_alt_path'])
                # F1 continues to measure table overlap independently of order.
                self.assertEqual(scores['f1'], 1.0)

    def test_reversed_path_does_not_match_requested_endpoints(self):
        self.assertFalse(self.score(list(reversed(self.gold)))['exact_match'])

    def test_repeated_table_does_not_match(self):
        self.assertFalse(self.score(['a', 'b', 'b', 'c', 'd'])['exact_match'])

    def test_valid_alternative_shortest_path_matches(self):
        scores = self.score(['a', 'e', 'c', 'd'])
        self.assertTrue(scores['exact_match'])
        self.assertTrue(scores['valid_alt_path'])
        self.assertFalse(scores['order_correct'])
        self.assertEqual(scores['f1'], 1.0)

    def test_valid_permutation_is_checked_as_alternative_path(self):
        # Same table set, same endpoints, different valid FK ordering.
        self.adj = {0: {1, 2}, 1: {2, 3}, 2: {3}}
        scores = self.score(['a', 'c', 'b', 'd'])
        self.assertTrue(scores['exact_match'])
        self.assertTrue(scores['valid_alt_path'])

    def test_alternative_path_requires_graph_validation(self):
        self.assertFalse(self.score(['a', 'e', 'c', 'd'], False)['exact_match'])


if __name__ == '__main__':
    unittest.main()
