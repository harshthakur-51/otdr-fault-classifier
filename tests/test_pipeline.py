import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from otdr_fault_classifier.cli import run
from otdr_fault_classifier.features import extract_features
from otdr_fault_classifier.traces import FAULT_CLASSES, generate_dataset


class PipelineTests(unittest.TestCase):
    def test_generator_contains_all_classes(self):
        dataset = generate_dataset(samples_per_class=2)
        self.assertEqual({label for _, label in dataset}, set(FAULT_CLASSES))

    def test_feature_vector_is_compact(self):
        trace, _ = generate_dataset(samples_per_class=1)[0]
        self.assertEqual(len(extract_features(trace)), 11)

    def test_neural_classifier_learns_fault_signatures(self):
        self.assertGreater(run(samples_per_class=55, epochs=75, seed=7), 0.88)


if __name__ == "__main__":
    unittest.main()
