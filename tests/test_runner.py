# coding: utf-8

from unittest import TestCase
from .context import vrtool

import os


class RunnerTest(TestCase):
    def setUp(self):
        self.runner = vrtool.Runner( 'config.ini', '../tests/testfiles')

    def test_get_available_corpora(self):
        corpora = self.runner.get_available_corpora()
        self.assertEqual(len(corpora), 3)
        self.assertEqual(sorted([c[0] for c in corpora]), sorted(
            ['subfolder2', 'negatives', 'subfolder1']))
        self.assertEqual([isinstance(c[1], vrtool.corpus_directory.CorpusDirectory)
                          for c in corpora], [True, True, True])

    def test_get_corpora_info(self):
        corpora = self.runner.get_available_corpora()
        corpora_info = self.runner.get_corpora_info(corpora)

        self.assertEqual(sorted([c['corpus_name'] for c in corpora_info]), sorted(
            ['subfolder2', 'subfolder1']))

        self.assertEqual(sorted([c['image_info'].empty for c in corpora_info]), sorted(
            [False, True]))

    def test_create_experiments(self):
        corpora = self.runner.get_available_corpora()
        corpora_info = self.runner.get_corpora_info(corpora)
        img_info = [el['image_info']
            for el in corpora_info if el['corpus_name'] == 'subfolder1'][0]
            
        with self.assertRaises(ValueError):
            self.runner.create_experiments(3, img_info)

        self.assertEqual(self.runner.create_experiments(2, img_info)[0]['train'].class_name.value_counts().values,[1])
        self.assertEqual(self.runner.create_experiments(2, img_info)[0]['test'].class_name.value_counts().values,[1])
        self.assertNotEqual(self.runner.create_experiments(2, img_info)[0]['test'].image.values, self.runner.create_experiments(2, img_info)[0]['train'].image.values)

    def test_train_k_classifiers(self):
        pass

    def test_test_k_classifiers(self):
        pass

    def test_train_classifier_class_file_tuple(self):
        pass

    def test_train_classifier_from_data_frame(self):
        pass

    def test_test_classifier_with_data_frame(self):
        pass

    def test_merge_predicted_and_target_labels(self):
        pass

    def test_get_confusion_matrix(self):
        pass

    def test_get_classification_report(self):
        pass

    def test_bootstrap_all(self):
        pass

    def bootstrap_specific_class(self):
        pass

    def duplicate_all(self):
        pass

    def duplicate_specific_class(self):
        pass
