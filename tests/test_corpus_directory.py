# coding: utf-8

from unittest import TestCase
from .context import vrtool

import os

class CorpusDirectoryTest(TestCase):
    def setUp(self):
        path = os.path.join(os.path.dirname(__file__),'testfiles')
        self.corpus_dir = vrtool.CorpusDirectory(path)

    def test_get_positive_examples(self):
        examples = self.corpus_dir.get_positive_examples('subfolder1')
        self.assertEqual(examples,[os.path.join(os.path.dirname(__file__),'testfiles','subfolder1','001.JPEG')])

        examples = self.corpus_dir.get_positive_examples('subfolder2')
        self.assertEqual(examples,[])

    def test_get_positive_examples_in_folder(self):
        path = os.path.join(os.path.dirname(__file__),'testfiles')
        examples = self.corpus_dir.get_positive_examples_in_folder('subfolder1', path)
        self.assertEqual(examples,[os.path.join(os.path.dirname(__file__),'testfiles','subfolder1','001.JPEG')])

    def test_get_all_positive_examples(self):
        class_names = ['subfolder1', 'subfolder2']
        positive_examples = self.corpus_dir.get_all_positive_examples(class_names)
        self.assertEqual(positive_examples[0]['class_name'], 'subfolder1')
        self.assertEqual(len(positive_examples[0]['images']), 1)
        self.assertEqual(positive_examples[1]['class_name'], 'subfolder2')
        self.assertEqual(positive_examples[1]['images'], [])

    def test_get_all_class_names(self):
        class_names = self.corpus_dir.get_all_class_names()
        self.assertEqual(sorted(class_names),['negatives', 'subfolder1', 'subfolder2'])

    def test_get_all_class_names_in_folder(self):
        path = os.path.join(os.path.dirname(__file__),'testfiles')
        class_names = self.corpus_dir.get_all_class_names()
        class_names_folder = self.corpus_dir.get_all_class_names_in_folder(path)
        self.assertEqual(sorted(class_names_folder), sorted(class_names))

    def test_get_all_class_images(self):
        images = self.corpus_dir.get_all_class_images()
        self.assertEqual(images[0]['class_name'], 'negatives')
        self.assertEqual(images[0]['image_name'], '002.JPG')
        self.assertEqual(images[1]['class_name'], 'subfolder1')
        self.assertEqual(images[1]['image_name'], '001.JPEG')

    def test_get_negative_example_images(self):
        images = self.corpus_dir.get_negative_example_images()
        self.assertEqual(images[0]['class_name'], 'negatives')
        self.assertEqual(images[0]['image_name'], '002.JPG')

