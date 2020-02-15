# coding: utf-8

from unittest import TestCase
from .context import vrtool

import os

class DirectoryTest(TestCase):
    def setUp(self):
        path = os.path.join(os.path.dirname(__file__),'testfiles')
        self.directory = vrtool.Directory(path)

    def test_get_image_files(self):
        path = os.path.join(os.path.dirname(__file__),'testfiles')
        files = self.directory.get_image_files(path)
        self.assertEqual(sorted(files), ['001.JPEG', '002.JPG', '003.jpeg', '004.jpg', '005.png','006.PNG'])

    def test_get_subfolders_all(self):
        path = os.path.join(os.path.dirname(__file__),'testfiles')
        subfolders = self.directory.get_sub_folders(path)
        self.assertEqual(sorted(subfolders), ['negatives', 'subfolder1', 'subfolder2'])

    def test_get_subfolders_regex(self):
        path = os.path.join(os.path.dirname(__file__),'testfiles')
        subfolders = self.directory.get_sub_folders(path, '.*1')
        self.assertEqual(subfolders, ['subfolder1'])

    def test_has_sub_folder(self):
        path = os.path.join(os.path.dirname(__file__),'testfiles')
        has_sub_folder = self.directory.has_sub_folder(path)
        self.assertTrue(has_sub_folder)

        path = os.path.join(os.path.dirname(__file__),'testfiles', 'subfolder2')
        has_sub_folder = self.directory.has_sub_folder(path)
        self.assertFalse(has_sub_folder)

    def test_has_sub_folders(self):
        path = os.path.join(os.path.dirname(__file__),'testfiles')
        has_sub_folders = vrtool.Directory.has_sub_folders(path)
        self.assertTrue(has_sub_folders)

        path = os.path.join(os.path.dirname(__file__),'testfiles', 'subfolder2')
        has_sub_folders = vrtool.Directory.has_sub_folders(path)
        self.assertFalse(has_sub_folders)

    