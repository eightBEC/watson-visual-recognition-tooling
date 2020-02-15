# coding: utf-8

from unittest import TestCase
from .context import vrtool

import os

class CorporaTest(TestCase):
    def setUp(self):
        path = os.path.join(os.path.dirname(__file__),'testfiles')
        self.corpora = vrtool.Corpora(path)

    def test_get_corpora(self):
        corpora = self.corpora.get_corpora()
        folders = sorted([f[0] for f in corpora])
        self.assertEqual(folders[0],'negatives')
        self.assertEqual(folders[1],'subfolder1')
        self.assertEqual(folders[2],'subfolder2')
    