# coding: utf-8

from unittest import TestCase
from .context import vrtool

import os
import zipfile

class ZipHelperTest(TestCase):
    def setUp(self):
        path = os.path.join(os.path.dirname(__file__),'testfiles')
        self.ziphelper = vrtool.ZipHelper()
        self.corpus_dir = vrtool.CorpusDirectory(path)

    def test_clean_up(self):
        self.ziphelper.clean_up()
        self.assertEqual(self.ziphelper.open_file_handles, [])
        self.assertEqual(self.ziphelper.files_created, [])
        #self.assertEqual(self.ziphelper.temp_folder,[] )

    def test_create_temp_zip_from_folder_content(self):
        testzipfile = None
        try:
            images = self.corpus_dir.get_positive_examples('subfolder1')
            zip_file = self.ziphelper.create_temp_zip_from_files('subfolder1',images)
            self.assertEqual(zip_file['class_name'], 'subfolder1')
        
            testzipfile = zipfile.ZipFile(zip_file['zip'])
            testresult = testzipfile.testzip()
            self.assertEqual(testresult, None)
        finally:
            if not testzipfile is None:
                testzipfile.close()
        self.ziphelper.clean_up()

    def test_create_temp_zip_from_files(self):
        pass

    def test_get_zip_file_list(self):
        pass

