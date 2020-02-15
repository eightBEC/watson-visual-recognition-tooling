# coding: utf-8

# Copyright 2018-2020 Jan Forster
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from os import listdir
from os.path import join, isdir, dirname, realpath
from .corpus_directory import CorpusDirectory

class Corpora(object):
    """
    Represents multiple corpus objects within a folder as a single corpora object.
    On the filesystem, a folder that contains image folders is considered a corpora object.
    """
    def __init__(self, corpus_dir_name):
        """Initializes the corpora object instance based on the folde provided.
        
        Arguments:
            corpus_dir_name {string} -- Path to the corpora folder
        """
        working_dir = join(dirname(realpath(__file__)), corpus_dir_name)
        self.corpora = [(f, CorpusDirectory(join(working_dir, f)))
                        for f in listdir(working_dir) if isdir(join(working_dir, f))]

    def get_corpora(self):
        return self.corpora
