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
import re
from os import listdir
from os import fsencode
from os.path import isfile, join, isdir, splitext


class Directory(object):
    """
    Provides convenience methods to access files and subfolders in a given directory.
    """

    def __init__(self, path):
        self.path = fsencode(path).decode('utf-8')
        self.files = []
        self.image_file_extensions = ['.jpg', '.jpeg', '.png']

    def get_image_files(self, path):
        """ Returns all files in a given directory. Does not return files in sub folders.

        Arguments:
            path {string} -- The directory for which the files should be returned.

        Returns:
            list -- A list of strings containing files with their fully qualified path name for the given folder.
        """
        self.files = [f for f in listdir(path) if isfile(
            join(path, f)) and self.is_image_file(join(path, f)) and not f.startswith('.')]
        return self.files

    def is_image_file(self, filepath):
        try:
            __, fext = splitext(filepath)
            if (fext.lower() in self.image_file_extensions):
                return True
            else:
                return False
        except Exception as __:
            return False

    def get_sub_folders(self, path, filter_regex=None):
        """ Returns a list of all subfolders for a given path.
        If a filter is set, the name of the sub folder must match the filter provided.

        Arguments:
            path {string} -- A path pointing to the directory for which the sub folders should be returned.
            filter_regex {string} -- An optional variable, that is specifying a filter for the sub folder names.

        Returns:
            list -- A list of subfolders with their fully qualified path for a given path.
        """
        if filter_regex is not None:
            return [f for f in listdir(path) if isdir(join(path, f)) and re.match(filter_regex, f)]
        else:
            return [f for f in listdir(path) if isdir(join(path, f))]

    def has_sub_folder(self, path):
        sub_folders = self.get_sub_folders(path)
        if sub_folders is not None and len(sub_folders) > 0:
            return True
        else:
            return False

    @staticmethod
    def has_sub_folders(path):
        sub_folders = [f for f in listdir(path) if isdir(join(path, f))]
        if sub_folders is not None and len(sub_folders) > 0:
            return True
        else:
            return False

    def get_path(self):
        return self.path
