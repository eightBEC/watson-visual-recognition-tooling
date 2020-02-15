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
from os.path import isfile, join, isdir, split
from .directory import Directory


class CorpusDirectory(Directory):
    """
    A helper class for corpus data that is organized in folders.
    Each subfolder represents an entity and its respective contents that can be used for training purposes.
    E.g.:
    ./image_corpus/
        ./car/
            car03.jpg
            car04.jpg
            car01.jpg
            car02.jpg
        ./truck/
            truck01.jpg
            truck02.jpg
    """

    def __init__(self, path):
        Directory.__init__(self, path)

    def get_positive_examples(self, class_name):
        return self.get_positive_examples_in_folder(class_name, self.get_path())

    def get_positive_examples_in_folder(self, class_name, path):
        img_class_folder = join(path, class_name)
        img_files = self.get_image_files(img_class_folder)

        if img_files:
            return [join(img_class_folder, f) for f in img_files]
        return []

    def get_all_positive_examples(self, class_names):
        results = []
        for class_name in class_names:
            positive_examples = self.get_positive_examples(class_name)
            results.append({'class_name': class_name,
                            'images': positive_examples})
        return results

    def get_all_class_names(self):
        return [f for f in listdir(self.get_path()) if isdir(join(self.get_path(), f)) and re.match('[A-Za-z0-9]+', f)]

    def get_all_class_names_in_folder(self, path):
        return [f for f in listdir(path) if isdir(join(path, f)) and re.match('[A-Za-z0-9]+', f)]

    def get_classifier_association(self, classifier_prefix='classifier'):
        result = []
        classifer_folders = [f for f in listdir(self.get_path()) if isdir(
            join(self.get_path(), f)) and f.startswith(classifier_prefix)]
        for classifier_folder in classifer_folders:
            classifier_path = join(self.get_path(), classifier_folder)
            classes = self.get_all_class_names_in_folder(classifier_path)
            for class_name in classes:
                positive_examples = self.get_positive_examples_in_folder(
                    class_name, classifier_path)
                for positive_example in positive_examples:
                    entry = {'classifier': classifier_folder,
                             'class_name': class_name, 'image': positive_example}
                    result.append(entry)
        return result

    def get_all_class_images(self):
        result = []
        classes = self.get_all_class_names()
        for class_name in classes:
            positive_examples = self.get_positive_examples(class_name)
            for positive_example in positive_examples:
                image_name = split(positive_example)[1]
                entry = {'class_name': class_name,
                         'image': positive_example, 'image_name': image_name}
                result.append(entry)
        return result

    def get_negative_example_images(self):
        result = []
        negatives = [f for f in listdir(self.get_path()) if isdir(
            join(self.get_path(), f)) and re.match('negative*', f)]
        for class_name in negatives:
            negative_examples = [join(self.get_path(), class_name, f)
                    for f in self.get_image_files(join(self.get_path(), class_name))]
            for negative_example in negative_examples:
                image_name = split(negative_example)[1]
                entry = {'class_name': class_name,
                         'image': negative_example, 'image_name': image_name}
                result.append(entry)
        return result
