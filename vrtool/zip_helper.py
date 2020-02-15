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
import zipfile
import tempfile
import shutil
from os.path import join, split
from random import randint
from pathlib import Path
from os import remove


class ZipHelper:
    """
    This class is providing convenience methods to create and delete
    temporary zip files from folders or list of files
    """

    def __init__(self):
        self.open_file_handles = []
        self.files_created = []
        self.temp_folder = tempfile.mkdtemp()

    def create_temp_zip_from_folder_content(self, folder):
        zip_files = []
        for class_images in folder:
            class_name = class_images["class_name"]
            images = class_images["images"]
            zip_file_dict = self.create_temp_zip_from_files(class_name, images)
            zip_files.append(zip_file_dict)
        return zip_files

    def create_temp_zip_from_files(self, class_name, list_of_files, limit=None):
        temp_file_name = class_name+'_positive_examples.zip'
        temp_file_path = self.to_wdc_path(
            join(self.temp_folder, temp_file_name))
        temp_zip_file = zipfile.ZipFile(
            temp_file_path, 'w', compression=zipfile.ZIP_STORED)

        for i, fileentry in enumerate(list_of_files):
            if(limit is not None and i >= limit):
                break
            fileentry = self.to_wdc_path(fileentry)
            temp_zip_file.write(fileentry,  split(fileentry)[1])
        temp_zip_file.close()

        f = open(temp_file_path, 'rb')
        self.open_file_handles.append(f)
        self.files_created.append(temp_file_path)
        return {"zip": f, "class_name": class_name}

    def to_wdc_path(self, path):
        return path.replace("\\", "/")

    def get_zip_file_list(self, class_name, images_in_class_folder, limit=700, chunksize=300):
        zip_file_list = []

        for i in range(0, len(images_in_class_folder), chunksize):
            if(i >= limit):
                break
            sublist = images_in_class_folder[i:i+chunksize]
            temp_file_name = class_name+'_' + \
                str(i)+'_'+str(randint(0, 1000))+'_positive_examples.zip'
            temp_file_path = join(self.temp_folder, temp_file_name)
            temp_file_path = self.to_wdc_path(temp_file_path)
            temp_zip_file = zipfile.ZipFile(
                temp_file_path, 'w', compression=zipfile.ZIP_STORED)

            for _, fileentry in enumerate(sublist):
                fileentry = self.to_wdc_path(fileentry)
                temp_zip_file.write(fileentry, split(fileentry)[1])
            temp_zip_file.close()

            f = open(temp_file_path, 'rb')
            self.open_file_handles.append(f)
            self.files_created.append(temp_file_path)
            zip_file_list.append({"zip": f, "class_name": class_name})

        return zip_file_list

    def clean_up(self):
        self.close_all_open_file_handles()
        self.remove_all_open_zips()
        self.clean_temp_folder()
        self.open_file_handles = []
        self.files_created = []

    def close_all_open_file_handles(self):
        for open_handles in self.open_file_handles:
            open_handles.close()

    def remove_all_open_zips(self):
        for open_file in self.files_created:
            remove(open_file)

    def clean_temp_folder(self):
        if(self.temp_folder is not None and Path(self.temp_folder).exists()):
            shutil.rmtree(self.temp_folder)
