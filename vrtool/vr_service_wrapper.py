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
import logging
import time
import json

from multiprocessing.pool import ThreadPool
from operator import itemgetter
from os import rename
from os.path import split, join

logging.basicConfig(level=logging.WARNING)


class VrServiceWrapper:
    """
    This service wrapper is providing methods to create and test Watson
    Visual Recognition Classifiers as well as providing helper functions
    for positive and negative image set creation.
    """

    def __init__(self, vr_instance, zip_helper):
        self.vr_instance = vr_instance
        self.zip_helper = zip_helper
        logging.debug("Initialized VrServiceWrapper")

    def create_classifier(self, classifier_name, positive_example_zips, negative_example_zip=None):
        img_param_chunks = []
        added_negative = False
        classifier_trained = False
        classifier_id = None

        if not isinstance(positive_example_zips, list):
            positive_example_zips = [positive_example_zips]

        img_param_chunks = [positive_example_zips[i:i + 5]
                            for i in range(0, len(positive_example_zips), 5)]

        for chunk in img_param_chunks:
            positive_img_params = {}
            negative_img_params = None
            for entry in chunk:
                if not added_negative:
                    if negative_example_zip is not None:
                        negative_img_params = negative_example_zip["zip"]
                        added_negative = True
                positive_img_params[entry['class_name']] = entry['zip']
            if not classifier_trained:
                classifier_info = self.vr_instance.create_classifier(
                    classifier_name, positive_examples=positive_img_params, negative_examples=negative_img_params)
                classifier_id = classifier_info.get_result()['classifier_id']
                successfully_trained = self.wait_for_finished_training(
                    classifier_id)
                if successfully_trained:
                    classifier_trained = True
                else:
                    logging.error(
                        "Error training classifier with classifier_id " + classifier_id)
                    raise Exception(
                        "Error training classifier with classifier_id " + classifier_id)

            else:
                if added_negative:
                    self.vr_instance.update_classifier(
                        classifier_id, positive_examples=positive_img_params)
                else:
                    self.vr_instance.update_classifier(
                        classifier_id, positive_examples=positive_img_params, negative_examples=negative_img_params)
                successfully_retrained = self.wait_for_finished_training(
                    classifier_id)
                if not successfully_retrained:
                    logging.error(
                        "Error retraining classifier with classifier_id " + classifier_id)
                    raise Exception(
                        "Error training classifier with classifier_id " + classifier_id)

    def wait_for_finished_training(self, classifier_id):
        status = self.get_training_status(classifier_id)
        while(status == 'training' or status == 'retraining'):
            time.sleep(10)
            status = self.get_training_status(classifier_id)

        if(status == 'ready'):
            return True
        else:
            return False

    def get_training_status(self, classifier_id):
        info = self.vr_instance.get_classifier(classifier_id)
        return info.get_result()['status']

    def test_classifier(self, classifier_id, images_file, threshold=0.5, use_default_classifier=None):
        classifier_ids = []

        if images_file is None:
            raise ValueError(
                "No valid zip file with testimages provided. Expecting a flat zip file with a maximum of 20 images.")

        if use_default_classifier is not None:
            classifier_ids.append('default')

        if classifier_id is not None and isinstance(classifier_id, list):
            classifier_ids = classifier_id
        elif classifier_id is not None:
            classifier_ids.append(classifier_id)
        else:
            if 'default' not in classifier_ids:
                classifier_ids.append('default')
            logging.info(
                "No classifier id provided for testing, using default classifier as fallback")

        images_file_content_type = "application/zip"
        res = self.vr_instance.classify(
            images_file=images_file, classifier_ids=classifier_ids,threshold=threshold, images_file_content_type=images_file_content_type)

        return res.get_result()

    def test_classifier_all_docs(self, classifier_id, test_data_zip_list, threshold=0.5, use_default_classifier=None):
        result_list = []
        pool = ThreadPool(len(test_data_zip_list))

        for test_data_zip in test_data_zip_list:
            result_list.append(pool.apply_async(self.test_classifier, args=(
                classifier_id, test_data_zip['zip'], threshold, use_default_classifier)))

        pool.close()
        pool.join()

        result_list = [r.get() for r in result_list]

        return result_list

    def get_img_class_as_example_zip(self, class_name, files):
        zip_file = self.zip_helper.create_temp_zip_from_files(
            class_name, files)

        return zip_file

    def bootstrap_classes(self, classifier_id, all_images_by_class):
        for class_images in all_images_by_class:
            images = class_images["images"]
            class_name = class_images["class_name"]
            self.bootstrap_class(classifier_id, class_name, images)

    def bootstrap_class(self, classifier_id, class_name, images_in_class_folder, threshold=0.2):
        zip_list = self.zip_helper.get_zip_file_list(
            class_name, images_in_class_folder)
        # classify all images from step 1
        result = self.test_classifier_all_docs(
            classifier_id, zip_list, threshold=threshold, use_default_classifier=None)
        # take class with highest probability and add class name as prefix

        for image in images_in_class_folder:
            image_name = split(image)[1]
            image_path = split(image)[0]
            most_likely_class, score = self.get_most_likely_class(
                result, image_name)
            if most_likely_class is not None:
                rename(image, join(image_path, most_likely_class +
                                   "_" + str(score) + "_" + image_name))

    def bootstrap_class_batch(self, classifier_id, class_name, images_in_class_folder, threshold=0.5, batch_size=10):
        zip_list = self.zip_helper.get_zip_file_list(
            class_name, images_in_class_folder)
        # classify all images from step 1
        num_images = len(zip_list)

        for i in range(0, num_images, batch_size):
            print("Bootstrapped " + str(i) + " of " +
                  str(num_images) + " batches")
            image_subst = zip_list[i:i + batch_size]
            result = self.test_classifier_all_docs(
                classifier_id, image_subst, threshold=threshold, use_default_classifier=None)
            # take class with highest probability and add class name as prefix

            for image in images_in_class_folder:
                image_name = split(image)[1]
                image_path = split(image)[0]
                most_likely_class, score = self.get_most_likely_class(
                    result, image_name)
                if most_likely_class is not None:
                    rename(image, join(image_path, most_likely_class +
                                       "_" + str(score) + "_" + image_name))

    def get_most_likely_classes(self, classification_results):
        parsed_results = self.parse_img_results(classification_results)
        for result in parsed_results:
            img = result.get("image")
            if img is not None:
                scores = result.get("results")
                if scores is not None and len(scores) > 0:
                    class_result = scores[0]
                    return class_result["class"], class_result["score"]
        return None, None

    def get_most_likely_class(self, classification_results, image_name):
        parsed_results = self.parse_img_results(classification_results)
        for result in parsed_results:
            img = result.get("image")
            result_image_name = split(img)[1]
            if img is not None and image_name == result_image_name:
                scores = result.get("results")
                if scores is not None and len(scores) > 0:
                    class_result = scores[0]
                    return class_result["class"], class_result["score"]
        return None, None

    def parse_img_results(self, results):
        parsed_results = []
        for result in results:
            for img in result["images"]:
                scores = []
                if "classifiers" in img:
                    for classi in img["classifiers"]:
                        for classes in classi["classes"]:
                            class_results = {}
                            class_results["score"] = classes["score"]
                            class_results["class"] = classes["class"]
                            scores.append(class_results)
                    predicted_score_1 = 0
                    predicted_class_1 = "None"
                    if scores is not None and len(scores) > 0:
                        scores = sorted(scores, key=itemgetter(
                            'score'), reverse=True)
                        predicted_score_1 = scores[0]['score']
                        predicted_class_1 = scores[0]['class']
                    image_name = split(img["image"])[1]
                    parsed_results.append(
                        {"image": img["image"], "results": scores, "image_name": image_name, "predicted_score_1": predicted_score_1, "predicted_class_1": predicted_class_1})
        return parsed_results

    def get_classifier_ids_by_name(self, classifier_name):
        matching_classifiers = []
        classifiers = self.vr_instance.list_classifiers().get_result()
        if("classifiers" in classifiers):
            matching_classifiers = [
                el for el in classifiers["classifiers"] if classifier_name in el['name']]
        return matching_classifiers
