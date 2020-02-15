# coding: utf-8

import configparser
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
import itertools
import pickle
import time
from multiprocessing.pool import ThreadPool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import VisualRecognitionV3
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold

from .corpora import Corpora
from .img_helper import duplicate_class
from .vr_service_wrapper import VrServiceWrapper
from .zip_helper import ZipHelper


class Runner():
    """
    Provides high-level functions to train and test visual recognition classifiers.
    """

    def __init__(self, corpus_dir, config_name=None, iam_api_key=None, service_instance_url=None):
        """ Initializes the Visual Recognition Runner. Creates instances of the Visual
        Recognition Service, the Zip helper and loads the corpus directory information.

        Arguments:
            config_name {string} -- The name of the config file.
            corpus_dir {string} -- The name of the folder that contains the corpus.
        """
        if config_name is None:
            config_iam_api_key = iam_api_key
            config_service_url = service_instance_url
        else:
            config_iam_api_key, config_service_url = self._get_config(
                config_name)

        authenticator = IAMAuthenticator(apikey=config_iam_api_key)

        self.vr_instance = VisualRecognitionV3(
            version='2018-03-19', authenticator=authenticator)
        self.vr_instance.set_service_url(config_service_url)

        self.zip_helper = ZipHelper()
        self.corpora = Corpora(corpus_dir)
        self.vr_service = VrServiceWrapper(self.vr_instance, self.zip_helper)

    def _get_config(self, config_name):
        """[summary]

        Arguments:
            config_name {string} -- The full name of the ini file that holds the config.

        Returns:
            tuple -- A tuple containing the old API key, the IAM API key and the service URL.
        """
        cfg = configparser.ConfigParser(defaults={})
        try:
            cfg.read(config_name)
            iam_api_key = cfg.get('vr', 'IAM_API_KEY')
            url = cfg.get('vr', 'URL')

            if(not url):
                url = 'https://gateway.watsonplatform.net/visual-recognition/api'
            if(not iam_api_key or iam_api_key is None or len(iam_api_key) < 44):
                iam_api_key = None

            if(iam_api_key):
                print("Using IAM authentication method.")
                print("IAM API key loaded:", iam_api_key[0:10], "***")
        except Exception as e:
            print(e)
            print("Please make sure that the config.ini file exists.")
        return (iam_api_key, url)

    ########################
    # Data Analysis
    ########################

    def get_corpora_info(self, corpora):
        """
        USED
        """
        corpora_info = []
        for corpus_name, corpus_dir in corpora:
            if("negative" not in corpus_name):
                imgs = corpus_dir.get_all_class_images()
                negatives = corpus_dir.get_negative_example_images()
                image_info = pd.DataFrame(imgs)
                negative_examples = pd.DataFrame(negatives)
                corpora_info.append({'corpus_name': corpus_name, 'corpus_dir': corpus_dir,
                                     'image_info': image_info, 'negative_examples': negative_examples})
                try:
                    print('------------------------------------------')
                    print("Corpus:", corpus_name)
                    class_stats = image_info.class_name.value_counts()
                    print(pd.DataFrame(
                        {'Classes': class_stats.index, 'Image Count': class_stats.values}).to_string(index=False))
                except AttributeError as e:
                    print(e)
        return corpora_info

    def get_available_corpora(self):
        """
        USED
        """
        return self.corpora.get_corpora()

    def create_experiments(self, splits, img_info):
        """ Create experiments for K-fold testing. Each experiment contains training and test data for a single fold.

        Arguments:
            splits {int} -- The number of folds.
            img_info {DataFrame} -- A DataFrame containing the image information

        Returns:
            array -- An array containing experiment dicts according to the number of splits.
        """
        experiments = []

        skf = StratifiedKFold(n_splits=splits)
        for train_index, test_index in skf.split(img_info, img_info['class_name']):
            experiment = {}
            experiment['train'] = img_info.iloc[train_index]
            experiment['test'] = img_info.iloc[test_index]
            experiments.append(experiment)

        print("Training Set:")
        print("-------------")
        for idx, experiment in enumerate(experiments):
            print("Split:", idx)
            print(experiment['train'].class_name.value_counts())

        print("")
        print("Test Set:")
        print("-------------")
        for idx, experiment in enumerate(experiments):
            print("Split:", idx)
            print(experiment['test'].class_name.value_counts())

        return experiments

    def save_experiments(self, experiments, corpus_to_train, splits):
        """Persist the experiments as pickle file to the modelconfigurations folder.

        Arguments:
            experiments {array} -- The array of experiments.
            corpus_to_train {string} -- The name of the corpus used for training and testing.
            splits {int} -- The number of folds.
        """
        with open("modelconfigurations/"+corpus_to_train+"_data_"+str(splits)+"_fold_" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".pkl", "wb") as fp:
            pickle.dump(experiments, fp)

    ########################
    # Training
    ########################

    def train_classifier_class_file_tuple(self, classifier_name, class_file_tuple_list):
        try:
            zip_files = []
            for (class_name, file_list) in class_file_tuple_list:
                zip_file_dict = self.zip_helper.create_temp_zip_from_files(
                    class_name, file_list)
                zip_files.append(zip_file_dict)
            self.vr_service.create_classifier(classifier_name, zip_files, None)
        finally:
            self.zip_helper.clean_up()

    def train_classifier_from_data_frame(self, classifier_name, df, limit=10):
        """
        USED in Train Test Split
        """
        zip_files = []
        negative_zip = None
        class_list = df['class_name'].unique()
        for class_name in class_list:
            class_rows = df[df['class_name'] == class_name]
            images = class_rows['image'].tolist()
            if len(images) >= limit:
                if (class_name == 'negative_examples'):
                    negative_zip = self.zip_helper.get_zip_file_list(
                        'negative_examples', images)[0]
                else:
                    zip_file_dict = self.zip_helper.get_zip_file_list(
                        class_name, images)
                    zip_files = zip_files + zip_file_dict

        self.vr_service.create_classifier(
            classifier_name, zip_files, negative_zip)

    def train_k_classifiers(self, experiments, corpus_to_train, splits):
        """Train classifiers according to the number of splits.This method actively waits until the classifiers are trained.
        Each resulting classifier will follow this naming convention: {corpus_to_train}_{actual_split}_{random_id_generated_by_Visual_Recognition}

        Arguments:
            experiments {array} -- The experiments used for training.
            corpus_to_train {string} -- The name of the corpus to be used. This will be the name of the resulting classifiers.
            splits {int} -- The number of folds.

        Returns:
            array -- An array containing the classification results of Visual Recognition.
        """
        start_time = time.time()

        results = []
        pool = ThreadPool(splits)

        for idx, experiment in enumerate(experiments):
            print("Starting training for iteration "+str(idx))
            results.append(pool.apply_async(self.train_classifier_from_data_frame, args=(
                corpus_to_train+"_"+str(idx), experiment['train'])))

        pool.close()
        pool.join()

        results = [r.get() for r in results]
        print("Finished Training after %s seconds." %
              (time.time() - start_time))
        return results

    ########################
    # Testing
    ########################

    def test_k_classifiers(self, experiments, classifier_ids, splits):
        """
        USED in K-Fold
        """
        start_time = time.time()

        results = []
        pool = ThreadPool(splits)

        for idx, experiment in enumerate(experiments):
            classifier_id = classifier_ids[idx]
            test_data = experiment['test']
            print("Starting Test "+str(idx))
            results.append(pool.apply_async(
                self.test_classifier_with_data_frame, args=(classifier_id, test_data)))

        pool.close()
        pool.join()

        results = [r.get() for r in results]
        print("Finished Testing after %s seconds." %
              (time.time() - start_time))
        return results

    def test_classifier_with_data_frame(self, classifier_id, df, threshold=0.3):
        """
        USED in Evaluation
        """
        results = []
        try:
            class_list = df['class_name'].unique()
            pool = ThreadPool(len(class_list))
            for class_name in class_list:
                class_rows = df[df['class_name'] == class_name]
                images = class_rows['image'].tolist()
                zip_list = self.zip_helper.get_zip_file_list(
                    class_name, images, chunksize=20)
                results.append(pool.apply_async(self.vr_service.test_classifier_all_docs, args=(
                    classifier_id, zip_list, threshold, None)))

            pool.close()
            pool.join()
            results = [r.get() for r in results]
            results = [item for sublist in results for item in sublist]
        finally:
            pass
            # self.zip_helper.clean_up()
        return results

    ########################
    # Evaluation
    ########################

    def get_confusion_matrix(self, y_actual, y_pred):
        """
        USED in Evaluation
        """
        return confusion_matrix(y_actual, y_pred)

    def print_cross_tab(self, evaluation, thresholds):
        for threshold in thresholds:
            y_actual, y_pred = self.get_y_values(evaluation, threshold)
            crosstab = pd.crosstab(y_actual, y_pred)
            if('None' in crosstab):
                none_col = crosstab['None']
                crosstab.drop(labels=['None'], axis=1, inplace=True)
                crosstab = crosstab.fillna(0)
                crosstab = crosstab.astype(int)
                crosstab.insert(len(crosstab.columns), 'None', none_col)

            print("Threshold: "+str(threshold))
            print(crosstab)

    def print_experiment_confusion_matrix(self, experiments, thresholds):
        for idx, experiment in enumerate(experiments):
            print("Iteration "+str(idx))
            self.print_cross_tab(experiment['evaluation'], thresholds)

    def print_consolidated_experiment_confusion_matrix(self, experiments, thresholds):
        for threshold in thresholds:
            y_actual, y_pred = self.get_consolidated_experiment_y_values(
                experiments, threshold)
            crosstab = pd.crosstab(y_actual, y_pred)
            if('None' in crosstab):
                none_col = crosstab['None']
                crosstab.drop(labels=['None'], axis=1, inplace=True)
                crosstab = crosstab.fillna(0)
                crosstab = crosstab.astype(int)
                crosstab.insert(len(crosstab.columns), 'None', none_col)

            print("Threshold: "+str(threshold))
            print(crosstab)

    def print_classification_report(self, evaluation, thresholds):
        print("Classification Report:")
        for threshold in thresholds:
            print("Threshold: "+str(threshold))
            y_actual, y_pred = self.get_y_values(evaluation, threshold)

            classification_report = self.get_classification_report(
                y_actual, y_pred)
            print(classification_report)

    def print_experiment_classification_report(self, experiments, thresholds):
        print("Classification Report for iterations:")
        for idx, experiment in enumerate(experiments):
            print("Iteration "+str(idx))
            for threshold in thresholds:
                print("Threshold: "+str(threshold))
                y_actual, y_pred = self.get_y_values(
                    experiment['evaluation'], threshold)
                classification_report = self.get_classification_report(
                    y_actual, y_pred)
                print(classification_report)

    def print_consolidated_experiment_classification_report(self, experiments, thresholds):
        print("Classification Report")
        for threshold in thresholds:
            print("Threshold: "+str(threshold))
            y_actual, y_pred = self.get_consolidated_experiment_y_values(
                experiments, threshold)
            classification_report = self.get_classification_report(
                y_actual, y_pred)
            print(classification_report)

    def print_consolidated_experiment_metrics(self, experiments, thresholds):
        for threshold in thresholds:
            print("Threshold: "+str(threshold))
            y_actual, y_pred = self.get_consolidated_experiment_y_values(
                experiments, threshold)
            print("Overall Accuracy: ", accuracy_score(y_actual, y_pred))

    def get_classification_report(self, y_actual, y_pred, labels=None):
        """
        USED
        """
        return classification_report(y_actual, y_pred, labels=labels)

    ########################
    # Helper
    ########################

    def get_y_values(self, evaluation, threshold=0.5):
        ev = evaluation.copy()
        ev.loc[ev['predicted_score_1'] < threshold,
               'predicted_class_1'] = 'None'
        y_actual = ev['class_name'].copy()
        y_actual = y_actual.replace("negative_examples", "None")
        y_actual = y_actual.replace("negative_example", "None")
        y_actual = y_actual.replace("negatives", "None")
        y_actual = y_actual.replace("negative", "None")

        y_pred = ev['predicted_class_1'].copy()
        y_pred = y_pred.replace("negative_examples", "None")
        y_pred = y_pred.replace("negative_example", "None")
        y_pred = y_pred.replace("negatives", "None")
        y_pred = y_pred.replace("negative", "None")

        return (y_actual, y_pred)

    def merge_predicted_and_target_labels(self, df, results):
        """
        USED in Evaluation
        """
        parsed_results = self.vr_service.parse_img_results(results)
        res_df = pd.DataFrame(parsed_results)
        return pd.merge(df, res_df, on="image_name")

    def get_consolidated_experiment_y_values(self, experiments, threshold):
        all_y_actual = None
        all_y_pred = None

        for _, experiment in enumerate(experiments):
            y_actual, y_pred = self.get_y_values(
                experiment['evaluation'], threshold)
            if(all_y_actual is None):
                all_y_actual = y_actual
                all_y_pred = y_pred
            else:
                all_y_actual = pd.concat([all_y_actual, y_actual])
                all_y_pred = pd.concat([all_y_pred, y_pred])

        return all_y_actual, all_y_pred

    ########################
    # Serialization
    ########################

    def confusion_matrix_to_csv(self, matrices, corpus_name):
        matrices_list = []
        merged_df = None

        for idx, value in enumerate(matrices):
            new_col = value[1].copy()
            new_col.insert(loc=0, column='Threshold', value=value[0])

            if (idx == 0):
                merged_df = new_col
            else:
                merged_df = pd.concat([merged_df, new_col])

        threshold_col = merged_df['Threshold']
        merged_df.drop(labels=['Threshold'], axis=1, inplace=True)
        merged_df = merged_df.fillna(0)
        merged_df = merged_df.astype(int)
        merged_df.insert(0, 'Threshold', threshold_col)

        name_placeholder = ""
        if (corpus_name is not None):
            name_placeholder = corpus_name+'_'

        merged_df.to_csv('./reports/confusion_matrices_report_' +
                         name_placeholder + time.strftime("%d-%m-%Y-%H-%M-%S") + '.csv', index=True)

    def classification_reports_to_csv(self, reports, corpus_name):
        report_data = []
        for threshold, report in reports:
            lines = report.split('\n')
            row = {}
            row['Threshold'] = threshold
            report_data.append(row)
            for line in lines[2:]:
                row = {}
                line = line.replace('micro avg', 'micro_avg').replace(
                    'macro avg', 'macro_avg').replace('weighted avg', 'weigthed_avg')
                row_data = line.split(' ')
                row_data = list(filter(None, row_data))
                if(len(row_data) > 0):
                    row['class'] = row_data[0]
                    row['precision'] = float(row_data[1])
                    row['recall'] = float(row_data[2])
                    row['f1_score'] = float(row_data[3])
                    row['support'] = float(row_data[4])
                    report_data.append(row)
        dataframe = pd.DataFrame.from_dict(report_data)

        name_placeholder = ""
        if (corpus_name is not None):
            name_placeholder = corpus_name+'_'

        dataframe.to_csv('./reports/classification_report_' +
                         name_placeholder + time.strftime("%d-%m-%Y-%H-%M-%S") + '.csv', index=False)

    def evaluation_result_to_csv(self, evaluation, corpus_name):
        evaluation_csv = evaluation.copy()

        cols = ['Target Class Name', 'Image Path in Corpus', 'Image Name',
                'Image Path in ZIP', 'Predicted Class', 'Predicted Score', 'Raw Results']
        ordered_cols = ['Image Name', 'Target Class Name', 'Predicted Class',
                        'Predicted Score', 'Image Path in Corpus', 'Image Path in ZIP', 'Raw Results']
        evaluation_csv.columns = cols

        name_placeholder = ""
        if (corpus_name is not None):
            name_placeholder = corpus_name+'_'

        evaluation_csv.to_csv("./reports/" + name_placeholder + "_result_" +
                              time.strftime("%d-%m-%Y-%H-%M-%S") + ".csv", columns=ordered_cols)

    ########################
    # Bootstrapping
    ########################

    def bootstrap_all(self, classifier_id, corpus_dir):   
        class_names = corpus_dir.get_all_class_names()
        bootstrap_images = corpus_dir.get_all_positive_examples(class_names)
        self.vr_service.bootstrap_classes(classifier_id, bootstrap_images)

    def bootstrap_specific_class(self, classifier_id, corpus_dir, class_name, threshold):
        unclassified_images = corpus_dir.get_unclassified_images(class_name)
        self.vr_service.bootstrap_class_batch(
            classifier_id, class_name, unclassified_images, threshold=threshold)

    def duplicate_all(self, corpus_dir, rotation_angle, lower_bound):
        class_names = corpus_dir.get_all_class_names()
        for cn in class_names:
            cn_images = corpus_dir.get_positive_examples(cn)
            if len(cn_images) <= lower_bound:
                duplicate_class(cn_images, rotation_angle)

    def duplicate_specific_class(self, corpus_dir, rotation_angle, class_name):
        class_images = corpus_dir.get_positive_examples(class_name)
        duplicate_class(class_images, rotation_angle)

    def plot_confusion_matrix(self, cm, y_actual, y_pred,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        y_actual = y_actual.unique()
        y_pred = y_pred.unique()
        y_actual_len = len(y_actual)
        if('None' in list(y_pred)):
            y_actual_len += 1
            y_actual = np.append('None', y_actual)

        tick_marks_y = np.arange(y_actual_len)
        tick_marks_x = np.arange(len(y_pred))
        plt.xticks(tick_marks_x, y_pred, rotation=45)
        plt.yticks(tick_marks_y, y_actual)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
