# Visual Recognition Tooling
This is a set of tools will should help to get up to speed when using IBM Watson Visual Recognition. It provides helpers to simplify the training, testing and evaluation of classifiers and tools to manipulate your image files.

## Features
- Creation of binary and multi class classifiers
- K-Fold Cross Validation
- Persisting train, test and result set
- Image augmentation
- Image manipulation (batch cropping, batch resizing)

## Installation

To install missing dependencies use the command:
```bash
pip install -r requirements.txt
```

## Configuration

Duplicate the dummy.config.ini to config.ini to get started.

Configure your config.ini by entering your IAM API key and URL of your Visual Recognition service instance.
```
IAM_API_KEY:your_IAM_api_key
URL:the_url_of_the_service
```

## Getting Started

All notebooks in this porject are templates. To train and evaluate your own classifiers, 
it is recommended to duplicate the Jupyter Notebooks and adjust them to your needs.

To start the Visual Recognition Tooling just run:
```bash
jupyter notebook
```

To train and test classifiers using IBM Watson Visual Recognition open the notebook `K-Fold Testing Tool.ipynb`. <br>
To evaluate trained  IBM Watson Visual Recognition classifiers open the notebook `Evaluation Tool.ipynb`.<br>
To augment images open the notebook `Augmentation Tool.ipynb`.<br>
To perform batch resize and cropping open the notebook `Resize & Cropping Tool.ipynb`.<br>

## Project Structure 
The project consists of the following modules and folders:

* ./corpus - The image data
* ./modelconfigurations - The saved training, testing, evaluation data set
* ./tests - Unit Tests
* ./vrtool - Helper Library
* Augmentation Tool.ipynb
* Dataset Helper And Evaluator K-Fold.ipynb
* dummy.config.ini
* image_licenses
* installation.md
* README.md
* requirements.txt
* Resize & Cropping Tool.ipynb

## Image Corpus Layout
Currently the tooling is working with image corpora that are file and folder based. An image corpus can consist of several folders. Each folder represents a class the respective classifier will be able to recognize. Each class folder  contains all images that will be used to train and test the classifier on this class. If only one class per classifier is given, a folder called negative_examples is needed as well.

In the corpus directory of this project you'll find examples of the folder structure.

To get a better understanding of the layout, take a look at this sample folder hierarchy:
```
 ./corpus
     /bmw_corpus
         /three
             320.jpg
             330.jpg
         /five
             530.jpg
             550.jpg
         /seven
             750.jpg
             760.jpg
     /audi_corpus
         /athree
             a3.jpg
         /afour
             a4.jpg
     /mercedes_corpus
         /sclass
             s500.jpg
         /negative_examples
             eclass.jpg
```

## Training and Testing Classifiers
To create new classifiers using IBM Watson Visual Recognition open the notebook `K-Fold Testing Tool.ipynb` in the Jupyter Browser window. All steps required to train the classifiers are described in the Jupyter Notebook.

### Integration into IBM Watson Studio

1. Distribute the library by running:
   ```
   python setup.py sdist
   ```
2. Upload the file `./dist/VisualRecognitionTooling-1.0.tar.gz` to your IBM Watson Studio project.
3. Include the library into your Notebook by inserting a project token. Click More>Insert project token (three dots in the top right of your Watson Studio Notebook).
4. Below the hidden cell that was created, insert the following code to load in the packaged version of this library:
   ```
   with open("VisualRecognitionTooling-1.0.tar.gz","wb") as f:
       f.write(project.get_file("VisualRecognitionTooling-1.0.tar.gz").read())
   !pip install VisualRecognitionTooling-1.0.tar.gz
   !pip install ibm-watson==4.2.1 --quiet
   !conda install opencv --quiet
   ```

### 
Created by Jan Forster