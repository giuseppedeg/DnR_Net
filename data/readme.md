# Data

This folder is where you need to place your datasets.

For each dataset you want to use, create a separate folder.
Inside each dataset folder, you should have:

* an `imgs` folder containing all the images
* a `data.json` file in COCO format with the training set annotations
* a `test.json` file in COCO format with the test set annotations
* *(Optional)* a `categories.json` file. If this file is present, its annotations will be used instead of the ones provided in the COCO files.
