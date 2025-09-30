# DnR-net: Detection and Recognition Network

DnR-net is a tool that lets you detect and classify Ancient Greek characters written on papyrus.

You can try it out using the dataset from the *ICDAR23 Competition on Detection and Recognition of Greek Letters on Papyri*:
[https://zenodo.org/records/13825619](https://zenodo.org/records/13825619)

---

### 1. Set up your environment

We recommend using **Conda** to manage the environment.

To create and install it, run:

```
conda env create -f environment.yml
```

When the installation is done, activate the environment:

```
activate dr_net
```

---

### 2. The `_config.py` file

This file controls all the global settings of the project (for example, file paths).
Each script also has its own parameters, usually at the top of the file, that you can adjust as needed.

---

## 3. Train the Resizer

The first step is to train the resizer model, which handles preprocessing of the images.

Run:

```
python train_resizer_model.py
```

When the script finishes, you’ll find the trained weights saved in the `checkpoints` folder.

Tip: The first few lines of the script let you change important parameters.

---

## 4. Train the DnR model

### Step 4.1: Preprocess the dataset

Before training the main model, it helps to preprocess the dataset to speed things up.

Run:

```
python dataset_gen_prepr.py
```

This will create a preprocessed version of your dataset inside the `data` folder.

Again, check the first lines of the script to customize parameters.

---

### Step 4.2: Train the main model

Now you’re ready to train the main DnR model.
Run:

```
python train_dr_FastRCNN.py
```

When the training is complete, the weights will be saved in the `checkpoints` folder.

As always, you can tweak parameters at the top of the script.

---

### Step 4.3: Train with cross-validation (optional)

If you want, you can also train the model using **cross-validation**.
In this mode, the training set is split into several folds. Each fold is used once as validation, and you’ll end up with one set of weights per fold.

Run:

```
python train_dr_FastRCNN_crossval.py
```

---

## 5. Run the model on new images

To test the model on new images:

1. Place the weights you want to use in the `data/weights` folder.

   * You can include more than one set of weights.
   * If you do, the network will run each set one by one and then combine the results into a final output.

2. Make sure the resizer weights are well indexed in the `_config.py` file.

3. Put the images you want to analyze inside `data/images_to_test`.

4. Run the prediction script:

```
python predict_Fast_RCNN.py
```

The output files will appear in the `out` folder.
