# Checkpoints

This folder is where the trained model weights are stored.

---

## The `weights` subfolder

Place the trained models you want to use for prediction inside the `weights` subfolder.

* If you include more than one weight file, the prediction step will run each model one by one.
* At the end, the results from all models will be merged into a single output.

---

## The `resizer` subfolder

Place the trained resizer models you want to use during prediction in the `resizer` subfolder.
These models are specifically used for the preprocessing resizer step.
