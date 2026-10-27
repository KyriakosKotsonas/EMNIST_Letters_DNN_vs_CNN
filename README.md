# EMNIST Handwritten Letters – Original Coursework (MSDS680)

This folder contains my original notebooks from the MSDS680 course project. Their content is **unchanged**, exactly as they were run in Google Colab; the only change is a step number added to the front of each file name, so they sort in the order described below. They show how the model's accuracy went from **about 44% to about 92%** on classifying handwritten letters.

A cleaned-up, single-notebook version of the final work is in the companion folder `EMNIST_2_clean_notebook`.

---

## The task

The [EMNIST Letters](https://www.nist.gov/itl/products-and-services/emnist-dataset) dataset contains 145,600 grayscale images of handwritten letters (28×28 pixels). There are 26 classes (a–z, upper and lower case merged), 124,800 training images and 20,800 test images. The goal was to build a neural network that recognises the letter in each image.

I wrote my own loader for the raw EMNIST files (the gzip-compressed IDX binary format). It reads the file headers (magic number, image count, rows, columns) and converts the bytes into NumPy arrays.

---

## How the accuracy improved, step by step

The order below is my reconstruction of how the work progressed (the notebooks weren't dated individually). The number at the start of each file name matches its step.

### Step 1: First DNN experiments, peak ~44% → `1_Kotsonas_CourseProject.ipynb`

- I built a custom data feed (`EMNISTVectorDataFeed`) using `tf.data`. It flattens each image to 784 values, scales pixels to 0–1, shuffles and batches.
- I built a configurable deep neural network (`DNNNormalized`): fully connected layers with batch normalisation, dropout and a softmax output.
- I ran several hyperparameter experiments (numbered 5, 10, 22 and 77) with different layer sizes, dropout rates (0.1–0.4), learning rates (0.001–0.05) and batch sizes (32–256).
- **Result:** the best run reached **43.9% validation accuracy after the first epoch**. Then training became unstable: the loss rose each epoch and accuracy fell to about 6%, so I stopped the run. A second run never got above about 4% (random guessing for 26 classes is 3.8%).

**What was wrong:**
1. **Label mismatch.** EMNIST labels letters as 1–26, but the network's 26 outputs are numbered 0–25. The labels were one-hot encoded without shifting them, so every "z" (label 26) became an all-zero target, and label 0 never appeared.
2. **Unstable training settings.** Some combinations of learning rate and batch size made the loss diverge instead of decrease.

### Step 2: Trying a CNN, still low (24–35%) → `2_cnn.ipynb`

- I switched to a convolutional neural network with two Conv2D + MaxPooling blocks, a 128-unit dense layer and dropout. I also added pixel standardisation.
- **Result:** 24–35% validation accuracy.
- **Lesson:** changing the architecture alone didn't help, because the labels were still encoded the same way. This pointed to the data pipeline as the real problem, not the model.

### Step 3: Fixing the labels, DNN reaches ~91% → `3_last.ipynb`

- I subtracted 1 from every label before one-hot encoding (`to_categorical(train_labels - 1, 26)`), so the letters map correctly to outputs 0–25.
- A straightforward Keras DNN (784 → 256 → 128 → 128 → 64 → 26, Adam optimizer, batch size 256, 25 epochs) reached **90.97% accuracy on the official 20,800-image test set**.
- The custom `DNNNormalized` model, retrained with the corrected labels, reached **88.2% validation accuracy** after 12 epochs.
- I also found a bug in my **evaluation code**. It reported 4.3% accuracy even though the model was working, because the true labels were collected by flattening the one-hot matrix (540,800 values instead of 20,800) rather than taking `argmax`. The model was fine; the metric calculation was wrong.

### Step 4: CNN with the fixed labels, ~75% → `4_Kotsonas_CNN.ipynb`

- I applied the label fix (`labels - 1`) to the CNN pipeline and moved the settings into a configuration dictionary.
- **Result:** 74.8% test accuracy. The CNN learned, but the learning rate (0.02 with Adam) was too high for it to settle into a good solution.

### Step 5: Final CNN, ~92% → `5_Kotsonas.ipynb` (and `Kotsonas.py`, the same notebook exported as a script)

- I kept the label fix and used Adam's default learning rate (0.001).
- I added **standardisation** (subtract the training-set mean, divide by its standard deviation) and **data augmentation**: each image is randomly cropped from 28×28 to 24×24, so the network sees slightly shifted letters and generalises better.
- I split the training data 80/20 into training (99,840 images) and validation (24,960 images) sets.
- The model: Conv2D (32 filters, 3×3) → MaxPooling → Flatten → Dense (128) → Dense (26, softmax), trained for 10 epochs.
- **Result: 91.6% validation accuracy.** Most letters scored an F1 of 0.91–0.97. The weakest were **i** (0.71) and **l** (0.74), which look almost identical when handwritten, followed by **g** (0.82) and **q** (0.84). I plotted a confusion matrix and printed a full classification report.

---

## Summary of results

| Stage | Notebook | Model | Accuracy |
|---|---|---|---|
| 1 | `1_Kotsonas_CourseProject.ipynb` | Custom DNN, labels not shifted | ~44% (then collapsed) |
| 2 | `2_cnn.ipynb` | CNN, labels not shifted | 24–35% |
| 3 | `3_last.ipynb` | Keras DNN, labels fixed | **91.0%** (official test set) |
| 3 | `3_last.ipynb` | Custom DNN, labels fixed | 88.2% (validation) |
| 4 | `4_Kotsonas_CNN.ipynb` | CNN, labels fixed, high learning rate | 74.8% |
| 5 | `5_Kotsonas.ipynb` | CNN, fixed labels + standardisation + augmentation | **91.6%** (validation) |

## Key takeaways

- **Check the data pipeline before tuning the model.** The biggest improvement came from correcting how the labels were encoded, not from a more complex network.
- **Learning rate matters.** The same CNN scored 75% with a learning rate of 0.02 and 92% with 0.001.
- **Validate the evaluation code too.** A metric bug can make a working model look broken.
- With these fixes, a simple DNN (91.0%) and a simple CNN (91.6%) performed similarly on this task.

## Note on the notebooks

These are the original files, kept exactly as they were run, including dead ends, debugging cells and error messages. They were written in Google Colab and depend on Google Drive paths and the course's `radnn` helper library, so they won't run as-is outside that environment. The clean, self-contained version is in `EMNIST_2_clean_notebook`.

**Tools:** Python, TensorFlow/Keras, NumPy, scikit-learn, Matplotlib, Google Colab.
