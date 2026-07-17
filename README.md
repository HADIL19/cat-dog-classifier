# Cat and Dog Image Classifier

A convolutional neural network built with TensorFlow 2.0 / Keras to classify images as cats or dogs, completed as part of the freeCodeCamp Machine Learning with Python certification.

## Project Overview

The goal was to build an image classifier that correctly labels photos of cats and dogs with at least 63% accuracy (70%+ for extra credit), using a dataset of:
- 2000 training images (1000 cats, 1000 dogs)
- 1000 validation images (500 cats, 500 dogs)
- 50 unlabeled test images

## What I Built

### 1. Data pipeline (`ImageDataGenerator` + `flow_from_directory`)
Created three image generators to feed data into the model:
- **Train / validation generators**: rescaled pixel values from `[0, 255]` to `[0, 1]`, and read images directly from labeled subfolders (`cats/`, `dogs/`) using `class_mode='binary'`.
- **Test generator**: required a workaround since the test folder has no class subfolders — pointed `directory` to the *parent* of the test folder and passed `classes=['test']` to treat it as one pseudo-class, with `class_mode=None` (no labels) and `shuffle=False` (to keep predictions aligned with the original image order).

### 2. Data augmentation
Since 2000 training images is a small dataset, there's a real risk of overfitting — the model memorizing exact training photos instead of learning general cat/dog features. To combat this, I recreated the training generator with random transformations applied to every image on the fly:
- Rotation
- Width/height shifts
- Zoom
- Horizontal flip

Each epoch, the model effectively sees a "new" variation of every image, artificially expanding the training data without collecting more photos.

### 3. CNN architecture
Built a `Sequential` model with three convolutional blocks of increasing depth:

```
Conv2D(16) → MaxPooling2D
Conv2D(32) → MaxPooling2D
Conv2D(64) → MaxPooling2D
Flatten
Dense(512, relu)
Dense(1, sigmoid)
```

Filter counts double at each block (16 → 32 → 64) because deeper layers need to represent a much larger number of possible feature *combinations* — early layers detect simple edges/colors, while deeper layers combine those into shapes and textures.

Compiled with:
- **Optimizer**: `adam`
- **Loss**: `binary_crossentropy` (standard pairing for a sigmoid output on a two-class problem)
- **Metric**: `accuracy`

### 4. Training
Trained using `model.fit()` with:
- `steps_per_epoch` / `validation_steps` computed as `ceil(total_images / batch_size)`, rounding up so no images get dropped from a partial final batch
- 15 epochs, batch size 128

### 5. Prediction
Ran `model.predict()` on the test generator to get a probability (0–1) for each of the 50 test images, then used `plotImages()` to display each image with its predicted label ("X% dog" or "X% cat").

## Key Concepts I Learned

- **Convolution & pooling**: how `Conv2D` layers slide small filters over an image to detect patterns, and how `MaxPooling2D` downsamples feature maps while keeping the strongest signals — and why stacking these layers lets a network build up from simple to complex visual features.
- **Overfitting and data augmentation**: why a small dataset risks memorization rather than generalization, and how random transformations act as a practical (not just theoretical) fix.
- **Sigmoid + binary crossentropy**: why this pairing is the standard choice for binary classification, and how to interpret a single output neuron as a probability.
- **`flow_from_directory` quirks**: that it expects class subfolders by default, and how to work around that for an unlabeled test set using `classes=[...]` and pointing at a parent directory.
- **Reading a model summary**: how to sanity-check an architecture by tracing spatial dimensions shrinking and channel depth growing layer by layer, and confirming the `Flatten` output size matches `height × width × channels` from the last conv block.

## Result

**Achieved 70.0% accuracy** on the test set — passing the 63% requirement and clearing the 70% extra-credit bar.

## Debugging Notes

One issue came up in Cell 11's grading logic: `model.predict()` returns predictions shaped like `(50, 1)` (a 2D array — each prediction wrapped in its own 1-element array), rather than a flat `(50,)` list of plain numbers. Python's built-in `round()` only works on plain floats, so looping over the raw predictions threw:
```
TypeError: type numpy.ndarray doesn't define __round__ method
```
The fix was adding `.flatten()` to the predictions in Cell 10:
```python
probabilities = model.predict(sample_test_images).flatten()
```
This is a good reminder that Keras/TensorFlow methods often return NumPy arrays with an extra dimension baked in (especially for single-output models), and it's worth checking `.shape` before assuming data is in the format a later step expects.

## Tools Used
- TensorFlow 2.x / Keras
- Google Colaboratory
- NumPy, Matplotlib
## Resources 
https://youtu.be/eg8DJYwdMyg?si=XzjPA3QdJQ6maONR 
