# Classify-Waste-Products-Using-TL-en-TF

## Overview

This project builds an image classifier that distinguishes between **organic** and **recyclable** waste using **transfer learning** and **fine-tuning** with the pre-trained **VGG16** convolutional neural network in TensorFlow/Keras.

Manual waste sorting is slow, costly, and error-prone. Automating it with computer vision can improve recycling rates, reduce contamination in recycling streams, and lower the amount of waste sent to landfills. Instead of training a deep network from scratch, which requires large datasets and significant computing power, this project reuses the visual features VGG16 learned from millions of ImageNet images and adapts them to the waste classification task.

## Objectives

- Apply transfer learning using VGG16 as a frozen feature extractor.
- Build and train a custom classification head for binary classification (organic vs. recyclable).
- Fine-tune the top convolutional layers of VGG16 to improve performance on the target domain.
- Compare the feature-extraction model against the fine-tuned model.
- Evaluate both models with standard metrics and visualize predictions on test images.

## Dataset

The dataset contains labeled waste images divided into two classes:

| Class | Label | Examples |
|-------|-------|----------|
| Organic | `O` | Food scraps, fruit and vegetable peels, plant material |
| Recyclable | `R` | Plastic, paper, cardboard, glass, metal |

Images are organized into `train` and `test` directories, with one subfolder per class, so they can be loaded directly with Keras data generators. Part of the training set is reserved for validation.

## Methodology

### 1. Data Preprocessing and Augmentation

- Images are resized to a fixed input size and pixel values are rescaled to the `[0, 1]` range.
- Data augmentation (horizontal flips, shifts, and zoom) is applied to the training set to increase variability and reduce overfitting.
- Validation and test images are only rescaled, so evaluation reflects real-world conditions.

### 2. Transfer Learning (Feature Extraction)

- VGG16 is loaded with ImageNet weights and `include_top=False`, removing its original classifier.
- All convolutional layers are frozen to preserve the pre-trained features.
- A custom head is added on top: `Flatten → Dense (ReLU) → Dropout → Dense (ReLU) → Dropout → Dense (sigmoid)`.
- The model is compiled with **binary cross-entropy** loss and trained with callbacks such as **EarlyStopping**, **ModelCheckpoint**, and a learning-rate scheduler.

### 3. Fine-Tuning

- The last convolutional block of VGG16 is unfrozen, allowing its high-level features to adapt to waste images.
- The model is retrained with a **low learning rate** to adjust the pre-trained weights gradually without destroying previously learned representations.

### 4. Evaluation

- Training and validation accuracy and loss curves are plotted for both models.
- Performance on the test set is measured with accuracy, precision, recall, and F1-score through a classification report.
- A confusion matrix shows where each model makes mistakes.
- Sample test images are displayed with their true labels and the predictions of both models for a visual comparison.

## Tech Stack

- **Python 3**
- **TensorFlow / Keras**: model building, training, and data loading
- **VGG16**: pre-trained CNN backbone (ImageNet weights)
- **NumPy**: numerical operations
- **Matplotlib**: visualization of curves and predictions
- **scikit-learn**: classification report and confusion matrix
- **Jupyter Notebook**: development environment

## Project Structure

```
Classify-Waste-Products-Using-TL-en-TF/
├── Clasificación final de residuos usando TL FT.ipynb   # Main notebook
└── README.md                                            # Project documentation
```

## Key Learnings

- Transfer learning makes it possible to reach strong performance with a relatively small dataset and limited compute.
- Freezing the base model first and fine-tuning later, with a low learning rate, produces more stable training.
- Data augmentation and dropout are effective tools against overfitting in image classification.

## Future Improvements

- Extend the classifier to more categories (plastic, glass, paper, metal).
- Compare VGG16 with lighter architectures such as MobileNetV2 or EfficientNet.
- Deploy the model as a web app or API for real-time classification.
- Optimize the model with TensorFlow Lite for mobile and edge devices.

## Acknowledgments

This project was developed as part of a hands-on deep learning laboratory on transfer learning and fine-tuning with Keras.

## Author

**Ricardo** — AI Engineering student
GitHub: [Richard002422](https://github.com/Richard002422)
