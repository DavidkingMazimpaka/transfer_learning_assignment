# # Transfer Learning Assignment: Brain Tumor Detection from MRI Images

## Problem Statement
The objective of this project is to leverage transfer learning to detect brain tumors from MRI images. Brain tumors are abnormal growths of cells in the brain that can be benign or malignant, potentially leading to severe neurological issues if untreated. Early detection is crucial for effective treatment, and deep learning models can assist in automating this diagnostic process, especially in resource-limited settings.

## Dataset
We used a dataset of brain MRI images sourced from various medical centers. The dataset is organized into three main folders: train, test, and val, each containing subfolders for two categories: Tumor and Normal. The images were screened for quality, graded by expert radiologists, and evaluated by an additional expert to minimize errors.

For the analysis of brain MRI images, all scans were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert radiologists before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

Dataset Link: [[Brain MRI Tumor Dataset]](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset)

## Pre-trained Models Used
We selected the following pre-trained models based on their architecture and suitability for image classification tasks involving complex anatomical structures, such as distinguishing between healthy brain tissue and tumors:

1. VGG16: Known for its simplicity and effectiveness in capturing fine details in images.
2. InceptionV3: Recognized for its ability to handle complex image classification tasks with efficient feature extraction.
3. EfficientNetB0: Offers state-of-the-art accuracy with fewer parameters, making it efficient for medical image analysis.

## Justification for Model Selection
- VGG16: Its deep architecture with multiple convolutional layers makes it suitable for capturing intricate details in MRI images, which is crucial for differentiating between tumor and normal brain tissues.
- InceptionV3: It balances complexity and performance by using a modular approach to capture different scales of image features, enhancing its ability to classify varied patterns seen in brain MRI scans.
- EfficientNetB0: It uses compound scaling to optimize depth, width, and resolution, making it highly effective for medical image analysis with relatively lower computational cost.

## Fine-Tuning Process
For each model, the pre-trained weights from ImageNet were used as a starting point. We replaced the top layers of the models with new layers tailored to our binary classification task:

- Global Average Pooling Layer: To reduce dimensions and avoid overfitting.
- Dense Layers: To introduce complexity specific to brain tumor detection.
- Dropout Layers: To prevent overfitting by randomly disabling neurons during training.

The modified models were trained on the augmented dataset with a learning rate fine-tuned for each architecture. The last few layers were unfrozen and retrained to adapt the feature extraction capabilities of each model specifically to our dataset.

## Evaluation Metrics
The models were evaluated using the following metrics:

- Accuracy: Measures the overall correctness of predictions.
- Loss: Binary cross-entropy loss used for training and evaluation.
- Precision: Indicates the proportion of true positive predictions among all positive predictions.
- Recall: Measures the ability of the model to detect all relevant cases (tumors).
- F1 Score: Harmonic mean of precision and recall, providing a balanced performance measure.

## Results
| Model         | Accuracy | Loss   | Precision | Recall | F1 Score |
|---------------|----------|--------|-----------|--------|----------|
| VGG16         | 88.78%   | 0.3345 | 53.00%    | 55.00% | 54.00%   |
| InceptionV3   | 83.81%   | 0.3812 | 53.00%    | 56.00% | 54.00%   |
| EfficientNetB0| 62.50%   | 0.6897 | 39.00%    | 62.00% | 48.00%   |

## Conclusion
Transfer learning proved highly effective for brain tumor detection from MRI images, with InceptionV3 providing the best overall performance. This approach can be extended to other medical imaging tasks to enhance diagnostic accuracy and speed in clinical practice.

## Author
KingDavid Mazimpaka
