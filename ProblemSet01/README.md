# Problem Set 01 – Medical Image Classification

## Dataset
The dataset contains 5,863 pediatric chest X-ray images (anterior-posterior view), categorized into two classes:
- **Normal**
- **Pneumonia**

The images are organized into three folders:
- `train/`
- `test/`
- `val/`

Each folder has subdirectories for the two categories.

## Approach & Methodology
1. **Data Preparation**
   - Used `ImageDataGenerator` for rescaling and augmentation.
   - Loaded images from the train, validation, and test directories.

2. **Model Architecture**
   - Convolutional Neural Network (CNN) with:
     - Multiple `Conv2D` + `MaxPooling2D` layers
     - Flatten layer
     - Dense hidden layers with ReLU activation
     - Output layer with Sigmoid activation for binary classification

3. **Training**
   - Trained for 10 epochs with batch size set appropriately.
   - Monitored both training and validation accuracy during training.

4. **Evaluation**
   - Evaluated the model on the test set.
   - Plotted **Training vs Validation Accuracy** and **Loss curves**.

## Results
- **Training Accuracy:** ~95–100%
- **Validation Accuracy:** fluctuated between ~70–95%
- **Test Accuracy:** (insert your actual test accuracy here)

## Findings
- The CNN successfully learned to distinguish between Normal and Pneumonia X-rays.
- High training accuracy but fluctuating validation accuracy suggests possible **overfitting**.
- Improvements could include:
  - Adding **Dropout layers** or **regularization**
  - Increasing the number of epochs
  - Fine-tuning hyperparameters

## Conclusion
This model demonstrates the potential of CNNs in medical image classification. While the results are promising, further optimization is needed to improve validation stability and generalization.
