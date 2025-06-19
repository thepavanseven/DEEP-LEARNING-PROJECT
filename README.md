# DEEP-LEARNING-PROJECT

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: RUTTALA PAVAN TEJA

*INTERN ID*: CT06DF466

*DOMAIN*: DATA SCIENCE

*DURATION*: 6 WEEKS

*MENTOR*: NEELA SANTOSH

## Overview
This project focuses on building an image classification system capable of distinguishing between two types of flowers: roses and tulips. The primary goal is to leverage deep learning techniques, specifically convolutional neural networks (CNNs), to automate the process of flower recognition from digital images. The project demonstrates the full workflow, from data preparation and augmentation to model training, evaluation, and visualization of results, using the PyTorch deep learning framework.

Platform and Tools Used
Platform:
The entire project was developed and executed on a local machine using Jupyter Notebook, which provides an interactive environment for writing and running Python code. Jupyter is particularly well-suited for exploratory data analysis, visualization, and iterative model development.

Programming Language:
Python 3.x

Key Libraries and Frameworks:

PyTorch: Used for building and training the neural network models.

Torchvision: Provided access to pre-trained models (like ResNet-18), image transformations, and dataset utilities.

Matplotlib & NumPy: Used for data visualization and numerical operations.

PIL (Python Imaging Library): For image loading and basic processing.

Hardware:
The code is designed to utilize GPU acceleration if available, but it can also run on standard CPUs.

Project Workflow
1. Data Preparation
The dataset is organized locally in a folder named flowers_dataset, with subfolders for training and validation splits. Each split contains two subfolders: roses and tulips, each holding relevant image files. This structure is compatible with PyTorch’s ImageFolder utility, which automatically labels images based on their folder names.

2. Data Augmentation and Loading
To improve the model’s generalization ability, various data augmentation techniques are applied, such as random cropping and horizontal flipping for the training set. Both training and validation images are normalized using the mean and standard deviation values commonly used for models pretrained on ImageNet. Data loaders are created to efficiently batch and shuffle the data during training and validation.

3. Model Selection and Transfer Learning
A pre-trained ResNet-18 model from Torchvision is used as the backbone for this classification task. Transfer learning is employed by freezing all layers except the final fully connected layer, which is re-trained to distinguish between roses and tulips. This approach leverages the feature extraction capabilities of models trained on large datasets, enabling effective learning even with a relatively small flower dataset.

4. Training and Evaluation
The model is trained for several epochs using stochastic gradient descent (SGD) as the optimizer and cross-entropy as the loss function. During each epoch, the model’s performance on both the training and validation sets is monitored, with accuracy and loss metrics logged for analysis. The best-performing model can be saved for future inference.

5. Visualization
To ensure the data pipeline is functioning correctly, and to provide insight into the model’s predictions, batches of images are visualized along with their predicted and true class labels using Matplotlib. This step is crucial for qualitative assessment and debugging.

Real-World Applications
Automated flower classification has numerous practical applications, including:

Mobile Applications:
Integrating this model into a smartphone app can allow users to identify flowers in real time by simply taking a photo.

Botanical Research:
Assisting botanists and researchers in cataloging and studying plant species by automating the identification process.

Agriculture:
Helping farmers and horticulturists monitor and manage crops, detect invasive species, or track flowering patterns.

Education:
Serving as an interactive tool for students and educators in biology and botany classes.

Environmental Monitoring:
Supporting conservation efforts by enabling large-scale, automated monitoring of plant biodiversity in various ecosystems.

Conclusion
This project showcases the power and flexibility of PyTorch and transfer learning for image classification tasks, even with limited data and computational resources. By following best practices in data preparation, model training, and evaluation, it is possible to build robust and efficient models for specialized applications like flower classification. The techniques demonstrated here can be readily adapted to other image recognition problems, making this project a valuable template for future work in computer vision and deep learning.

#output

<img width="658" alt="Image" src="https://github.com/user-attachments/assets/baf13349-1886-48a1-a905-1fc7ac47fd37" />
