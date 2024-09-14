Lung Cancer Classification
Overview

This project introduces INAX-Net, a deep learning-based model designed for the classification of various types of lung cancer. The model integrates the strengths of Inception V4 and AlexNet architectures, combined with a multi-class SVM (Support Vector Machine) classifier, to achieve accurate detection of lung cancer types.

The primary focus of this model is to classify:

    Adenocarcinoma
    Large Cell Carcinoma
    Squamous Cell Carcinoma
    Normal Healthy Tissue

The model utilizes the LIDC-IDRI CT Scan Dataset to extract multi-scale features via Inception V4 and high-level features through AlexNet, providing a seamless flow of information across its layers. This combination ensures both detailed feature extraction and robust classification.
Key Features

    Inception V4: Multi-scale feature extraction capacity, ensuring the capture of fine details in medical imaging data.
    AlexNet: Strong high-level feature extraction for effective classification.
    Multi-class SVM Classifier: Ensures precise categorization of different lung cancer types.
    Dataset: Employs the LIDC-IDRI CT Scan Dataset for training and validation, which is well-known for lung cancer research.

Performance

The proposed model demonstrates superior performance over traditional methods with:

    Accuracy: 99.43%
    Specificity: 99.512%

This high performance highlights the potential of combining deep learning architectures with SVM classifiers for lung cancer detection, greatly enhancing precision, sensitivity, and specificity.
Conclusion

INAX-Net proves that deep learning techniques, when combined with powerful classifiers like SVM, can significantly improve the accuracy and reliability of lung cancer detection systems. This innovative approach offers a promising solution for enhancing early detection and treatment planning in the medical field.
Keywords

Lung Cancer • Deep Learning • AlexNet • Inception V4 • Machine Learning • SVM • CT Scans • Medical Imaging
