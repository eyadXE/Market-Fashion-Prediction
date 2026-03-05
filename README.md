# Fashion Image Classification with Deep Learning

## Overview


### Demo Video

[![Watch the demo](thumbnail.gif)](demo.gif)

hugging face : https://huggingface.co/spaces/eyad222/Fashion-resnet/tree/main
This project builds an end-to-end **Deep Learning system for fashion image classification**. The system takes an input image of a clothing item and predicts its category using a trained convolutional neural network.

The project demonstrates the **complete machine learning lifecycle**:

* Dataset acquisition and exploration
* Handling class imbalance
* Data preprocessing and augmentation
* Model architecture selection
* Training and optimization
* Model evaluation
* Building an inference API
* Creating a user interface
* Containerization with Docker
* Preparing the system for deployment

The final model achieved **~95% classification accuracy** on the validation dataset.

---

## Project Architecture

The system consists of three main components:

1. **Deep Learning Model (PyTorch)**
   Responsible for feature extraction and classification.

2. **Backend API (FastAPI)**
   Serves the trained model and exposes a prediction endpoint.

3. **Frontend Interface (Streamlit)**
   Allows users to upload an image and receive predictions interactively.

Deployment is made reproducible using **Docker containers**.

---

## Dataset

The dataset contains images of different fashion categories such as clothing items and accessories. Each image is labeled with its corresponding class.

Typical dataset characteristics:

* RGB images
* Multiple fashion classes
* Class imbalance present across categories

### Dataset Preparation

Steps performed:

1. Downloaded dataset
2. Organized images by class folders
3. Checked dataset distribution
4. Identified class imbalance

To address imbalance we applied **resampling techniques and augmentation**.

---

## Handling Class Imbalance

Real-world datasets often contain **uneven class distributions**, which can bias models toward majority classes.

Techniques used:

* Class distribution analysis
* Oversampling minority classes
* Data augmentation

Tools used:

* Python
* PyTorch utilities
* Image transformations

> **Reference:**
> He, H., & Garcia, E. (2009). *Learning from Imbalanced Data*. IEEE Transactions on Knowledge and Data Engineering.
> https://doi.org/10.1109/TKDE.2008.239

---

## Data Preprocessing

Before training, images must be converted into a numerical format suitable for neural networks.

The preprocessing pipeline included:

* Image resizing to **128 × 128**
* Conversion to tensor
* Normalization of pixel values

Example normalization:

```
Mean = [0.5, 0.5, 0.5]
Std  = [0.5, 0.5, 0.5]
```

Libraries used:

* `torchvision.transforms`
* `Pillow`

> **Reference:**
> PyTorch Vision Transforms Documentation
> https://pytorch.org/vision/stable/transforms.html

---

## Data Augmentation

To improve model generalization, we applied **data augmentation** during training.

Transformations included:

* Random horizontal flips
* Random rotations
* Random cropping
* Color jitter
* Resizing

Benefits:

* Reduces overfitting
* Simulates real-world variations
* Increases effective dataset size

> **Reference:**
> Shorten, C., & Khoshgoftaar, T. (2019). *A survey on Image Data Augmentation for Deep Learning.*
> https://doi.org/10.1186/s40537-019-0197-0

---

## Model Architecture

The model is based on **ResNet18**, a deep convolutional neural network.

ResNet introduces **skip connections (residual connections)** that allow training very deep networks efficiently.

> **Original paper:**
> He, K., Zhang, X., Ren, S., Sun, J. (2016). *Deep Residual Learning for Image Recognition.*
> https://arxiv.org/abs/1512.03385

Architecture used:

* Pretrained ResNet18 backbone
* Custom final fully connected layer
* Output size = number of dataset classes

Advantages:

* Strong feature extraction
* Efficient training
* Good performance on image classification tasks

> **Reference:**
> PyTorch Model Zoo
> https://pytorch.org/vision/stable/models.html

---

## Training Process

Training was conducted using **PyTorch**.

| Parameter | Details |
|---|---|
| **Loss Function** | Cross Entropy Loss |
| **Optimizer** | Adam Optimizer |
| **Learning Rate** | Adjusted during experiments to achieve stable convergence |
| **Batch Size** | Configured according to available GPU memory |
| **Hardware** | GPU when available |

> **Reference:**
> Kingma, D., & Ba (2015). *Adam: A Method for Stochastic Optimization.*
> https://arxiv.org/abs/1412.6980

---

## Model Evaluation

Evaluation metrics included:

* Accuracy
* Validation loss
* Model generalization on unseen data

### Final Result

> ✅ **Validation Accuracy: ~95%**

This indicates that the model successfully learned discriminative visual features for fashion classification.

> **Reference:**
> Powers, D. (2011). *Evaluation: From Precision, Recall and F-measure to ROC.*
> https://doi.org/10.48550/arXiv.2010.16061

---

## Model Saving

The trained model was saved using PyTorch serialization.

Two formats were produced:

| File | Contents |
|---|---|
| `final_model_state_dict.pth` | Model weights only |
| `final_model_with_classes.pth` | Model weights + class labels |

> ⚠️ The second format is recommended for deployment because it preserves class mappings.

> **Reference:**
> PyTorch Serialization Documentation
> https://pytorch.org/docs/stable/generated/torch.save.html

---

## Backend API

A REST API was built using **FastAPI**.

FastAPI is a modern Python framework designed for high performance machine learning APIs.

**Prediction Endpoint:**

```
POST /predict
```

**Input:** Image file

**Output:** Predicted class label

**Example response:**

```json
{
  "predicted_class": "t-shirt"
}
```

> **Reference:**
> FastAPI Documentation
> https://fastapi.tiangolo.com/

---

## Frontend Interface

A user interface was created using **Streamlit**.

The interface allows users to:

* Upload an image
* View the uploaded image
* Receive a prediction from the model

Streamlit simplifies building ML demos and dashboards.

> **Reference:**
> Streamlit Documentation
> https://docs.streamlit.io/

---

## Containerization

The project was containerized using **Docker**.

Benefits of containerization:

* Reproducible environments
* Easier deployment
* Dependency isolation
* Platform independence

Docker builds an image containing:

* Python runtime
* Required dependencies
* Model files
* Application code

> **Reference:**
> Docker Documentation
> https://docs.docker.com/

---

## Deployment

The project is prepared for deployment on platforms such as:

* Hugging Face Spaces
* Cloud services
* Container orchestration systems

Using Docker ensures consistent execution across environments.

> **Reference:**
> Hugging Face Spaces Documentation
> https://huggingface.co/docs/hub/spaces

---

## Project Structure

```
project/
│
├── streamlit_app.py
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
│
├── final_model_state_dict.pth
└── final_model_with_classes.pth
```

---

## Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python |
| **Deep Learning** | PyTorch, Torchvision |
| **API** | FastAPI |
| **Frontend** | Streamlit |
| **Containerization** | Docker |
| **Utilities** | NumPy, Pillow |

---

## Future Improvements

Potential extensions for this project include:

* Top-k prediction probabilities
* Model explainability using Grad-CAM
* Larger datasets
* Hyperparameter tuning
* Cloud deployment
* CI/CD integration

---

## References

| Resource | Link |
|---|---|
| PyTorch Documentation | https://pytorch.org/docs/stable/index.html |
| FastAPI Documentation | https://fastapi.tiangolo.com/ |
| Streamlit Documentation | https://docs.streamlit.io/ |
| Docker Documentation | https://docs.docker.com/ |
| He et al., Deep Residual Learning | https://arxiv.org/abs/1512.03385 |
| Kingma & Ba, Adam Optimizer | https://arxiv.org/abs/1412.6980 |
| Shorten & Khoshgoftaar, Augmentation Survey | https://doi.org/10.1186/s40537-019-0197-0 |
| He & Garcia, Learning from Imbalanced Data | https://doi.org/10.1109/TKDE.2008.239 |
