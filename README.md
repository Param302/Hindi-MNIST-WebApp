# Hindi MNIST WebApp

A web application for **Hindi** handwritten digit recognition using a custom **Convolutional Neural Network** (CNN).

---

## Dataset

- **Source:** [Hindi MNIST Dataset (Kaggle)](https://www.kaggle.com/datasets/imbikramsaha/hindi-mnist/data)
- **Description:** Contains grayscale images (32x32) of handwritten Hindi digits, split into train, validation, and test sets.

---

## Model Architecture

The model is a simple CNN with the following architecture:

- Conv2d(1, 32, kernel_size=3) + ReLU
- MaxPool2d(kernel_size=2)
- Conv2d(32, 64, kernel_size=3) + ReLU
- MaxPool2d(kernel_size=2)
- Flatten
- Linear(64*6*6, 128) + ReLU
- Dropout(0.5)
- Linear(128, 10)
- Softmax(dim=1)

**Total parameters:** 315K (315,146)
**Accuracy:** ~98% on the test set

---

## Modeling Notebook

All data processing, training, and evaluation are performed in [`Hindi_MNIST_CNN.ipynb`](Hindi_MNIST_CNN.ipynb).

---

## Steps to Run

### 1. Clone the repository

```sh
git clone https://github.com/Param302/Hindi-MNIST-WebApp
cd Hindi-MNIST-WebApp
```

### 2. Create a virtual environment

**Python version required:** 3.10

#### Windows

```sh
python -m venv venv
venv\Scripts\activate
```

#### Linux/Mac

```sh
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```sh
pip install -r requirements.txt
```

### 4. Run the app

```sh
streamlit run main.py
```


---

## Requirements

See [`requirements.txt`](requirements.txt) for all dependencies.

---

## Contact
For any questions or contributions, feel free to reach out:
[**Parampreet Singh**](https://parampreetsingh.me)  
Email: [connectwithparam.30@gmail.com](mailto:connectwithparam.30@gmail.com)
