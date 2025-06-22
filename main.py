import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
from PIL import Image
import streamlit as st

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),   # out: 26×26
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),        # out: 13×13

            nn.Conv2d(32, 64, kernel_size=3, padding=0),  # out: 11×11
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),        # out: 5×5

            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


transformer = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if not st.session_state.get("device"):
    st.session_state.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')


@st.cache_resource(show_spinner=False)
def load_model():
    model = torch.load(
        "SimpleCNN_epoch5_98pct.pt", map_location=st.session_state.device, weights_only=False).to(device=st.session_state.device)
    model.eval()
    return model


def predict_digit(img, model):
    img_t = transformer(img).unsqueeze(0).to(
        st.session_state.device)  # Add batch dimension
    with torch.no_grad():
        output = model(img_t)
        pred = torch.argmax(output, dim=1).item()
        conf = torch.max(output).item()
    return pred, conf


def sidebar_details():
    st.sidebar.header("Dataset")
    st.sidebar.markdown(
        '[Hindi MNIST Dataset (Kaggle)](https://www.kaggle.com/datasets/imbikramsaha/hindi-mnist)'
    )

    st.sidebar.header("Model Details")
    st.sidebar.markdown("""**Architecture:** Simple CNN  
**Total parameters:** 315k (315,146)
""")
    st.sidebar.markdown("""**Layers:**  
- Conv2d(1, 32, 3) (ReLU)
- MaxPool2d(2)
- Conv2d(32, 64, 3) (ReLU)
- MaxPool2d(2)
- Flatten
- ReLU(64*6*6, 128)
- Dropout(0.5)
- Softmax(128, 10)""")

    with st.sidebar.expander("Show code"):
        st.code(
            '''class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),   # out: 26×26
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),        # out: 13×13

            nn.Conv2d(32, 64, kernel_size=3, padding=0),  # out: 11×11
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),        # out: 5×5

            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)
''',
            language="python")

    st.sidebar.subheader("Overall Metrics")
    st.sidebar.markdown("""- **Accuracy:** 98%
- **Avg Precision:** 0.98
- **Avg Recall:** 0.98
- **Avg F1 Score:** 0.98""")


def metrics_table():
    data = {
        "Class": tuple(range(10)),
        "Precision": [0.94, 0.97, 0.97, 1.00, 1.00, 0.97, 1.00, 1.00, 1.00, 1.00],
        "Recall":    [1.00, 1.00, 1.00, 1.00, 0.97, 1.00, 0.93, 0.93, 1.00, 1.00],
        "F1 Score":  [0.97, 0.98, 0.98, 1.00, 0.98, 0.98, 0.97, 0.97, 1.00, 1.00]
    }
    df = pd.DataFrame(data)
    st.dataframe(df, hide_index=True, use_container_width=True)


st.set_page_config(page_title="Hindi MNIST Digit Recognition", layout="wide")
st.title("Hindi MNIST Digit Recognition")

with st.spinner("Loading model..."):
    model = load_model()

sidebar_details()

# --- Main Layout ---
col1, col2 = st.columns([1, 1])

# Two-column structure for expanders
exp_col1, exp_col2 = st.columns([1, 1])

with exp_col1:
    with st.expander("Show Sample Hindi MNIST Dataset"):
        st.image("mnist_sample.png", use_container_width=True)

with exp_col2:
    with st.expander("Show Model Metrics & Training Graph"):
        st.subheader("Model Metrics")
        metrics_table()
        st.image("loss_acc_graph.png",
                 caption="Training Loss & Accuracy", use_container_width=True)

with col1:
    st.subheader("Upload a Digit Image")
    uploaded_file = st.file_uploader(
        "",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    st.subheader("Or Capture from Camera")
    capture = st.camera_input("", label_visibility="collapsed")

    if "img" not in st.session_state:
        st.session_state.img = None

    if uploaded_file is not None:
        st.session_state.img = Image.open(uploaded_file).convert("RGB")
    elif capture is not None:
        st.session_state.img = Image.open(capture).convert("RGB")

    img = st.session_state.img

with col2:
    col_head, col_cross = st.columns([6, 1])
    with col_head:
        st.subheader("Preview")
    with col_cross:
        if st.session_state.img is not None:
            if st.button("✖", key="clear_img_btn", help="Clear Image"):
                st.session_state.img = None
    img = st.session_state.img

    if img is not None:
        col_img, col_pred = st.columns([2, 3])
        with col_img:
            st.image(img, caption="Uploaded Image", width=180)
        with col_pred:
            pred, conf = predict_digit(img, model)
            st.markdown(
                f"<div style='font-size:2.2rem; font-weight:bold;'>Predicted as {pred}</div>"
                f"<div style='font-size:1.1rem; color:gray;'>with a confidence of {conf:.2f}</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("Upload or capture a digit image to preview and predict.")

st.markdown("""
---
<center>
Made by <a href="https://parampreetsingh.me" target="_blank" style="text-decoration:none;"><strong>Parampreet Singh</strong></a> | <a href="https://github.com/Param302/Hindi-MNIST-WebApp" target="_blank" style="text-decoration:none;"><strong>GitHub</strong></a>
</center>
""", unsafe_allow_html=True)
