import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

# Mapping for Cassava dataset
idx2label = {
    0: 'cbb',
    1: 'cbsd',
    2: 'cgm',
    3: 'cmd',
    4: 'healthy',
}

# LeNet model for MNIST


class LeNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                               kernel_size=5, padding='same')
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.avgpool1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.avgpool2(outputs)
        outputs = F.relu(outputs)
        outputs = self.flatten(outputs)
        outputs = self.fc_1(outputs)
        outputs = self.fc_2(outputs)
        outputs = self.fc_3(outputs)
        return outputs


# LeNet model for Cassava Leaf
class LeNetClassifier2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,
                               kernel_size=5, padding='same')
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(16 * 35 * 35, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.avgpool1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.avgpool2(outputs)
        outputs = F.relu(outputs)
        outputs = self.flatten(outputs)
        outputs = self.fc_1(outputs)
        outputs = self.fc_2(outputs)
        outputs = self.fc_3(outputs)
        return outputs


@st.cache_resource
def load_model(model_path, num_classes, dataset):
    if dataset == "MNIST":
        model = LeNetClassifier(num_classes)
    else:
        model = LeNetClassifier2(num_classes)
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


# Inference function
def inference(image, model, dataset):
    if dataset == "MNIST":
        w, h = image.size
        if w != h:
            crop = transforms.CenterCrop(min(w, h))
            image = crop(image)
            wnew, hnew = image.size
        img_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
        img_new = img_transform(image)
        img_new = img_new.expand(1, 1, 28, 28)
    else:  # Cassava
        img_transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
        ])
        img_new = img_transform(image)
        img_new = torch.unsqueeze(img_new, 0)

    with torch.no_grad():
        predictions = model(img_new)
    preds = nn.Softmax(dim=1)(predictions)
    p_max, yhat = torch.max(preds.data, 1)
    return p_max.item() * 100, yhat.item()


def main():
    st.title("Multi-Dataset Image Classification")
    st.sidebar.title("Configuration")
    dataset = st.sidebar.selectbox(
        "Select Dataset", ("MNIST", "Cassava Leaf Disease"))

    # Load model based on dataset
    model_path = './model/lenet_model_mnist.pt' if dataset == "MNIST" else './model/lenet_model_cassava.pt'
    num_classes = 10 if dataset == "MNIST" else 5
    model = load_model(model_path, num_classes, dataset)

    st.subheader(f"Model: LeNet. Dataset: {dataset}")
    option = st.selectbox('How would you like to give the input?',
                          ('Upload Image File', 'Run Example Image'))

    if option == "Upload Image File":
        file = st.file_uploader("Please upload an image", type=["jpg", "png"])
        if file is not None:
            image = Image.open(file)
            p, idx = inference(image, model, dataset)
            if dataset == "MNIST":
                label = idx
            else:
                label = idx2label[idx]
            st.image(image)
            st.success(
                f"The uploaded image is classified as {label} with {p: .2f} % probability.")
    elif option == "Run Example Image":
        example_image = './example/demo_8.png' if dataset == "MNIST" else './example/demo_cbsd.jpg'
        image = Image.open(example_image)
        p, idx = inference(image, model, dataset)
        if dataset == "MNIST":
            label = idx
        else:
            label = idx2label[idx]
        st.image(image)
        st.success(
            f"The example image is classified as {label} with {p: .2f} % probability.")


if __name__ == '__main__':
    main()
