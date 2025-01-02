# Multi-Dataset Image Classification (LeNet Streamlit)

This project provides an implementation of image classification for multiple datasets using the LeNet architecture. It supports MNIST and Cassava Leaf Disease datasets and includes a Streamlit-based user interface for convenient interaction.

## Features
- **LeNet Model**: Implements two variations of the LeNet architecture for grayscale (MNIST) and RGB (Cassava) images.
- **Dataset Support**:
  - MNIST: Classifies handwritten digits.
  - Cassava Leaf Disease: Identifies various diseases in cassava leaves and healthy leaves.
- **Streamlit UI**: A simple web application to classify uploaded or example images.

## Prerequisites
Ensure the following are installed:
- Python 3.7+
- Required Python libraries (see [Requirements](#requirements))
- Pretrained model weights:
  - MNIST: `lenet_model_mnist.pt`
  - Cassava: `lenet_model_cassava.pt`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/OxyzGiaHuy/LeNet_Streamlit.git
   cd LeNet_Streamlit
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Run the Application
Start the Streamlit application with the following command:
```bash
streamlit run app.py
```

### User Interface
1. **Dataset Selection**:
   - Choose between MNIST and Cassava Leaf Disease in the sidebar.
2. **Input Options**:
   - Upload an image file (JPG/PNG).
   - Run an example image provided in the repository.
3. **Output**:
   - Displays the image.
   - Shows the predicted label and confidence percentage.

## Model Details
### LeNet for MNIST
- Input: Grayscale images (28x28)
- Layers:
  - Conv2D (1 -> 6)
  - Average Pooling
  - Conv2D (6 -> 16)
  - Average Pooling
  - Fully Connected Layers

### LeNet for Cassava Leaf
- Input: RGB images (150x150)
- Layers:
  - Conv2D (3 -> 6)
  - Average Pooling
  - Conv2D (6 -> 16)
  - Average Pooling
  - Fully Connected Layers

## Requirements
- torch
- torchvision
- streamlit
- Pillow

Install dependencies using:
```bash
pip install torch torchvision streamlit Pillow
```

## Directory Structure
```
project/
|
|-- app.py                     # Main application script
|-- model/
|   |-- lenet_model_mnist.pt   # Pretrained MNIST model
|   |-- lenet_model_cassava.pt # Pretrained Cassava model
|-- example/
    |-- demo_8.png             # Example MNIST image
    |-- demo_cbsd.jpg          # Example Cassava image
```

## Future Enhancements
- Support for additional datasets.
- Integration with cloud services for model hosting.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/)
- [Cassava Dataset](https://www.kaggle.com/c/cassava-leaf-disease-classification)
