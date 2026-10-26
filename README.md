# OncoVision: AI-Powered Cancer Cell Detection Web App

OncoVision is a web application that detects cancer cells in medical images using deep learning. A user uploads an image, and the app shows which cells look cancerous and which look normal.

> **Note:** This is an educational project. It is not a medical tool and must not be used for real diagnosis.

## Features

- Upload a medical image through a web interface
- Image preprocessing with OpenCV
- Cell segmentation with a U-Net model
- Classification of cells as cancerous or normal
- Results shown directly in the web interface

## How it works

1. **Upload:** the user uploads an image in the web app.
2. **Preprocess:** the image is cleaned and prepared with OpenCV.
3. **Segment:** a U-Net model separates the cells from the background.
4. **Classify:** the trained model labels the cells as cancerous or normal.
5. **Display:** the result is shown to the user in the browser.

## Tech stack

- Python
- U-Net (deep learning, image segmentation)
- OpenCV (image preprocessing)
- Web interface: Streamlit

## Dataset

**Dataset used from the warwick University a real lizard dataset**

## Installation and usage

```bash
# 1. Clone the repository
git clone https://github.com/syedtalhaimran1/OncoVision.git
cd OncoVision

# 2. Install the requirements
pip install -r requirements.txt

# 3. Run the app
python run app.py
```
or streamlit run app.py 

Then open the address shown in the terminal in your browser.

If you do not have a `requirements.txt` file yet, create one with the libraries your project uses.

## Author

**Syed Talha Imran**
Software Engineering graduate (2026), Lahore Garrison University
Email: syedtalhaimran1@gmail.com
LinkedIn: [linkedin.com/in/syed-talhaimran-10510724a](https://linkedin.com/in/syed-talhaimran-10510724a)
GitHub: [github.com/syedtalhaimran1](https://github.com/syedtalhaimran1)
