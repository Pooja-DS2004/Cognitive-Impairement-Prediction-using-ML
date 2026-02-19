_Cognitive Impairment Prediction System_
An end-to-end Machine Learning solution designed for the early detection and classification of cognitive impairment (including Alzheimer's and Dementia) using neuroimaging data.

_Key Features_
Deep Learning Inference: Leverages a fine-tuned MobileNetV2 model for high-accuracy MRI scan classification.
Lightweight Architecture: Optimized for efficiency, making it suitable for deployment on standard hardware.
Web-Based Interface: A user-friendly Flask dashboard for uploading scans and viewing real-time diagnostic reports.
Clinical Insight: Provides classification categories (e.g., Non-Demented, Mild, Moderate) to assist caregivers and medical professionals.

_Technical Stack_
Component -- Technology
Model Architecture -- MobileNetV2 (Transfer Learning)
Backend Framework -- Flask (Python)
Deep Learning -- TensorFlow / Keras
Image Processing -- OpenCV / PIL
Frontend -- "HTML5, CSS3, JavaScript, Flask"

_Model Architecture_
We utilize MobileNetV2 because of its inverted residual structure and linear bottlenecks, which allow the model to remain lightweight without sacrificing the depth necessary for medical image feature extraction.

_Installation & Setup_
1. Prerequisites
Ensure you have Python 3.8+ installed.
2. Clone the Repository
3. git clone https://github.com/Pooja-DS2004/cognitive-prediction-ml.git
cd cognitive-prediction-ml
3. Install Dependencies
   pip install -r requirements.txt
   Tensorflow, Keras, andother ML libraries.
4. Run the Application
   python app.py

_How It Works_
Upload: The user uploads a brain MRI slice (JPG/PNG) via the Flask interface.
Preprocessing: The image is resized to 224*224 and normalized to the range [0,1].
Prediction: The Flask backend passes the image to the loaded .h5 model.
Result: The UI displays the classification label and a confidence score.

_Output_
<img width="1917" height="871" alt="image" src="https://github.com/user-attachments/assets/6040fd3d-a38c-45c7-a403-c62dc67cd3d3" />
<img width="1897" height="862" alt="image" src="https://github.com/user-attachments/assets/f075a374-c3ac-4992-8a53-b4b939619d4a" />
<img width="1901" height="862" alt="image" src="https://github.com/user-attachments/assets/a6e5fe71-9d81-4f43-a4e5-def855b514bd" />
<img width="1900" height="860" alt="image" src="https://github.com/user-attachments/assets/5feb7458-d3e8-41cc-9448-17e36154161c" />
<img width="1896" height="860" alt="image" src="https://github.com/user-attachments/assets/8c21fa13-974a-4549-9b5b-09078937b3af" />
<img width="1897" height="867" alt="image" src="https://github.com/user-attachments/assets/383f5694-6b30-42d7-8c8b-6893aa6acfad" />
<img width="1888" height="857" alt="image" src="https://github.com/user-attachments/assets/b0515edc-5327-44bc-8ff3-0be7db90ed55" />






