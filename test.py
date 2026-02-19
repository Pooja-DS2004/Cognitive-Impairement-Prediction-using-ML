

import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load trained model
model = load_model("best_model.h5")
img_size = 224

# Function to predict image
def predict_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    
    if not file_path:
        return
    
    # Show selected image
    img = Image.open(file_path)
    img_resized = img.resize((200, 200))  
    img_tk = ImageTk.PhotoImage(img_resized)
    panel.config(image=img_tk)
    panel.image = img_tk

    # Preprocess for model
    img_for_model = load_img(file_path, target_size=(img_size, img_size))
    img_array = img_to_array(img_for_model) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        result_label.config(text=f"Prediction: Affected (Dementia)", fg="red")
    else:
        result_label.config(text=f"Prediction: Normal", fg="green")


# ============================
# Tkinter GUI
# ============================
root = tk.Tk()
root.title("Dementia Detection")
root.geometry("400x400")

# Button
btn = Button(root, text="Select Image", command=predict_image, font=("Arial", 12), bg="lightblue")
btn.pack(pady=10)

# Image panel
panel = Label(root)
panel.pack()

# Prediction result
result_label = Label(root, text="No image selected", font=("Arial", 12))
result_label.pack(pady=10)

# Run GUI
root.mainloop()
