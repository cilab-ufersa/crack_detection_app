import numpy as np
import base64
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from fpdf import FPDF 
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet import preprocess_input
from keras.models import load_model, Model
import streamlit.components.v1 as components
from utils.loss_metrics import (
    Weighted_Cross_Entropy,
    Precision_dil,
    F1_score,
    F1_score_dil,
)

loss = Weighted_Cross_Entropy(10)
precision_dil = Precision_dil
f1_score = F1_score
f1_score_dil = F1_score_dil

model = load_model(
    "models/unet_mobilenet.h5",
    custom_objects={
        "loss": loss,
        "Precision_dil": precision_dil,
        "F1_score": f1_score,
        "F1_score_dil": f1_score_dil,
    },
)


model_characterization = load_model("models/resnet50_characterization.h5")

def save_pdf(image, overlay, binary, interpolation, negative, positive, user_description, characterization_class):
    """ Saves the input image, overlay image, binary image and the classification probabilities in a PDF file

    Args:
        image (Path): input image
        overlay (Path): overlay image of the input image with the segmented crack
        binary (Path): binary image of the segmented crack
        interpolation (Path): interpolation image of the input image
        negative (float): probability of the image not containing a crack
        positive (float): probability of the image containing a crack
        user_description (str): user description of the image
        characterization_class (str): classification of the crack
        
    Returns:
        pdf (str): PDF file with the input image, overlay image, binary image and the classification probabilities
    """
    pdf = FPDF()
    pdf.add_page()

    pdf.set_title("Surface Crack Detection Report")
    pdf.set_font("Times", size=14)
    pdf.cell(200, 10, txt="Surface Crack Detection Report", ln=True, align="C")

    pdf.set_font("Times", size=12)
    pdf.cell(200, 10, txt="1. Result Images", ln=True, align="L")

    # Overlay image (top-left)
    pdf.image(overlay, x=10, y=30, w=90)
    pdf.set_y(30 + 90 + 5)  # Positioning below the image
    pdf.set_x(10)
    pdf.cell(90, 10, txt="Overlay Image", ln=False, align="C")

    # Binary image (top-right)
    pdf.image(binary, x=110, y=30, w=90)
    pdf.set_y(30 + 90 + 5)  # Positioning below the image
    pdf.set_x(110)
    pdf.cell(90, 10, txt="Binary Image", ln=True, align="C")

    pdf.set_y(30 + 90 + 20)  # Adjusting y-position for the next row of images

    if positive > negative:
        pdf.set_fill_color(230, 83, 83)
        crack = "Containing Crack."
    else:
        pdf.set_fill_color(83, 230, 83)
        crack = "Not Containing Crack."

    # Interpolation image (bottom-left)
    pdf.image(interpolation, x=10, y=pdf.get_y() + 5, w=90)
    pdf.set_y(pdf.get_y() + 90 + 10)  # Positioning below the image
    pdf.set_x(10)
    pdf.cell(90, 10, txt="Interpolation Image", ln=False, align="C")

    # Final image (bottom-right)
    pdf.image(image, x=110, y=pdf.get_y() - 90, w=90)
    pdf.set_y(pdf.get_y() + 10)
    pdf.set_x(110)
    pdf.cell(90, 10, txt="Input Image", ln=True, align="C")

    pdf.set_y(pdf.get_y() + 15)  # Adjusting y-position for the classification results
    pdf.set_font("Times", size=12)
    pdf.cell(200, 10, txt="2. Classification Result", ln=True, align="L")
    pdf.cell(200, 10, txt=f"Probability of Containing Crack: {round(positive, 2)}%", ln=True, align="L", fill=False)
    pdf.cell(200, 10, txt=f"Probability of Not Containing Crack: {round(negative, 2)}%", ln=True, align="L", fill=False)
    pdf.cell(52, 10, txt=f"Result: {crack}", ln=True, align="L", fill=True)

    if positive > negative:
        pdf.cell(200, 10, txt=f"Crack Classification: {characterization_class}", ln=True, align="L")
    pdf.ln(13)
    
    pdf.set_font("Times", size=12)
    pdf.cell(200, 10, txt="3. User description", ln=True, align="L")
    pdf.cell(200, 10, txt=f"Description: {user_description}", ln=True, align="L")
    
    pdf.ln(20)

    return pdf.output(dest="S").encode("latin-1")
    
    
def segmentation(path):
    """
    Generates an overlay image of the input image with the segmented crack

    Args:
        path (str): receives the path of the images for segmentation

    Returns:
        overlay (PIL.Image): overlay image of the input image with the segmented crack
        binary (PIL.Image): binary image of the segmented crack
    """

    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    segmented_output = Model(
        inputs=model.input, outputs=model.get_layer(name="sigmoid").output
    )

    y_pred = segmented_output.predict(np.expand_dims(img, axis=0))
    
    binary_mask = np.where(y_pred > 0.5, 1, 0).astype(np.uint8)
    binary_mask = np.squeeze(binary_mask)
    
    color_mask = np.zeros((224, 224, 3), dtype=np.uint8)
    color_mask[binary_mask == 1] = (0, 0, 255)  
   
    img = img.astype('float32')
    color_mask = color_mask.astype('float32')
    overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    overlay = cv2.resize(overlay, (224, 224))
    
    overlay = Image.fromarray((overlay * 255).astype(np.uint8))
    
    binary_array = (binary_mask * 255).astype(np.uint8)
    binary = Image.fromarray(binary_array)
    
    return overlay, binary


def classification(path):
    """Classifies the input image as containing a crack or not

    Args:
        Args:
        path (str): receives the path of the images for classification

    Returns:
        negative (float): probability of the image not containing a crack
        positive (float): probability of the image containing a crack
    """

    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    y_pred = model.predict(img)

    negative = y_pred[0][0] * 100
    positive = y_pred[0][1] * 100

    return (negative, positive)

def download_images(image, overlay, binary):
    """Saves the input image, overlay image, binary image and the classification probabilities in a zip file
    
    Args:
        image (Path): input image
        overlay (Path): overlay image of the input image with the segmented crack
        binary (Path): binary image of the segmented crack
    
    Returns:
        zip_path (str): path of the zip file
    """
    
    import zipfile
    zip_path = os.path.join("app/temp", "images.zip")
    images = [image, overlay, binary]
    
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for img in images:
            # Get the base name of the image file
            img_name = os.path.basename(img)
            # Write the image file to the zip file with the base name
            zipf.write(img, arcname=img_name)

    return zip_path

def white_pixels(binary_img):
    white_pixels = np.where(binary_img == 255)
    x_coords = white_pixels[1]
    y_coords = white_pixels[0]

    return x_coords, y_coords

def fit_line(ax, binary_img, color):
    x_coords, y_coords = white_pixels(binary_img)

    coeffs = np.polyfit(x_coords, y_coords, 1)
    poly_func = np.poly1d(coeffs)
    arctan = np.arctan(coeffs[0])
    degree = np.degrees(arctan)

    x_values = np.linspace(min(x_coords)-10, max(x_coords)+50, len(x_coords))

    ax.scatter(x_coords, y_coords, c='k', marker='o')
    ax.plot(x_values, poly_func(x_values), c=color, label=f"Angle: {(degree * (- 1)):.2f}")

    ax.legend()

    ax.set_xlim(0, max(x_coords)+50)
    ax.set_ylim(min(y_coords),max(y_coords)+50)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)

def count_white_segments(binary_img, filename):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    fig, ax = plt.subplots(figsize=(5,5))
    color = ['r', 'g', 'c', 'b', 'm', 'y', 'orange', 'brown', 'pink']

    for i, contour in enumerate(contours):
        blank_image = np.zeros_like(binary_img)
        segment = cv2.drawContours(blank_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        fit_line(ax, segment, color[i % len(color)])

    ax.set_xlim(0, binary_img.shape[1])
    ax.set_ylim(200, 0)
    ax.set_xlabel('Pixels')
    ax.set_ylabel('Pixels')

    plt.savefig(f"app/temp/{filename}", format='png')

def save_interpolation(img_path, filename):
    image = Image.open(img_path)

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Binary Image')

    count_white_segments(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), filename=filename)

def save_points(binary_img, filename):
    x_coords, y_coords = white_pixels(binary_img)

    with open(filename, 'w') as f:
        f.write('x,y\n')
        for x, y in zip(x_coords, y_coords):
            f.write(f'{x},{y}\n')

    print(f"Points saved to {filename}")

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

def show_image_result(path):
    image = image_to_base64(path)

    html_string = f"""
                <style>
                .container {{
                  position: relative;
                  width: 400px;
                  height: 400px;
                  overflow: hidden;
                }}

                .original {{
                  width: 100%;
                  height: 100%;
                  transition: transform 0.2s ease;
                }}

                .container:hover .original {{
                  transform: scale(2);  /* Zoom */
                }}
                </style>
                <div class='container'>
                    <img id='zoom-image' class='original' src="data:image/png;base64,{image}" />
                </div>

                <script>
                const container = document.querySelector('.container');
                const img = document.getElementById('zoom-image');

                container.addEventListener('mousemove', function(e) {{
                  const rect = container.getBoundingClientRect();
                  const x = ((e.clientX - rect.left) / rect.width) * 100;
                  const y = ((e.clientY - rect.top) / rect.height) * 100;

                  img.style.transformOrigin = `${{x}}% ${{y}}%`;
                }});
                </script>
                """

    components.html(html_string, width=400, height=400)

def characterization(img_path):
    img = load_img(img_path, target_size=(227, 227))
    
    img_array = img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    
    prediction = model_characterization.predict(img_preprocessed)
    predicted_class = np.argmax(prediction)
    
    classes = {0: "Disseminated",
               1: "Isolated"}
    
    prediction_str = classes[predicted_class]
    
    return prediction_str

def sliding_windows_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        files = os.listdir(folder_path)
        for file in files:
            os.remove(f'{folder_path}/{file}')

def janelamento(img_path):
    image = Image.open(img_path)
    image = np.array(image)

    height, width, _ = image.shape
    window_size = (224, 224)
    
    num_rows = height // window_size[0]
    num_cols = width // window_size[1]
    
    
    original_path = 'app/temp/windows/original/'
    segmented_mask_path = 'app/temp/windows/segmented_mask/'
    segmented_binary_path = 'app/temp/windows/segmented_binary/'
    
    sliding_windows_folder(original_path)
    sliding_windows_folder(segmented_mask_path)
    sliding_windows_folder(segmented_binary_path)
    
    for contador in range(height // window_size[0]):
        for contador2 in range(width // window_size[1]):
            x = contador * window_size[0]
            y = contador2 * window_size[1]
            window = image[x:x + window_size[0], y:y + window_size[1]]
            
            window = cv2.cvtColor(window, cv2.COLOR_RGB2BGR)
            
            label = f"window_{contador2}_{contador}.jpg"
            window_path = f"{original_path}{label}"
            cv2.imwrite(window_path, window)
            mask, binary = segmentation(window_path)
        
            mask_path = os.path.join(segmented_mask_path, f"{label}")
            binary_path = os.path.join(segmented_binary_path, f"{label}")

            mask.save(mask_path)
            binary.save(binary_path)
                

    print("Janelamento concluído com sucesso!")

    
def concatenacao(img_path, mask = None, binary = None):
    
    image = Image.open(img_path)
    image = np.array(image)

    height, width, _ = image.shape
    window_size = (224, 224)
    
    num_rows = height // window_size[0]
    num_cols = width // window_size[1]
    
    
    if mask == True:
        final_image = np.zeros((num_rows * window_size[0], num_cols * window_size[1], 3), dtype=np.uint8)
    elif binary == True:
        final_image = np.zeros((num_rows * window_size[0], num_cols * window_size[1], 3), dtype=np.uint8)
 
    for row in range(num_rows):
        for col in range(num_cols):
            x = row * window_size[0]
            y = col * window_size[1]
            if mask == True:
                folder = "segmented_mask"
            elif binary == True:
                folder = "segmented_binary"
            
            image = Image.open(f'app/temp/windows/{folder}/window_{col}_{row}.jpg')
            window = np.array(image)
            window = cv2.cvtColor(window, cv2.COLOR_RGB2BGR)
            
            final_image[x:x + window_size[0], y:y + window_size[1]] = window
   
    if mask == True:
        final_image_name = "concatenated_mask_image.jpg"
    elif binary == True:
        final_image_name = "concatenated_binary_image.jpg"
    
    path = f'app/temp/windows/{final_image_name}'
        
    cv2.imwrite(path, final_image)
    
    return path
        
