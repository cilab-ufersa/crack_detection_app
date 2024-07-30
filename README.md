# Crackit: Crack analysis and diagnosis on buildings elements

Crack detection is an important task in the field of civil engineering. 
Surface cracks can be a sign of structural damage and can lead to 
catastrophic failure. This app uses a deep learning model to detect cracks 
in images of concrete surfaces. You can upload an image or take a picture 
from your camera to see the model in action. 
It will segment the image to highlight the cracks and classify 
the image as containing a crack or not. You can also download a report 
of the results.

Also, you can analyze the crack by selecting two points on the image. 
The app will calculate the line equation that passes through the points and 
the angle of the line with the x-axis. This can be useful to determine the 
orientation of the crack.

## App preview

https://github.com/user-attachments/assets/4520cde5-f5dc-42ea-9b00-a568d74ed57a

## Prerequisites

What things you need to have to be able to run:

  * Python 3.11
  * Pip 3+
  * VirtualEnvWrapper is recommended but not mandatory


## Requirements 

```bash
    $ pip install -r requirements.txt
```

## Run

```bash
    $ streamlit run app/main.py
```


## About the model 

We used a pre-trained model called U-Net to segment the cracks in the images, and MobileNet to classify the images.
Moreover, we used a custom algorithm to calculate the angle of the crack, which is based on the line equation that passes through two points selected by the user.
The software also distinguishes between isolated cracks and map cracks, which can be useful for further analysis. For the distinction, we used ResNet50 model. 


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details. 

## Acknowledgments

* This project was developed as part of the project "Crack Detection and Characterization on Building Elements" at the Federal University of Semi-Árido, Brazil.

* The dataset used to train the models was provided by the University of Stuttgart, Germany. The dataset is available at: [Link](https://data.mendeley.com/datasets/5y9wdsg2zt/1)

