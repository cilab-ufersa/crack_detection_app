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
