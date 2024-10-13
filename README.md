
# Quantifying the Tumour Vasculature Environment from CD-31 IHC Images Using Deep Learning Semantic Segmentation

This repository contains the code accompanying the research paper **"Quantifying the tumour vasculature environment from CD-31 immunohistochemistry images of breast cancer using deep learning-based semantic segmentation"**. The repository provides the code for the deep learning models, image processing, and quantification algorithms discussed in the paper.

## Overview

The main goal of this project is to automatically quantify the tumour vasculature environment from CD-31 immunohistochemistry (IHC) images of breast cancer using semantic segmentation based on a U-Net architecture. The code extracts vascular parameters like density, area, circularity, and thickness from segmented images.

## Features

- **Deep Learning Models**: U-Net-based models trained to segment CD-31 IHC images.
- **Custom Quantification Algorithms**: Extract various vascular parameters such as density and circularity.
- **Preprocessing and Postprocessing**: Code to prepare IHC images and process segmentation outputs.
- **Statistical Analysis**: Scripts for performing paired and unpaired statistical tests to evaluate the significance of measurements.

## Requirements

- **Python 3.7**
- **TensorFlow 2.4.4**
- **Keras 2.4.0**
- **Keras Preprocessing 1.1.2**
- **NumPy 1.19.5**
- **Pandas 1.3.4**
- **Scikit-Image 0.18.3**
- **Matplotlib 3.5.0**
- **Seaborn 0.11.2**
- **OpenCV 4.5.4.60**
- **SciPy 1.7.1**
- **PyQt5 5.15.6**
- **Requests 2.26.0**
- **tqdm 4.62.3**

Detailed library versions and dependencies are available in the `requirements.txt` file.

## Data

Due to privacy concerns, the patient data used in the research is not publicly available. However, if you are interested in using the models or data, you can request access from the corresponding author.

## License

This work is licensed under a **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License**. 

**You are free to:**
- **Share** — copy and redistribute the material in any medium or format.
- **Adapt** — remix, transform, and build upon the material.

**Under the following terms:**
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial** — You may not use the material for commercial purposes.

Read the full license [here](https://creativecommons.org/licenses/by-nc/4.0/).

## Citation

If you use this code in your research, please cite the paper:

```
@article{whitmarsh2024,
  title={Quantifying the tumour vasculature environment from CD-31 immunohistochemistry images of breast cancer using deep learning-based semantic segmentation},
  author={Whitmarsh, T. and Cope, W. and Carmona-Bozo, J. and Manavaki, R. and Sammut, S.-J. and Woitek, R. and Provenzano, E. and Brown, E. L. and Bohndiek, S. E. and Gallagher, F. A. and Caldas, C. and Gilbert, F. J. and Markowetz, F.},
  journal={N/A, under review},
  year={2024},
  publisher={N/A}
}
```

## Contact

For any questions or requests, feel free to contact the corresponding author.
