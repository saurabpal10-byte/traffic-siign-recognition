
🚦 Traffic Sign Recognition using Convolutional Neural Networks (CNNs)

This project implements an advanced Traffic Sign Recognition (TSR) system using **Convolutional Neural Networks** and deep learning techniques. It classifies traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset, with high accuracy and performance, making it suitable for integration into autonomous vehicle systems and driver assistance technologies.



 📌 Project Overview

 **Objective:** To build a robust and accurate traffic sign classifier using CNNs.
 **Dataset Used:** [GTSRB - German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_news.html)
 **Tools & Frameworks:** Python, TensorFlow/Keras, OpenCV, NumPy, Matplotlib
 **Accuracy Achieved:** Over 98% on validation set



📂 Project Structure

```
├── data/                 # Contains the dataset (GTSRB)
├── models/               # Saved model weights and architecture
├── src/
│   ├── preprocessing.py  # Preprocessing and augmentation scripts
│   ├── train.py          # Model training script
│   ├── evaluate.py       # Evaluation and result generation
│   └── utils.py          # Helper functions
├── notebooks/            # Jupyter notebooks for testing and visualization
├── README.md             # Project documentation
└── requirements.txt      # Required Python packages
```

---

🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/traffic-sign-recognition-cnn.git
   cd traffic-sign-recognition-cnn
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and preprocess the dataset**
   - Download GTSRB from the [official site](https://benchmark.ini.rub.de/gtsrb_news.html)
   - Extract it into the `data/` directory

4. **Train the model**
   ```bash
   python src/train.py
   ```

5. **Evaluate the model**
   ```bash
   python src/evaluate.py
   ```



 📊 Results

- Training Accuracy: 98.6%
- Validation Accuracy: 99.25%



🧠 Model Architecture

- 3 Convolutional Layers (ReLU)
- Max Pooling + Dropout Layers
- Fully Connected Dense Layer
- Softmax Output (43 classes)



📌 Future Enhancements

- Real-time traffic sign detection from video feed using OpenCV
- Model optimization for mobile and embedded deployment (TensorRT, TFLite)
- Support for multilingual/multinational sign datasets



📚 References

- **GTSRB Dataset**: [https://benchmark.ini.rub.de/gtsrb_news.html](https://benchmark.ini.rub.de/gtsrb_news.html)
- **Stallkamp et al., JMLR 2012**: [PDF](https://www.jmlr.org/papers/volume13/stallkamp12a/stallkamp12a.pdf)
- **Sermanet & LeCun, IJCNN 2011**: [PDF](https://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)



✨ Acknowledgements

Thanks to the creators of the GTSRB dataset and the open-source deep learning community.



📬 Contact

If you have any questions or suggestions, feel free to open an issue or reach out!
