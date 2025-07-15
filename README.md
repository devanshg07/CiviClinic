# CiviClinic: Wound & Disease Classification with Deep Learning

A full-stack medical image and symptom recognition application that uses deep learning for wound image classification and machine learning for disease prediction, with a modern interface for healthcare support.

## 🎯 How It Works

### Architecture
- **Backend**: Python API with PyTorch (for wound image classification) and scikit-learn (for disease prediction)
- **AI**: ResNet50-based neural network for wound classification; Random Forest for symptom-based disease prediction
- **Data**: Custom wound image dataset and symptom CSVs

### Process Flow
1. User uploads a wound image or enters symptoms
2. Image is preprocessed and passed to the neural network
3. Symptoms are encoded and passed to the Random Forest model
4. Models predict wound type or disease
5. Results are returned to the user with probabilities

## 🛠️ Tools & Technologies Used

### Backend
- **Python 3.x**
- **PyTorch** - Deep learning framework (wound classification)
- **torchvision** - Pretrained models and transforms
- **scikit-learn** - Random Forest for disease prediction
- **Flask** (if API is used)
- **Pandas, NumPy** - Data processing
- **Pillow (PIL)** - Image handling
- **Matplotlib** - Visualization (for training/analysis)
- **pickle** - Model serialization

### Data & Training
- **Custom wound image dataset** (multiple wound types)
- **Symptom CSVs** for disease prediction

## 🚀 How to Deploy

### Prerequisites
- Python 3.7+
- pip package manager

### 1. Clone/Download the Project
```bash
git clone <your-repo-url>
cd CiviClinic
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Data
- Place wound images in `dataset/dataset/train/` and `dataset/dataset/test/`
- Ensure `train_disease.csv` and `test_disease.csv` are in `dataset/dataset/`

### 4. Train the Models (First Time Only)
- **Wound Classification:**
  ```bash
  python wound_classification.py
  ```
  This will train and save a ResNet50-based model.
- **Disease Prediction:**
  Run the Jupyter notebook:
  ```bash
  jupyter notebook diseaseprediction.ipynb
  ```
  This will train and save a Random Forest model as `model.pkl`.

### 5. Run the Prediction API (if available)
```bash
python wound_prediction.py
```

## 📁 Project Structure
```
CiviClinic/
├── wound_classification.py        # Wound image classifier training
├── wound_prediction.py            # Wound image prediction API/util
├── diseaseprediction.ipynb        # Symptom-based disease prediction notebook
├── model.pkl                      # Trained Random Forest model
├── dataset/
│   ├── dataset/
│   │   ├── train/                 # Training wound images
│   │   ├── test/                  # Test wound images
│   │   ├── train_disease.csv      # Symptom-disease training data
│   │   └── test_disease.csv       # Symptom-disease test data
│   └── sympton_list.txt           # List of symptoms
└── README.md                      # This file
```

## 🎨 Features

### User Interface (if applicable)
- **Modern Design**: Clean, medical-themed UI (if frontend is present)
- **Image Upload**: For wound classification
- **Symptom Input**: For disease prediction
- **Probability Output**: Shows confidence for each class

### AI Capabilities
- **Wound Classification**: Deep ResNet50 model, multiple wound types
- **Disease Prediction**: Random Forest, multi-symptom input
- **Robust Preprocessing**: Handles various image and symptom formats

## 🔧 Customization

### Model Architecture
- Edit `wound_classification.py` to change backbone, layers, or training parameters
- Edit `diseaseprediction.ipynb` for Random Forest hyperparameters

### Data
- Add new wound types by creating new folders in `train/` and `test/`
- Update CSVs for new symptoms/diseases

## 🐛 Troubleshooting

### Common Issues
1. **Missing model files**: Train models as described above
2. **Import errors**: Ensure all dependencies are installed
3. **CUDA errors**: If no GPU, code will fall back to CPU
4. **Poor predictions**: Check data quality and retrain

### Performance Tips
- Use a GPU for faster training (PyTorch auto-detects)
- Ensure images are clear and well-labeled
- Balance classes for best results

## 📊 Model Performance
- **Wound Classifier**: ResNet50, accuracy depends on dataset (see logs)
- **Disease Predictor**: Random Forest, ~97% accuracy on test set (see notebook)

## 🤝 Contributing
Feel free to submit issues, feature requests, or pull requests to improve the project!

## 📄 License
This project is open source and available under the MIT License.
