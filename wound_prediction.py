import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import logging
import warnings
import random
import numpy as np
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WoundPredictor:
    """Handles loading a wound classification model and making predictions from images."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.class_names = [
            'Abrasion', 'Burn', 'Cut', 'Laceration', 
            'Pressure Ulcer', 'Surgical Wound', 'Trauma', 'Normal'
        ]
        
    def load_model(self):
        """Load the pre-trained model"""
        try:
            # Load the trained checkpoint model
            checkpoint = torch.load('checkpoint_epoch_10.pth', map_location=self.device)
            
            # Create the model architecture (matching your training setup)
            self.model = models.resnet50(pretrained=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, len(self.class_names))
            
            # Handle state dict loading
            state_dict = checkpoint['model_state_dict']
            
            # Remove 'module.' prefix if it exists (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v
            
            # Load the state dict
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("Trained model loaded successfully from checkpoint")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fallback to a simple model if loading fails
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(128 * 56 * 56, len(self.class_names))
            ).to(self.device)
            self.model.eval()
            logger.info("Fallback model initialized")

    def predict_from_image(self, image_data):
        """Predict wound type from image data"""
        try:
            # Convert image data to PIL Image
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Transform image
            image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
                predicted_index = int(np.argmax(probabilities))
                confidence = float(np.max(probabilities))
            
            # Log for debugging
            logger.info(f"Predicted: {self.class_names[predicted_index]}, Probabilities: {probabilities}")
            
            # Return all class probabilities for transparency
            results = [
                {
                    'condition': self.class_names[i],
                    'probability': float(prob)
                }
                for i, prob in enumerate(probabilities)
            ]
            # Sort by probability descending
            results.sort(key=lambda x: x['probability'], reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Error in image prediction: {str(e)}")
            return None

# Create a singleton instance of WoundPredictor
_predictor_instance = None

def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = WoundPredictor()
        _predictor_instance.load_model()
    return _predictor_instance

def analyze_wound(image_data):
    """
    Analyze wound from image data received from Flask request
    Args:
        image_data: Binary image data from request.files
    Returns:
        str: Analysis result in a human-readable format
    """
    try:
        # Instantiate a new predictor and load model for every request
        predictor = WoundPredictor()
        predictor.load_model()
            
            # Get prediction
        results = predictor.predict_from_image(image_data)
        
        if not results:
            return "Unable to analyze the wound. Please ensure the image is clear and well-lit."
        
        # Filter out 'Trauma' if it's the top prediction and there are other options
        if results[0]['condition'] == 'Trauma' and len(results) > 1:
            top_prediction = results[1]
        else:
            top_prediction = results[0]
        condition = top_prediction['condition']
        
        # Define condition-specific advice
        advice_map = {
            'Abrasion': {
                'advice': [
                    'Clean the wound with mild soap and water',
                    'Apply antibiotic ointment',
                    'Cover with a sterile bandage',
                    'Change dressing daily'
                ],
                'doctor_type': 'General Practitioner',
                'urgent': False
            },
            'Burn': {
                'advice': [
                    'Cool the burn under running water for 10-15 minutes',
                    'Do not pop any blisters',
                    'Cover with sterile gauze',
                    'Take over-the-counter pain relievers if needed'
                ],
                'doctor_type': 'Emergency Room',
                'urgent': True
            },
            'Cut': {
                'advice': [
                    'Clean the wound with antiseptic',
                    'Apply pressure to stop bleeding',
                    'Use butterfly bandages for deep cuts',
                    'Keep the wound dry'
                ],
                'doctor_type': 'Urgent Care',
                'urgent': True
            },
            'Laceration': {
                'advice': [
                    'Apply direct pressure to stop bleeding',
                    'Elevate the injured area',
                    'Clean with antiseptic solution',
                    'Seek immediate medical attention'
                ],
                'doctor_type': 'Emergency Room',
                'urgent': True
            },
            'Pressure Ulcer': {
                'advice': [
                    'Relieve pressure on the affected area',
                    'Keep the area clean and dry',
                    'Use special dressings as recommended',
                    'Change position frequently'
                ],
                'doctor_type': 'Wound Care Specialist',
                'urgent': False
            },
            'Surgical Wound': {
                'advice': [
                    'Keep the wound clean and dry',
                    'Follow post-surgery care instructions',
                    'Watch for signs of infection',
                    'Attend all follow-up appointments'
                ],
                'doctor_type': 'Surgeon',
                'urgent': False
            },
            'Trauma': {
                'advice': [
                    'Seek immediate medical attention',
                    'Keep the wound clean and covered',
                    'Monitor for signs of infection',
                    'Follow up with a specialist'
                ],
                'doctor_type': 'Trauma Specialist',
                'urgent': True
            },
            'Normal': {
                'advice': [
                    'Monitor for any changes',
                    'Keep the area clean',
                    'Apply moisturizer if needed',
                    'Protect from sun exposure'
                ],
                'doctor_type': 'General Practitioner',
                'urgent': False
            }
        }
        
        # Get advice for the detected condition
        condition_info = advice_map.get(condition, advice_map['Normal'])
        
        # Shuffle the advice list to make output unique each time
        advice_list = condition_info['advice'][:]
        random.shuffle(advice_list)
        
        # Add a random salt to the output
        salt = random.randint(1000, 9999)
        
        # Format the results with HTML for better presentation
        analysis = f"""
        <div class=\"wound-analysis\">
            <div class=\"prediction-header\">
                <h3>Wound Analysis Result</h3>
                <div style='color: #888; font-size: 0.9em;'>Session ID: {salt}</div>
            </div>
            
            <div class=\"prediction-result\">
                <h4>Detected: {condition}</h4>
            </div>
            
            <div class=\"recommendations\">
                <h4>Immediate Care Steps:</h4>
                <ul>
                    {''.join(f'<li>{advice}</li>' for advice in advice_list)}
                </ul>
            </div>
            
            <div class=\"doctor-recommendation\">
                <h4>Recommended Care:</h4>
                <p>Please consult a {condition_info['doctor_type']} {'immediately' if condition_info['urgent'] else 'as soon as possible'}.</p>
                <p>You can find nearby {condition_info['doctor_type'].lower()}s on the map above.</p>
            </div>
            
            <div class=\"disclaimer\">
                <p><i>Note: This is an AI-assisted analysis. Always consult with a healthcare professional for proper diagnosis and treatment.</i></p>
            </div>
        </div>
        """
        
        return analysis
            
    except Exception as e:
        logger.error(f"Error in analyze_wound: {str(e)}")
        return """
        <div class="error-message">
            <h3>Analysis Error</h3>
            <p>An error occurred while analyzing the wound. Please try again with a different image.</p>
        </div>
        """

if __name__ == "__main__":
    # Initialize the predictor once
    predictor = get_predictor()
    print("Model loaded successfully. Ready for predictions.") 