from flask import Flask, render_template, request, redirect, session, jsonify
import requests
from flask_session import Session
from datetime import datetime
import pytz
from sql import * #Used for database connection and management
from SarvAuth import * #Used for user authentication functions
from auth import auth_blueprint
from doctors import doctors_blueprint
from dotenv import load_dotenv
import openai
import os
import random
import time
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

app = Flask(__name__)

app.config["SESSION_PERMANENT"] = True
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY')
OPENAI_KEY = os.getenv('OPENAI_KEY')

openai.api_key = OPENAI_KEY

autoRun = True #Change to True if you want to run the server automatically by running the app.py file
port = 5000 #Change to any port of your choice if you want to run the server automatically
authentication = True #Change to False if you want to disable user authentication

if authentication:
    app.register_blueprint(auth_blueprint, url_prefix='/auth')

app.register_blueprint(doctors_blueprint, url_prefix='/doctors')

#This route is the base route for the website which renders the index.html file
@app.route("/", methods=["GET", "POST"])
def index():
    if not authentication:
        return render_template("index.html")
    else:
        if not session.get("name"):
            return render_template("index.html", authentication=True)
        else:
            id = session.get("id")
            db = SQL("sqlite:///users.db")
            user = db.execute("SELECT * FROM users WHERE id = :id", id=id)
            print(user)
            if user[0]["role"] == "patient":
                return render_template("patient_dashboard.html")
            return render_template("/auth/loggedin.html")

@app.route("/termsandconditions")
def termsandconditions():
    return render_template("termsandconditions.html")

@app.route("/details")
def details():
    return render_template("implications.html")

@app.route('/autocomplete/cities')
def autocomplete_cities():
    query = request.args.get('q')
    if not query:
        return jsonify([])

    url = "https://wft-geo-db.p.rapidapi.com/v1/geo/cities"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "wft-geo-db.p.rapidapi.com"
    }
    params = {
        "namePrefix": query,
        "limit": 5
    }

    response = requests.get(url, headers=headers, params=params)
    cities = response.json().get("data", [])

    results = [f"{city['city']}, {city['country']}" for city in cities]
    return jsonify(results)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyze_symptoms_with_retry(client, messages):
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=800
    )

@app.route("/analyze-symptoms", methods=["POST"])
def analyze_symptoms():
    if not session.get("name"):
        return jsonify({"error": "Not authenticated"}), 401
        
    symptoms = request.json.get("symptoms")
    location = request.json.get("location", "")  # Get location from request
    
    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400
        
    try:
        # Check if OpenAI key is configured
        if not OPENAI_KEY:
            return jsonify({"error": "OpenAI API key not configured"}), 500
            
        client = openai.OpenAI(api_key=OPENAI_KEY)
        messages = [
            {"role": "system", "content": """You are a medical assistant. Analyze the symptoms provided and give a preliminary assessment. 
            Format your response in HTML with the following structure:
            <div class='analysis'>
                <h4>Possible Conditions:</h4>
                <ul>
                    <li>Condition 1 (probability)</li>
                    <li>Condition 2 (probability)</li>
                </ul>
                
                <h4>Recommendations:</h4>
                <ul>
                    <li>Recommendation 1</li>
                    <li>Recommendation 2</li>
                </ul>
                
                <h4>When to Seek Medical Attention:</h4>
                <ul>
                    <li>Warning sign 1</li>
                    <li>Warning sign 2</li>
                </ul>
            </div>
            
            Note: This is a preliminary assessment only. Always consult with a healthcare professional for proper diagnosis and treatment."""},
            {"role": "user", "content": f"Please analyze these symptoms and provide a preliminary assessment: {symptoms}"}
        ]
        
        response = analyze_symptoms_with_retry(client, messages)
        analysis = response.choices[0].message.content
        
        # Get relevant helplines based on location
        helplines = get_helplines(location)
        
        return jsonify({
            "analysis": analysis,
            "helplines": helplines
        })
        
    except openai.AuthenticationError:
        return jsonify({"error": "Invalid OpenAI API key"}), 500
    except openai.RateLimitError:
        return jsonify({"error": "OpenAI API rate limit exceeded. Please try again in a few minutes."}), 429
    except Exception as e:
        print(f"Error analyzing symptoms: {str(e)}")
        return jsonify({"error": "Failed to analyze symptoms. Please try again later."}), 500

def get_helplines(location):
    """Get relevant helplines based on location"""
    # Default helplines (Canada)
    helplines = {
        "emergency": "911",
        "poison_control": "1-800-268-9017",
        "mental_health": "1-833-456-4566",
        "health_info": "811"
    }
    
    # Add location-specific helplines
    if location:
        location = location.lower()
        if "ontario" in location or "toronto" in location:
            helplines.update({
                "telehealth": "1-866-797-0000",
                "covid": "1-888-999-6488",
                "mental_health": "1-866-531-2600"
            })
        elif "british columbia" in location or "vancouver" in location:
            helplines.update({
                "telehealth": "811",
                "covid": "1-888-268-4319",
                "mental_health": "1-800-784-2433"
            })
        elif "alberta" in location or "calgary" in location or "edmonton" in location:
            helplines.update({
                "telehealth": "811",
                "covid": "1-844-343-0971",
                "mental_health": "1-877-303-2642"
            })
        elif "quebec" in location or "montreal" in location:
            helplines.update({
                "telehealth": "811",
                "covid": "1-877-644-4545",
                "mental_health": "1-866-277-3553"
            })
    
    return helplines

@app.route("/dashboard")
def dashboard():
    if not session.get("name"):
        return redirect("/auth/login")
        
    id = session.get("id")
    db = SQL("sqlite:///users.db")
    user = db.execute("SELECT * FROM users WHERE id = :id", id=id)
    
    if user[0]["role"] == "patient":
        return render_template("patient_dashboard.html")
    else:
        # Get doctor's information
        db = SQL("sqlite:///doctors.db")
        doctor = db.execute("SELECT * FROM doctors WHERE userID = :id", id=id)
        
        if not doctor:
            return redirect("/doctors/update")
            
        # Get nearby patients
        db = SQL("sqlite:///users.db")
        patients = db.execute("SELECT * FROM users WHERE role = 'patient'")
        
        # Calculate age for each patient
        for patient in patients:
            dob = datetime.strptime(patient["dateOfBirth"], "%Y-%m-%d")
            today = datetime.now()
            patient["age"] = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            
            # Add dummy condition for now (in a real app, this would come from patient records)
            patient["condition"] = "General Checkup"
            
            # Add dummy coordinates for now (in a real app, this would come from geocoding)
            patient["lat"] = 43.6532 + (random.random() - 0.5) * 0.1
            patient["lng"] = -79.3832 + (random.random() - 0.5) * 0.1
        
        return render_template("doctor_dashboard.html", doctor=doctor[0], patients=patients)

if autoRun:
    if __name__ == '__main__':
        app.run(debug=True, port=port)
