from flask import Flask, render_template, request, redirect, session, jsonify, Blueprint
from flask_session import Session
from datetime import datetime
import pytz
from sql import * #Used for database connection and management
from SarvAuth import * #Used for user authentication functions
import requests
import os
from dotenv import load_dotenv

load_dotenv()

doctors_blueprint = Blueprint('doctors', __name__)

@doctors_blueprint.route("/", methods=["GET"])
def doctors_list():
    if not session.get("name"):
        return redirect("/auth/login")
        
    # Get search parameters
    location = request.args.get("location", "")
    specialty = request.args.get("specialty", "")
    
    try:
        db = SQL("sqlite:///doctors.db")
        
        # First check if the doctors table exists
        tables = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='doctors'")
        if not tables:
            return render_template("doctors.html", doctors=[], specialties=[], 
                                current_location=location, current_specialty=specialty,
                                error="No doctors found in the database.")
        
        # Base query - modified to not depend on users table
        query = "SELECT * FROM doctors WHERE 1=1"
        params = {}
        
        if location:
            query += " AND city LIKE :location"
            params["location"] = f"%{location}%"
        
        if specialty:
            query += " AND specialty LIKE :specialty"
            params["specialty"] = f"%{specialty}%"
        
        doctors = db.execute(query, **params)
        
        # Get unique specialties for filter
        specialties = db.execute("SELECT DISTINCT specialty FROM doctors")
        specialties = [s["specialty"] for s in specialties]
        
        return render_template("doctors.html", doctors=doctors, specialties=specialties, 
                             current_location=location, current_specialty=specialty)
                             
    except Exception as e:
        print(f"Database error: {str(e)}")
        return render_template("doctors.html", doctors=[], specialties=[], 
                             current_location=location, current_specialty=specialty,
                             error="An error occurred while fetching doctors.")

@doctors_blueprint.route("/search")
def search_doctors():
    location = request.args.get("location", "")
    specialty = request.args.get("specialty", "")
    
    try:
        db = SQL("sqlite:///doctors.db")
        
        query = "SELECT * FROM doctors WHERE 1=1"
        params = {}
        
        if location:
            query += " AND city LIKE :location"
            params["location"] = f"%{location}%"
        
        if specialty:
            query += " AND specialty LIKE :specialty"
            params["specialty"] = f"%{specialty}%"
        
        doctors = db.execute(query, **params)
        return jsonify(doctors)
        
    except Exception as e:
        print(f"Database error: {str(e)}")
        return jsonify({"error": "Failed to fetch doctors"}), 500

@doctors_blueprint.route("/update", methods=["GET", "POST"])
def update():
    if not session.get("name"):
        return redirect("/")
        
    id = session.get("id")
    print(id)
    db = SQL("sqlite:///users.db")
    user = db.execute("SELECT * FROM users WHERE id = :id", id=id)
    if user[0]["role"] == "patient":
        return redirect("/")

    if request.method == "GET":
        return render_template("updateInfo.html")
    else:
        specialty = request.form.get("specialty")
        license_number = request.form.get("license_number")
        years_experience = request.form.get("years_experience")
        hospital_affiliation = request.form.get("hospital_affiliation")
        degrees_certifications = request.form.get("degrees_certifications")
        location = request.form.get("location")
        address = request.form.get("address")
        phoneNumber = request.form.get("phoneNumber")
        languages_spoken = request.form.get("languages_spoken")
        bio = request.form.get("bio")

        id = session.get("id")
        db = SQL("sqlite:///doctors.db")
        
        try:
            # Create doctors table if it doesn't exist
            db.execute("""
                CREATE TABLE IF NOT EXISTS doctors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    userID INTEGER,
                    specialty TEXT,
                    license_number TEXT,
                    years_experience INTEGER,
                    hospital_affiliation TEXT,
                    degrees_certifications TEXT,
                    city TEXT,
                    address TEXT,
                    phoneNumber TEXT,
                    bio TEXT,
                    languages_spoken TEXT
                )
            """)
            
            user = db.execute("SELECT * FROM doctors WHERE userID = :id", id=id)

            if len(user) == 0:
                db.execute("""
                    INSERT INTO doctors (
                        userID, specialty, license_number, years_experience, 
                        hospital_affiliation, degrees_certifications, city, 
                        address, phoneNumber, bio, languages_spoken
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """, id, specialty, license_number, years_experience, 
                    hospital_affiliation, degrees_certifications, location, 
                    address, phoneNumber, bio, languages_spoken)
            else:
                db.execute("""
                    UPDATE doctors SET 
                        specialty = :specialty, 
                        license_number = :license_number, 
                        years_experience = :years_experience, 
                        hospital_affiliation = :hospital_affiliation, 
                        degrees_certifications = :degrees_certifications, 
                        city = :city, 
                        address = :address, 
                        phoneNumber = :phoneNumber, 
                        bio = :bio, 
                        languages_spoken = :languages_spoken 
                    WHERE userID = :id
                """, specialty=specialty, license_number=license_number, 
                    years_experience=years_experience, 
                    hospital_affiliation=hospital_affiliation, 
                    degrees_certifications=degrees_certifications, 
                    city=location, address=address, phoneNumber=phoneNumber, 
                    bio=bio, languages_spoken=languages_spoken, id=id)

            return render_template("message.html", message="You have successfully updated your information!")
            
        except Exception as e:
            print(f"Database error: {str(e)}")
            return render_template("message.html", message="An error occurred while updating your information. Please try again.")
