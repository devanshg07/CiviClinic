import sqlite3
import os

database = open('doctors.db', 'w')
database.truncate(0)  
database.close()
connection = sqlite3.connect("doctors.db")
crsr = connection.cursor()

fields = [
          "userID", #Connecting to their authentication credentials
          "specialty",
          "license_number",
          "years_experience",
          "hospital_affiliation",
          "degrees_certifications",
          "city",
          "address",
          "phoneNumber",
          "bio",
          "verified",
          "languages_spoken" 
        ]


#Easily convertible to MySQL or other databases due to iterative strategy as opposed to hardcoding the db create string, also improves readability and ease of maintenance and adding new fields

dbCreateString = "CREATE TABLE doctors (id INTEGER, "

for field in fields:
    dbCreateString += field+", "

dbCreateString+="PRIMARY KEY(id))"

crsr.execute(dbCreateString)
connection.commit()
crsr.close()
connection.close()

