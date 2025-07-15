import sqlite3
import os

database = open('users.db', 'w')
database.truncate(0)  
database.close()
connection = sqlite3.connect("users.db")
crsr = connection.cursor()

fields = [
    #Personal Information for Account Setup and Maintenance
          "username NOT NULL", 
          "password NOT NULL",
          "dateJoined",
          "salt",
          "accountStatus",
          "role", #hierarchy for possible admins, doctor or patient
          "twoFactorAuth",
          "lastLogin",
          "emailAddress",
          "phoneNumber",
          "name", #Full Name
          "dateOfBirth",
          "gender", #Prefer Not to Say or Other
          "location",
        ]


#Easily convertible to MySQL or other databases due to iterative strategy as opposed to hardcoding the db create string, also improves readability and ease of maintenance and adding new fields

dbCreateString = "CREATE TABLE users (id INTEGER, "

for field in fields:
    dbCreateString += field+", "

dbCreateString+="PRIMARY KEY(id))"

crsr.execute(dbCreateString)
connection.commit()
crsr.close()
connection.close()

