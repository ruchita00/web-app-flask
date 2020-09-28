import csv

with open('D:\\indian_liver_patient.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)