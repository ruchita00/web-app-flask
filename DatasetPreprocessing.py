import csv
import numpy
filename = 'D:\\indian_liver_patient.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = numpy.array(x)
print(data.shape)