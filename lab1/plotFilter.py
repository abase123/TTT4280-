import csv
import matplotlib.pyplot as plt

header = []
data = []


filename = "filter.csv"
with open(filename) as csvfile:
    csvreader = csv.reader(csvfile)

    header = next(csvreader)

    for datapoint in csvreader:

        values = [float(value) for value in datapoint]
        data.append(values)

print(header)
print(data[0])
print(data[1])

time = [p[0] for p in data]
ch1 = [p[1] for p in data]


plt.xlabel('Freq [Hz]')
plt.ylabel('Output fitler [dB]')

plt.plot(time,ch1)
plt.show()

    