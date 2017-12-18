import numpy as np

csvfile = './data/mobility_encoding.csv'
task = 'mobility'

data = np.genfromtxt(csvfile, dtype=float, delimiter=',')
mmsi = data[:,0].astype(int)
labels = data[:,data.shape[1]-1]
features = data[:, 1:data.shape[1]-1]

mmsi_length = np.unique(mmsi, return_counts=True)
num_persons = len(mmsi_length[0])
max_mmsi_len = max(mmsi_length[1])
num_features = features.shape[1]

padding_data = np.zeros((max_mmsi_len * num_persons, num_features))
for person in range(num_persons):
    person_data = np.zeros((max_mmsi_len, num_features))
    person_data[0:mmsi_length[1][person],:] = features[mmsi == mmsi_length[0][person],:]
    padding_data[person*max_mmsi_len : (person+1)*max_mmsi_len,:] = person_data
    

padding_labels = np.zeros((max_mmsi_len * num_persons))
for person in range(num_persons):
    person_data = np.zeros((max_mmsi_len))
    person_data[0:mmsi_length[1][person]] = labels[mmsi == mmsi_length[0][person]]
    padding_labels[person*max_mmsi_len : (person+1)*max_mmsi_len] = person_data
    


x = np.reshape(padding_data, (num_persons, max_mmsi_len, num_features))
y = np.reshape(padding_labels, (num_persons, max_mmsi_len))
y = y.astype(int)
np.save('./data/x_'+task+'.npy', x)
np.save('./data/y_'+task+'.npy', y)
np.save('./data/mmsi_'+task+'.npy', mmsi_length)


