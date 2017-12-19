# TrajectoryNet

### Slides
For a quick overview of the paper, please refer to the [slides](https://docs.google.com/presentation/d/1nabGnSxAEvjNKHVVCXxMD_NRNJc6983BorVIX36GeaU/edit?usp=sharing)

### Title
TrajectoryNet: An Embedded GPS Trajectory Representation for Point-based Classification Using Recurrent Neural Networks

**Authors**: Xiang Jiang, Erico N de Souza, Ahmad Pesaranghader, Baifan Hu, Daniel L. Silver and Stan Matwin

**Abstract**:
Understanding and discovering knowledge from GPS (Global Positioning System) traces of human activities is an essential topic in mobility-based urban computing. We propose TrajectoryNet-a neural network architecture for point-based trajectory classification to infer real world human transportation modes from GPS traces. To overcome the challenge of capturing the underlying latent factors in the low-dimensional and heterogeneous feature space imposed by GPS data, we develop a novel representation that embeds the original feature space into another space that can be understood as a form of basis expansion.
We also enrich the feature space via segment-based information and use Maxout activations to improve the predictive power of Recurrent Neural Networks (RNNs). We achieve over 98% classification accuracy when detecting four types of transportation modes, outperforming existing models without additional sensory data or location-based prior knowledge.

**Contact**

xiang.jiang@dal.ca

Please feel free to email me in case you have any questions with the code.
There are a lot of redundant codes that I have not got a chance to clean up. Sorry about this clutter!

### Dependencies
The follwoing softwares are required for this source code.
- **Python 3.5**
- **Tensorflow 1.4.1**
- sklean 0.18.1
- numpy 1.12.0
- R (optional for preprocessing)

### Data
The original dataset can be found [here](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/).
We preprocessed a subset of the data in the `data` directory.
This dataset contains features in `.csv` format as well as `.npy` format:
- `csvdata.csv`: include data of four classes (car, bike, drive and walk) with longitudes, latitudes, speeds and other features
- `encodedcsv.zip`: the same data after discretization and one-hot-encoding
- `npy.zip`: the same data in `.npy` format. This can be used to train the model directly.

If you want to train the model without any preprocessing, please `unzip` the numpy data before training the model:

```
cd ./data
unzip npy.zip
```

### Network Configurations
The file config.json includes the configurations of the model:
- training, validation and test set selection
- number of hidden nodes in each layer
- learning rate
- mini-batch size
- number of layers
- number of epochs
- whether to checkpoint the model after training
- whether to restore the model before training
- the size of truncated sequences
- frequency to evaluate the model while training
- number of threads
- whether to use GPU during training
- etc.

### Network Training
`python trajectoryNet.py`

### Preprocessing
In case you are interested in the preprocessing and discretization of the data, please refer to file `preprocess.R`.

After the preprocess data are stored in a `.csv` file, it is required to run `create_npy.py` to transform the data into `.npy` format to get ready to import to Tensorflow.
