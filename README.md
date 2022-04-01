# LyapunovAutoEncode
## System Requirements
Our code is run on Ubuntu 18.04. No non-standard hardware is required.

## Installation Guide
### requirement
- Python 3.8
- Pytorch 1.10.2
- Additional requirements in requirements.txt
  - ```bash
    # installation could take a couple minutes
    pip install -r requirements.txt
## Demo
The following code demo outlines the steps to analyze AeLLE for the CharRNN task.

Utilizing the Lyapunov AutoEncoder requires four steps:
1. Train Various Models (with varying hyperparameters)
2. Calculate Lyapunov Exponents
3. Train Lyapunov AutoEncoder
4. Analyze AeLLE

### Step 1: Train Models (Trials)
Before we begin any training, we pre-process our data by running the following code:
~~~
python test_set_prep.py
~~~

We start by training a number of recurrent models with different hyperparameters.
To do this, we run 
~~~
#CharRNN_trials.py -model [MODEL TYPE] -evals [NO. EVALUATIONS]
python CharRNN_trials.py -model lstm evals 300
~~~

For MODEL_TYPE, select either 'lstm', 'gru', or 'rnn' (default: 'lstm')

For NO. EVALUATIONS, select an integer for the number of trials/networks of each size to train. (default: 20)

This code will train [NO. EVALUATIONS] instances of [MODEL TYPE] models each of size 64, 128, 256, 512.

The relevant training outputs for all models of a given size (model parameters, training and validation loss, etc.) will be stored in a Trials object.

### Step 2: Calculate Lyapunov Exponents
Running the code for Step 1 will also calculate the Lyapunov Exponents (LEs) of each of these models and store them in the Trial object generated in step 1.

If you wish to adjust the hyperparameters of the LE calculation, you can open CharRNN_Trials.py and change the values under the header 'LE Hyperparameters'.
The hyperparameters that you can change are:
- le_batch_size : The number of LEs to be calcualated in parallel and averaged. Larger values allow for more accurate LEs, but take more time and memory. (Default: 15)
- le_seq_length : Length of sequence used to calculate the LEs. Length of sequence needed can depend on the input data distribution. To calibrate, you can observe the convergence of the LEs over sequence length (not included in this demo). (Default: 100)
- ON_step : Steps between orthonormalization. Can be increased above 1 to increase speed, but decreases precision of lower-valued (more negative) LEs (Default: 1)
- warmup: Number of steps over which to evolve the system to allow it to relax onto attractor before calculating LEs. Can help with convergence rates. (Default: 500)

Once this is done, we run
~~~
# python AE_utils.py -model [MODEL TYPE]
python AE_utils.py -model lstm
~~~
Use the same MODEL TYPE as above. This will combine the LEs across all network sizes into a single-sized dataset by interpolating the values of the smaller networks.
The output of this step will be a single dictionary file containing all the LEs split into a training, validation, and test set. 
The name of this file will be [MODEL TYPE]_data_split_vfrac0.2.p.

### Step 3: Train Lyapunov AutoEncoder
Once the LEs are calculated for all the models, we use an Autoencoder to project the LEs into a lower dimension and relate them to performance (as measured by validation loss).

We have the following arguments for our Lyapunov Autoencoder training:
- model : Same [MODEL TYPE] (str) as above (default: 'lstm')
- latent : The size (int) of the latent space of the autoencoder. (default: 32)
- alphas: This should be a list ([]) of at least 1 number. This is the weighting assigned to the prediction loss versus the reconstruction loss. Best to start low and increase. (default: [5, 5, 10, 20])
- epochs: The number (int) of epochs the network is trained on for each value of alpha (default: 1000)
- ae_lr : The initial learning rate of the optimizer. (default: 1e-5)

Running the following code will train the autoencoder:
~~~
#AE_train.py -model [MODEL TYPE] -latent [LATENT SIZE] -alphas [ALPHA LIST] -epochs [EPOCHS] -ae_lr [INITIAL LEARNING RATE]
python AE_train.py -model lstm -latent 32 -alphas [10, 10, 10, 10] -epochs 500 -ae_lr 0.01
~~~
The model will save the reconstruction loss (standard for autoencoders) as well as the prediction loss, for which it uses the latent LE representation to predict the validation loss of the corresponding network.
A figure showing the loss as a function of training epoch will be saved in a folder titled 'Figures/'

### Step 4: Analyze Latent Lyapunov Exponents (AeLLE)
Once the autoencoder is trained, we would like to analyze its latent representation of the LEs (AeLLE). We can do this by taking the first 2 principal components of the AeLLE and comparing the performance of these networks.

Use the following code to generate this PCA plot:
~~~
#LE_clustering.py -model [MODEL TYPE] -thresh [THRESHOLD] -latent [LATENT SIZE]
python LE_clustering.py -model lstm -thresh 1.0 -latent 32
~~~

This will generate plots of the AeLLE with different the performance of the corresponding network indicated by color.

Expected output
Expected run time for demo on "normal" desktop computer

## Instructions for Use
How to run software on your data
Reproduction instructions