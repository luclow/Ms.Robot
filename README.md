# Mrs.Robot Fashion Modelling 👩🏻‍🔬

**＊ ✿ ❀ Training a variational autoencoder on the Fashion MNIST dataset ❀ ✿ ＊**


<div>
  
  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/lucylow/Mrs.Robot.svg)](https://github.com/lucylow/Mrs.Robot/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/lucylow/Mrs.Robot.svg)](https://github.com/lucylow/Mrs.Robot/pulls)
  [![License](https://img.shields.io/bower/l/bootstrap)]()

</div>


---

## Table_of_Contents &#x1F49C;

* [Motivation](#Motivation-)
* [Autoencoders](#Autoencoders-)
* [Label descriptions](#Label_descriptions-)
* [Download the fashion data](#Download_the_fashion_data-)
* [Run the training script](#Run_the_training_script-) 
* [Loss error function](#Loss_error_function-)
* [TensorBoard monitoring model training](#TensorBoard_monitoring_model_training-)
* [Model Discussion](#Model_Discussion-)
* [Conclusion](#Conclusion-)
* [References](#references-) 

---

## Motivation &#x1F49C;

* **Train a variational autoenconder (VAE) using TensorFlow.js on Node with technical requirements:**
  * Tensorflow==1.12.0
  * Keras==2.2.4
  * TensorflowJS==0.6.7
  * AzureML-SDK==1.0.41.
  
* The model will be trained on the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset 

    ![Gender bias](https://github.com/lucylow/Mrs.Robot/blob/master/images/google_search.png)

    *Image. Gender Bias. Google Search's auto suggestions when user types in "**How to get my daughter into...**" Also reference video https://www.ted.com/talks/cameron_russell_looks_aren_t_everything_believe_me_i_m_a_model/up-next 👩🏻‍*

---

## Autoencoders &#x1F49C;

 ![Autoencoders yay ](https://github.com/lucylow/Mrs.Robot/blob/master/images/autoencoder.jpg)

  *Image. How autoencoders work using the MNIST data set with the number "2"*

* "Autoencoding" == **Data compression algorithm** with compression and decompression functions
* User defines the parameters in the function using variational autoencoder
* Self-supervised learning where target models are generated from input data
* Implemented with **neural networks** - useful for problems in unsupervised learning (no labels)

---

## Variational Autoencoders (VAE) &#x1F49C;

* Variational autoencoders are autoencoders, but with more constraints
* **Generative model** with parameters of a probability distribution modeling the data
* The encoder, decoder, and VAE are 3 models that share weights. After training the VAE model, **the encoder can be used to generate latent vectors**
* [Refer to Keras tutorial for variational autoenconder (MNIST digits)](https://blog.keras.io/building-autoencoders-in-keras.html) except we will be using Fashion data instead :)

  ![Spark Joy!!!](https://github.com/lucylow/Mrs.Robot/blob/master/images/marie_kondo.jpg)
  
  *Image. Marie Kondo sparking joy with the wonders of variational autoencoders 👩🏻‍🔬*


---

## Variational Autoencoder (VAE) Example &#x1F49C;

Example of **encoder network maping inputs to latent vectors**:

* Input samples x into two parameters in latent space = **z_mean and z_log_sigma** 
* Randomly sample points z from latent normal distribution to generate data
* z = z_mean + exp(z_log_sigma) * epsilon, where epsilon is a **random normal tensor**
* **Decoder network maps latent space** points back to the original input data

```python

 x = Input(batch_shape=(batch_size, original_dim))
 
 h = Dense(intermediate_dim, activation='relu')(x)
 
 z_mean = Dense(latent_dim)(h)
 
 z_log_sigma = Dense(latent_dim)(h)
```
*Sample Code for VAE encoder network*


---


## Label_descriptions &#x1F49C;

**Mrs.Robot has the following fashion pieces in her wardrobe:**

0.	T-shirt/top
1.	Trouser
2.	Pullover
3.	Dress
4.	Coat
5.	Sandal
6.	Shirt
7.	Sneaker
8.	Bag
9.	Ankle boot


  ![Plot of subset Images from Fashion MNIST dataset](https://github.com/lucylow/Mrs.Robot/blob/master/images/mnist%20labels.png)
    
   *Image. The 0 to 9 label descriptions for the Fashion MNIST dataset*
  
---
  
## Prepare_the_node_environment &#x1F49C;


```sh
yarn
# Or
npm install
```

----

## Node.js Applications on Microsoft's Azure &#x1F49C;

* Mrs. Robot application built using Node.js is hosted on Azure with deployments automatically connecting to the github repo and syncing to Azure
* Applications managed with VS code or web browser
* Quick deployment of full-stack applications on Linux service (ex. Docker container)
* Video tutorial "Build and deploy Node.js and React apps with Visual Studio Code, Azure App Service and Cosmos DB" https://www.youtube.com/watch?v=mvW0tIsdnHI&list=PLLasX02E8BPBsxTwjnhjcPYHWAbzUZvf7

---


## Azure Training ML Models in the Cloud &#x1F49C;

* The machine learning model is run on remote compute resources using [Microsoft's Azure ML SDK](https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-train-models-with-aml?WT.mc_id=aisummit-github-amynic)
* [Azure SDK for Node.js developers](https://github.com/Azure/azure-sdk-for-node):
  * Machine Learning	npm install azure-arm-machinelearning
  * Machine Learning Compute	npm install azure-arm-machinelearningcompute
* Connect to Azure's "workspace object"

---

## Download_the_fashion_data &#x1F49C;

* Download **Mrs.Robot's fashion dataset with over 60,000 fashion training set images** [train-images-idx3-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz) from [here](https://github.com/zalandoresearch/fashion-mnist#get-the-data)
* Uncompress the large file size (26 MBytes)
* Move the uncompressed file `train-images-idx3-ubyte` into `dataset` folder in the example folder of this repo

---

## Run_the_training_script &#x1F49C;

* Can not feed all the data to the model at once due to computer memory limitations so **data is split into "batches"** 
* When all batches are fed exactly once, an "epoch" is completed. As training script runs, **preview images afer every epoch will show**
* At the end of each epoch the preview image should look more and more like an item of clothing for Mrs.Robot

```sh
yarn train
```

---

## Loss_error_function &#x1F49C;

* **Loss function to account for error in training** since Mrs.Robot is picky about her fashion pieces 
* Two loss function options: The default **binary cross entropy (BCE)** or **mean squared error (MSE)**
* The loss from a good training run will be approx 40-50 range whereas an average training run will be close to zero

  ![Example loss curve from training](https://github.com/lucylow/Mrs.Robot/blob/master/images/vae_tensorboard2.png)

    *Image of loss curve with the binary cross entropy error function*


---

### TensorBoard_monitoring_model_training &#x1F49C;

Use `--logDir` flag of `yarn train` command. Log the **batch-by-batch loss values** to a log directory

```sh
yarn train --logDir /tmp/vae_logs
```

Start TensorBoard in a separate terminal to  print an **http:// URL to the console**. The training process can then be **monitored in the browser by Mrs.Robot:**

```sh
pip install tensorboard 
tensorboard --logdir /tmp/vae_logs
```

![Tensorboard Monitoring](https://github.com/lucylow/Mrs.Robot/blob/master/images/tensorboard%20web%20view.png)

*Image. Tensorboard's monitoring interface.*

![Tensorboard](https://github.com/lucylow/Mrs.Robot/blob/master/images/tensorboard%20curves.png)

*Image. Tensorboard's monitoring interface.*


---

## Model_Discussion &#x1F49C;

**VAE is a generative mode**l which means it can be used to **generate new fashion pieces for Mrs.Robot**. This is done by scanning the latent plane, sampling the latent points at regular intervals, to generate the corresponding fashion piece for each point. Run to serve the model and the training web page 👩🏻‍🔬:


```sh
yarn watch
```


Refer to image below for a **visualization of the latent manifold** that was **"generated"**:


![screenshot of results on fashion MNIST. A 30x30 grid of small images](https://github.com/lucylow/Mrs.Robot/blob/master/images/fashion-mnist-vae-scr.png)

  *Image of completed training results on fashion MNIST 30x30 grid of small images for Mrs.Robot*
  
---

## Conclusion &#x1F49C;


Our results show that a **generative model with parameters of a probability distribution** (VAE) is capable of **achieving results on a highly challenging dataset** of over 60,000 fashion set images using machine learning.


---

## References &#x1F49C;
* [Tensorflow's tutorial with tf.keras, a high-level API to train Fashion MNIST] https://www.tensorflow.org/tutorials/keras/basic_classification
* [Zaiando Research Fashion MNIST data] http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/
* [Microsoft Documentation. Node.js on Azure]https://azure.microsoft.com/en-ca/develop/nodejs/
* Learning generative visual models from training examples: An
incremental bayesian approach tested on 101 object categories. 
* [Google Scholar - Publications on Fashion MNIST data sets] https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=fashion-mnist&btnG=&oq=fas
* [Building Autoencoders in Keras using DL for Python] https://blog.keras.io/building-autoencoders-in-keras.html
* Microsoft Azure Train image classification data in cLOUD https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-train-models-with-aml?WT.mc_id=aisummit-github-amynic
* L. Fei-Fei, R. Fergus, and P. Perona. Learning generative visual models from few training examples. Computer Vision and Image Understanding. 2007.
* [Kaggle Data Science competitions with fashion data set] https://www.kaggle.com/zalando-research/fashionmnist
* Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. Han Xiao, Kashif Rasul, Roland Vollgraf. arXiv:1708.07747
* Kingma, Diederik P., and Max Welling. "Auto-Encoding Variational Bayes." https://arxiv.org/abs/1312.6114
