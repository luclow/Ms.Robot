# AI-powered Fashion Models 👩🏻‍🔬

**＊ ✿ ❀Fashion MNIST dataset ❀ ✿ ＊**


<div>
  
  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/lucylow/Ms.Robot.svg)](https://github.com/lucylow/Ms.Robot/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/lucylow/Ms.Robot.svg)](https://github.com/lucylow/Ms.Robot/pulls)
  [![License](https://img.shields.io/aur/license/android-studio.svg)]()

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
* [Conclusion Model Discussion](#Conclusion_Model_Discussion-)
* [References](#references-) 

---

## Motivation &#x1F49C;

* **Train a variational autoenconder (VAE) using TensorFlow.js on Node with technical requirements:**
  * Tensorflow==1.12.0
  * Keras==2.2.4
  * TensorflowJS==0.6.7
  
* The model will be trained on the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset 



![](https://github.com/lucylow/Ms.Robot/blob/master/images/Screen%20Shot%202021-05-10%20at%208.22.41%20PM.png)

![](https://github.com/lucylow/Ms.Robot/blob/master/images/Screen%20Shot%202021-05-10%20at%208.22.52%20PM.png)

---
## AI-powered Mobile App &#x1F49C;

![](https://github.com/lucylow/Ms.Robot/blob/master/images/Screen%20Shot%202021-05-10%20at%208.25.18%20PM.png)
![](https://github.com/lucylow/Ms.Robot/blob/master/images/Screen%20Shot%202021-05-10%20at%208.25.31%20PM.png)
![](https://github.com/lucylow/Ms.Robot/blob/master/images/Screen%20Shot%202021-05-10%20at%208.25.48%20PM.png)
![](https://github.com/lucylow/Ms.Robot/blob/master/images/Screen%20Shot%202021-05-10%20at%208.25.40%20PM.png)
![](https://github.com/lucylow/Ms.Robot/blob/master/images/Screen%20Shot%202021-05-10%20at%208.26.07%20PM.png)
![](https://github.com/lucylow/Ms.Robot/blob/master/images/Screen%20Shot%202021-05-10%20at%208.26.13%20PM.png)
![](https://github.com/lucylow/Ms.Robot/blob/master/images/Screen%20Shot%202021-05-10%20at%208.26.53%20PM.png)
![](https://github.com/lucylow/Ms.Robot/blob/master/images/Screen%20Shot%202021-05-10%20at%208.26.23%20PM.png)
![](https://github.com/lucylow/Ms.Robot/blob/master/images/Screen%20Shot%202021-05-10%20at%208.26.45%20PM.png)
![](https://github.com/lucylow/Ms.Robot/blob/master/images/Screen%20Shot%202021-05-10%20at%208.27.28%20PM.png)

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

**Ms.Robot has the following fashion pieces in her wardrobe:**

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


  ![Plot of subset Images from Fashion MNIST dataset](https://github.com/lucylow/Ms.Robot/blob/master/images/mnist%20labels.png)
    
   *Image. The 0 to 9 label descriptions for the Fashion MNIST dataset*
  
---
  
## Prepare_the_node_environment &#x1F49C;


```sh
yarn
# Or
npm install
```

---

## Download_the_fashion_data &#x1F49C;

* Download **Ms.Robot's fashion dataset with over 60,000 fashion training set images** [train-images-idx3-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz) from [here](https://github.com/zalandoresearch/fashion-mnist#get-the-data)
* Uncompress the large file size (26 MBytes)
* Move the uncompressed file `train-images-idx3-ubyte` into `dataset` folder in the example folder of this repo

---

## Run_the_training_script &#x1F49C;

* Can not feed all the data to the model at once due to computer memory limitations so **data is split into "batches"** 
* When all batches are fed exactly once, an "epoch" is completed. As training script runs, **preview images afer every epoch will show**
* At the end of each epoch the preview image should look more and more like an item of clothing for Ms.Robot

```sh
yarn train
```

---

## Loss_error_function &#x1F49C;

* **Loss function to account for error in training** since Ms.Robot is picky about her fashion pieces 
* Two loss function options: The default **binary cross entropy (BCE)** or **mean squared error (MSE)**
* The loss from a good training run will be approx 40-50 range whereas an average training run will be close to zero

  ![Example loss curve from training](https://github.com/lucylow/Ms.Robot/blob/master/images/vae_tensorboard2.png)

    *Image of loss curve with the binary cross entropy error function*


---

### TensorBoard_monitoring_model_training &#x1F49C;

Use `--logDir` flag of `yarn train` command. Log the **batch-by-batch loss values** to a log directory

```sh
yarn train --logDir /tmp/vae_logs
```

Start TensorBoard in a separate terminal to  print an **http:// URL to the console**. The training process can then be **monitored in the browser by Ms.Robot:**

```sh
pip install tensorboard 
tensorboard --logdir /tmp/vae_logs
```

![Tensorboard Monitoring](https://github.com/lucylow/Ms.Robot/blob/master/images/tensorboard%20web%20view.png)

*Image. Tensorboard's monitoring interface.*

![Tensorboard](https://github.com/lucylow/Ms.Robot/blob/master/images/tensorboard%20curves.png)

*Image. Tensorboard's monitoring interface.*


---

## Conclusion_Model_Discussion &#x1F49C;

Results show a generative model with parameters of a probability distribution variational autoencoder (VAE) is capable of achieving results on a highly challenging dataset of over 60,000 fashion set images using machine learning. The variational autoencoder VAE is a generative model which means it can be used to generate new fashion pieces for Ms.Robot. This is done by scanning the latent plane, sampling the latent points at regular intervals, to generate the corresponding fashion piece for each point. Run to serve the model and the training web page. Intelligence never goes out of style. 

```sh
yarn watch
```

![screenshot of results on fashion MNIST. A 30x30 grid of small images](https://github.com/lucylow/Ms.Robot/blob/master/images/fashion-mnist-vae-scr.png)

  *Image of completed training results on fashion MNIST 30x30 grid of small images for Ms.Robot. Visualization of the latent manifold** that was "generated" by the Ms.Robot generative model.


---

## References &#x1F49C;
![Gender/Brilliance bias](https://github.com/lucylow/Mrs.Robot/blob/master/images/google_search.png)


* "How to get my daughter into modeling?" https://duckduckgo.com/?q=how+to+get+my+daughter+into+modeling&t=hx&ia=web
* Tensorflow's tutorial with tf.keras, a high-level API to train Fashion MNIST https://www.tensorflow.org/tutorials/keras/basic_classification
* Genius Bias Gender Paper https://fermatslibrary.com/s/gender-stereotypes-about-intellectual-ability-emerge-early-and-influence-children-s-interests#email-newsletter
* Zaiando Research Fashion MNIST data http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/
* Repo name inspiration from Mr.Robot https://mrrobot.fandom.com/wiki/Elliot_Alderson
* Google Scholar - Publications on Fashion MNIST data sets https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=fashion-mnist&btnG=&oq=fas
* Building Autoencoders in Keras using DL for Python https://blog.keras.io/building-autoencoders-in-keras.html
* L. Fei-Fei, R. Fergus, and P. Perona. Learning generative visual models from few training examples. Computer Vision and Image Understanding. 2007.
* Kaggle Data Science competitions with fashion data set https://www.kaggle.com/zalando-research/fashionmnist
* Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. Han Xiao, Kashif Rasul, Roland Vollgraf. arXiv:1708.07747
* Kingma, Diederik P., and Max Welling. "Auto-Encoding Variational Bayes." https://arxiv.org/abs/1312.6114
