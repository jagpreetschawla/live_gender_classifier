# live_gender_classifier
This project classifies faces as a male or female face on a live camera feed. It is powered be a very basic Neural Network model, served using tensorflow-serving

![Live gender classifier](http://www.cylopsis.com/img/live_gender_classifier.png)

![Gender Classifier](http://www.cylopsis.com/img/gender_classifier.png)

Project requirements:
* TensorFlow (v1.6.0)
* Numpy (v1.14.2)
* Scikit-Image (v0.13.1)
* OpenCV (v3.4.0.12)
* TensorFlow-Serving (v1.6.0)

The ipython notebook `nn_classifier.ipynb` contains code to train and export the NN, and python script `mf_reco_live.py` detects the faces (using opencv) and classifies them on a live camera feed using exported model which is served by tensorflow-serving.
To run the live classifier, you must have model being server by tensorflow-serving on port 9000. To learn more about tensorflow-serving see <https://www.tensorflow.org/serving/> and <https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/serving_basic.md>.

`models/NN/1524339561` contains the model that I trained and exported, and can be used directly.

The data set used is taken from <https://github.com/MinhasKamal/DeepGenderRecognizer>. Minas Kamal gathered these images from LFW dataset and different magzines. It is a quite small dataset, feel free to use a larger dataset for better performance.

A step by step guide for training this NN can be found [here](http://www.cylopsis.com/post/neural-network/gender-classification/).
