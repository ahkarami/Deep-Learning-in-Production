# Deep-Learning-in-Production
In this repository, I will share some useful notes and references about deploying deep learning-based models in production.

<p align="center">
  <img src="./Final-Logo.jpg?raw=true" alt="Logo"/>
</p>

## Convert PyTorch Models in Production:
- [The road to 1.0: production ready PyTorch](https://pytorch.org/2018/05/02/road-to-1.0.html)
- [PyTorch model recognizing hotdogs and not-hotdogs deployed on flask](https://github.com/jaroslaw-weber/hotdog-not-hotdog)
- [Flask application to support pytorch model prediction](https://github.com/craigsidcarlson/PytorchFlaskApp)
- [Serving PyTorch Model on Flask Thread-Safety](https://discuss.pytorch.org/t/serving-pytorch-model-on-flask-thread-safety/13921)
- [Serving PyTorch Models on AWS Lambda with Caffe2 & ONNX](https://machinelearnings.co/serving-pytorch-models-on-aws-lambda-with-caffe2-onnx-7b096806cfac)
- [Serving PyTorch Models on AWS Lambda with Caffe2 & ONNX (Another Version)](https://blog.waya.ai/deploy-deep-machine-learning-in-production-the-pythonic-way-a17105f1540e)
- [WebDNN: Fastest DNN Execution Framework on Web Browser](https://github.com/mil-tokyo/webdnn)

## Convert PyTorch Models to C++:
- [ATen: A TENsor library](https://github.com/pytorch/pytorch/tree/master/aten)
- [Important Issue about PyTorch-like C++ interface](https://github.com/pytorch/pytorch/issues/3335)
- [PyTorch C++ API Test](https://github.com/pytorch/pytorch/tree/master/test/cpp/api)
- [PyTorch C++ Library](https://github.com/warmspringwinds/pytorch-cpp)
- [Direct C++ Interface to PyTorch](https://github.com/ebetica/autogradpp)
- [A Python module for compiling PyTorch graphs to C](https://github.com/lantiga/pytorch2c)

## Deploy TensorFlow Models in Production:
- [How to deploy Machine Learning models with TensorFlow - _Part1_](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-1-make-your-model-ready-for-serving-776a14ec3198)
- [How to deploy Machine Learning models with TensorFlow - _Part2_](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-2-containerize-it-db0ad7ca35a7)
- [How to deploy Machine Learning models with TensorFlow - _Part3_](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-3-into-the-cloud-7115ff774bb6)
- [Creating REST API for TensorFlow models](https://becominghuman.ai/creating-restful-api-to-tensorflow-models-c5c57b692c10)
- ["How to Deploy a Tensorflow Model in Production" by _Siraj Raval_ on YouTube](https://www.youtube.com/watch?v=T_afaArR0E8)
- [Code for the "How to Deploy a Tensorflow Model in Production" by _Siraj Raval_ on YouTube](https://github.com/llSourcell/How-to-Deploy-a-Tensorflow-Model-in-Production)
- [How to deploy an Object Detection Model with TensorFlow serving](https://medium.freecodecamp.org/how-to-deploy-an-object-detection-model-with-tensorflow-serving-d6436e65d1d9) [_Very Good Tutorial_]
- [Freeze Tensorflow models and serve on web](http://cv-tricks.com/how-to/freeze-tensorflow-models/) [_Very Good Tutorial_]
- [How Zendesk Serves TensorFlow Models in Production](https://medium.com/zendesk-engineering/how-zendesk-serves-tensorflow-models-in-production-751ee22f0f4b)
- [TensorFlow Serving Example Projects](https://github.com/Vetal1977/tf_serving_example)
- [Serving Models in Production with TensorFlow Serving](https://www.youtube.com/watch?v=q_IkJcPyNl0) [_TensorFlow Dev Summit 2017 Video_]
- [Building TensorFlow as a Standalone Project](https://tuatini.me/building-tensorflow-as-a-standalone-project/)
- [TensorFlow C++ API Example](https://github.com/jhjin/tensorflow-cpp)
- [TensorFlow.js](https://js.tensorflow.org/)
- [Introducing TensorFlow.js: Machine Learning in Javascript](https://medium.com/tensorflow/introducing-tensorflow-js-machine-learning-in-javascript-bf3eab376db)

## Convert Keras Models in Production:
- [Deep learning in production with Keras, Redis, Flask, and Apache](https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/) [_Rank: 1st & General Usefult Tutorial_]
- [Deploying your Keras model](https://medium.com/@burgalon/deploying-your-keras-model-35648f9dc5fb)
- [Deploying your Keras model using Keras.JS](https://becominghuman.ai/deploying-your-keras-model-using-keras-js-2e5a29589ad8)
- ["How to Deploy a Keras Model to Production" by _Siraj Raval_ on Youtube](https://github.com/llSourcell/how_to_deploy_a_keras_model_to_production)
- [Deploy Keras Model with Flask as Web App in 10 Minutes](https://github.com/mtobeiyf/keras-flask-deploy-webapp) [Good Repository]
- [keras2cpp](https://github.com/pplonski/keras2cpp)

## Deploy MXNet Models in Production:
- [Model Server for Apache MXNet](https://github.com/awslabs/mxnet-model-server)
- [Running the Model Server](https://github.com/awslabs/mxnet-model-server/blob/master/docs/server.md)
- [Exporting Models for Use with MMS](https://github.com/awslabs/mxnet-model-server/blob/master/docs/export.md)
- [Single Shot Multi Object Detection Inference Service](https://github.com/awslabs/mxnet-model-server/blob/master/examples/ssd/README.md)
- [How can we serve MXNet models built with gluon api](https://discuss.mxnet.io/t/how-can-we-serve-mxnet-models-built-with-gluon-api/684)
- [MXNet C++ Package](https://github.com/apache/incubator-mxnet/tree/master/cpp-package)
- [MXNet C++ Package Examples](https://github.com/apache/incubator-mxnet/tree/master/cpp-package/example)
- [MXNet Image Classification Example of C++](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification/predict-cpp)
- [MXNet C++ Tutorial](http://mxnet.incubator.apache.org/tutorials/c%2B%2B/basics.html)
- [An introduction to the MXNet API](https://becominghuman.ai/an-introduction-to-the-mxnet-api-part-1-848febdcf8ab) [Very Good Tutorial for Learning MXNet]
- [GluonCV](https://gluon-cv.mxnet.io/)

## ONNX (Open Neural Network Exchange):
- [Tutorials for using ONNX](https://github.com/onnx/tutorials)

## Some Caffe2 Tutorials:
- [Mnist using caffe2](http://vast.uccs.edu/~adhamija/blog/MNIST_singleGPU.html)
- [Make Transfer Learning of SqueezeNet on Caffe2](https://medium.com/@KazamiXHayato/make-transfer-learning-in-caffe2-21d96c47ba0e)
- [Build Basic program by using Caffe2 framework in C++](https://medium.com/@KazamiXHayato/write-caffe2-program-in-c-5519e2646382)

## Some Useful Resources for Designing UI:
- [ReactJS vs Angular5 vs Vue.js](https://medium.com/@TechMagic/reactjs-vs-angular5-vs-vue-js-what-to-choose-in-2018-b91e028fa91d)
- [A Guide to Becoming a Full-Stack Developer](https://medium.com/coderbyte/a-guide-to-becoming-a-full-stack-developer-in-2017-5c3c08a1600c) [_Very Good Tutorial_]
- [Roadmap to becoming a web developer in 2018](https://github.com/kamranahmedse/developer-roadmap) [_Very Good Repository_]
- [Modern Frontend Developer in 2018](https://medium.com/tech-tajawal/modern-frontend-developer-in-2018-4c2072fa2b9c)
- [Build A Real World Beautiful Web APP with Angular 6](https://medium.com/@hamedbaatour/build-a-real-world-beautiful-web-app-with-angular-6-a-to-z-ultimate-guide-2018-part-i-e121dd1d55e)

## Other:
- [Some PyTorch Workflow Changes](https://github.com/pytorch/pytorch/issues/6032)
- [PyTorch and Caffe2 repos getting closer together](https://github.com/caffe2/caffe2/issues/2439#issuecomment-391155017)
- [PyTorch or TensorFlow?](https://awni.github.io/pytorch-tensorflow/)
- [Choosing a Deep Learning Framework in 2018: Tensorflow or Pytorch?](http://cv-tricks.com/deep-learning-2/tensorflow-or-pytorch/)
- [Making your C library callable from Python](https://medium.com/@shamir.stav_83310/making-your-c-library-callable-from-python-by-wrapping-it-with-cython-b09db35012a3)
- [MIL WebDNN](https://mil-tokyo.github.io/webdnn/)

