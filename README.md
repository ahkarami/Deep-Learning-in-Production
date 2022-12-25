# Deep-Learning-in-Production
In this repository, I will share some useful notes and references about deploying deep learning-based models in production.

<p align="center">
  <img src="./Final-Logo.jpg?raw=true" alt="Logo"/>
</p>

## Convert PyTorch Models in Production:
- [PyTorch Production Level Tutorials](https://pytorch.org/tutorials/#production-usage) [_Fantastic_]  
- [The road to 1.0: production ready PyTorch](https://pytorch.org/2018/05/02/road-to-1.0.html)
- [PyTorch 1.0 tracing JIT and LibTorch C++ API to integrate PyTorch into NodeJS](http://blog.christianperone.com/2018/10/pytorch-1-0-tracing-jit-and-libtorch-c-api-to-integrate-pytorch-into-nodejs/) [_Good Article_]
- [Model Serving in PyTorch](https://pytorch.org/blog/model-serving-in-pyorch/)
- [PyTorch Summer Hackathon](https://pytorch.devpost.com/) [_Very Important_]
- [Deploying PyTorch and Building a REST API using Flask](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html) [_Important_]
- [PyTorch model recognizing hotdogs and not-hotdogs deployed on flask](https://github.com/jaroslaw-weber/hotdog-not-hotdog)
- [Serving PyTorch 1.0 Models as a Web Server in C++ ](https://github.com/Wizaron/pytorch-cpp-inference) [_Useful Example_]
- [PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/)  [_Interesting & Useful Article_]  
- [Flask application to support pytorch model prediction](https://github.com/craigsidcarlson/PytorchFlaskApp)
- [Serving PyTorch Model on Flask Thread-Safety](https://discuss.pytorch.org/t/serving-pytorch-model-on-flask-thread-safety/13921)
- [Serving PyTorch Models on AWS Lambda with Caffe2 & ONNX](https://machinelearnings.co/serving-pytorch-models-on-aws-lambda-with-caffe2-onnx-7b096806cfac)
- [Serving PyTorch Models on AWS Lambda with Caffe2 & ONNX (Another Version)](https://blog.waya.ai/deploy-deep-machine-learning-in-production-the-pythonic-way-a17105f1540e)
- [Deep Dive into ONNX Runtime](https://medium.com/@mohsen.mahmoodzadeh/a-deep-dive-into-onnx-onnx-runtime-part-1-874517c66ffc)  
- [EuclidesDB - _multi-model machine learning feature database with PyTorch_](https://euclidesdb.readthedocs.io/en/latest/)
- [EuclidesDB - GitHub](https://github.com/perone/euclidesdb/)
- [WebDNN: Fastest DNN Execution Framework on Web Browser](https://github.com/mil-tokyo/webdnn)
- [FastAI PyTorch Serverless API (with AWS Lambda)](https://github.com/alecrubin/pytorch-serverless/)
- [FastAI PyTorch in Production (discussion)](http://forums.fast.ai/t/fastai-pytorch-in-production/16928)   
- [OpenMMLab Model Deployment Framework](https://github.com/open-mmlab/mmdeploy)  
- [TorchServe](https://github.com/pytorch/serve) [Great Tool]    
- [TorchServe Video Tutorial](https://www.youtube.com/watch?v=XlO7iQMV3Ik)  

## Convert PyTorch Models to C++:
- [**Loading a PyTorch Model in C++**](https://pytorch.org/tutorials/advanced/cpp_export.html) [_**Fantastic**_]
- [**PyTorch C++ API**](https://pytorch.org/cppdocs/index.html) [_Bravo_]
- [An Introduction To Torch (Pytorch) C++ Front-End](https://radicalrafi.github.io/posts/pytorch-cpp-intro/) [_Very Good_]
- [Blogs on using PyTorch C++ API](https://discuss.pytorch.org/t/a-series-of-blogs-on-pytorch-c-api-transfer-learning-jupyter-notebook-with-libtorch-xeus-cling-and-more/54628) [_Good_]
- [ATen: A TENsor library](https://github.com/pytorch/pytorch/tree/master/aten)
- [Important Issue about PyTorch-like C++ interface](https://github.com/pytorch/pytorch/issues/3335)
- [PyTorch C++ API Test](https://github.com/pytorch/pytorch/tree/master/test/cpp/api)
- [PyTorch via C++](https://discuss.pytorch.org/t/pytorch-via-c/19234) [_Useful Notes_]
- [AUTOGRADPP](https://github.com/pytorch/pytorch/tree/master/torch/csrc/api)
- [PyTorch C++ Library](https://github.com/warmspringwinds/pytorch-cpp)
- [Direct C++ Interface to PyTorch](https://github.com/ebetica/autogradpp)
- [A Python module for compiling PyTorch graphs to C](https://github.com/lantiga/pytorch2c)

## Deploy TensorFlow Models in Production:
- [How to deploy Machine Learning models with TensorFlow - _Part1_](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-1-make-your-model-ready-for-serving-776a14ec3198)
- [How to deploy Machine Learning models with TensorFlow - _Part2_](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-2-containerize-it-db0ad7ca35a7)
- [How to deploy Machine Learning models with TensorFlow - _Part3_](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-3-into-the-cloud-7115ff774bb6)
- [Neural Structured Learning (NSL) in TensorFlow](https://medium.com/tensorflow/introducing-neural-structured-learning-in-tensorflow-5a802efd7afd) [_Great_]
- [Building Robust Production-Ready Deep Learning Vision Models](https://medium.com/google-developer-experts/building-robust-production-ready-deep-learning-vision-models-in-minutes-acd716f6450a)  
- [Creating REST API for TensorFlow models](https://becominghuman.ai/creating-restful-api-to-tensorflow-models-c5c57b692c10)
- ["How to Deploy a Tensorflow Model in Production" by _Siraj Raval_ on YouTube](https://www.youtube.com/watch?v=T_afaArR0E8)
- [Code for the "How to Deploy a Tensorflow Model in Production" by _Siraj Raval_ on YouTube](https://github.com/llSourcell/How-to-Deploy-a-Tensorflow-Model-in-Production)
- [How to deploy an Object Detection Model with TensorFlow serving](https://medium.freecodecamp.org/how-to-deploy-an-object-detection-model-with-tensorflow-serving-d6436e65d1d9) [_Very Good Tutorial_]
- [Freeze Tensorflow models and serve on web](http://cv-tricks.com/how-to/freeze-tensorflow-models/) [_Very Good Tutorial_]
- [How to deploy TensorFlow models to production using TF Serving](https://medium.freecodecamp.org/how-to-deploy-tensorflow-models-to-production-using-tf-serving-4b4b78d41700) [_Good_]
- [How Zendesk Serves TensorFlow Models in Production](https://medium.com/zendesk-engineering/how-zendesk-serves-tensorflow-models-in-production-751ee22f0f4b)
- [TensorFlow Serving Example Projects](https://github.com/Vetal1977/tf_serving_example)
- [Serving Models in Production with TensorFlow Serving](https://www.youtube.com/watch?v=q_IkJcPyNl0) [_TensorFlow Dev Summit 2017 Video_]
- [Building TensorFlow as a Standalone Project](https://tuatini.me/building-tensorflow-as-a-standalone-project/)
- [TensorFlow C++ API Example](https://github.com/jhjin/tensorflow-cpp)
- [TensorFlow.js](https://js.tensorflow.org/)
- [Introducing TensorFlow.js: Machine Learning in Javascript](https://medium.com/tensorflow/introducing-tensorflow-js-machine-learning-in-javascript-bf3eab376db)

## Convert Keras Models in Production:
- [Deep learning in production with Keras, Redis, Flask, and Apache](https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/) [_Rank: 1st & General Usefult Tutorial_]
- [Deploying a Keras Deep Learning Model as a Web Application in Python](https://towardsdatascience.com/deploying-a-keras-deep-learning-model-as-a-web-application-in-p-fc0f2354a7ff) [_Very Good_]
- [Deploying a Python Web App on AWS](https://towardsdatascience.com/deploying-a-python-web-app-on-aws-57ed772b2319) [_Very Good_]
- [Deploying Deep Learning Models Part 1: Preparing the Model](https://blog.paperspace.com/deploying-deep-learning-models-flask-web-python/)  
- [Deploying your Keras model](https://medium.com/@burgalon/deploying-your-keras-model-35648f9dc5fb)
- [Deploying your Keras model using Keras.JS](https://becominghuman.ai/deploying-your-keras-model-using-keras-js-2e5a29589ad8)
- ["How to Deploy a Keras Model to Production" by _Siraj Raval_ on Youtube](https://github.com/llSourcell/how_to_deploy_a_keras_model_to_production)
- [Deploy Keras Model with Flask as Web App in 10 Minutes](https://github.com/mtobeiyf/keras-flask-deploy-webapp) [Good Repository]
- [Deploying Keras Deep Learning Models with Flask](https://towardsdatascience.com/deploying-keras-deep-learning-models-with-flask-5da4181436a2)
- [keras2cpp](https://github.com/pplonski/keras2cpp)

## Deploy MXNet Models in Production:
- [Model Server for Apache MXNet](https://github.com/awslabs/mxnet-model-server)
- [Running the Model Server](https://github.com/awslabs/mxnet-model-server/blob/master/docs/server.md)
- [Multi Model Server (MMS) Documentation](https://github.com/awslabs/multi-model-server/tree/master/docs)  
- [Introducing Model Server for Apache MXNet](https://aws.amazon.com/blogs/machine-learning/introducing-model-server-for-apache-mxnet/)  
- [Single Shot Multi Object Detection Inference Service](https://github.com/awslabs/mxnet-model-server/blob/master/examples/ssd/README.md)
- [Amazon SageMaker](https://aws.amazon.com/sagemaker/)
- [How can we serve MXNet models built with gluon api](https://discuss.mxnet.io/t/how-can-we-serve-mxnet-models-built-with-gluon-api/684)
- [MXNet C++ Package](https://github.com/apache/incubator-mxnet/tree/master/cpp-package)
- [MXNet C++ Package Examples](https://github.com/apache/incubator-mxnet/tree/master/cpp-package/example)
- [MXNet Image Classification Example of C++](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification/predict-cpp)
- [MXNet C++ Tutorial](http://mxnet.incubator.apache.org/tutorials/c%2B%2B/basics.html)
- [An introduction to the MXNet API](https://becominghuman.ai/an-introduction-to-the-mxnet-api-part-1-848febdcf8ab) [Very Good Tutorial for Learning MXNet]
- [GluonCV](https://gluon-cv.mxnet.io/)
- [GluonNLP](http://gluon-nlp.mxnet.io/)
- [Model Quantization for Production-Level Neural Network Inference](https://medium.com/apache-mxnet/model-quantization-for-production-level-neural-network-inference-f54462ebba05) [_Excellent_]

## Deploy Machine Learning Models with Go:
- [Cortex: Deploy machine learning models in production](https://github.com/cortexlabs/cortex)  
- [Cortex - Main Page](https://www.cortex.dev/)  
- [Why we deploy machine learning models with Go — not Python](https://towardsdatascience.com/why-we-deploy-machine-learning-models-with-go-not-python-a4e35ec16deb)  
- [Go-Torch](https://github.com/orktes/go-torch)  
- [Gotch - Go API for PyTorch](https://github.com/sugarme/gotch)  
- [TensorFlow Go Lang](https://www.tensorflow.org/install/lang_go)  
- [Go-onnx](https://github.com/dhdanie/goonnx)  

## General Deep Learning Deployment Toolkits:
- [OpenVINO Toolkit - Deep Learning Deployment Toolkit repository](https://github.com/openvinotoolkit/openvino) [_Great_]   
- [ClearML - ML/DL development and production suite](https://github.com/allegroai/clearml)  
- [Model Deployment Using Heroku: A Complete Guide on Heroku](https://www.analyticsvidhya.com/blog/2021/10/a-complete-guide-on-machine-learning-model-deployment-using-heroku/) [Good]   
- [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) [**Great**]      
- [NVIDIA Triton Inference Server - GitHub](https://github.com/triton-inference-server/server) [**Great**]   
- [Cohere Boosts Inference Speed With NVIDIA Triton Inference Server](https://txt.cohere.ai/nvidia-boosts-inference-speed-with-cohere/)  
- [NVIDIA Deep Learning Examples for Tensor Cores](https://github.com/NVIDIA/DeepLearningExamples) [Interesting]  
- [Deploying the Jasper Inference model using Triton Inference Server](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechRecognition/Jasper/triton) [Useful]   
- [Nvidia MLOPs Course via Triton](https://analyticsindiamag.com/nvidia-is-offering-a-four-hour-self-paced-course-on-mlops/)  
- [Awesome Production Machine Learning](https://github.com/EthicalML/awesome-production-machine-learning) [Great]     

## Huawei Deep Learning Framework:
- [MindSpore - Huawei Deep Learning Framework](https://github.com/mindspore-ai/mindspore)  
- [MindSpore - Tutorial](https://www.mindspore.cn/tutorial/en/0.1.0-alpha/quick_start/quick_start.html)  

## General Deep Learning Compiler Stack:
- [TVM Stack](https://tvm.ai/)

## Model Conversion between Deep Learning Frameworks:
- [ONNX (Open Neural Network Exchange)](https://onnx.ai/)
- [Tutorials for using ONNX](https://github.com/onnx/tutorials)
- [MMdnn](https://github.com/Microsoft/MMdnn) [_Fantastic_]  
- [Convert Full ImageNet Pre-trained Model from MXNet to PyTorch](https://blog.paperspace.com/convert-full-imagenet-pre-trained-model-from-mxnet-to-pytorch/) [_Fantastic_, & Full ImageNet model means the model trained on ~ 14M images] 

## Some Caffe2 Tutorials:
- [Mnist using caffe2](http://vast.uccs.edu/~adhamija/blog/MNIST_singleGPU.html)
- [Caffe2 C++ Tutorials and Examples](https://github.com/leonardvandriel/caffe2_cpp_tutorial)
- [Make Transfer Learning of SqueezeNet on Caffe2](https://medium.com/@KazamiXHayato/make-transfer-learning-in-caffe2-21d96c47ba0e)
- [Build Basic program by using Caffe2 framework in C++](https://medium.com/@KazamiXHayato/write-caffe2-program-in-c-5519e2646382)

## Some Useful Resources for Designing UI (Front-End Development):
- [ReactJS vs Angular5 vs Vue.js](https://medium.com/@TechMagic/reactjs-vs-angular5-vs-vue-js-what-to-choose-in-2018-b91e028fa91d)
- [A comparison between Angular and React and their core languages](https://medium.freecodecamp.org/a-comparison-between-angular-and-react-and-their-core-languages-9de52f485a76)
- [A Guide to Becoming a Full-Stack Developer](https://medium.com/coderbyte/a-guide-to-becoming-a-full-stack-developer-in-2017-5c3c08a1600c) [_Very Good Tutorial_]
- [Roadmap to becoming a web developer in 2018](https://github.com/kamranahmedse/developer-roadmap) [_Very Good Repository_]
- [Modern Frontend Developer in 2018](https://medium.com/tech-tajawal/modern-frontend-developer-in-2018-4c2072fa2b9c)
- [Roadmap to becoming a React developer in 2018](https://github.com/adam-golab/react-developer-roadmap)
- [2019 UI and UX Design Trends](https://uxplanet.org/2019-ui-and-ux-design-trends-92dfa8323225) [_Good_]
- [Streamlit](https://streamlit.io/) [_The fastest way to build custom ML tools_]  
- [Gradio](https://www.gradio.app/) [**Good**]   
- [Web Developer Monthly](https://medium.com/@andreineagoie/web-developer-monthly-july-2018-513e02f15fb6)
- [23 Best React UI Component Frameworks](https://hackernoon.com/23-best-react-ui-component-libraries-and-frameworks-250a81b2ac42)
- [9 React Styled-Components UI Libraries for 2018](https://blog.bitsrc.io/9-react-styled-components-ui-libraries-for-2018-4e1a0bd3e179)
- [35 New Tools for UI Design](https://blog.prototypr.io/35-new-tools-for-ui-design-412cf1d701fd)
- [5 Tools To Speed Up Your App Development](https://medium.com/swlh/5-tools-to-speed-up-your-app-development-6979d0e49e34) [_Very Good_]
- [How to use ReactJS with Webpack 4, Babel 7, and Material Design](https://medium.freecodecamp.org/how-to-use-reactjs-with-webpack-4-babel-7-and-material-design-ff754586f618)
- [Adobe Typekit](https://typekit.com/) [_Great fonts, where you need them_]
- [Build A Real World Beautiful Web APP with Angular 6](https://medium.com/@hamedbaatour/build-a-real-world-beautiful-web-app-with-angular-6-a-to-z-ultimate-guide-2018-part-i-e121dd1d55e)
- [You Don't Know JS](https://github.com/getify/You-Dont-Know-JS)
- [JavaScript Top 10 Articles](https://medium.mybridge.co/javascript-top-10-articles-for-the-past-month-v-sep-2018-8f27a300d6c5)
- [Web Design with Adobe XD](https://medium.freecodecamp.org/a-developers-guide-to-web-design-for-non-designers-1f64ce28c38d)
- [INSPINIA Bootstrap Web Theme](https://wrapbootstrap.com/theme/inspinia-responsive-admin-theme-WB0R5L90S)
- [A Learning Tracker for Front-End Developers](https://github.com/Syknapse/My-Learning-Tracker-first-ten-months)
- [The best front-end hacking cheatsheets — all in one place](https://medium.freecodecamp.org/modern-frontend-hacking-cheatsheets-df9c2566c72a) [_Useful & Interesting_]
- [GUI-fying the Machine Learning Workflow (Machine Flow)](https://towardsdatascience.com/gui-fying-the-machine-learning-workflow-towards-rapid-discovery-of-viable-pipelines-cab2552c909f)
- [Electron - Build cross platform desktop apps with JavaScript](https://electronjs.org/) [_Very Good_]
- [Opyrator - Turns Python functions into microservices with web API](https://github.com/ml-tooling/opyrator) [**Great**]    
- [A First Look at PyScript: Python in the Web Browser](https://realpython.com/pyscript-python-in-browser/) [**Interesting**]  

## Mobile & Embedded Devices Development:
- [PyTorch Mobile](https://pytorch.org/mobile/home/) [_Excellent_]  
- [Mobile UI Design Trends In 2018](https://uxplanet.org/mobile-ui-design-trends-in-2018-ccd26031dfd8)  
- [ncnn - high-performance neural network inference framework optimized for the mobile platform](https://github.com/Tencent/ncnn) [_Useful_]  
- [Alibaba - MNN](https://github.com/alibaba/MNN)  
- [Awesome Mobile Machine Learning](https://github.com/fritzlabs/Awesome-Mobile-Machine-Learning)  
- [EMDL - Embedded and Mobile Deep Learning](https://github.com/EMDL/awesome-emdl)  
- [Fritz - machine learning platform for iOS and Android](https://www.fritz.ai/)  
- [TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/)  
- [Tiny Machine Learning: The Next AI Revolution](https://medium.com/@matthew_stewart/tiny-machine-learning-the-next-ai-revolution-495c26463868)  
- [TLT - NVIDIA Transfer Learning Toolkit](https://developer.nvidia.com/transfer-learning-toolkit)  
- [NVIDIA Jetson Inference](https://github.com/dusty-nv/jetson-inference)  [_Great_]

## Back-End Development Part:
- [Modern Backend Developer in 2018](https://medium.com/tech-tajawal/modern-backend-developer-in-2018-6b3f7b5f8b9)
- [Deploying frontend applications — the fun way](https://hackernoon.com/deploying-frontend-applications-the-fun-way-bc3f69e15331) [_Very Good_]
- [RabbitMQ](https://www.rabbitmq.com/) [_Message Broker Software_]
- [Celery](http://www.celeryproject.org/) [_Distributed Task Queue_]
- [Kafka](https://kafka.apache.org/) [_Distributed Streaming Platform_]
- [Docker training with DockerMe](https://github.com/AhmadRafiee/Docker_training_with_DockerMe)  
- [Kubernetes - GitHub](https://github.com/kubernetes/kubernetes)
- [Deploy Machine Learning Pipeline on Google Kubernetes Engine](https://towardsdatascience.com/deploy-machine-learning-model-on-google-kubernetes-engine-94daac85108b)  
- [An introduction to Kubernetes for Data Scientists](https://www.jeremyjordan.me/kubernetes/)  
- [Jenkins and Kubernetes with Docker Desktop](https://medium.com/@garunski/jenkins-and-kubernetes-with-docker-desktop-53a853486f7c)
- [Helm: The package manager for Kubernetes](https://helm.sh/)  
- [Create Cluster using docker swarm](https://medium.com/tech-tajawal/create-cluster-using-docker-swarm-94d7b2a10c43)  
- [deepo - Docker Image for all DL Framewors](https://github.com/ufoym/deepo)  
- [Kubeflow](https://www.kubeflow.org/)  [_deployments of ML workflows on Kubernetes_]  
- [kubespray - Deploy a Production Ready Kubernetes Cluster](https://github.com/kubernetes-sigs/kubespray)  
- [KFServing - Kubernetes for Serving ML Models](https://github.com/kubeflow/kfserving)  
- [Deploying a HuggingFace NLP Model with KFServing](http://www.pattersonconsultingtn.com/blog/deploying_huggingface_with_kfserving.html) [_Interesting_]  
- [Seldon Core - Deploying Machine Learning Models on Kubernetes](https://www.seldon.io/tech/products/core/)  
- [Seldon Core - GitHub](https://github.com/SeldonIO/seldon-core)  
- [Machine Learning: serving models with Kubeflow on Ubuntu, Part 1](https://ubuntu.com/blog/ml-serving-models-with-kubeflow-on-ubuntu-part-1)  
- [CoreWeave Kubernetes Cloud](https://github.com/coreweave/kubernetes-cloud/tree/master/online-inference/)  
- [MLOps References](https://github.com/visenger/mlops-references)  [_DevOps for ML_]
- [Data Version Control - DVC](https://dvc.org/)  [_Great_]  
- [MLEM: package and deploy machine learning models](https://github.com/iterative/mlem)  
- [PySyft - A library for encrypted, privacy preserving deep learning](https://github.com/OpenMined/PySyft)  
- [LocalStack - A fully functional local AWS cloud stack](https://github.com/localstack/localstack)  
- [poetry: Python packaging and dependency management](https://python-poetry.org/)  

## GPU Management Libraries:
- [GPUtil](https://github.com/anderskm/gputil)
- [py3nvml](https://github.com/fbcotter/py3nvml) [_Python 3 binding to the NVIDIA Management Library_]
- [PyCUDA - GitHub](https://github.com/inducer/pycuda)
- [PyCUDA](https://mathema.tician.de/software/pycuda/)
- [PyCUDA Tutorial](https://documen.tician.de/pycuda/)
- [setGPU](https://github.com/bamos/setGPU)
- [Monitor your GPUs](https://github.com/msalvaris/gpu_monitor) [**Excellent**]  
- [GPU-Burn - Multi-GPU CUDA stress test](https://github.com/wilicc/gpu-burn) [_Useful_]   
- [Grafana - Monitoring and Observability](https://github.com/grafana/grafana) [**Excellent**]  
- [Prometheus](https://prometheus.io/) [_Excellent for monitoring solution & extract required metrics_]  
- [OpenAI Triton: Open-Source GPU Programming for Neural Networks](https://openai.com/blog/triton/)  

## Speed-up & Scalabale Python Codes:
- [Numba - makes Python code fast](http://numba.pydata.org/)
- [Dask - natively scales Python](https://dask.org/)
- [What is Dask](https://medium.com/better-programming/what-is-dask-and-how-can-it-help-you-as-a-data-scientist-72adec7cec57)  
- [Ray - running distributed applications](https://github.com/ray-project/ray)  
- [Neural Network Distiller](https://github.com/NervanaSystems/distiller/) [_Distillation & Quantization of Deep Learning Models in PyTorch_]
- [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)  
- [Can you remove 99% of a neural network without losing accuracy? - An introduction to weight pruning](https://towardsdatascience.com/can-you-remove-99-of-a-neural-network-without-losing-accuracy-915b1fab873b)  
- [PocketFlow - An Automatic Model Compression (AutoMC) framework](https://github.com/Tencent/PocketFlow) [**Great**]  
- [Introducing the Model Optimization Toolkit for TensorFlow](https://medium.com/tensorflow/introducing-the-model-optimization-toolkit-for-tensorflow-254aca1ba0a3)  
- [TensorFlow Model Optimization Toolkit — Post-Training Integer Quantization](https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-post-training-integer-quantization-b4964a1ea9ba)  
- [TensorFlow Post-training Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)  
- [Dynamic Quantization in PyTorch](https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html)  
- [Static Quantization in PyTorch](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)  
- [NVIDIA DALI - highly optimized data pre-processing in deep learning](https://github.com/NVIDIA/dali)  
- [Horovod - Distributed training framework](https://github.com/horovod/horovod)  
- [ONNX Float32 to Float16](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/float32_float16_onnx.ipynb)  
- [Speeding Up Deep Learning Inference Using TensorRT](https://devblogs.nvidia.com/speeding-up-deep-learning-inference-using-tensorrt/)  
- [Speed up Training](https://ai.googleblog.com/2020/05/speeding-up-neural-network-training.html)  
- [Native PyTorch automatic mixed precision for faster training on NVIDIA GPUs](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/)  
- [JAX - Composable transformations of Python+NumPy programs](https://github.com/google/jax)  
- [TensorRTx - popular DL networks with tensorrt](https://github.com/wang-xinyu/tensorrtx)  
- [Speeding up Deep Learning Inference Using TensorFlow, ONNX, and TensorRT](https://devblogs.nvidia.com/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/)  
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)  
- [How to Convert a Model from PyTorch to TensorRT and Speed Up Inference](https://www.learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/) [_Good_]   

## Hardware Notes for Deep Learning:  
- [Hardware for Deep Learning](https://blog.inten.to/hardware-for-deep-learning-part-3-gpu-8906c1644664)  

## MLOPs Courses & Resources:  
- [MLOps-Basics](https://github.com/graviraja/MLOps-Basics) [Great]  
- [MLOPs-Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) [Great]   
- [A collection of resources to learn about MLOPs](https://github.com/dair-ai/MLOPs-Primer) [Great]  
- [Awesome MLOPs](https://github.com/visenger/awesome-mlops) [Great]  
- [Data Science Topics & MLOPs](https://github.com/khuyentran1401/Data-science#mlops) [Great]  
- [MLEM: package and deploy machine learning models](https://github.com/iterative/mlem)  
- [DevOps Exercises](https://github.com/bregman-arie/devops-exercises)  
- [MlOPs Sample Project](https://github.com/AntonisCSt/Mlops_project_semicon)  
- [prefect: Orchestrate and observe all of your workflows](https://www.prefect.io/)  
- [DataTalks Club: The place to talk about data](https://datatalks.club/)  

## Other:
- [A Guide to Production Level Deep Learning](https://github.com/alirezadir/Production-Level-Deep-Learning)  
- [Facebook Says Developers Will Love PyTorch 1.0](https://medium.com/syncedreview/facebook-says-developers-will-love-pytorch-1-0-ba2f89ebc9cc)
- [Some PyTorch Workflow Changes](https://github.com/pytorch/pytorch/issues/6032)
- [wandb - A tool for visualizing and tracking your machine learning experiments](https://github.com/wandb/client)  
- [PyTorch and Caffe2 repos getting closer together](https://github.com/caffe2/caffe2/issues/2439#issuecomment-391155017)
- [PyTorch or TensorFlow?](https://awni.github.io/pytorch-tensorflow/)
- [Choosing a Deep Learning Framework in 2018: Tensorflow or Pytorch?](http://cv-tricks.com/deep-learning-2/tensorflow-or-pytorch/)
- [Deep Learning War between PyTorch & TensorFlow](https://hub.packtpub.com/can-a-production-ready-pytorch-1-0-give-tensorflow-a-tough-time/)
- [Embedding Machine Learning Models to Web Apps (Part-1)](https://towardsdatascience.com/embedding-machine-learning-models-to-web-apps-part-1-6ab7b55ee428)
- [Deploying deep learning models: Part 1 an overview](https://towardsdatascience.com/deploying-deep-learning-models-part-1-an-overview-77b4d01dd6f7)
- [Machine Learning in Production](https://medium.com/contentsquare-engineering-blog/machine-learning-in-production-c53b43283ab1)
- [how you can get a 2–6x speed-up on your data pre-processing with Python](https://towardsdatascience.com/heres-how-you-can-get-a-2-6x-speed-up-on-your-data-pre-processing-with-python-847887e63be5)
- [Making your C library callable from Python](https://medium.com/@shamir.stav_83310/making-your-c-library-callable-from-python-by-wrapping-it-with-cython-b09db35012a3)
- [MIL WebDNN](https://mil-tokyo.github.io/webdnn/)
- [Multi-GPU Framework Comparisons](https://medium.com/@iliakarmanov/multi-gpu-rosetta-stone-d4fa96162986) [_Great_]  

