## TO-DO
 - [ ] Utils: Random checks, utilities should be added here.
 - [ ] Utils: Separate data utilities from data, config parser, etc. and add it under a subsection in utils. OR call them helpers and utilities can be generic stuff.
 - [ ] Utils: Either helpers/utilities can manage yolov3 code related stuff here.
 - [ ] Data handlers need to be added. Procedure providing multiple options to load data. TFRecords/Arrangements as images/... labels/... .
 - [ ] PreTrained Models: Feature to load pretrained models created from the repository itself or darknet pretrained models.
 - [ ] PreTrained Models: Manage leavy_conv and conv namespaces in Tensorflow SavedModel for loading a pretrained model.
 - [ ] PreTrained Models: Capability to load darknet yolov3 trained models
 - [ ] Config: Capability to read darknet config
 - [ ] Config: Needs to be more easily reusable. Probably using protobufs
 - [ ] Capability to design backbones
 - [ ] Modularize backbones, anchors, loss layer, etc.
 - [ ] Separation for train, eval, test and predict.
 - [ ] Handle CPU/GPU/multi-GPU training
 - [ ] Data: Add Augmentations.
 - [ ] Modularize: Create a newly inherited Conv2D layer for implementation
   purposes. With this layer you can sequentially implement it as Conv2d,
   followed by a BN/Activation as it pleases and also leaky relu is not an
   option and forces this kind of implementation. Some references"
    - [BatchNorm after RELU](https://github.com/gcr/torch-residual-networks/issues/5)
    - [Where do I call BatchNorm in Keras - StackOverflow](https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras)
    - [FCHOLLET enlightening us on BN application. Basically it's a little irrelevant to him](https://github.com/keras-team/keras/issues/1802)
 - [ ] TensorRT: An optimization idea would be folding batch norm params into
   the Conv2D to increase FPS. FPS LOVER ALL THE WAY!
 - [ ] Prediction/Evaluation: Add NMS Implementation.

 ## PROGRESS
 - Keeping Protobufs aside. But, currently what can be done is a label map like implementation, 
   with each layer type (Convolution, Route, etc.) associated with an id and an input. 
   So that when we write a .pbtxt for defining a network, we can define a layer, it's properties (args), 
   associate it with an id, an input (the id of a layer who's outputs are going in as inputs) to the
   current layer.
 - Fixing Residual Blocks for Now.
 - Using batch normalization after convolution blocks makes sense. Because,
   assuming images are provided as normalized as inputs to a network,
   (considering input to be a layer) it means each layer's output is
   normalized. And going by that, batch norm is applied at the output of each
   layer.
 - Inherit the Conv2D layer from keras and create a new one called
   DarknetConv2D because of the use of LeakyRelu and providing flexibility
