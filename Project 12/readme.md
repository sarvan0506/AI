fenwicks:

fenwicks provides necessary, easy-to-use utility functions and abstractions for DNN from simple tfRecord transformations to building complex model architectures and also provides efficient memory usage for TPU

##
import fenwicks as fw
##
- loading the fenwicks library

##
fw.colab_tpu.setup_gcs()
##
- Setting up Google Cloud Storage for TPU reference
- By specifying the TPU network address we can achieve this

##
data_dir, work_dir = fw.io.get_gcs_dirs(BUCKET, PROJECT)
##
- Setting up 
            
            data directory(data_dir): Since storing huge data in memory will be a problem for big datasets like cifar10 we need a directory in GCS to store the train and test data

            and

            working directory(work_dir): This location will be used to store intermediate files generated during training such as storing weights and all trainable parameters in the network.
##
train_fn = os.path.join(data_dir, "train.tfrec")
test_fn = os.path.join(data_dir, "test.tfrec"
##
- defining and including tfRecord file destination for training and test dataset

##
fw.io.numpy_tfrecord(X_train, y_train, train_fn)
fw.io.numpy_tfrecord(X_test, y_test, test_fn)
##
- numpy_tfrecord provides a simple solution to convert numpy data tp tfRecord data and writing it to the tfrec paths defined before
    
    TFRecords        --> TFRecord format is a simple format for storing a sequence of binary records.

    tf.Example       --> message (or protobuf) is a flexible message type that represents a {"string": value} mapping. It is designed for use                        with TensorFlow and is used throughout the higher-level APIs such as TFX.
    t.train.Features --> message type can accept one of the following three types (See the .proto file for reference).                                               Most other generic types can be coerced into one of these:

                         tf.train.BytesList (the following types can be coerced)

                            string
                            byte
                        
                         tf.train.FloatList (the following types can be coerced)

                            float (float32)
                            double (float64)

                         tf.train.Int64List (the following types can be coerced)

                            bool
                            enum
                            int32
                            uint32
                            int64
                            uint64
    
    - numpy_tfrecord does
        
        1. Converting all the numpy array values to tf.train.Features of type float
        2. Writing the converted values to the tfrec destination using TFRecordWriter

##
def parser_train(tfexample):
 x, y = fw.io.tfexample_numpy_image_parser(tfexample, img_size,
 img_size)
 x = fw.transform.ramdom_pad_crop(x, 4)
 x = fw.transform.random_flip(x)
 x = fw.transform.cutout(x, 8, 8)
return x, y

parser_test = lambda x: fw.io.tfexample_numpy_image_parser(x, img_size, img_size)
##
- parser_train() is a pipeline for DavidNet Data Pre-Processing
- Pipeline operations
        1. fw.io.tfexample_numpy_image_parser := Expand the samples to [h, w, c] format
        2. fw.transform.ramdom_pad_crop := Randomly pad the image by `pad_size` at each border (top, bottom, left, right). Then, crop the padded image to its original size. This is used to prepare the dataset for a cut-out region of h * w.
        3. fw.transform.random_flip := Used to flip an image horizontally. The flip parameters allows to determine probability of the number of images that needs this transformation. Vetical flip is optional
        4  fw.transform.cutout := Finally implementing cut-out for better generalization of the model. Randomly cuts a h by w whole in the image, and fill the whole with zeros.

##
train_input_func = lambda params: fw.io.tfrecord_ds(train_fn, parser_train, batch_size=params['batch_size'], training=True)
eval_input_func = lambda params: fw.io.tfrecord_ds(test_fn, parser_test, batch_size=params['batch_size'], training=False)
##
- fw.io.tfrecords_ds := this function fetches data from tfrecords and apply the parser efficiently using dataset.interleave(). This also takes                          care of prefetch operation wherein the next batch of records are parallelly fetched and applied the necessary                                   transformations while TPU is busy.

##
def build_nn(c=64, weight=0.125):
 model = fw.Sequential()
 model.add(fw.layers.ConvBN(c, **fw.layers.PYTORCH_CONV_PARAMS))
 model.add(fw.layers.ConvResBlk(c*2, res_convs=2,
 **fw.layers.PYTORCH_CONV_PARAMS))
 model.add(fw.layers.ConvBlk(c*4, **fw.layers.PYTORCH_CONV_PARAMS))
 model.add(fw.layers.ConvResBlk(c*8, res_convs=2,
 **fw.layers.PYTORCH_CONV_PARAMS))
 model.add(tf.keras.layers.GlobalMaxPool2D())
 model.add(fw.layers.Classifier(n_classes, 
 kernel_initializer=fw.layers.init_pytorch, weight=0.125))
return model
##
- This function builds a custom network with the flexibility of setting the initial weights. We can achieve the necessary ResNet architecture using functions like

        1. fw.layers.ConvBN --> A Conv2D followed by BatchNormalization and ReLU activation.
        2. fw.layers.ConvBlk --> A block of `ConvBN` layers, followed by a pooling layer.
        3. fw.layers.ConvResBlk --> A `ConvBlk` with additional residual `ConvBN` layers.

##
lr_func = fw.train.triangular_lr(LEARNING_RATE/BATCH_SIZE, total_steps, warmup_steps=WARMUP*steps_per_epoch)
fw.plt.plot_lr_func(lr_func, total_steps
##
- This implements cyclic-learning rate for faster convergence
        
        One cycle learning rate schedule.
        :param init_lr: peak learning rate.
        :param total_steps: total number of training steps.
        :param warmup_steps: number of steps in the warmup phase, during which the learning rate increases linearly.
        :param decay_sched: learning rate decay function.
        :return: learning rate schedule function satisfying the above descriptions.


##
opt_func = fw.train.sgd_optimizer(lr_func, mom=MOMENTUM, wd=WEIGHT_DECAY*BATCH_SIZE)
model_func = fw.tpuest.get_clf_model_func(build_nn, opt_func, reduction=tf.losses.Reduction.SUM)
##
- building the SGD optimizer and model function for TPUEstimator
        1. initializing the SGD with the learning function generated previously
        2. Attach the optimizer to the model built for classification purpose.

##
est = fw.tpuest.get_tpu_estimator(n_train, n_test, model_func, work_dir, trn_bs=BATCH_SIZE)
est.train(train_input_func, steps=total_steps)
##
- initializing a tpu estimator with the custom build model for training and train.

This way the fw package can help in simplifying the essential operations for achieving the best training time on any model run on TPU


#####################################################################

David Page

Objective - Training a ResNet(18 layer) on cifar10 to state-of-art accuracy of 94% under 100s(DAWNBench 341s) using single GPU

Steps:

1. Baseline: Improving the state-of-art model performance from Baseline by
             1. Removing consecutive duplicate BN-ReLU blocks.
             2. Fixing learning function by smoothing sudden climbs in the learning graph.
             3. Preprocessing jobs like Padding, Normalizing and transpositions are done once before training using PyTorch Dataloader
             4. Making bulk calls to random number generator used for data augumentation

             reduced training time: 297s

2. Mini-batches:
             1. Increasing the batch size from 128 to 512
             2. Increasing the learning rate by 10%(emphirical)

             reduced training time: 256s

3. Regularization:
             1. Setting BN weights to single precision
             2. Reducing learning rate schedule to 30 epochs
             3. Manual optimization of learning rate schedule to achieve the peak faster and a linear decay from there
             4. momentum = 0.9 and weight decay = 5e-4

             reduced training time: 154s

4. Network Architecture:
             1. 9 layer architecture
             2. 2 residual blocks in Layer1 and Layer2
             3. epochs 24

             reduced training time: 79s

5. Hyperparameters:
             1. batch size = 512
             2. 1-momentum = 0.9
             3. weight decay = 5e-4

             reduced training time: has no significant improvement by tuning from baseline. More of a study to understand hyperparameter dynamics

6. Weight Decay: Comparing LARS with SGD+Momentum with Weight Decay. Conlcluding with the later as tuning along this line does not have any                      considerable significance

7. Batch Normalization: Forming a hypothesis around Batch Normalization and proving that BN with First Order Optimizer like SGD is best way to                          go

8. Bag of Tricks:
             1. Moving Data Processing to GPU(70s)
             2. Moving MaxPooling Layer before BN & Activation(64s)
             3. Applying Label Smoothing(59s)
             4. Replacing ReLU with CELU activation for smoothing(52s)
             5. Implementing Ghost Batch Normalization(46s)
             6. Implementing Frozen Batch Norm by fixing scale and bias(43s)
             7. Input Patch Whitening(36s)
             8. Adding Exponential Moving Average to Learing Rate Schedule(34s)
             9. Test-Time Augumentation(26s)

             Reduced training time: accuracy-94.1, time-26s, epochs-10