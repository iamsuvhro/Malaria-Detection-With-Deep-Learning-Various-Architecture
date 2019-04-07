global_start=dt.now()

#Dimensions of our flicker images is 256 X 256
img_width, img_height = 256, 256

#Declaration of parameters needed for training and validation
train_data_dir = 'cell_images/train'
validation_data_dir = 'cell_images/test'
epochs = 40
batch_size = 16

#Get the bottleneck features by  Weights.T * Xi
def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1./255)

    #Load the pre trained VGG16 model from Keras, we will initialize only the convolution layers and ignore the top layers.
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator_tr = datagen.flow_from_directory(train_data_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            class_mode=None, #class_mode=None means the generator won't load the class labels.
                                            shuffle=False) #We won't shuffle the data, because we want the class labels to stay in order.
    nb_train_samples = len(generator_tr.filenames) #3600. 1200 training samples for each class
    bottleneck_features_train = model.predict_generator(generator_tr, nb_train_samples // batch_size)
    np.save('weights/bottleneck_features_train.npy',bottleneck_features_train) #bottleneck_features_train is a numpy array

    generator_ts = datagen.flow_from_directory(validation_data_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            class_mode=None,
                                            shuffle=False)
    nb_validation_samples = len(generator_ts.filenames) #1200. 400 training samples for each class
    bottleneck_features_validation = model.predict_generator(generator_ts, nb_validation_samples // batch_size)
    np.save('weights/bottleneck_features_validation.npy',bottleneck_features_validation)
    print("Got the bottleneck features in time: ",dt.now()-global_start)
    
    num_classes = len(generator_tr.class_indices)
    
    return nb_train_samples,nb_validation_samples,num_classes,generator_tr,generator_ts
    
nb_train_samples,nb_validation_samples,num_classes,generator_tr,generator_ts=save_bottlebeck_features()
