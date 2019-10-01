# Import ??


def deep_feature_extraction(img):
    """
    extract feature from images using transfer learning.
    :img string of image name
    
    feature: np.array of dim 1x25088
    """
    
    try:
        img = Image.open(img)
        target_size = (224, 224)
        img = img.resize(target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        model = VGG16(weights='imagenet', include_top=False)
        preds = model.predict(x)
        features = preds.reshape((preds.shape[0], 7 * 7 * 512))
        return features
    except FileNotFoundError:
        print("missing picture number " + img)
        pass