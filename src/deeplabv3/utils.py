import numpy as np

# sets labels for classes, for now it should just be bg vs foreground
def set_class_labels(all_classes, classes_to_train):
    class_values = [all_classes.index(cls.lower()) for cls in classes_to_train]
    return class_values


# segmentation mask is just a value for each pixel, representing which
# class it belongs to
def get_label_mask(mask, class_values, label_colors_list):
    """
    This function encodes the pixels belonging to the same class
    in the image into the same label

    :param mask: NumPy array, segmentation mask.
    :param class_values: List containing class values, e.g background = 0 receipt = 1
    :param label_colors_list: List containing RGB color value for each class.
    """
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for value in class_values:
        for ii, label in enumerate(label_colors_list):
            if value == label_colors_list.index(label):
                label = np.array(label)
                rows, cols = np.where(np.all(mask == label, axis=-1))
                label_mask[rows, cols] = value
                # label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = value
    label_mask = label_mask.astype(int)
    return label_mask
