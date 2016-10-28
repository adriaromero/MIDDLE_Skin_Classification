import os
import skimage
from skimage import data, draw
from skimage import transform, util
import numpy as np
from skimage import filters, color
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu

# NOTE:  This script runs for about 45 seconds for a single image



############################################################# TEST CONSTANTS
# test image for sample
file_path = './input_test_center.jpg'

# desired size
x_axis = 500
y_axis = 500


############################################################# Utility functions
# read an image from a file path and return the image object
def return_image_from_file(file_path):
    img_original = skimage.io.imread(file_path)
    return img_original


# create a directory if it does not already exist
def ensure_directory_exist(directory_name):
    exist_bool = os.path.isdir('./' + directory_name)
    if not exist_bool:
        os.mkdir(directory_name)


def return_image_x_y(image_current):
    height, width, channels = image_current.shape
    return height,width

############################################################# [END] Utility functions


############################################################# Crop length helper

# determine if x is the longest axis
# Input:
#   - both axis lengths
# Output:
#   - [Bool]: True if x is the longest axis
def is_x_longest_edge(image_size_y, image_size_x):
    if image_size_x >= image_size_y:
        return True
    else:
        return False


# returns the length to remove from the longest side
# Input:
#   - bool = True if x-axis is longer
#   - both axis lengths
# Output:
#   - length to remove from side
def get_length_to_remove(x_longer, image_size_x, image_size_y):
    if x_longer:
        remove_to_square_length = image_size_x - image_size_y
    else:
        remove_to_square_length = image_size_y - image_size_x
    return remove_to_square_length

# determine the number of pixels(as col or row) to remove from an image
# if the full lenth is an odd value, one side will receive an extra pixel to ensure
# that the final size is the expected full pixel value
def determine_remove_values(full_length):
    # when even, we can remove the same amount from each side
    if full_length % 2 == 0:
        remove_a = int(full_length/2)
        remove_b = int(full_length/2)
    # when odd, we can have to pick a side to remove an extra pixel from
    # currently, the end side is selected
    else:
        remove_a = int(full_length/2)
        remove_b = int(full_length/2) + 1

    return remove_a, remove_b

############################################################# [END]Crop length helper


############################################################# Photo manipulations

# crop the image according to input specs
def crop_image(image_current, image_size_x, image_size_y, remove_to_square_length, x_longer):
    x_width = image_size_x
    y_width = image_size_y

    if x_longer:
        x_crop_length = remove_to_square_length
        y_crop_length = 0
    else:
        x_crop_length = 0
        y_crop_length = remove_to_square_length

    x_remove_start, x_remove_end = determine_remove_values(x_crop_length)
    y_remove_start, y_remove_end = determine_remove_values(y_crop_length)

    # set pixel start and end
    x_start = 0 + x_remove_start
    x_end = x_width - x_remove_end
    y_start = 0 + y_remove_start
    y_end = y_width - y_remove_end

    # perform crop of the image
    cropped_image = image_current[y_start:y_end,x_start:x_end] # this is opposite what is expected, but works
    plot_image_save_to_file("standard_cropped_image",cropped_image)
    return cropped_image


# resize image, the resize uses the values provided on the global level
def resize_image_to_global_spec(image_current):
    img_resized = transform.resize(image_current, (x_axis, y_axis))
    return img_resized


# seam carve image using sobel edge
def scarve_edge_image(image_current, size, x_longer):
    edges = filters.sobel(color.rgb2gray(image_current))
    # vertical=decrease width, horizontal=decrease height
    plot_image_save_to_file("scarve_edge", edges)
    if x_longer:
        img_seam_carved = transform.seam_carve(image_current, edges, 'vertical', size)
    else:
        img_seam_carved = transform.seam_carve(image_current, edges, 'horizontal', size)

    return img_seam_carved


# seam carve image using otsu global (experimental)
def scarve_otsu_global_image(image_current, size, x_longer):
    # perform global otsu
    gray_image = color.rgb2gray(image_current)
    threshold_global_otsu = threshold_otsu(gray_image)
    # print(threshold_global_otsu)
    global_otsu = gray_image >= threshold_global_otsu # need to make lesion the 'important values'
    # global_otsu = [1 for pixel in image_current if pixel>=threshold_global_otsu]
    plot_image_save_to_file("scarve_global_otsu", global_otsu)
    if x_longer:
        img_otsu_carved = transform.seam_carve(image_current, global_otsu, 'vertical', size)
    else:
        img_otsu_carved = transform.seam_carve(image_current, global_otsu, 'horizontal', size)
    return img_otsu_carved

############################################################# [END] Photo manipulations

############################################################# Plot figures
# --------- create sample image for specifed input image
# Input:
#   - images to plot
# Output:
#   - figure with all images (currently a 3x2)
def plot_all_images(img_original, img_resized, img_scarve_resize, img_crop_resize, img_scarve_otsu_global_resize):
    fig = plt.figure()
    # fig.suptitle("Test Different Resizing Methods", fontsize=14)

    # original
    fig_1 = fig.add_subplot(321)
    fig_1.set_title("Original", fontsize=10)
    fig_1.imshow(img_original)

    # resize
    fig_2 = fig.add_subplot(322)
    fig_2.set_title("Resize(Brute Force)", fontsize=10)
    fig_2.imshow(img_resized)

    # scarve
    fig_3 = fig.add_subplot(323)
    fig_3.set_title("Scarve(Sobel Edge) + Resize", fontsize=10)
    fig_3.imshow(img_scarve_resize)

    fig_4 = fig.add_subplot(324)
    fig_4.set_title("Scarve(Otsu_Global) + Resize", fontsize=10)
    fig_4.imshow(img_scarve_otsu_global_resize)

    # crop
    fig_5 = fig.add_subplot(325)
    fig_5.set_title("Crop(center on long) + Resize", fontsize=10)
    fig_5.imshow(img_crop_resize)

    # save the figure
    sample_photo_path = 'current_image_sample.png'  # currently set for current dir
    fig.tight_layout()  # will give warning, but adds padding to the figure
    fig.savefig(sample_photo_path)


# --------- save a 'named' individual photo to the sampleImages dir
def plot_image_save_to_file(name, img_cur):
    # make the figure
    plt.figure()
    plt.title(name)
    plt.imshow(img_cur)

    # ensure a directory is present/build if necessary
    save_directory = "sampleImages"
    ensure_directory_exist(save_directory)

    # build full path and save
    file_name = name + '.png'
    full_path = os.path.join(save_directory, file_name)
    plt.savefig(full_path)

#############################################################[END] Plot figures


# Main outside wrapper
def main():

    # read in image
    img_original = return_image_from_file(file_path)

    # get image information
    image_size_y, image_size_x = return_image_x_y(img_original)
    x_longer = is_x_longest_edge(image_size_y, image_size_x)   # if true, then x=longer
    remove_to_square_length = get_length_to_remove(x_longer, image_size_x, image_size_y)

    # --- "original image"
    plot_image_save_to_file("original", img_original)

    # ------ resized image
    img_resized = resize_image_to_global_spec(img_original)
    plot_image_save_to_file("classic_resized", img_resized)

    # ------ scarve otsu + resized image
    img_scarve_otsu_global_resize = scarve_otsu_global_image(img_original, remove_to_square_length, x_longer)
    img_scarve_otsu_global_resize = resize_image_to_global_spec(img_scarve_otsu_global_resize)
    plot_image_save_to_file("scarve_otsu_resized", img_scarve_otsu_global_resize)

    # ------ scarved (edges) + resized image
    img_scarve_resize = scarve_edge_image(img_original, remove_to_square_length, x_longer)
    img_scarve_resize = resize_image_to_global_spec(img_scarve_resize)
    plot_image_save_to_file("scarve_edge_resized", img_scarve_resize)

    # ------ crop + resized image
    img_crop_resize = crop_image(img_original, image_size_x, image_size_y, remove_to_square_length, x_longer)
    img_crop_resize = resize_image_to_global_spec(img_crop_resize)
    plot_image_save_to_file("crop_resized", img_crop_resize)

    # plot all images in the same figure for easy viewing
    plot_all_images(img_original, img_resized, img_scarve_resize, img_crop_resize, img_scarve_otsu_global_resize)

main()
