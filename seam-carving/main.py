from seam_carving import SeamCarver


IMAGE_FILE = 'input.jpg'


def scale_square_example():
    sc = SeamCarver(IMAGE_FILE)
    image = sc.resize(500, 500)
    image.save("seam_carved.jpg")

if __name__ == '__main__':
    scale_square_example()
