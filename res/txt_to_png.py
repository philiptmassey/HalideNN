import sys
import numpy
from PIL import Image

OFFSET = 10000000

def float_to_tuple(value):
    positive = value > 0
    adjusted = int(OFFSET * abs(value))
    R = (adjusted & 0xFF000000) >> 24
    G = ((adjusted & 0xFF0000) >> 16) & 0xFF
    B = ((adjusted & 0xFF00) >> 8) & 0xFF
    A = adjusted & 0xFF

    # The last bit represents the sign of the number
    # 1 means positive, 0 means negative
    if positive:
        A = A | 1
    else:
        A = (A >> 1) << 1

    return (R, G, B, A)

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print "Usage: python txt_to_png.py <input filename> <output_filename>"

    image = []
    with open(sys.argv[1], 'r') as inputfile:
        for line in inputfile:
            image.append([float(x) for x in line.split()])

    im = [[float_to_tuple(y) for y in x] for x in image]
    im = numpy.asarray(im, dtype=numpy.uint8)
    im = Image.fromarray(im)
    im.save(sys.argv[2])
