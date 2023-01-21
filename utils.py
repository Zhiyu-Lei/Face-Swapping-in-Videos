import numpy as np
import cv2
import dlib
from scipy import signal
from scipy.spatial import Delaunay
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

hogFaceDetector = dlib.get_frontal_face_detector()
facePredictor = dlib.shape_predictor('code/shape_predictor_68_face_landmarks.dat')


def interp2(v, xq, yq):
    dim_input = 1
    if len(xq.shape) == 2 or len(yq.shape) == 2:
        dim_input = 2
        q_h = xq.shape[0]
        q_w = xq.shape[1]
        xq = xq.flatten()
        yq = yq.flatten()

    h = v.shape[0]
    w = v.shape[1]
    if xq.shape != yq.shape:
        raise 'query coordinates Xq Yq should have same shape'

    x_floor = np.floor(xq).astype(np.int32)
    y_floor = np.floor(yq).astype(np.int32)
    x_ceil = np.ceil(xq).astype(np.int32)
    y_ceil = np.ceil(yq).astype(np.int32)

    x_floor[x_floor < 0] = 0
    y_floor[y_floor < 0] = 0
    x_ceil[x_ceil < 0] = 0
    y_ceil[y_ceil < 0] = 0

    x_floor[x_floor >= w-1] = w-1
    y_floor[y_floor >= h-1] = h-1
    x_ceil[x_ceil >= w-1] = w-1
    y_ceil[y_ceil >= h-1] = h-1

    v1 = v[y_floor, x_floor]
    v2 = v[y_floor, x_ceil]
    v3 = v[y_ceil, x_floor]
    v4 = v[y_ceil, x_ceil]

    lh = yq - y_floor
    lw = xq - x_floor
    hh = 1 - lh
    hw = 1 - lw

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw

    interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

    if dim_input == 2:
        return interp_val.reshape(q_h, q_w)
    return interp_val


def matrixABC(sparse_control_points, elements):
    output = np.zeros((3, 3))
    for i, element in enumerate(elements):
        output[0:2, i] = sparse_control_points[element, :]
    output[2, :] = 1
    return output


def get_mask(size_H, size_W, target_pts):
    Tri = Delaunay(target_pts)
    x, y = np.meshgrid(np.arange(size_W), np.arange(size_H))
    x = x.reshape(-1)
    y = y.reshape(-1)
    simplices = Tri.find_simplex(np.array(list(zip(x, y))))
    mask = (simplices != -1)
    return mask.reshape((size_H, size_W))


def corner_points(image, rect):
    size_H = image.shape[0]
    size_W = image.shape[1]
    left = max(rect[0] - (rect[2] - rect[0]) // 4, 0)
    right = min(rect[2] + (rect[2] - rect[0]) // 4, size_W - 1)
    top = max(rect[1] - (rect[3] - rect[1]) // 4, 0)
    bottom = min(rect[3] + (rect[3] - rect[1]) // 4, size_H - 1)
    return np.array([[left, top], [left, bottom], [right, top], [right, bottom]])


def generate_warp(size_H, size_W, Tri, ABC_Inter_inv_set, ABC_im_set, image):
    # Generate x,y meshgrid
    x, y = np.meshgrid(np.arange(size_W), np.arange(size_H))
    x = x.reshape(-1)
    y = y.reshape(-1)

    # Zip the flattened x, y and Find Simplices
    simplices = Tri.find_simplex(np.array(list(zip(x, y))))

    # Filter out outside pixels
    x = x[simplices != -1]
    y = y[simplices != -1]
    simplices = simplices[simplices != -1]

    # Compute alpha, beta, gamma for all the color layers(3)
    abg = np.matmul(ABC_Inter_inv_set[simplices], np.dstack((x, y, np.ones(simplices.shape))).reshape((-1, 3, 1)))

    # Find all x and y coordinates
    xy = np.matmul(ABC_im_set[simplices], abg)

    # Generate Warped Images (Use function interp2) for each of 3 layers
    generated_pic = np.zeros((size_H, size_W, 3), dtype=np.uint8)
    for i in range(3):
        generated_pic[y, x, i] = interp2(image[:, :, i], xy[:, 0, 0], xy[:, 1, 0])

    return generated_pic


def ImageMorphingTriangulation(source, target, source_rect, source_pts, target_rect, target_pts):
    # compute the H,W of the target image
    size_H = target.shape[0]
    size_W = target.shape[1]

    # compute mask
    mask = get_mask(size_H, size_W, target_pts)

    # add corner points
    source_pts = np.vstack((source_pts, corner_points(source, source_rect)))
    target_pts = np.vstack((target_pts, corner_points(target, target_rect)))

    # create a triangulation of the target points
    Tri = Delaunay(target_pts)

    # No. of Triangles
    nTri = Tri.simplices.shape[0]

    # Initialize the Triangle Matrices for all the triangles in image
    ABC_Inter_inv_set = np.zeros((nTri, 3, 3))
    ABC_source_set = np.zeros((nTri, 3, 3))

    for ii, element in enumerate(Tri.simplices):
        ABC_Inter_inv_set[ii, :, :] = np.linalg.inv(matrixABC(target_pts, element))
        ABC_source_set[ii, :, :] = matrixABC(source_pts, element)

    assert ABC_Inter_inv_set.shape[0] == nTri

    # generate warp pictures for each of the two images
    warp_source = generate_warp(size_H, size_W, Tri, ABC_Inter_inv_set, ABC_source_set, source)
    warp_source = warp_source.astype(np.uint8)

    return warp_source, mask


def getIndexes(mask):
    maskH, maskW = mask.shape
    x, y = np.meshgrid(np.arange(maskW), np.arange(maskH))
    x = x[mask]
    y = y[mask]
    indexes = np.zeros(mask.shape)
    indexes[y, x] = np.arange(1, x.shape[0] + 1)
    return indexes


def getCoefficientMatrix(indexes):
    N = indexes.max().astype(int)
    A = sparse.csr_matrix((np.repeat(4, N), (np.arange(N), np.arange(N))), shape=(N, N))

    indexesH, indexesW = indexes.shape
    indexesX, indexesY = np.meshgrid(np.arange(indexesW), np.arange(indexesH))
    x = indexesX[indexes > 0]
    y = indexesY[indexes > 0]
    y_neighbor = np.arange(N)

    # Left neighbors
    left_neighbor = indexes[y, x - 1]
    left_y_neighbor = y_neighbor[left_neighbor > 0]
    left_x_neighbor = (left_neighbor[left_y_neighbor] - 1).astype(int)
    A[left_y_neighbor, left_x_neighbor] = -1

    # Right neighbors
    right_neighbor = indexes[y, x + 1]
    right_y_neighbor = y_neighbor[right_neighbor > 0]
    right_x_neighbor = (right_neighbor[right_y_neighbor] - 1).astype(int)
    A[right_y_neighbor, right_x_neighbor] = -1

    # Up neighbors
    up_neighbor = indexes[y - 1, x]
    up_y_neighbor = y_neighbor[up_neighbor > 0]
    up_x_neighbor = (up_neighbor[up_y_neighbor] - 1).astype(int)
    A[up_y_neighbor, up_x_neighbor] = -1

    # Down neighbors
    down_neighbor = indexes[y + 1, x]
    down_y_neighbor = y_neighbor[down_neighbor > 0]
    down_x_neighbor = (down_neighbor[down_y_neighbor] - 1).astype(int)
    A[down_y_neighbor, down_x_neighbor] = -1

    return A


def getSolutionVect(indexes, source, target):
    # 1. get Laplacian part of b from source image
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    source_laplacian = signal.convolve2d(source, laplacian, "same")

    indexesH, indexesW = indexes.shape
    indexesX, indexesY = np.meshgrid(np.arange(indexesW), np.arange(indexesH))
    x = indexesX[indexes > 0]
    y = indexesY[indexes > 0]

    laplacian_b = source_laplacian[y, x]

    # 2. get pixel part of b from target image
    y_b = np.arange(x.shape[0])

    # Left neighbors
    left_neighbor = indexes[y, x - 1]
    left_y_zero = y[left_neighbor == 0]
    left_x_zero = (x - 1)[left_neighbor == 0]
    left_val = target[left_y_zero, left_x_zero]
    left_b = np.zeros(x.shape[0])
    left_b[y_b[left_neighbor == 0]] = left_val

    # Right neighbors
    right_neighbor = indexes[y, x + 1]
    right_y_zero = y[right_neighbor == 0]
    right_x_zero = (x + 1)[right_neighbor == 0]
    right_val = target[right_y_zero, right_x_zero]
    right_b = np.zeros(x.shape[0])
    right_b[y_b[right_neighbor == 0]] = right_val

    # Up neighbors
    up_neighbor = indexes[y - 1, x]
    up_y_zero = (y - 1)[up_neighbor == 0]
    up_x_zero = x[up_neighbor == 0]
    up_val = target[up_y_zero, up_x_zero]
    up_b = np.zeros(x.shape[0])
    up_b[y_b[up_neighbor == 0]] = up_val

    # Down neighbors
    down_neighbor = indexes[y + 1, x]
    down_y_zero = (y + 1)[down_neighbor == 0]
    down_x_zero = x[down_neighbor == 0]
    down_val = target[down_y_zero, down_x_zero]
    down_b = np.zeros(x.shape[0])
    down_b[y_b[down_neighbor == 0]] = down_val

    # add two parts together to get b
    b = laplacian_b + left_b + right_b + up_b + down_b

    return b


def reconstructImg(indexes, blue, green, red, target):
    # get nonzero component in indexes
    indexesH, indexesW = indexes.shape
    indexesX, indexesY = np.meshgrid(np.arange(indexesW), np.arange(indexesH))
    x = indexesX[indexes > 0]
    y = indexesY[indexes > 0]

    # stack three channels together with numpy dstack
    pixels = np.dstack((blue, green, red))

    # copy new pixels in the indexes area to the target image
    resultImg = target
    resultImg[y, x] = pixels

    return resultImg


def seamlessCloningPoisson(sourceImg, targetImg, mask):
    # index replacement pixels
    indexes = getIndexes(mask)

    # compute the Laplacian matrix A
    A = getCoefficientMatrix(indexes)

    # for each color channel, compute the solution vector b
    blue, green, red = [
        getSolutionVect(indexes, sourceImg[:, :, i], targetImg[:, :, i]).T for i in range(3)
    ]

    # solve for the equation Ax = b to get the new pixels in the replacement area
    new_blue, new_green, new_red = [
        spsolve(A, channel) for channel in [blue, green, red]
    ]

    # reconstruct the image with new color channel
    resultImg = reconstructImg(indexes, new_blue, new_green, new_red, targetImg)
    return resultImg


def skin_color_adjustment(im1, im2, mask=None):
    """
    color adjustment
    :param im1: image1
    :param im2: image2
    :param mask: fase mask. if exists, substitute with average color, else, gaussian blur
    :return: image1 with im2's color
    """
    if mask is None:
        im1_ksize = 55
        im2_ksize = 55
        im1_factor = cv2.GaussianBlur(im1, (im1_ksize, im1_ksize), 0).astype(np.float)
        im2_factor = cv2.GaussianBlur(im2, (im2_ksize, im2_ksize), 0).astype(np.float)
    else:
        im1_face_image = cv2.bitwise_and(im1, im1, mask=mask.astype(np.uint8))
        im2_face_image = cv2.bitwise_and(im2, im2, mask=mask.astype(np.uint8))

        im1_factor = np.mean(im1_face_image, axis=(0, 1))
        im2_factor = np.mean(im2_face_image, axis=(0, 1))

    im1_face_image = np.clip((im1_face_image.astype(float) * im2_factor / np.clip(im1_factor, 1e-6, None)), 0,
                             255).astype(np.uint8)
    im1[mask] = im1_face_image[mask]
    return im1
