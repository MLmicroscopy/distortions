import subprocess
import cv2

import numpy as np
import io

import skimage
import uuid

from skimage.transform import PolynomialTransform

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


def compute_score(segment, ebsd):
    """
    Compute Dice measure (or f1 score) on the segmented pixels

    :param segment: segmented electron image (uint8 format)
    :param ebsd:  speckle segmented out of the ebsd file (uint8 format)
    :return: Dice score
    """

    segmented_ebsd = ebsd >= 128
    segmented_segment = segment >= 128

    co_segmented = (segmented_ebsd & segmented_segment).sum()
    normalization = segmented_segment.sum() + segmented_ebsd.sum()

    score = 2 * co_segmented / normalization

    return score


# create a uuid for the current session to generate tmp png files. (nasty hack!)
tmp_uiid = uuid.uuid1()


def apply_distortion(ebsd, sources, targets, polynom, use_img_magic=False, verbose=False):
    """
    Use a distorded mesh to estimate the polynomial distortion on a ebsd.

    Cf sklearn doc for polynomial parameters format. (Butit si kind of intuitive )

    :param ebsd: speckle segmented out of the ebsd file (image with uint8 format)
    :param sources: flat initial mesh (I,J format)
    :param targets: flat target mesh (I,J format) use to compute the distortion
    :param polynom: degree of the polynomial distortion
    :param use_img_magic: Use image_magic (instead of sklearn) to learn the distorsion (faster)
    :param verbose: display distortion parameters
    :return: distorded ebsd speckles, polynomial parameters
    """

    if use_img_magic:

        points = np.concatenate([sources, targets], axis=1)

        # define in/out path
        in_ebsd = "/tmp/img.align.{}.png".format(tmp_uiid)
        out_ebsd = "/tmp/img.distord.{}.png".format(tmp_uiid)

        # turn the point into string to feed image_magic
        str_buffer = io.BytesIO()
        np.savetxt(str_buffer, points, fmt='%i')

        # save tmp image
        cv2.imwrite(in_ebsd, ebsd)

        if verbose:
            verbose_str = '-verbose'
        else:
            verbose_str = ''

        subprocess.run('convert {input_ebsd} {verbose} -virtual-pixel Black '
                       '-distort Polynomial "{polynom} {points}" '
                       '{output_distord}'
                       .format(input_ebsd=in_ebsd,
                               output_distord=out_ebsd,
                               points=str_buffer.getvalue(),
                               polynom=polynom,
                               verbose=verbose_str), shell=True)

        ebsd_distord = cv2.imread(out_ebsd, 0)
        params = None

    else:

        # Define the polynomial regression
        model_i = Pipeline([('poly', PolynomialFeatures(degree=polynom, include_bias=True)),
                            ('linear', LinearRegression(fit_intercept=False, normalize=False))])

        model_j = Pipeline([('poly', PolynomialFeatures(degree=polynom, include_bias=True)),
                            ('linear', LinearRegression(fit_intercept=False, normalize=False))])

        # Solve the regression system
        model_i.fit(sources, targets[:, 0])
        model_j.fit(sources, targets[:, 1])

        # Define the image transformation
        params = np.stack([model_i.named_steps['linear'].coef_,
                           model_j.named_steps['linear'].coef_], axis=0)

        # Distord image
        transform = skimage.transform._geometric.PolynomialTransform(params)
        ebsd_distord = skimage.transform.warp(ebsd, transform,
                                              cval=0,  # new pixels are black pixel
                                              order=0,  # k-neighbour
                                              preserve_range=True)

        if verbose:
            print("Polynomial regression parameters:")
            print(" - xx : {} \n - yy : {}".format(params[0], params[1]))

    return ebsd_distord, params


class Aligner(object):
    """
    Helper method to apply linear transformation (crop, rotate, translate) to an image
    """

    def __init__(self, rescale=(1, 1)):
        self.rescale = rescale

    @staticmethod
    def rotate(segment, angle):

        if angle != 0:
            center_rotation = (segment.shape[0] / 2, segment.shape[1] / 2)
            m_rot = cv2.getRotationMatrix2D(center_rotation, angle, 1)
            segment = cv2.warpAffine(segment, m_rot, segment.shape[::-1])

        return segment

    @staticmethod
    def rescale(segment, rescale):
        if rescale[0] != 1 and rescale[1] != 1:
            segment = cv2.resize(segment, None,
                                 fx=rescale[0],
                                 fy=rescale[1],
                                 interpolation=cv2.INTER_AREA)
        return segment

    @staticmethod
    def translate(segment, tx, ty, shape):

        # Apply affine transformation
        m_aff = np.float32([[1, 0, tx], [0, 1, ty]])
        segment = cv2.warpAffine(segment, m_aff, shape)

        return segment
