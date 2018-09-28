import argparse
import numpy as np
from ang.write_core_ang import Ang
from ang.phase import Phase

from matplotlib import pyplot as plt, cm

import skimage
from skimage.transform import PolynomialTransform

from misc.tools import Aligner, compute_score

import cv2
import json
import os
import collections

DataPoints = collections.namedtuple('DataPoints', ['index', 'points'])


def __main__(args=None):
    if args is None:
        parser = argparse.ArgumentParser('Learn Polynomial distortion with CMA-ES!')

        parser.add_argument("-seg_ref_path", type=str, required=True, help="Path to the segmented image")
        parser.add_argument("-ebsd_ref_path", type=str, required=True, help="Path to the grain image")
        parser.add_argument("-ang_ref_path", type=str, required=True, help="Path to the ang file")
        parser.add_argument("-out_dir", type=str, required=True, help="Directory with input image")

        parser.add_argument("-align", type=str, required=True, help="Path to the align file")
        parser.add_argument("-distord", type=str, required=True, help="Path to the distord file")

        parser.add_argument("-phase_name", type=str, default="Prec", help="Define the phase name for the ang file")
        parser.add_argument("-phase_formula", type=str, default="NiAl", help="Define the phase formula for the ang file")

        args = parser.parse_args()

    segment = cv2.imread(args.seg_ref_path, 0)
    ebsd = cv2.imread(args.ebsd_ref_path, 0)

    # Step 1: Align segment
    with open(args.align, "rb") as f:
        align_conf = json.loads(f.read())

        segment_align = Aligner.rescale(segment, rescale=align_conf["rescale"])
        segment_align = Aligner.rotate(segment_align, align_conf["angle"])
        segment_align = Aligner.translate(segment=segment_align,
                                          tx=align_conf["translate"][0], ty=align_conf["translate"][1],
                                          shape=ebsd.shape[::-1])

        score_align = compute_score(segment=segment_align, ebsd=ebsd)
        print("Score (Align): {0:.4f}".format(score_align))

    # Step 2: Distord data
    with open(args.distord, "rb") as f:
        distord_conf = json.loads(f.read())

        # Distord image
        params = np.array(distord_conf["params"])
        transform = skimage.transform._geometric.PolynomialTransform(params)
        ebsd_distord = skimage.transform.warp(ebsd, transform,
                                              cval=0,  # new pixels are black pixel
                                              order=0,  # k-neighbour
                                              preserve_range=True)

        score_align = compute_score(segment=segment_align, ebsd=ebsd_distord)
        print("Score (Distord): {0:.4f}".format(score_align))

    # Step 4: Dump results
    print("Dump results...")
    # Plot the final segment
    fig = plt.figure(figsize=(15, 8))
    plt.imshow(segment_align)
    fig.savefig(os.path.join(args.out_dir, "segment_align.png"))

    # Plot the final ebsd
    fig = plt.figure(figsize=(15, 8))
    plt.imshow(ebsd_distord)
    fig.savefig(os.path.join(args.out_dir, "ebsd_distord.png"))

    # Plot how ebsd/segment overlap
    fig = plt.figure(figsize=(15, 8))
    plt.imshow(segment_align, interpolation='nearest', cmap=cm.gray)
    plt.imshow(ebsd_distord, interpolation='nearest', cmap=cm.jet, alpha=0.5)
    fig.savefig(os.path.join(args.out_dir, "overlap.distord.png"))

    # Step 3: Create ang file
    print("Dump ang...")
    ang = Ang(args.ang_ref_path)
    ang.warp(params)

    phase = Phase.create_from_phase(id=2, name=args.phase_name, formula=args.phase_formula, phase=ang.phases[0])
    ang.add_phase(segment_align=segment_align, phase=phase)
    ang.dump_file(os.path.join(args.out_dir, os.path.basename(args.ang_ref_path)))


if __name__ == "__main__":
    __main__()
