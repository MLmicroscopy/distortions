import os
import cv2
import json
import numpy as np

from matplotlib import pyplot as plt, cm
from misc.tools import Aligner, compute_score
import argparse


def get_range(conf):
    return np.arange(conf["start"], conf["end"], conf["step"])


def __main__(args=None):

    if args is None:

        parser = argparse.ArgumentParser('Image align input!')

        parser.add_argument("-seg_ref_path", type=str, required=True, help="Path to the segmented image")
        parser.add_argument("-ebsd_ref_path", type=str, required=True, help="Path to the grain image")
        parser.add_argument("-conf_path", type=str, required=True, help="Path to the configuration file")
        parser.add_argument("-align_dir", type=str, required=True, help="Output directory (segment crop)")
        parser.add_argument("-out_dir", type=str, required=True, help="Output directory (image overlap + affine.pkl)")

        parser.add_argument("-id_xp", type=int, default=0, help="Define the xp id to save the results")

        args = parser.parse_args()

    # Load configuration
    with open(args.conf_path, "r") as conf_file:
        conf = json.loads(conf_file.read())

    # Load for the segment/esbd
    segment = cv2.imread(args.seg_ref_path, 0)
    segment = Aligner.rescale(segment, rescale=(conf["rescale"]["x"], conf["rescale"]["y"]))

    ebsd = cv2.imread(args.ebsd_ref_path, 0)

    # Look for best alignment
    print("Look for best alignment...")
    best_score = 0
    best_val, best_segment = None, None

    for angle in get_range(conf["grid_search"]["rotate"]):

        # use a private method to avoid rotating at each iteration
        init_segment = np.copy(segment)
        rot_segment = Aligner.rotate(init_segment, angle)

        for tx in get_range(conf["grid_search"]["translate_x"]):
            for ty in get_range(conf["grid_search"]["translate_y"]):

                align_segment = Aligner.translate(segment=rot_segment,
                                                  tx=tx, ty=ty,
                                                  shape=ebsd.shape[::-1])

                score = compute_score(segment=align_segment, ebsd=ebsd)

                if score > best_score:
                    best_score = score
                    best_segment = np.copy(align_segment)
                    best_val = (best_score, (tx, ty), angle)
                    print("Score: {0:.4f}, tx: {1}, ty: {2}, angle: {3}".format(float(best_score), tx, ty, angle))

    # Display results
    (best_score, (tx, ty), angle) = best_val
    print("----------------------------------------------")
    print("Best score: ", best_score)
    print(" - tx  :  ", tx)
    print(" - ty  :  ", ty)
    print(" - rot : ", angle)
    print("----------------------------------------------")

    # Plot results
    out_image = os.path.join(args.out_dir, "overlap.align.{}.png".format(args.id_xp))
    fig = plt.figure(figsize=(15, 8))
    plt.imshow(best_segment, interpolation='nearest', cmap=cm.gray)
    plt.imshow(ebsd, interpolation='nearest', cmap=cm.jet, alpha=0.5)
    fig.savefig(out_image)

    # Store align segment
    filename_out = os.path.join(args.align_dir, "segment.align.{}.png".format(args.id_xp))
    cv2.imwrite(filename_out, best_segment)

    # Dump affine.pkl with all the required information to perform the affine transformation
    with open(os.path.join(args.out_dir, "affine.{}.json".format(args.id_xp)), "wb") as f:
        data = dict(score=best_score,
                    ebsd=os.path.basename(args.ebsd_ref_path),
                    segment=os.path.basename(args.seg_ref_path),
                    conf=conf,
                    rescale=(conf["rescale"]["x"], conf["rescale"]["y"]),
                    translate=[int(tx), int(ty)],
                    angle=int(angle))
        results_json = json.dumps(data)

        f.write(results_json.encode('utf8', 'replace'))

    return best_score, best_segment, data


if __name__ == "__main__":
    __main__()
