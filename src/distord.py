import numpy as np

import os
import cv2

from distutils.util import strtobool
from matplotlib import pyplot as plt, cm
import argparse

from misc.tools import compute_score, apply_distortion
from ang.write_core_ang import Ang
from ang.phase import Phase

import json
import cma
import collections

DataPoints = collections.namedtuple('DataPoints', ['index', 'points'])


def __main__(args=None):
    if args is None:
        parser = argparse.ArgumentParser('Learn Polynomial distortion with CMA-ES!')

        parser.add_argument("-seg_ref_path", type=str, required=True, help="Path to the segmented image")
        parser.add_argument("-ebsd_ref_path", type=str, required=True, help="Path to the grain image")
        parser.add_argument("-ang_ref_path", type=str, required=True, help="Path to the ang file")

        parser.add_argument("-out_dir", type=str, required=True, help="Directory with input image")

        parser.add_argument("-mesh_step", type=float, default=65, help="Use an percentage step size instead of no_points")
        parser.add_argument("-mesh_std", type=float, default=5, help="How far are going to look around the mesh ground (% of the image dimension)")
        parser.add_argument("-num_sampling", type=int, default=2500, help="How far are going to look around the mesh ground (% of the image dimension)")
        parser.add_argument("-polynom", type=int, default=3, help="Order of the polynom to compute distorsion")

        parser.add_argument("-image_magic", type=lambda b: bool(strtobool(b)), default="False", help="use image magick")

        parser.add_argument("-phase_name", type=str, default="Prec", help="Define the phase name for the ang file")
        parser.add_argument("-phase_formula", type=str, default="NiAl", help="Define the phase formula for the ang file")

        parser.add_argument("-id_xp", type=int, default=0, help="Define the xp id to save the results")
        parser.add_argument("-seed", type=int, default=None, help="Define random seed")

        args = parser.parse_args()

    # Load ebsd/segment. Note that the segment must be preprocess with align.py
    segment_align = cv2.imread(args.seg_ref_path, 0)
    ebsd = cv2.imread(args.ebsd_ref_path, 0)

    assert segment_align.shape == ebsd.shape, "Grain and shape must be of the same dimension"

    # Compute initial score
    init_score = compute_score(segment=segment_align, ebsd=ebsd)
    print("Init score : {0:.4f}".format(init_score))

    # Create initial mesh grid
    i_mesh = np.arange(0, segment_align.shape[1], args.mesh_step)
    j_mesh = np.arange(0, segment_align.shape[0], args.mesh_step)

    ii, jj = np.meshgrid(i_mesh, j_mesh)
    ii = ii.reshape(-1)
    jj = jj.reshape(-1)

    initial_points = np.concatenate((ii, jj)).astype(np.int32)

    ##############
    # CMA-ES
    ##############

    no_samples, best_score = 0, init_score
    es = cma.CMAEvolutionStrategy(initial_points, args.mesh_std, {'seed': args.seed})

    all_scores = []

    # Start black-box optimization
    while not es.stop() and no_samples < args.num_sampling:
        solutions = es.ask()

        # turn floating mesh into integer one (pixels are integer!)
        mesh_int = np.array(solutions, dtype=np.int32)

        # prepare data for multi-threading
        scores = []

        for s in mesh_int:
            ii_dist = s[:len(ii)]
            jj_dist = s[len(jj):]

            sources = np.stack([ii, jj]).transpose().astype(np.int32)
            targets = np.stack([ii_dist, jj_dist]).transpose().astype(np.int32)

            ebsd_distord, _ = apply_distortion(ebsd=ebsd,
                                               sources=sources,
                                               targets=targets,
                                               polynom=args.polynom,
                                               use_img_magic=args.image_magic)

            score = compute_score(segment=segment_align, ebsd=ebsd_distord)
            all_scores.append(score)

            # Make the score negative as it is a minimization process
            score *= -1
            scores.append(score)

        # store no_samples
        no_samples += len(scores)

        es.tell(solutions, scores)
        es.disp()

    ##############
    # Retrieve Best distortion
    ##############

    # retrieve the best mesh
    final_mesh = es.result.xbest

    ii_dist = final_mesh[:len(ii)]
    jj_dist = final_mesh[len(jj):]
    sources = np.stack([ii, jj]).transpose().astype(np.int32)
    targets = np.stack([ii_dist, jj_dist]).transpose().astype(np.int32)

    # Define path to save results
    out_distord = '{out_dir}/ebsd_distord.{index}.png'.format(out_dir=args.out_dir, index=args.id_xp)
    out_mesh = '{out_dir}/mesh_distord.{index}.png'.format(out_dir=args.out_dir, index=args.id_xp)
    out_image = os.path.join(args.out_dir, "overlap.distord.{}.png".format(args.id_xp))
    params_path = os.path.join(args.out_dir, "params.{}.json".format(args.id_xp))

    # Recompute best transformation
    final_ebsd, params = apply_distortion(ebsd=ebsd,
                                          sources=sources,
                                          targets=targets,
                                          polynom=args.polynom,
                                          verbose=True)

    best_score = compute_score(segment=segment_align,
                               ebsd=final_ebsd)
    print("Final best score : {0:.4f}".format(best_score))

    ##############
    # Dump results
    ##############

    # Plot the final ebsd
    fig = plt.figure(figsize=(15, 8))
    plt.imshow(final_ebsd)
    fig.savefig(out_distord)

    # Plot how ebsd/segment overlap
    fig = plt.figure(figsize=(15, 8))
    plt.imshow(segment_align, interpolation='nearest', cmap=cm.gray)
    plt.imshow(final_ebsd, interpolation='nearest', cmap=cm.jet, alpha=0.5)
    fig.savefig(out_image)

    # Plot the mesh over the distorted image
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(final_ebsd)
    ax.plot(targets[:, 0], targets[:, 1], '.')
    fig.savefig(out_mesh)

    # Store points for transformation
    with open(params_path, "w") as params_file:
        params_file.write(json.dumps(dict(
            ebsd=os.path.basename(args.ebsd_ref_path),
            segment=os.path.basename(args.seg_ref_path),
            score=best_score,
            polynom=args.polynom,
            sources=sources.tolist(),
            targets=targets.tolist(),
            params=params.tolist())))

    # Create ang file
    ang = Ang(args.ang_ref_path)
    ang.warp(params)

    phase = Phase.create_from_phase(id=2, name=args.phase_name, formula=args.phase_formula, phase=ang.phases[0])
    ang.add_phase(segment_align=segment_align, phase=phase)
    ang.dump_file(os.path.join(args.out_dir, os.path.basename(args.ang_ref_path)))

    return best_score, final_ebsd, params, all_scores


if __name__ == "__main__":
    __main__()
