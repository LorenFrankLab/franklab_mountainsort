'''
This module calls the helper functions defined in p2p (`proc2py`) that in turn,
call Mountain Sort processors. This should be a collection of common processing
steps that are standard across the lab, although parameters can be changed
flexibly. Default parameters are defined in the arguments of each pypline, but
can be overwritten by the user.

These pyplines should be called by a python batch script, which will manage
running the steps on particular animal, days, and ntrodes.

AKGillespie based on code from JMagland
Demetris Roumis

# WARNING: Before anything else, must concat all eps together becuase ms4 no
           longer handles the prv list of mdas
'''
import json
import math
import os
import subprocess
from logging import getLogger

from franklab_mountainsort.ms4_franklab_proc2py import (
    bandpass_filter, clear_seg_files, compute_cluster_metrics,
    get_epoch_offsets, get_mda_list, mask_out_artifacts, ms4alg,
    pyms_extract_clips, pyms_extract_segment, read_dataset_params,
    tagged_curation, whiten)
from franklab_msdrift.p_anneal_segments import \
    anneal_segments as pyms_anneal_segs

from franklab_mstaggedcuration.p_add_curation_tags import \
    add_curation_tags as pyms_add_curation_tags
from franklab_mstaggedcuration.p_merge_burst_parents import \
    merge_burst_parents as pyms_merge_burst_parents

logger = getLogger(__name__)


def concat_epochs(dataset_dir, mda_list=None, opts=None, mda_opts=None):
    '''Concatenate all epochs in a day and save as raw.mda.

    Saves the raw.mda to the output dir, which serves as src for subsequent
    steps.

    Runs 'ms3.concat_timeseries' using either:
       1: mda_list provided
       2: mda_list empty and date, ntrode specified in opts
       3: string path to prv file containing entries for mda files

    Parameters
    ----------
    dataset_dir : str
    mda_list : None or list, optional
    opts : None or dict, optional
    mda_opts : None or dict, optional

    Notes
    -----
    Format for input and output of ms3.concat_timeseries is:
        'timeseries_list:{path} timeseries_list:{path} ...'
    There cannot be a space between the colon and the path.

    '''

    if mda_list is None:
        mda_list = []
    if opts is None:
        opts = {}
    if mda_opts is None:
        mda_opts = {}

    has_opts_keys = (
        {'anim', 'date', 'ntrode', 'data_location'}.issubset(mda_opts))
    if isinstance(mda_list, list) and len(mda_list) > 0:
        logger.info('...Using provided list of mda files')
        str_start = [f'timeseries_list:{entry}' for entry in mda_list]
    elif len(mda_list) == 0 and has_opts_keys:
        logger.info(
            f'...Finding list of mda files from mda directories of '
            f'date: {mda_opts["date"]}, ntrode: {mda_opts["ntrode"]}')
        mda_list = get_mda_list(
            mda_opts['anim'], mda_opts['date'], mda_opts['ntrode'], mda_opts['data_location'])
        str_start = [f'timeseries_list:{entry}' for entry in mda_list]
    elif isinstance(mda_list, str):
        logger.info('...Using mda files listed in prv file')
        with open(mda_list) as f:
            mdalist = json.load(f)
        str_start = []
        for entries in mdalist['files']:
            prv_path = entries['prv']['original_path']
            str_start.append(f'timeseries_list:{prv_path}')

    joined = ' '.join(str_start)
    out_path = os.path.join(f'timeseries_out:{dataset_dir}', 'raw.mda')
    subprocess.run(['ml-run-process', 'ms3.concat_timeseries',
                    '--inputs', joined, '--outputs', out_path], check=True)


def filt_mask_whiten(dataset_dir, output_dir, freq_min=300, freq_max=6000,
                     artifacts_interval_size=2000, artifacts_threshold=5,
                     mask_artifacts=True, opts=None):
    '''

    Parameters
    ----------
    dataset_dir : str
    output_dir : str
    freq_min : float, optional
    freq_max : float, optional
    mask_artifacts : bool, optional
    opts : None or dict, optional

    '''
    if opts is None:
        opts = {}
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Dataset parameters
    ds_params = read_dataset_params(dataset_dir)

    # Bandpass filter
    bandpass_filter(
        timeseries=os.path.join(dataset_dir, 'raw.mda'),
        timeseries_out=os.path.join(output_dir, 'filt.mda.prv'),
        samplerate=ds_params['samplerate'],
        freq_min=freq_min,
        freq_max=freq_max,
        opts=opts
    )
    # Mask out artifacts
    if mask_artifacts:
        mask_out_artifacts(
            timeseries=os.path.join(output_dir, 'filt.mda.prv'),
            timeseries_out=os.path.join(output_dir, 'filt.mda.prv'),
            threshold=artifacts_threshold,
            interval_size=artifacts_interval_size,
            opts=opts
        )
    # Whiten
    whiten(
        timeseries=os.path.join(output_dir, 'filt.mda.prv'),
        timeseries_out=os.path.join(output_dir, 'pre.mda.prv'),
        opts=opts
    )


def ms4_sort_full(dataset_dir, output_dir, geom=None, adjacency_radius=-1,
                  detect_threshold=3, detect_interval=10, detect_sign=False,
                  num_workers=2, opts=None):
    '''Sort the entire file as one mda

    Parameters
    ----------
    dataset_dir : str
    output_dir : str
    geom : None or list, optional
    adjacency_radius : float, optional
    detect_threshold : float, optional
    detect_interval : int, optional
    detect_sign : bool, optional
    num_workers : int, optional
    opt : dict or None, optional

    '''
    if geom is None:
        geom = []
    if opts is None:
        opts = {}
    # Fetch dataset parameters
    ds_params = read_dataset_params(dataset_dir)

    ms4alg(
        timeseries=os.path.join(output_dir, 'pre.mda.prv'),
        geom=geom,
        firings_out=os.path.join(output_dir, 'firings_raw.mda'),
        adjacency_radius=adjacency_radius,
        detect_sign=int(detect_sign),
        detect_threshold=detect_threshold,
        detect_interval=detect_interval,
        num_workers=num_workers,
        opts=opts
    )

    # Compute cluster metrics
    compute_cluster_metrics(
        timeseries=os.path.join(output_dir, 'pre.mda.prv'),
        firings=os.path.join(output_dir, 'firings_raw.mda'),
        metrics_out=os.path.join(output_dir, 'metrics_raw.json'),
        samplerate=ds_params['samplerate'],
        opts=opts
    )


def ms4_sort_on_segs(dataset_dir, output_dir, geom=None,
                     adjacency_radius=-1, detect_threshold=3.0,
                     detect_interval=10,
                     detect_sign=False, rm_segment_intermediates=True,
                     num_workers=2, opts=None, mda_opts=None):
    '''Sort by timesegments, then join any matching clusters

    Parameters
    ----------
    dataset_dir : str
    output_dir : str
    geom : None or list, optional
    adjacency_radius : float, optional
    detect_threshold : float, optional
    detect_interval : int, optional
    detect_sign : bool, optional
    rm_segment_intermediates : bool, optional
    num_workers : int, optional
    opt : dict or None, optional
    mda_opt : dict or None, optional

    '''
    if geom is None:
        geom = []
    if opts is None:
        opts = {}
    if mda_opts is None:
        mda_opts = {}

    # Fetch dataset parameters
    ds_params = read_dataset_params(dataset_dir)
    has_keys = {'anim', 'date', 'ntrode', 'data_location'}.issubset(mda_opts)

    if has_keys:
        logger.info(
            'Finding list of mda file from mda directories of '
            f'date:{mda_opts["date"]}, ntrode:{mda_opts["ntrode"]}')
        mda_list = get_mda_list(
            mda_opts['anim'], mda_opts['date'], mda_opts['ntrode'], mda_opts['data_location'])
        # calculate time_offsets and total_duration
        sample_offsets, total_samples = get_epoch_offsets(
            dataset_dir=dataset_dir, opts={'mda_list': mda_list})

    else:
        # calculate time_offsets and total_duration
        sample_offsets, total_samples = get_epoch_offsets(
            dataset_dir=dataset_dir)

    # break up preprocesed data into segments and sort each
    firings_list = []
    timeseries_list = []
    for segind in range(len(sample_offsets)):
        t1 = math.floor(sample_offsets[segind])
        if segind == len(sample_offsets) - 1:
            t2 = total_samples - 1
        else:
            t2 = math.floor(sample_offsets[segind + 1]) - 1

        t1_min = t1 / ds_params['samplerate'] / 60
        t2_min = t2 / ds_params['samplerate'] / 60
        logger.info(f'Segment {segind + 1}: t1={t1}, t2={t2}, '
                    f't1_min={t1_min:.3f}, t2_min={t2_min:.3f}')

        pre_outpath = os.path.join(dataset_dir, f'pri-{segind + 1}.mda')
        pyms_extract_segment(
            timeseries=os.path.join(output_dir, 'pre.mda.prv'),
            timeseries_out=pre_outpath,
            t1=t1,
            t2=t2,
            opts=opts)

        firings_outpath = os.path.join(
            dataset_dir, f'firings-{segind + 1}.mda')
        ms4alg(
            timeseries=pre_outpath,
            firings_out=firings_outpath,
            geom=geom,
            detect_sign=int(detect_sign),
            adjacency_radius=adjacency_radius,
            detect_threshold=detect_threshold,
            detect_interval=detect_interval,
            num_workers=num_workers,
            opts=opts)

        firings_list.append(firings_outpath)
        timeseries_list.append(pre_outpath)

    firings_out_final = os.path.join(output_dir, 'firings_raw.mda')
    # sample_offsets have to be converted into a string to be properly passed
    # into the processor
    str_sample_offsets = ','.join(map(str, sample_offsets))
    logger.info(str_sample_offsets)

    # Drift tracking
    pyms_anneal_segs(
        timeseries_list=timeseries_list,
        firings_list=firings_list,
        firings_out=firings_out_final,
        dmatrix_out=[],
        k1_dmatrix_out=[],
        k2_dmatrix_out=[],
        dmatrix_templates_out=[],
        time_offsets=str_sample_offsets
    )

    # clear the temp pre and firings files if specified
    if rm_segment_intermediates:
        clear_seg_files(
            timeseries_list=timeseries_list,
            firings_list=firings_list
        )

    # Compute cluster metrics
    compute_cluster_metrics(
        timeseries=os.path.join(output_dir, 'pre.mda.prv'),
        firings=os.path.join(output_dir, 'firings_raw.mda'),
        metrics_out=os.path.join(output_dir, 'metrics_raw.json'),
        samplerate=ds_params['samplerate'],
        opts=opts
    )


def merge_burst_parents(dataset_dir, output_dir):
    '''

    Parameters
    ----------
    dataset_dir : str
    output_dir : str

    '''
    pyms_merge_burst_parents(
        firings=os.path.join(output_dir, 'firings_raw.mda'),
        metrics=os.path.join(output_dir, 'metrics_raw.json'),
        firings_out=os.path.join(output_dir, 'firings_burst_merged.mda'))

    ds_params = read_dataset_params(dataset_dir)
    # Compute cluster metrics
    compute_cluster_metrics(
        timeseries=os.path.join(output_dir, 'pre.mda.prv'),
        firings=os.path.join(output_dir, 'firings_burst_merged.mda'),
        metrics_out=os.path.join(output_dir, 'metrics_merged.json'),
        samplerate=ds_params['samplerate'])


def add_curation_tags(dataset_dir, output_dir, firing_rate_thresh=0.01,
                      isolation_thresh=0.95, noise_overlap_thresh=0.03,
                      peak_snr_thresh=1.5, metrics_input='metrics_raw.json',
                      metrics_output='metrics_tagged.json'):
    '''

    Parameters
    ----------
    dataset_dir : str
    output_dir : str
    firing_rate_thresh : float, optional
    isolation_thresh : float, optional
    noise_overlap_thresh : float, optional
    peak_snr_thresh : float, optional
    metrics_input : str, optional
    metrics_output : str, optional
    opts : None or dict, optional

    Notes
    -----
    This is split out and not included after metrics calculation
    because of a bug in ms3.combine_cluster_metrics - doesn't work if anything
    follows it.

    '''
    pyms_add_curation_tags(
        metrics=os.path.join(dataset_dir, metrics_input),
        metrics_tagged=os.path.join(output_dir, metrics_output),
        firing_rate_thresh=firing_rate_thresh,
        isolation_thresh=isolation_thresh,
        noise_overlap_thresh=noise_overlap_thresh,
        peak_snr_thresh=peak_snr_thresh, mv2file='')


def recalc_metrics(dataset_dir, output_dir, firings_in='firings_processed.mda',
                   metrics_to_update='metrics_processed.json',
                   firing_rate_thresh=0.01, isolation_thresh=0.95,
                   noise_overlap_thresh=0.03,
                   peak_snr_thresh=1.5, mv2_file='', manual_only=True):
    '''post-merge, should recalculate metrics and update tags (both tags based
    on thresholds and any manually added ones, stored in the mv2). If manual_only==True, 
    adds the tag “mua” to clusters that do not meet the criteria to be considered as single units

    Parameters
    ----------
    dataset_dir : str
    output_dir : str
    firings_in : str, optional
    metrics_to_update : str, optional
    firing_rate_thresh : float, optional
        Clusters less than the firing rate threshold is excluded (spikes / s )
    isolation_thresh : float, optional
        Distance to a cluster of noise.
    noise_overlap_thresh : float, optional
        Fraction of “noise events” in a cluster.
    peak_snr_thresh : float, optional
    mv2_file : str, optional
    manual_only :bool, weather to only keep manual tags or further threshold (see above), optional

    '''

    ds_params = read_dataset_params(dataset_dir)

    compute_cluster_metrics(
        timeseries=os.path.join(output_dir, 'pre.mda.prv'),
        firings=os.path.join(output_dir, firings_in),
        metrics_out=os.path.join(output_dir, metrics_to_update),
        samplerate=ds_params['samplerate'])

    tagged_curation(
        cluster_metrics=os.path.join(dataset_dir, metrics_to_update),
        metrics_tagged=os.path.join(output_dir, metrics_to_update),
        firing_rate_thresh=firing_rate_thresh,
        isolation_thresh=isolation_thresh,
        noise_overlap_thresh=noise_overlap_thresh,
        peak_snr_thresh=peak_snr_thresh,
        mv2file=mv2_file,
        manual_only=manual_only
    )

def recalc_metrics_epoch_electrode(params,
                    opts=None,rm_segment_intermediates=True,
                    updated_mda='firings_processed.mda',
                    mv2_file='manualcuration',metrics_to_update='metrics_processed_epoch',manual_only=True):
    '''Sort by timesegments, then join any matching clusters

    Parameters
    ----------
    dataset_dir : str
    output_dir : str
    geom : None or list, optional
    rm_segment_intermediates : bool, optional
    opt : dict or None, optional
    updated_mda:
    mda_opt : dict or None, optional
    metrics_to_update: DO NOT include .json as it will be appended in the code

    '''
    output_dir=params[0]
    mda_opts=params[1]

    if mda_opts is None:
        mda_opts = {}
    if opts is None:
        opts = {}

    # Fetch dataset parameters
    ds_params = read_dataset_params(output_dir)
    has_keys = {'anim', 'date', 'ntrode', 'data_location'}.issubset(mda_opts)


    if has_keys:
        anim=mda_opts['anim']
        date=mda_opts['date']
        ntrode=mda_opts['ntrode']
        log_file = os.path.join(output_dir, f'{anim}_{date}_nt{ntrode}.log')
        logger_ = getLogger(log_file)
        logger_.info(
            'Finding list of mda file from mda directories of'
            f'date:{date}, ntrode:{ntrode}')
        mda_list = get_mda_list(
            date, ntrode, mda_opts['data_location'])
        # calculate time_offsets and total_duration
        sample_offsets, total_samples = get_epoch_offsets(
            dataset_dir=output_dir, opts={'mda_list': mda_list})

    else:
        # calculate time_offsets and total_duration
        sample_offsets, total_samples = get_epoch_offsets(
            dataset_dir=output_dir)

    # break up preprocesed data into segments
    firings_list = []
    timeseries_list = []
    for segind in range(len(sample_offsets)):
        t1 = math.floor(sample_offsets[segind])
        if segind == len(sample_offsets) - 1:
            t2 = total_samples - 1
        else:
            t2 = math.floor(sample_offsets[segind + 1]) - 1

        t1_min = t1 / ds_params['samplerate'] / 60
        t2_min = t2 / ds_params['samplerate'] / 60
        logger.info(f'Segment {segind + 1}: t1={t1}, t2={t2}, '
                    f't1_min={t1_min:.3f}, t2_min={t2_min:.3f}')

        pre_outpath = os.path.join(output_dir, f'pri-{segind + 1}.mda')
        pyms_extract_segment(
            timeseries=os.path.join(output_dir,updated_mda),
            timeseries_out=pre_outpath,
            t1=t1,
            t2=t2,
            opts=opts)

        firings_outpath = os.path.join(
            output_dir, f'firings-{segind + 1}.mda')

        firings_list.append(firings_outpath)
        timeseries_list.append(pre_outpath)
        
        recalc_metrics(output_dir,output_dir,firings_in=firings_outpath,
            metrics_to_update=metrics_to_update+{segind + 1}+'.json',mv2_file=mv2_file,manual_only=True)

    if rm_segment_intermediates:
        clear_seg_files(
            timeseries_list=timeseries_list,
            firings_list=firings_list
        )
        
        



def extract_clips(dataset_dir, output_dir, clip_size=45, opts=None):
    '''

    Parameters
    ----------
    dataset_dir : str
    output_dir : str
    clip_size : float, optional
    opts : None or dict, optional

    '''
    if opts is None:
        opts = {}
    opts['clip_size'] = clip_size

    pyms_extract_clips(
        timeseries=os.path.join(dataset_dir, 'pre.mda.prv'),
        firings=os.path.join(dataset_dir, 'firings_raw.mda'),
        clips_out=os.path.join(output_dir, 'clips.mda'),
        opts=opts)


def extract_marks(dataset_dir, output_dir, opts=None):
    '''

    Parameters
    ----------
    dataset_dir : str
    output_dir : str
    opts : None or dict, optional

    '''
    if opts is None:
        opts = {}
    opts['clip_size'] = 1

    pyms_extract_clips(
        timeseries=os.path.join(dataset_dir, 'pre.mda.prv'),
        firings=os.path.join(dataset_dir, 'firings_raw.mda'),
        clips_out=os.path.join(output_dir, 'marks.mda'),
        opts=opts)
