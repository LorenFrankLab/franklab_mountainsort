'''
This module acts as the most basic interface between python and mountainlab
processors. Defining a processors inputs, outputs, and params here allows
processors written in any language (as long as they follow mountainlab
conventions) to be made into a callable python function with passable params.

These functions serve as the building blocks for subpipelines, defined in
`ms4_franklab_pyplines`

In general, each function corresponds to one processor, except in the case
where multiple processors always function together

Note that NO default params are set here. This is to prevent the use of any
default values unknowingly. Default values should be provided in the pyplines
script

AKGillespie based on code from JMagland
'''

import glob
import json
import os

from mountainlab_pytools import mdaio
from mountainlab_pytools import mlproc as mlp


def read_dataset_params(dataset_dir):
    '''

    Parameters
    ----------
    dataset_dir : str

    Returns
    -------
    parameter_file_text : str

    '''
    params_fname = mlp.realizeFile(os.path.join(dataset_dir, 'params.json'))
    if not os.path.exists(params_fname):
        raise FileNotFoundError(
            f'Dataset parameter file does not exist: {params_fname}')
    with open(params_fname) as f:
        return json.load(f)


def bandpass_filter(timeseries, timeseries_out, samplerate, freq_min,
                    freq_max, opts=None):
    '''

    Parameters
    ----------
    timeseries : str
        Filepath to .mda file
    timeseries_out : str
        Filepath to filt.mda.prv file
    samplerate : float
    freq_min : float
    freq_max : float
    opts : None or dict, optional

    '''
    if opts is None:
        opts = {}
    return mlp.runProcess(
        'ephys.bandpass_filter',
        {
            'timeseries': timeseries
        }, {
            'timeseries_out': timeseries_out
        },
        {
            'samplerate': samplerate,
            'freq_min': freq_min,
            'freq_max': freq_max
        },
        opts
    )


def whiten(timeseries, timeseries_out, opts=None):
    '''

    Parameters
    ----------
    timeseries : str
        Filepath to 'filt.mda.prv' file
    timeseries_out : str
        Filepath to 'pre.mda.prv' file
    opts : None or dict, optional

    '''
    if opts is None:
        opts = {}
    return mlp.runProcess(
        'ephys.whiten',
        {
            'timeseries': timeseries
        },
        {
            'timeseries_out': timeseries_out
        },
        {},
        opts
    )


def mask_out_artifacts(timeseries, timeseries_out, threshold, interval_size,
                       opts=None):
    '''

    Parameters
    ----------
    timeseries : str
        Filepath to filt.mda.prv file
    timeseries_out : str
        Filepath to filt.mda.prv file
    threshold : float
    interval_size : int
    opts : None or dict, optional

    '''
    if opts is None:
        opts = {}
    return mlp.runProcess(
        'ms3.mask_out_artifacts',
        {
            'timeseries': timeseries
        },
        {
            'timeseries_out': timeseries_out
        },
        {
            'threshold': threshold,
            'interval_size': interval_size
        },
        opts
    )


def ms4alg(timeseries, geom, firings_out, detect_sign, adjacency_radius,
           detect_threshold, detect_interval, num_workers, opts=None):
    '''
    Parameters
    ----------
    timeseries : str
        Filepath to `pre.mda.prv` file
    geom : list
    firings_out : str
        Filepath to `firings_raw.mda`
    detect_sign : int
    detect_threshold : float
    opts : None or dict, optional

    '''
    if opts is None:
        opts = {}
    pp = {'detect_sign': detect_sign,
          'adjacency_radius': adjacency_radius,
          'detect_interval': detect_interval,
          'detect_threshold': detect_threshold,
          'num_workers': num_workers,
          }
    return mlp.runProcess(
        'ms4alg.sort',
        {
            'timeseries': timeseries,
            'geom': geom
        },
        {
            'firings_out': firings_out
        },
        pp,
        opts
    )


def compute_cluster_metrics(timeseries, firings, metrics_out, samplerate,
                            opts=None):
    '''

    Parameters
    ----------
    timeseries : str
        Filepath to `pre.mda.prv` file
    firings : str
        Filepath to `firings_raw.mda` file
    metrics_out : str
        Filepath to the output metrics file
    samplerate : float
    opts : None or dict, optional

    '''
    if opts is None:
        opts = {}
    metrics1 = mlp.runProcess(
        'ms3.cluster_metrics',
        {
            'timeseries': timeseries,
            'firings': firings
        },
        {
            'cluster_metrics_out': True
        },
        {
            'samplerate': samplerate
        },
        opts
    )['cluster_metrics_out']

    metrics2 = mlp.runProcess(
        'ms3.isolation_metrics',
        {
            'timeseries': timeseries,
            'firings': firings
        },
        {
            'metrics_out': True
        },
        {
            'compute_bursting_parents': 'true'
        },
        opts
    )['metrics_out']

    return mlp.runProcess(
        'ms3.combine_cluster_metrics',
        {
            'metrics_list': [metrics1, metrics2]
        },
        {
            'metrics_out': metrics_out  # True
        },
        {},
        opts
    )


# UNTESTED?UNUSED BY AKG
def automated_curation(firings, cluster_metrics, firings_out, opts=None):
    '''

    Parameters
    ----------
    firings : str
        Filepath
    cluster_metrics : str
        Filepath
    firings_out : str
        Filepath
    '''
    if opts is None:
        opts = {}
    label_map = mlp.runProcess(
        'ms4alg.create_label_map',
        {
            'metrics': cluster_metrics
        },
        {
            'label_map_out': True
        },
        {},
        opts
    )['label_map_out']
    return mlp.runProcess(
        'ms4alg.apply_label_map',
        {
            'label_map': label_map,
            'firings': firings
        },
        {
            'firings_out': firings_out
        },
        {},
        opts
    )


def pyms_merge_burst_parents(firings, metrics, firings_out, opts=None):
    '''

    Parameters
    ----------
    firings : str
        Filepath to `firings_raw.mda`
    metrics : str
        Filepath to `metrics_raw`
    firings_out : str
        Filepath to `firings_burst_merged.mda`
    opts : None or dict, optional

    '''
    if opts is None:
        opts = {}
    return mlp.runProcess(
        'pyms.merge_burst_parents',
        {
            'firings': firings,
            'metrics': metrics
        },
        {
            'firings_out': firings_out
        },
        opts
    )

def tagged_curation(cluster_metrics, metrics_tagged,
                    firing_rate_thresh=0.01, isolation_thresh=0.95,
                    noise_overlap_thresh=0.03, peak_snr_thresh=1.5,
                    mv2file='', opts=None):
    '''
    Parameters
    ----------
    cluster_metrics : str
        Filepath
    metrics_tagged : str
        Filepath
    firing_rate_thresh : float, optional
    isolation_thresh : float, optional
    noise_overlap_thresh : float, optional
    peak_snr_thresh : float, optional
    mv2file : str, optional
    opts : None or dict, optional
    '''
    if opts is None:
        opts = {}
    return mlp.runProcess(
        'pyms.add_curation_tags',
        {
            'metrics': cluster_metrics
        },
        {
            'metrics_tagged': metrics_tagged
        },
        {
            'firing_rate_thresh': firing_rate_thresh,
            'isolation_thresh': isolation_thresh,
            'noise_overlap_thresh': noise_overlap_thresh,
            'peak_snr_thresh': peak_snr_thresh,
            'mv2file': mv2file,
        },
        opts
    )



def get_mda_list(anim, date, ntrode, data_location):
    '''

    Parameters
    ----------
    date : int
    ntrode : int
    data_location : str

    Returns
    -------
    mda_list : list of str

    '''
    date = str(date)
    mda_list = glob.glob(
        os.path.join(data_location, date, '*.mda', f'{date}_{animal}_*.nt{ntrode}.mda'))
    mda_list.sort()

    return mda_list


def get_epoch_offsets(dataset_dir, opts=None):
    '''
    Parameters
    ----------
    dataset_dir : str
    opts : None or dict, optional

    Returns
    -------
    sample_offsets : ???
    total_samples : ???

    '''
    if opts is None:
        opts = {}
    if 'mda_list' in opts:
        # initialize with 0 (first start time)
        lengths = [0]

        for idx in range(len(opts['mda_list'])):
            ep_path = opts['mda_list'][idx]
            ep_mda = mdaio.DiskReadMda(ep_path)
            # get length of the mda (N dimension)
            samplength = ep_mda.N2()
            # add to prior sum and append
            lengths.append(samplength + lengths[(idx)])

    else:

        prv_list = os.path.join(dataset_dir, 'raw.mda.prv')

        with open(prv_list, 'r') as f:
            ep_files = json.load(f)

        # initialize with 0 (first start time)
        lengths = [0]

        for idx in range(len(ep_files['files'])):
            ep_path = ep_files['files'][idx]['prv']['original_path']
            ep_mda = mdaio.DiskReadMda(ep_path)
            # get length of the mda (N dimension)
            samplength = ep_mda.N2()
            # add to prior sum and append
            lengths.append(samplength + lengths[(idx)])

    # first entries (incl 0) are starttimes; last is total time
    total_samples = lengths[-1]
    sample_offsets = lengths[0:-1]

    return sample_offsets, total_samples


def pyms_extract_segment(timeseries, timeseries_out, t1, t2, opts=None):
    '''

    Parameters
    ----------
    timeseries : str
        Filepath
    timeseries_out : str
        Filepath
    t1 : float
    t2 : float
    opts : None or dict, optional

    '''
    if opts is None:
        opts = {}

    return mlp.runProcess(
        'pyms.extract_timeseries',
        {
            'timeseries': timeseries
        },
        {
            'timeseries_out': timeseries_out
        },
        {
            't1': t1,
            't2': t2
        },
        opts
    )


def pyms_anneal_segs(timeseries_list, firings_list, firings_out,
                     dmatrix_out, k1_dmatrix_out, k2_dmatrix_out,
                     dmatrix_templates_out, time_offsets, opts=None):
    '''

    Parameters
    ----------
    timeseries_list : list of str
        Filepaths
    firings_list : list of str
        Filepaths
    firings_out : list of str
        Filepaths
    dmatrix_out : list
    k1_dmatrix_out : list
    k2_dmatrix_out : list
    time_offsets : str
    opts : None or dict, optional

    '''
    if opts is None:
        opts = {}

    return mlp.runProcess(
        'pyms.anneal_segments',
        {
            'timeseries_list': timeseries_list,
            'firings_list': firings_list
        },
        {
            'firings_out': firings_out,
            'dmatrix_out': dmatrix_out,
            'k1_dmatrix_out': k1_dmatrix_out,
            'k2_dmatrix_out': k2_dmatrix_out,
            'dmatrix_templates_out': dmatrix_templates_out
        },
        {
            'time_offsets': time_offsets
        },
        opts
    )


def clear_seg_files(timeseries_list, firings_list, opts=None):
    '''

    Parameters
    ----------
    timeseries_list : list of str
        Filepaths
    firings_list : list of str
    opts : None or dict

    '''
    if opts is None:
        opts = {}
    for file in timeseries_list:
        os.remove(file)

    for file in firings_list:
        os.remove(file)


def pyms_extract_clips(timeseries, firings, clips_out, opts=None):
    '''

    Parameters
    ----------
    timeseries : str
        Filepath to `pre.mda.prv` file
    firings : str
        Filepath to `firings_raw.mda` file
    clips_out : str
        Filepath to `clips.mda` file
    opts : None or dict, optional

    '''
    if opts is None:
        opts = {}
    return mlp.runProcess(
        'pyms.extract_clips',
        {
            'timeseries': timeseries,
            'firings': firings
        },
        {
            'clips_out': clips_out
        },
        opts
    )


def pyms_extract_marks(timeseries, firings, marks_out, markstimes_out,
                       opts=None):
    '''

    Parameters
    ----------
    timeseries : str
        Filepath to `pre.mda.prv`
    firings : str
        Filepath to `firings_raw.mda`
    marks_out : str
        Filepath to `marks.mda`
    markstimes_out : str
        Filepath
    opts : None or dict, optional
    '''
    if opts is None:
        opts = {}
    return mlp.runProcess(
        'pyms.extract_marks',
        {
            'timeseries': timeseries,
            'firings': firings
        },
        {
            'marks_out': marks_out,
            'markstimes_out': markstimes_out
        },
        opts
    )


def synthesize_sample_dataset(dataset_dir, samplerate=30000, duration=600,
                              num_channels=4, opts=None):
    '''

    Parameters
    ----------
    dataset_dir : str
    samplerate : float
    duration : float
    num_channels : int
    opts : None or dict, optional

    '''
    if opts is None:
        opts = {}
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    M = num_channels
    mlp.runProcess(
        'ephys.synthesize_random_waveforms',
        {},
        {
            'geometry_out': os.path.join(dataset_dir, 'geom.csv'),
            'waveforms_out': os.path.join(dataset_dir, 'waveforms_true.mda')
        },
        {
            'upsamplefac': 13,
            'M': M,
            'average_peak_amplitude': 100
        },
        opts
    )
    mlp.runProcess(
        'ephys.synthesize_random_firings',
        {},
        {
            'firings_out': os.path.join(dataset_dir, 'firings_true.mda')
        },
        {
            'duration': duration
        },
        opts
    )
    mlp.runProcess(
        'ephys.synthesize_timeseries',
        {
            'firings': os.path.join(dataset_dir, 'firings_true.mda'),
            'waveforms': os.path.join(dataset_dir, 'waveforms_true.mda')
        },
        {
            'timeseries_out': os.path.join(dataset_dir, 'raw.mda.prv')
        }, {
            'duration': duration,
            'waveform_upsamplefac': 13,
            'noise_level': 10
        },
        opts
    )
    params = {
        'samplerate': samplerate,
        'spike_sign': 1
    }
    with open(os.path.join(dataset_dir, 'params.json'), 'w') as outfile:
        json.dump(params, outfile, indent=4)
