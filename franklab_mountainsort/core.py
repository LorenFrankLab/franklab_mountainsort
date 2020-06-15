import glob
import json
import logging
import os
import pprint
import functools
import multiprocessing

import franklab_mountainsort.ms4_franklab_pyplines as pyp
import numpy as np
import pandas as pd

MS_IN_SECOND = 1000


def spike_sort_all(mda_file_info, mountainlab_output_folder=None,
                   firing_rate_thresh=0.01, isolation_thresh=0.97,
                   noise_overlap_thresh=0.03, peak_snr_thresh=1.5,
                   extract_marks=True, extract_clips=True, clip_time=1.5,
                   freq_min=300, freq_max=6000, adjacency_radius=-1,
                   detect_threshold=3, detect_interval=10, detect_sign=-1,
                   artifacts_interval_size=2000, artifacts_threshold=5,
                   sampling_rate=30000, drift_track=True, burst_merge=False,
                   num_workers=2):
    '''Runs mountain sort on all electrodes in `mda_file_info`

    Parameters
    ----------
    mda_file_info : pandas.DataFrame
        Dataframe generated using function `get_mda_files_dataframe`
    mountainlab_output_folder : None or str, optional
        If None, goes to default location in animal directory. Else it should
        be a path to a directory.
    firing_rate_thresh : float, optional
        Clusters less than the firing rate threshold is excluded (spikes / s )
    isolation_thresh : float, optional
        Distance to a cluster of noise.
    noise_overlap_thresh : float, optional
         Fraction of “noise events” in a cluster.
    peak_snr_thresh : float, optional
    extract_marks : bool, optional
    extract_clips : bool, optional
        Extract the spike waveform around a spike.
    clip_time : float, optional
         Time (in ms) to extract around each spike.
    freq_min : float, optional
        The highpass or low cutoff of the filter in Hz.
    freq_max : float, optional
        The lowpass or high cutoff of the filter in Hz.
    adjacency_radius : float, optional
        The radius in µm that defines a neighborhood of channels on which to
        sort (default -1 to ignore and not require a geom.csv file, useful for
        tetrodes).
    detect_threshold : float, optional
        Spike detection threshold in standard deviations.
    detect_interval : int, optional
        Minimum number of timepoints between events detected on the same
        channel.
    detect_sign : int, optional
         The direction of spike to detect (-1 for negative, 1 for positive,
         0 for both). -1 is recommended for most recordings.
    artifacts_threshold : float, optional
    artifacts_interval_size : int, optional
    sampling_rate : int, optional
        Number of samples per second.
    drift_track : bool, optional
        Use drift tracking.
    burst_merge : bool, optional
        Automatic merging of bursts.
    num_workers : int, optional
        Number of compute threads to use for sorting.
    '''
    electrodes = mda_file_info.groupby(
        ['animal', 'date', 'electrode_number'])
    logging.info(f'Processing {len(electrodes)} electrodes...')
    logging.info(f'Temp directory: {os.getenv("ML_TEMPORARY_DIRECTORY")}')

    if len(electrodes) == 0:
        logging.warn('No electrodes for sorting found. Check to see if your '
                     'input path is correctly pointing to the .mda files.')
        return

    for (animal, date, electrode_number), electrodes_df in electrodes:
        try:
            geom_file = electrodes_df.geom_filepath.unique()[0]
        except AttributeError:
            geom_file = electrodes_df.geom_filepath
        mda_filename = electrodes_df.mda_filepath.tolist()[0]
        preprocessing_folder = os.path.abspath(
            os.path.join(mda_filename, os.pardir, os.pardir, os.pardir))

        if mountainlab_output_folder is None:
            animal_folder = os.path.join(preprocessing_folder, os.pardir)
            mountainlab_output_folder = os.path.abspath(
                os.path.join(animal_folder, 'mountainlab_output'))

        spike_sort_electrode(
            animal, date, electrode_number, preprocessing_folder,
            mountainlab_output_folder, firing_rate_thresh, isolation_thresh,
            noise_overlap_thresh, peak_snr_thresh,
            extract_marks, extract_clips, clip_time, freq_min,
            freq_max, adjacency_radius, detect_threshold, detect_interval,
            detect_sign, artifacts_interval_size, artifacts_threshold,
            sampling_rate, geom=geom_file,
            drift_track=drift_track, burst_merge=burst_merge,
            num_workers=num_workers)


def spike_sort_electrode(animal, date, electrode_number, preprocessing_folder,
                         mountainlab_output_folder, firing_rate_thresh=0.01,
                         isolation_thresh=0.97,
                         noise_overlap_thresh=0.03, peak_snr_thresh=1.5,
                         extract_marks=True, extract_clips=True,
                         clip_time=1.5, freq_min=300, freq_max=6000,
                         adjacency_radius=-1, detect_threshold=3,
                         detect_interval=10, detect_sign=-1,
                         artifacts_interval_size=2000, artifacts_threshold=5,
                         sampling_rate=30000, geom=None, drift_track=True,
                         burst_merge=False, num_workers=2):
    '''Runs mountain sort on all electrodes in `mda_file_info`

    Parameters
    ----------
    animal : str
    date : str
    electrode_number : int
    preprocessing_folder : str
    mountainlab_output_folder : str
    firing_rate_thresh : float, optional
    isolation_thresh : float, optional
    noise_overlap_thresh : float, optional
    peak_snr_thresh : float, optional
        Lower threshold for peak SNR. Peak SNR is the peak absolute amplitude of
        the average waveform divided by the peak standard deviation.
    extract_marks : bool, optional
    extract_clips : bool, optional
    clip_time : float, optional
         Time (in ms) to extract around each spike.
    freq_min : float, optional
        The highpass or low cutoff of the filter in Hz.
    freq_max : float, optional
        The lowpass or high cutoff of the filter in Hz.
    adjacency_radius : float, optional
        The radius in µm that defines a neighborhood of channels on which to
        sort (default -1 to ignore and not require a geom.csv file, useful for
        tetrodes).
    detect_threshold : float, optional
        Spike detection threshold in standard deviations.
    detect_interval : int, optional
        Minimum number of timepoints between events detected on the same
        channel.
    detect_sign : int, optional
         The direction of spike to detect (-1 for negative, 1 for positive,
         0 for both). -1 is recommended for most recordings.
    sampling_rate : int, optional
        Number of samples per second.
    geom : ndarray or None, shape (n_contacts, 2), optional
        Geometry of the electrode. Important for probes.
    drift_track : bool, optional
        Use drift tracking.
    burst_merge : bool, optional
        Automatic merging of bursts.
    num_workers : int, optional
        Number of compute threads to use for sorting.
    '''
    date = str(date)

    # Setup log file
    mountain_out_electrode_dir = os.path.join(
        mountainlab_output_folder, date, f'nt{electrode_number}')
    os.makedirs(mountain_out_electrode_dir, exist_ok=True)

    logger = create_file_logger(
        animal, date, electrode_number, mountain_out_electrode_dir)

    logger.info(
        f'Processing animal: {animal}, date: {date}, '
        f'electrode: {electrode_number}')
    logger.info(f'Parameters: \n{pprint.pformat(locals())}')

    mda_opts = {'anim': animal,
                'date': date,
                'ntrode': electrode_number,
                'data_location': preprocessing_folder}
    raw_mda = os.path.join(mountain_out_electrode_dir, 'raw.mda')

    # Concatenate mda files from the same epoch
    if not os.path.isfile(raw_mda):
        logger.info('Concatenated .mda file does not exist.'
                    f'Creating {raw_mda}')
        # create params if it doesn't exist
        params_file = os.path.join(
            mountain_out_electrode_dir, 'params.json')
        if not os.path.isfile(params_file):
            params = {'samplerate': sampling_rate}
            with open(params_file, 'w') as f:
                json.dump(params, f, indent=4, sort_keys=True)
        pyp.concat_epochs(dataset_dir=mountain_out_electrode_dir,
                          mda_opts=mda_opts)

    logger.info(
        f'{animal} {date} nt{electrode_number} preprocessing waveforms...')
    pyp.filt_mask_whiten(
        dataset_dir=mountain_out_electrode_dir,
        output_dir=mountain_out_electrode_dir,
        freq_min=freq_min,
        freq_max=freq_max,
        artifacts_threshold=artifacts_threshold,
        artifacts_interval_size=artifacts_interval_size)

    logger.info(
        f'{animal} {date} nt{electrode_number} sorting spikes...')
    if drift_track:
        pyp.ms4_sort_on_segs(
            dataset_dir=mountain_out_electrode_dir,
            output_dir=mountain_out_electrode_dir,
            geom=geom,
            adjacency_radius=adjacency_radius,
            detect_threshold=detect_threshold,
            detect_interval=detect_interval,
            detect_sign=detect_sign,
            num_workers=num_workers,
            mda_opts=mda_opts)
    else:
        pyp.ms4_sort_full(
            dataset_dir=mountain_out_electrode_dir,
            output_dir=mountain_out_electrode_dir,
            geom=geom,
            adjacency_radius=adjacency_radius,
            detect_threshold=detect_threshold,
            detect_interval=detect_interval,
            num_workers=num_workers,
            detect_sign=detect_sign)

    if burst_merge:
        logger.info(
            f'{animal} {date} nt{electrode_number} merging burst parents...')
        pyp.merge_burst_parents(
            dataset_dir=mountain_out_electrode_dir,
            output_dir=mountain_out_electrode_dir)

        logger.info(
            f'{animal} {date} nt{electrode_number} adding curation tags...')
        pyp.add_curation_tags(
            dataset_dir=mountain_out_electrode_dir,
            output_dir=mountain_out_electrode_dir,
            metrics_input='metrics_merged.json',
            metrics_output='metrics_merged_tagged.json',
            firing_rate_thresh=firing_rate_thresh,
            isolation_thresh=isolation_thresh,
            noise_overlap_thresh=noise_overlap_thresh,
            peak_snr_thresh=peak_snr_thresh)

    if extract_marks:
        logger.info(
            f'{animal} {date} nt{electrode_number} extracting marks...')
        pyp.extract_marks(dataset_dir=mountain_out_electrode_dir,
                          output_dir=mountain_out_electrode_dir)

    if extract_clips:
        logger.info(
            f'{animal} {date} nt{electrode_number} extracting clips...')
        clip_size = np.ceil(clip_time * sampling_rate /
                            MS_IN_SECOND).astype(int)
        pyp.extract_clips(dataset_dir=mountain_out_electrode_dir,
                          output_dir=mountain_out_electrode_dir,
                          clip_size=clip_size)
    logger.info(f'{animal} {date} nt{electrode_number} done...')

def recalc_metrics_epoch(raw_mda_file_info,updated_mda='firings_processed.mda',
        mountainlab_output_folder=None,mv2_file='manualcuration',
        num_workers=4,rm_segment_intermediates=True,metrics_to_update='metrics_processed_epoch',opts=None,manual_only=True):
    '''post (manual) merge/delete, recalculate metrics for each epoch. This function carries manual 
    tags (stored in mv2) if you have one but also update based on thresholds.)

    Parameters
    ----------
    dataset_dir : str
    output_dir : str
    firings_in : str, optional
    metrics_to_update (OUTPUT): str, optional

    firing_rate_thresh : float, optional
    Clusters less than the firing rate threshold is excluded (spikes / s )
    isolation_thresh : float, optional
    Distance to a cluster of noise.
    noise_overlap_thresh : float, optional
    Fraction of “noise events” in a cluster.
    peak_snr_thresh : float, optional

    mv2_file: str, optional
    opts : None or dict, optional, can be used to control verbose or no

    '''
    electrodes = raw_mda_file_info.groupby(
        ['animal', 'date', 'electrode_number'])
    logging.info(f'Recalculating metrics for {len(electrodes)} electrodes...')
    logging.info(f'Temp directory: {os.getenv("ML_TEMPORARY_DIRECTORY")}')

    if len(electrodes) == 0:
        logging.warn('No electrodes for sorting found. Check to see if your '
                     'input path is correctly pointing to the .mda files.')
        return

    params={}
    params_count=0
    for (animal, date, electrode_number), electrodes_df in electrodes:

        #get preprocessing_folder, mountain_out_electrode_dir
        mda_filename = electrodes_df.mda_filepath.tolist()[0]
        preprocessing_folder = os.path.abspath(
            os.path.join(mda_filename, os.pardir, os.pardir, os.pardir))
        if mountainlab_output_folder is None:
            animal_folder = os.path.join(preprocessing_folder, os.pardir)
            mountainlab_output_folder = os.path.abspath(
                os.path.join(animal_folder, 'mountainlab_output'))
        date = str(date)
        mountain_out_electrode_dir = os.path.join(
            mountainlab_output_folder, date, f'nt{electrode_number}')
        
        #get raw mda address        
        raw_mda_opts = {'anim': animal,
                    'date': date,
                    'ntrode': electrode_number,
                    'data_location': preprocessing_folder}
        
        # Setup log file
        log_file = os.path.join(mountainlab_output_folder, f'{animal}_{date}_nt{electrode_number}.log')
        logger_e = logging.getLogger(log_file)
        logger_e.info(
            f'Recalculating animal: {animal}, date: {date}, '
            f'electrode: {electrode_number}')
        logger_e.info(f'Parameters: \n{pprint.pformat(locals())}')

        params[params_count]=(mountain_out_electrode_dir,raw_mda_opts,opts)
        params_count=params_count+1

    # Make partial so that multiprocessing can iterate through
    helper=functools.partial(pyp.recalc_metrics_epoch_electrode,params,opts=opts,rm_segment_intermediates=rm_segment_intermediates,
                    updated_mda=updated_mda,mv2_file=mv2_file,metrics_to_update=metrics_to_update,manual_only=manual_only)

    pool = multiprocessing.Pool(num_workers)
    pool.map(helper, range(0,params_count))
    pool.close()
    pool.join()

    

def get_mda_files_dataframe(data_path, recursive=False):
    '''

    Parameters
    ----------
    data_path : str
    recursive : bool, optional
        Recursive search using glob.

    Returns
    -------
    mda_files_dataframe : pandas.DataFrame

    Notes
    -----
    Geom file is expected to be in the format:
        {animal}/preprocessing/{date}/{date}_{animal}.mountain/nt{ntrode_number}/geom.csv

    Examples
    --------
    ```
    all_animal_info = get_mda_files_dataframe(
        '/data2/data1_backup/anna/*/preprocessing/')
    ```

    ```
    single_animal_info = get_mda_files_dataframe(
        '/data2/data1_backup/anna/remy/preprocessing/')
    ```
    '''

    mda_files = glob.glob(os.path.join(data_path, '*', '*.mda', '*.*.mda'),
                          recursive=recursive)
    file_info = [_get_mda_file_information(mda_file) for mda_file in mda_files
                 if _get_mda_file_information(mda_file) is not None]
    COLUMNS = ['animal', 'date', 'epoch', 'electrode_number', 'task',
               'mda_filepath']
    mda_df = (pd.DataFrame(file_info, columns=COLUMNS)
              .set_index(['animal', 'date', 'epoch', 'electrode_number'])
              .sort_index())

    geom_df = get_geom_files_dataframe(data_path, recursive=recursive)

    return (mda_df
            .join(geom_df)
            .replace(dict(geom_filepath={np.nan: None})))


def _get_mda_file_information(mda_file):
    mda_re = re.compile(
        "^(?:(\d*)_)(?:(\w*)_)(\d*)(?:_(\w*)){0,1}(?:\.[a-zA-Z]*(\d*))\.\w*$"
    )
    try:
        match_re = mda_re.match(os.path.basename(mda_file))
        date, animal, epoch, task, electrode_number = match_re.groups()
        date, epoch, electrode_number = int(
            date), int(epoch), int(electrode_number)
        return animal, date, epoch, electrode_number, task, mda_file
    except (ValueError, AttributeError):
        pass


def get_geom_files_dataframe(data_path, recursive=False):
    '''

    Parameters
    ----------
    data_path : str
    recursive : bool, optional
        Recursive search using glob.

    Returns
    -------
    geom_files_dataframe : pandas.DataFrame

    '''

    geom_files = glob.glob(os.path.join(data_path, '*', '*', '*', 'geom.csv'),
                           recursive=recursive)
    file_info = [_get_geom_file_information(geom_file)
                 for geom_file in geom_files]
    COLUMNS = ['animal', 'date', 'electrode_number',
               'geom_filepath']
    return (pd.DataFrame(file_info, columns=COLUMNS)
            .set_index(['animal', 'date', 'electrode_number'])
            .sort_index())


def _get_geom_file_information(geom_file):
    try:
        *_, animal, _, date, _, electrode_number, _ = geom_file.split('/')
        electrode_number = int(electrode_number.strip('nt'))
        date = int(date)
        geom_filepath = os.path.abspath(geom_file)

        return animal, date, electrode_number, geom_filepath
    except ValueError:
        pass


def create_file_logger(animal, date, electrode_number, log_directory,
                       level=logging.DEBUG):
    log_file = os.path.join(
        log_directory, f'{animal}_{date}_nt{electrode_number}.log')
    logger = logging.getLogger(f'{animal}_{date}_nt{electrode_number}')
    logger.setLevel(level)
    format_string = '%(asctime)s — %(name)s — %(message)s'
    log_format = logging.Formatter(format_string, datefmt='%d-%b-%y %H:%M:%S')

    # Creating and adding the file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger
