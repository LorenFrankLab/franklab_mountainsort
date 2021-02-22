# franklab_mountainsort

### Installation
1. Check if miniconda or anaconda is installed
https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

2. Clone git repository
```bash
git clone https://github.com/LorenFrankLab/franklab_mountainsort.git
```
3. Go into the franklab_mountainsort folder
```bash
cd franklab_mountainsort
```
4. Create conda environment
```bash
conda env create -f environment.yml
```
5. activate conda environment
```bash
conda activate franklab_mountainsort
```
6. Install package code
```bash
python setup.py develop
```
7. Check if everything installed correctly. Open jupyter notebook, jupyter lab, or python console. Try to import franklab_mountainsort.
```python
import franklab_mountainsort
```

### Usage
1. Define input data directory, output data, temp directory
2. Create a pandas dataframe with `.mda` file information (using function `get_mda_files_dataframe`)
3. Run spike sorting on all or a subset of the DataFrame (using function `spike_sort_all`)
   - This will concatenate all epochs in a day
   - Bandpass filter and whiten the data
   - Run the isosplit clustering algorithm
   - Merge clusters that look like they are from one cell that is bursting (optional).
   - Add curation metrics and tags (statistics about each cluster and whether it is multiunit or not.).
   - Extract clip waveforms (optional)
   - Extract marks (optional)
4. Open qt-mountainview (make sure you've activated the conda environment. If not type `conda activate franklab_mountainsort` into the terminal.)
```
qt-mountainview --pre=pre.mda.prv --firings=firings_raw.mda --samplerate=30000 --cluster_metrics=metrics_raw.json
```

### Common Issues
+ Make sure the version of trodes you used to extract the data is the same that you recorded with.
+ UnhandledPromiseRejectionWarning: The `sumit.json` file, located in `~/.mountainlab`, can get corrupted if something interrupts writing to that file, breaking the `.json` formatting. If this is the case, either fix the json formatting or simply remove it. Note that removing the git repo or conda environment will not fix this problem because the `.mountainlab` folder is not removed when removing these two things.

### What does `spike_sort_all` do?
For each set of channels on an electrode:
1. Concatenate timeseries over epochs
2. Bandpass filter
	+ between 300 and 6000 Hz by default
	+ relevant arguments: `freq_min`, `freq_max`, `sampling_rate`
3. Mask out high voltage artifacts
	+ remove chunks of data with amplitudes outside of the expected range by replacing these data by zeros
	+ probably should expose these parameters
4. Whiten
	+ decorrelate signals

If `drift_track`:

5. Identify clusters on each epoch individually
	+ using the isosplit clustering algorithm
	+ relevant arguments: `adjacency_radius`, `detect_threshold`, `detect_interval`, `detect_sign`, `num_workers`
6. Anneal segments
	+ stich epochs back together while matching clusters from the different epochs
	+ compares identified clusters across all epochs

##
7. Compute cluster metrics
	+ `firing rate`
	+ `isolation`: how well separated (in feature space) the cluster is from other nearby clusters
	+ `noise_overlap`: estimates the fraction of “noise events” in a cluster, i.e., above-threshold events not associated with true firings of this or any of the other clustered units.
	+ `peak_snr`: peak signal to noise ratio

If `burst_merge`:

8. Merge burst parents
	+ automatically merge clusters that are demeed to have come from the same bursting cell
##
9. Add curation tags
	+ Add tags to the metrics file to reflect which clusters should be
    rejected based on curation criteria.
    + `rejected`, `accepted`, `mua```
    + relevant arguments: `firing_rate_thresh`, `isolation_thresh`, `noise_overlap_thresh`, `peak_snr_thresh`

If `extract_marks`:

10. Extract marks
	+ extract the peak amplitude at the time of the spike

If `extract_clips`:

11. Extract clips
	+ extract the waveforms around the time of the spike
	+ relevant arguments: `clip_time`, `sampling_rate`

### Inputs to MountainSort
+ `*.mda`: time series for each recording.
+ `params.json`: contains information about the parameters to use for the sort. These can be overwritten by specifying them in the call to run the sort (in the batch script)
+ `geom.csv` (optional): contains information about the location of contacts for that ntrode; used in concert with adjacency_radius to determine the neighborhoods to sort on. In the case of tetrodes, this is not necessary because all the contacts of a tetrode should be sorted together as a single neighborhood. This can be specified by not providing a geom.csv, setting adjacency_radius to -1, or both.

### Note: `.prv` vs.`.mda` files
A `.prv` file is a text file with pointers to the binary file.
A `.mda` file is a binary file containing the actual data.

They can be used interchangeably when using `qt-mountainview`.


### Outputs from MountainSort
For each electrode:
+ `raw.mda.prv`: The concatenated time series for one day of recording. It is located in the `../<animal>/mountainlab_output/<date>/<date>_<animal>.mountain` folder
+ `clips.mda`: The waveforms around the spike times. Located in the `../<animal>/mountainlab_output/<date>/nt<number>/` folder.
+ `filt.mda.prv`: The time series after it has been bandpass filtered.  Located in the `../<animal>/mountainlab_output/<date>/nt<number>/` folder.
+ `firings_burst_merged.mda`: The firings file after burst merge processing.  Located in the `../<animal>/mountainlab_output/<date>/nt<number>/` folder.
+ `firings_raw.mda`: This contains the actual spike timing info that you care most about [electrode; time;label x #events….] for ALL detected events, regardless of cluster quality.  Located in the `../<animal>/mountainlab_output/<date>/nt<number>/` folder.
+ `metrics_merged.json`: This contains the metrics for the curated clusters, such as isolation scores, noise overlap, SNR, and more. Located in the `../<animal>/mountainlab_output/<date>/nt<number>/` folder.
+ `metrics_merged_tagged.json`: Same as metrics merged but with tags. Located in the `../<animal>/mountainlab_output/<date>/nt<number>/` folder.
+ `metrics_raw.json`: Metrics for all the original clusters. Located in the `../<animal>/mountainlab_output/<date>/nt<number>/` folder.
+ `pre.mda.prv`: The time series after it has been bandpass filtered and whitened. Located in the `../<animal>/mountainlab_output/<date>/nt<number>/` folder.
