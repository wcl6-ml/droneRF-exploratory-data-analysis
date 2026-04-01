# This markdown file record the structure of stored .h5 file
## Two keys in the .h5 file
### metadata
There are 10 keys in metadata with each key contains 45400, which is the number of total segments

keys:
```
band: "L" or "H"

bui: Drone BUI specified in the paper, i.e. 00000, 00110,...

drone_type: Phantom, background, ..., etc

file_idx: The number of a .csv file for each drone_type

fs_hz: Sampling frequency, 40MHz as the paper suggested

label: 0-4 label, can be derived from drone_type

seg_within_file: The ith of segment in a .csv file

recording_id: bui_file-idx_seg-withing-file

segment_id: ith segments in the total dataset
```


### segments
Each segment is a segment of I/Q data with length 100000.

The actual numpy array of a segment is indexed by:
```python
h5['segments']['00000']['signal']
```

Each segment has its own metadata recorded in 
```python
h5['segments']['00000'].attrs[key]
```
There are 10 keys in segment with one additional "interleaved_iq" highlighting the mode of collected I/Q data. 
