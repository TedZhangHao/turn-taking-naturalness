# Manifest format

Training CSVs (`train.csv`, `dev.csv`, and `test1.csv`) use one stereo recording per row:

```csv
id,session_id,source_type,audio_path,json_path,duration_sec,split
sample_001,sample_001,file,/path/to/stereo.wav,/path/to/meta.json,24.95,train
```

Alternatively, Seamless Interaction participant pairs may use
`participant1_relpath_abs`, `participant2_relpath_abs`,
`participant1_json_abs`, and `participant2_json_abs` columns.

Naturalness manifests use the schema demonstrated by `samples/manifest.csv`:
each row points to an edited recording and its paired natural recording.
