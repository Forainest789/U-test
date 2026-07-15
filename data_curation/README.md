# Data Curation

TransNetV2 is required for data curation: https://github.com/soCzech/TransNetV2

Install the optional Python dependencies before running the curation scripts:

```bash
pip install -r requirements_optional.txt
```

Place the TransNetV2 checkout at `data_curation/TransNetV2/` or pass `--model_path` to the exported `transnetv2-weights` directory.

## Pipeline

1. Detect shots and build clip records:

```bash
python data_curation/detect_shots.py \
  --path /path/to/raw_videos \
  --output_dir /path/to/curation_workdir \
  --model_path /path/to/transnetv2-weights \
  --export_clips
```

This writes `processed_shots.csv`, `clips_record/*_clips.json`, and optionally extracted clips under the output directory.

Output structure:

```text
/path/to/curation_workdir/
├── processed_shots.csv
├── abnormal_mp4name.log
├── clips_record/
│   └── Top001_clips.json
└── Top001/                         # only when --export_clips is used
    └── group_0/
        ├── clip1.mp4
        ├── clip2.mp4
        └── last_clip.mp4
```

`processed_shots.csv` sample:

```csv
video_id,shot_groups,filtered_shot_groups,filtered_shot_groups_avg_brightness_ge_1_8
Top001,"[[150, 360], ...]","[[150, 360], ...]","[(150, 360), ...]"
```

`clips_record/Top001_clips.json` sample:

```json
{
  "video_id": "Top001",
  "fps": 23.976023976023978,
  "groups": [
    {
      "group_index": 0,
      "shot_range": [150, 360],
      "clips": [
        {"name": "clip1", "start_frame": 150, "end_frame": 230},
        {"name": "clip2", "start_frame": 230, "end_frame": 310},
        {"name": "last_clip", "start_frame": 310, "end_frame": 360}
      ]
    }
  ]
}
```

2. Extract clips from existing clip records, if clips were not exported in step 1:

```bash
python data_curation/extract_clips.py \
  --record_dir /path/to/curation_workdir/clips_record \
  --video_dir /path/to/raw_videos \
  --output_dir /path/to/curation_workdir/video \
  --workers 4 \
  --ffmpeg_hwaccel none
```

Use `--ffmpeg_hwaccel cuda` only when the local ffmpeg build and GPU driver support CUDA decoding.

Output structure:

```text
/path/to/curation_workdir/video/
└── Top001/
    ├── group_0/
    │   ├── clip1.mp4
    │   ├── clip2.mp4
    │   └── last_clip.mp4
    └── group_1/
        ├── clip1.mp4
        └── last_clip.mp4
```

3. Recalculate clip boundaries with first/last-frame overlap:

```bash
python data_curation/recalc_clip_overlap.py \
  --input_dir /path/to/curation_workdir/clips_record \
  --output_dir /path/to/curation_workdir/clips_record_cross
```

Optional candidate filtering can be enabled by also passing `--candidate_groups_csv`, `--character_lists_dir`, and `--video_root`.

Output structure:

```text
/path/to/curation_workdir/clips_record_cross/
└── Top001_clips.json
```

`clips_record_cross/Top001_clips.json` sample:

```json
{
  "video_id": "Top001",
  "fps": 23.976023976023978,
  "groups": [
    {
      "group_index": 0,
      "shot_range": [150, 360],
      "clips": [
        {"name": "clip1", "start_frame": 150, "end_frame": 230},
        {"name": "clip2", "start_frame": 230, "end_frame": 310}
      ]
    }
  ]
}
```

4. Generate group-level and chunk-level captions:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1

python data_curation/caption_shot_groups.py \
  --clips_dir /path/to/curation_workdir/video \
  --output_dir /path/to/curation_workdir/caption \
  --model_name your-vlm-model \
  --workers 1
```

The prompt files are stored in `data_curation/prompts/` and can be overridden with `--global_prompt_path` and `--chunk_prompt_path`.

Output structure:

```text
/path/to/curation_workdir/caption/
├── Top143.json
└── sampled_frames/
    └── Top143/
        └── group_57/
            ├── frame_000.jpg
            ├── frame_001.jpg
            └── frame_002.jpg
```

`caption/Top143.json` sample:

```json
{
  "video_id": "Top143",
  "complete": true,
  "selected_group_indices": [57, 59, 60],
  "shots": [
    {
      "group_index": 57,
      "group_caption": "standing girl: A young woman with curly blonde hair wearing a light blue polo shirt stands up in a classroom; pink-shirted girl: A girl with long brown hair wearing a pink patterned shirt sits at the front...",
      "clips": [
        {
          "clip_index": 1,
          "caption": "Pink-shirted girl maintains a steady gaze toward the front; standing girl looks forward with a focused expression; scarfed boy and striped-shirted boy watch her intently...",
          "error": ""
        }
      ]
    }
  ]
}
```

5. Build character lists and candidate groups:

```bash
python data_curation/build_character_lists.py \
  --caption_dir /path/to/curation_workdir/caption
```

Output structure:

```text
/path/to/curation_workdir/caption/
├── candidate_groups.csv
└── character_lists/
    └── Top143.json
```

`caption/character_lists/Top143.json` sample:

```json
{
  "video_id": "Top143",
  "complete": true,
  "selected_group_indices": [57, 59, 60],
  "shots": [
    {
      "group_index": 57,
      "group_caption": "standing girl: A young woman with curly blonde hair wearing a light blue polo shirt stands up in a classroom; pink-shirted girl: A girl with long brown hair...",
      "clips": [
        {
          "clip_index": 1,
          "caption": "Pink-shirted girl maintains a steady gaze toward the front; standing girl looks forward with a focused expression...",
          "error": "",
          "characters": ["standing girl", "pink-shirted girl", "scarfed boy", "striped-shirted boy"],
          "overlapping_clip_indices": [2, 3, 4, 5]
        }
      ],
      "characters": ["standing girl", "pink-shirted girl", "scarfed boy", "striped-shirted boy"],
      "candidate_clips": [1, 2, 3, 4, 5]
    }
  ]
}
```

`caption/candidate_groups.csv` sample:

```csv
video_id,group_index,candidate_clips
Top143,57,1|2|3|4|5
Top143,59,1|2|3|4|5
```

6. Build stage-2 target/memory/update triplets:

```bash
python tools/build_stage2_candidate_groups.py \
  --character-lists-dir /path/to/curation_workdir/caption/character_lists \
  --video-root /path/to/curation_workdir/video \
  --output-csv /path/to/curation_workdir/caption/stage2_candidate_groups.csv \
  --require-videos
```

Output structure:

```text
/path/to/curation_workdir/caption/
├── candidate_groups.csv
├── stage2_candidate_groups.csv
└── character_lists/
    └── Top143.json
```

`caption/stage2_candidate_groups.csv` sample:

```csv
video_id,group_index,candidate_clips,target_clip,memory_clip,update_memory_clip,shared_characters
Top143,57,1|2|3,1,2,3,standing girl;pink-shirted girl;scarfed boy;striped-shirted boy
Top143,57,1|2|4,1,2,4,standing girl;pink-shirted girl;scarfed boy;striped-shirted boy
```

For training, set `DATA_ROOT` so the launchers can resolve:

```text
${DATA_ROOT}/caption/candidate_groups.csv
${DATA_ROOT}/caption/stage2_candidate_groups.csv
${DATA_ROOT}/caption/character_lists/
${DATA_ROOT}/video/
```
