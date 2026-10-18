# Archive: template matching (abandoned)

The first approach, kept because the report discusses why it was rejected. These scripts
are historical: they read the old flat `outputs/*.csv` layout and are not maintained.

- `make_template.py` cut tag templates (`templates/`) from frames with a clear tag.
- `tag_matcher.py` and `run_matching.py` matched those templates against every person crop
  at several scales and rotations.
- `visualize_matches.py` and `track_summary.py` inspected the scores.
- `score_plot.png` is the result: the best match score per frame is about the same whether
  or not a tag is visible, so no threshold separates the two.
- `make_dataset.py` turned the highest-scoring matches into training labels. The detector
  trained on them learned the staff member's appearance rather than the tag, which is why
  the final dataset was labelled by hand.
