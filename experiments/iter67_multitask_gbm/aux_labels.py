"""Re-parses the two raw log CSVs in the exact same concatenation order as
iter63's `_load_raw_time` (train file then valid+test file, no date
filtering applied here) to recover is_like/is_follow/is_comment/is_forward
per row. Row position in this loop equals `orig_idx` as defined in
iter63/data_ext.py's IDX (`len(rows)` at append time, before date
filtering) -- so `like[orig_idx]` etc. give the correct engagement labels
for any row already indexed by orig_idx.
"""
import csv, os
import numpy as np

FILES = ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv')


def load_aux_labels(data_dir):
    like, follow, comment, forward, longview = [], [], [], [], []
    for f in FILES:
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                like.append(1 if r['is_like'] != '0' else 0)
                follow.append(1 if r['is_follow'] != '0' else 0)
                comment.append(1 if r['is_comment'] != '0' else 0)
                forward.append(1 if r['is_forward'] != '0' else 0)
                longview.append(1 if r['long_view'] != '0' else 0)
    return dict(
        is_like=np.array(like, dtype=np.int8), is_follow=np.array(follow, dtype=np.int8),
        is_comment=np.array(comment, dtype=np.int8), is_forward=np.array(forward, dtype=np.int8),
        long_view=np.array(longview, dtype=np.int8))
