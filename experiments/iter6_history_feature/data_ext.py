"""iter6: KuaiRand-Pure 数据加载 + 编码，在原始 5 个特征域基础上新增第 6 个域
`hist_affinity` —— 该用户此前对该 author 的 long_view=1 交互次数（严格因果，只看更早的
date）。不修改 ../../data.py，这里复制+改造它的 load()/encode() 逻辑。

因果性设计（最重要的正确性约束）：
  1. 把 train+valid+test 三个 split 的原始行合并成一个"全局时间线"。因为
     SPLITS 的日期区间本身就是 train < valid < test 不重叠、单调递增的，所以
     "合并后按 date 排序，用严格 `<` 比较" 天然保证了：
       - train 的行只能看到更早的 train 行（不可能看到 valid/test，因为它们的
         date 全部更大）
       - valid/test 的行可以看到所有更早日期的行（含 train 的全部历史，以及
         同 split 内更早日期的行），符合"部署时模型只能看到过去"的语义
  2. 按 date 分组处理（而不是按行处理）：同一天内的所有行，先一次性读取
     "进入这一天之前"的计数器状态（这一步内，同天的行互相看不见对方，也看
     不见自己），处理完这一天的全部行之后，才把这一天里 label=1 的行更新进
     计数器。这保证了"同一 date 视为同时发生，不区分先后，严格 `<` 而非 `<=`"。
  3. 计数器 key 是 (user_id, author_id)，累加的是历史上 label==1 的次数，
     不含当前行自身。

这样得到的 hist_affinity 计数在写回每一行时，天然满足"该计数是该行发生前已确定
的信息"，不会用到同一行或未来行的 label，因此不会泄漏标签。
"""
import os, sys, csv, collections
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data import load as _load_raw, SPLITS  # noqa: E402  (reuse original raw loader, untouched)

LABEL = 'long_view'
FIELDS_EXT = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'hist_affinity']


def compute_history_counts(rows):
    """rows: list of (date, user_id, video_id, author_id, tab, duration_ms, label),
    in ANY order (does not need to be pre-sorted).
    Returns: list[int] aligned index-for-index with `rows` — for each row, the
    number of strictly-earlier-date rows with the same (user_id, author_id) and
    label==1. Same-date rows never count each other (strict `<`, not `<=`)."""
    n = len(rows)
    order = sorted(range(n), key=lambda i: rows[i][0])
    counts = [0] * n
    counter = collections.defaultdict(int)
    i = 0
    while i < n:
        j = i
        d = rows[order[i]][0]
        while j < n and rows[order[j]][0] == d:
            j += 1
        # 1) 先读取"这一天开始之前"的计数器状态，赋给这一天所有的行
        for idx in order[i:j]:
            r = rows[idx]
            counts[idx] = counter[(r[1], r[3])]  # (user_id, author_id)
        # 2) 再用这一天里 label==1 的行去更新计数器（下一天才会看到）
        for idx in order[i:j]:
            r = rows[idx]
            if r[6] == 1:
                counter[(r[1], r[3])] += 1
        i = j
    return counts


def bucket_history_count(c):
    """0/1/2/3-5/6+ —— 5 个桶。放弃分位数分桶：该计数分布高度偏零（绝大多数
    user-author 组合此前从未有过 long_view=1 交互），分位数在这种零堆积分布下
    会退化（多个分位点重合在 0 上，桶数实际不到 5-10 个且区分度很差）。改用
    固定的、按业务直觉设计的桶：0 次 / 1 次 / 2 次 / 3-5 次 / 6+ 次，能在低计数
    区间保留最大区分度（那里样本最密集），同时把长尾压缩掉。"""
    if c <= 0:
        return 0
    elif c == 1:
        return 1
    elif c == 2:
        return 2
    elif c <= 5:
        return 3
    else:
        return 4


def load_ext(data_dir):
    """返回按 (train/valid/test) 划分好的行列表，每行在原始 7 元组基础上追加
    (hist_count, hist_bucket) 两个字段 —— 即 9 元组：
    (date, user_id, video_id, author_id, tab, duration_ms, label, hist_count, hist_bucket)
    """
    splits = _load_raw(data_dir)  # 原始 5 特征域用的同一个 loader，未修改
    order = ('train', 'valid', 'test')
    flat = []
    owner = []
    for name in order:
        for r in splits[name]:
            flat.append(r)
            owner.append(name)

    counts = compute_history_counts(flat)
    buckets = [bucket_history_count(c) for c in counts]

    ext = {name: [] for name in order}
    for r, name, c, b in zip(flat, owner, counts, buckets):
        ext[name].append(r + (c, b))
    return ext


def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])


def encode_ext(splits):
    """splits: 来自 load_ext() 的 dict，每行 9 元组。
    返回 (enc, field_dims_sum)，enc[name] = (X, y, users)，X 为 (N, 6) int32。
    与 data.py 的 encode() 逻辑完全一致，只是 raw() 多拼一个 hist_bucket 域，
    且该域的桶边界是固定规则（bucket_history_count），dur_bucket 依旧是分位数
    （只用 train 行拟合，与原版一致）。"""
    tr = splits['train']
    edges = _bucket_edges([x[5] for x in tr])

    def raw(x):
        return [x[1], x[2], x[3], x[4],
                str(int(np.searchsorted(edges, x[5]))),
                str(x[8])]  # x[8] = hist_bucket，已经是离散值

    vocabs = [dict() for _ in FIELDS_EXT]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS_EXT)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))


if __name__ == '__main__':
    # 因果性人工抽查：打印几条样本行 (user_id, author_id, date, hist_count)，
    # 手动对照原始日志验证。
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.join(os.path.dirname(__file__), '..', '..',
                                                         'KuaiRand-Pure', 'data'))
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    ext = load_ext(a.data_dir)
    print({k: len(v) for k, v in ext.items()})

    flat = ext['train'] + ext['valid'] + ext['test']
    # 找几个 hist_count > 0 的例子，手动验证
    examples = [r for r in flat if r[7] >= 2][:5]
    print("\n--- sample rows with hist_count >= 2 (manual sanity check) ---")
    for r in examples:
        date, uid, vid, aid, tab, dur, label, hc, hb = r
        print(f"date={date} user={uid} author={aid} label={label} hist_count={hc} hist_bucket={hb}")
        # 手动重算：数一下 flat 中该 (uid, aid) 且 date < date 且 label==1 的行数
        manual = sum(1 for rr in flat if rr[1] == uid and rr[3] == aid and rr[0] < date and rr[6] == 1)
        same_date_pos = sum(1 for rr in flat if rr[1] == uid and rr[3] == aid and rr[0] == date and rr[6] == 1)
        print(f"    manual recount (date < {date}): {manual}  "
              f"(same-date positives, correctly excluded: {same_date_pos})")
        assert manual == hc, "CAUSALITY BUG: recount mismatch!"
    print("\nAll manual recounts match. No same-date or future leakage detected.")

    # 额外抽查：随便挑一个 hist_count == 0 的行，确认它确实没有更早的正例
    zero_examples = [r for r in flat if r[7] == 0][:3]
    print("\n--- sample rows with hist_count == 0 ---")
    for r in zero_examples:
        date, uid, vid, aid, tab, dur, label, hc, hb = r
        manual = sum(1 for rr in flat if rr[1] == uid and rr[3] == aid and rr[0] < date and rr[6] == 1)
        print(f"date={date} user={uid} author={aid} label={label} hist_count={hc}  manual={manual}")
        assert manual == 0
    print("OK.")

    dist = collections.Counter(r[8] for r in flat)
    print("\nhist_bucket distribution across all rows:", dict(sorted(dist.items())))
