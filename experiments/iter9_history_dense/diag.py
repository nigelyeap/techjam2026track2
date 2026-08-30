"""One-off diagnostic (not part of the pipeline): compute raw (unbucketed) values
for the candidate iter9 features across the full combined timeline and print
coverage / distribution stats, so bucket edges can be chosen sensibly before
committing to data_ext.py. Reuses data.load() unmodified."""
import os, sys, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from data import load

def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'KuaiRand-Pure', 'data')
    splits = load(data_dir)
    order = ('train', 'valid', 'test')
    flat = []
    for name in order:
        flat.extend(splits[name])
    n = len(flat)
    print(f"total rows: {n}")

    idx_sorted = sorted(range(n), key=lambda i: flat[i][0])
    activity = [0] * n       # count of prior rows (any label) for this user
    tab_pos = [0] * n        # count of prior label=1 rows, same user+tab
    prior_pos = [0] * n      # count of prior label=1 rows, this user (any tab)
    prior_total = [0] * n    # count of prior rows (any label) for this user (== activity, kept separate name for clarity)

    user_total = collections.defaultdict(int)
    user_pos = collections.defaultdict(int)
    user_tab_pos = collections.defaultdict(int)

    i = 0
    while i < n:
        j = i
        d = flat[idx_sorted[i]][0]
        while j < n and flat[idx_sorted[j]][0] == d:
            j += 1
        day_idx = idx_sorted[i:j]
        # read pre-day state
        for idx in day_idx:
            r = flat[idx]
            u, tab, label = r[1], r[4], r[6]
            activity[idx] = user_total[u]
            prior_total[idx] = user_total[u]
            prior_pos[idx] = user_pos[u]
            tab_pos[idx] = user_tab_pos[(u, tab)]
        # update with this day's rows
        for idx in day_idx:
            r = flat[idx]
            u, tab, label = r[1], r[4], r[6]
            user_total[u] += 1
            if label == 1:
                user_pos[u] += 1
                user_tab_pos[(u, tab)] += 1
        i = j

    activity = np.array(activity)
    tab_pos = np.array(tab_pos)
    prior_pos = np.array(prior_pos)
    prior_total = np.array(prior_total)

    print("\n--- feature 1: user activity (prior rows, any label) ---")
    print(f"coverage (nonzero): {np.mean(activity > 0)*100:.2f}%")
    print(f"quantiles [0,10,25,50,75,90,99,100]%: {np.percentile(activity, [0,10,25,50,75,90,99,100])}")
    print(f"value counts (first 20): {collections.Counter(activity.tolist()).most_common(20)}")

    print("\n--- feature 2: user-tab affinity (prior label=1 same tab) ---")
    print(f"coverage (nonzero): {np.mean(tab_pos > 0)*100:.2f}%")
    print(f"quantiles [0,10,25,50,75,90,99,100]%: {np.percentile(tab_pos, [0,10,25,50,75,90,99,100])}")
    print(f"value counts (first 20): {collections.Counter(tab_pos.tolist()).most_common(20)}")

    print("\n--- feature 3: user prior positive rate (Laplace smoothed) ---")
    alpha = 1.0
    rate = (prior_pos + alpha) / (prior_total + 2 * alpha)
    print(f"coverage (prior_total>0, i.e. not the global-prior default): {np.mean(prior_total > 0)*100:.2f}%")
    print(f"rate quantiles [0,10,25,50,75,90,99,100]%: {np.percentile(rate, [0,10,25,50,75,90,99,100])}")

    # sanity: how many distinct tab values?
    tabs = set(r[4] for r in flat)
    print(f"\ndistinct tab values: {len(tabs)} -> {sorted(tabs)[:20]}")
    users = set(r[1] for r in flat)
    print(f"distinct users: {len(users)}, rows/user avg: {n/len(users):.1f}")

if __name__ == '__main__':
    main()
