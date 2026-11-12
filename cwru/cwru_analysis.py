from pathlib import Path
from dataclasses import dataclass, asdict, replace
import argparse
import hashlib
import json
import math
import platform
import urllib.request
import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parent
STATISTICS = ['Count', 'Block sum', 'Energy', 'Capped energy', 'Interval scan', 'Run statistic']
METHODS = ['Direct count', 'Record count'] + STATISTICS[1:]
SEED = 20261110


@dataclass(frozen=True)
class Design:
    name: str = 'Primary'
    fit_pairs: int = 2000
    point_scores: int = 200
    r: int = 10
    q: int = 10
    m: int = 20
    K: int = 199
    gap: int = 50
    law: str = 'Gaussian'
    spaced_evaluation: bool = False

    @property
    def span(self):
        return self.r + (self.m - 1) * self.q + 1

    @property
    def stride(self):
        return self.span + self.gap if self.spaced_evaluation else 5 * self.r


DESIGNS = [Design(), Design(name='r5', r=5, q=5), Design(name='r20', r=20, q=20),
           Design(name='m40', m=40), Design(name='gap500', gap=500),
           Design(name='K399', K=399), Design(name='Long training', fit_pairs=20000, point_scores=2000),
           Design(name='Student score', law='Student'), Design(name='Spaced evaluation', spaced_evaluation=True)]


def record_floor(m, budget=.1, gamma=1.):
    pmf = np.array([1.])
    for j in range(1, m + 1):
        probability = 1 / j if gamma == 1 else -np.expm1(-np.log(gamma)) / -np.expm1(-j * np.log(gamma))
        pmf = np.convolve(pmf, [1 - probability, probability])
    return next(k for k in range(1, m + 2) if pmf[k:].sum() <= budget), pmf


def load_series(item, data_dir):
    path = data_dir / f"{item['file_id']}.mat"
    if not path.exists():
        with urllib.request.urlopen(item['url'], timeout=90) as response:
            path.write_bytes(response.read())
    raw = path.read_bytes()
    if item.get('sha256') and hashlib.sha256(raw).hexdigest() != item['sha256']:
        raise ValueError(f"Source checksum changed: {path.name}")
    data = loadmat(path)
    y = np.asarray(data[item['channel']], dtype=float).ravel()
    if len(y) != item['observations'] or not np.all(np.isfinite(y)):
        raise ValueError(f"Invalid series: {path.name} / {item['channel']}")
    return y


def fit_model(y, design):
    train = y[:design.fit_pairs + 1]
    center, scale = train.mean(), train.std(ddof=1)
    if not scale > 0:
        raise ValueError('Zero training scale')
    x = (train - center) / scale
    lag, response = x[:-1], x[1:]
    matrix = np.column_stack((np.ones(len(lag)), lag, lag * np.exp(-.5 * lag ** 2)))
    b = np.linalg.lstsq(matrix, response, rcond=None)[0]
    residual = response - matrix @ b
    v = np.linalg.lstsq(np.column_stack((np.ones(len(lag)), lag ** 2)), residual ** 2, rcond=None)[0]
    v_raw = v.copy()
    v[0], v[1] = max(v[0], .1), max(v[1], 0.)
    fisher = 1. if design.law == 'Gaussian' else 1.25
    information = fisher * np.mean(1 / (v[0] + v[1] * lag ** 2))
    return {'center':float(center), 'scale':float(scale), 'b':b.tolist(), 'v':v.tolist(),
            'v_raw':v_raw.tolist(), 'information':float(information), 'condition_number':float(np.linalg.cond(matrix))}


def score_series(y, model, design):
    x = (y - model['center']) / model['scale']
    b, v = model['b'], model['v']
    mean = b[0] + b[1] * x[:-1] + b[2] * x[:-1] * np.exp(-.5 * x[:-1] ** 2)
    sd = np.sqrt(v[0] + v[1] * x[:-1] ** 2)
    innovation = (x[1:] - mean) / sd
    psi = innovation if design.law == 'Gaussian' else 6 * innovation / (3 + innovation ** 2)
    contribution = psi / sd / np.sqrt(model['information'])
    z = np.full(len(y), np.nan)
    z[design.r:] = np.convolve(contribution, np.ones(design.r) / np.sqrt(design.r), mode='valid')
    return z, innovation


def blocks(z, starts, design):
    return z[starts[:, None] + design.r + design.q * np.arange(design.m)[None, :]]


def statistics(s, c0):
    absolute = np.abs(s)
    count = (absolute > c0).sum(axis=1)
    block_sum = np.abs(s.sum(axis=1)) / np.sqrt(s.shape[1])
    energy = (s ** 2).sum(axis=1)
    capped = np.minimum(s ** 2, c0 ** 2).sum(axis=1)
    cs = np.column_stack((np.zeros(len(s)), np.cumsum(s, axis=1)))
    scan = np.zeros(len(s))
    for width in range(3, s.shape[1] + 1):
        scan = np.maximum(scan, np.abs(cs[:, width:] - cs[:, :-width]).max(axis=1) / np.sqrt(width))
    run = np.minimum(np.minimum(absolute[:, :-2], absolute[:, 1:-1]), absolute[:, 2:]).max(axis=1)
    return np.column_stack((count, block_sum, energy, capped, scan, run))


def rank_cutoff(v, alpha):
    order = int(np.ceil((len(v) + 1) * (1 - alpha) - 1e-12))
    return (np.sort(v, axis=0)[order - 1] if order <= len(v) else np.full(v.shape[1], np.inf)), order


def calibrate(y, design):
    model = fit_model(y, design)
    z, innovation = score_series(y, model, design)
    point_lo = design.fit_pairs + 1 + design.gap
    point_ends = point_lo + design.r * np.arange(1, design.point_scores + 1)
    c0 = float(np.quantile(np.abs(z[point_ends]), .95, method='linear'))
    cal_lo = int(point_ends[-1]) + design.gap + 1
    starts = cal_lo + (design.span + design.gap) * np.arange(design.K)
    cal_hi = int(starts[-1] + design.span - 1)
    holdout_lo = cal_hi + 2 * design.gap + 1
    if holdout_lo + design.span > len(y):
        raise ValueError(f'Insufficient normal data for {design.name}')
    s = blocks(z, starts, design)
    v = statistics(s, c0)
    lower, h_lower = rank_cutoff(v, .05)
    upper, h_upper = rank_cutoff(v, .01)
    a = int(lower[0])
    kd = int(max(a + 2, upper[0] + 1))
    floor, pmf = record_floor(design.m)
    kr = max(kd, floor)
    model.update({'design':asdict(design), 'c0':c0, 'a':a, 'k_direct':kd, 'record_floor':floor, 'k_record':kr,
                  'lower':lower.tolist(), 'upper':upper.tolist(), 'h_lower':h_lower, 'h_upper':h_upper,
                  'point_lo':point_lo, 'point_hi':int(point_ends[-1]), 'cal_lo':cal_lo,
                  'cal_hi':cal_hi, 'holdout_lo':holdout_lo, 'span':design.span, 'stride':design.stride})
    return model, z, innovation, starts, s, v


def labels(v, model):
    lo, hi = np.asarray(model['lower']), np.asarray(model['upper'])
    count = v[:, 0]
    direct = np.where(count >= model['k_direct'], 2, np.where(count > model['a'], 1, 0))
    record = np.where(count >= model['k_record'], 2, np.where(count > model['a'], 1, 0))
    others = np.where(v[:, 1:] > hi[1:], 2, np.where(v[:, 1:] > lo[1:], 1, 0))
    return np.column_stack((direct, record, others)).astype(np.int8)


def acf(x, max_lag=100):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    result = fftconvolve(x, x[::-1], mode='full')[len(x) - 1:len(x) + max_lag]
    return result / result[0] if result[0] > 0 else np.full(max_lag + 1, np.nan)


def summarize(item, design, d, counts, model):
    rows = []
    for j, method in enumerate(METHODS[:d.shape[1]]):
        row = {k:item[k] for k in ['file_id', 'label', 'load_hp', 'fault_type', 'diameter_in', 'outer_position', 'cohort']}
        row.update(design=design.name, method=method, blocks=len(d), low=int((d[:, j] == 0).sum()),
                   intermediate=int((d[:, j] == 1).sum()), high=int((d[:, j] == 2).sum()),
                   warning=int((d[:, j] > 0).sum()), high_rate=float(np.mean(d[:, j] == 2)),
                   warning_rate=float(np.mean(d[:, j] > 0)), mean_count=float(counts.mean()),
                   a=model['a'], k_direct=model['k_direct'], k_record=model['k_record'])
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--designs', nargs='+', choices=[d.name for d in DESIGNS],
                        help='Optional exact design names; default runs every design')
    args = parser.parse_args()
    data_dir, out = ROOT / 'data', ROOT / 'results'
    data_dir.mkdir(exist_ok=True); out.mkdir(exist_ok=True)
    manifest = json.loads((data_dir / 'data_manifest.json').read_text())
    normal = {x['load_hp']:x for x in manifest if x['cohort'] == 'normal'}
    all_series = {x['file_id']:load_series(x, data_dir) for x in manifest}
    selected_designs = [d for d in DESIGNS if args.designs is None or d.name in args.designs]
    metrics, thresholds, coefficients, calibrations, matched, gamma_rows, acfs, diagnostic_rows, stability = [], [], [], [], [], [], [], [], []
    primary_frames = []
    for design in selected_designs:
        for load in range(4):
            base_item = normal[load]; base_y = all_series[base_item['file_id']]
            model, z0, innovation, cal_starts, cal_scores, cal_stats = calibrate(base_y, design)
            model_file = out / f'model_{design.name.replace(" ", "_")}_{load}.json'
            model_file.write_text(json.dumps(model, indent=2))
            thresholds.append(dict(design=design.name, load_hp=load, **asdict(replace(design, name=design.name)),
                                   c0=model['c0'], a=model['a'], k_direct=model['k_direct'],
                                   record_floor=model['record_floor'], k_record=model['k_record'],
                                   record_active=model['k_record'] > model['k_direct'], span=design.span,
                                   stride=design.stride, holdout_start_row=model['holdout_lo'] + 1))
            coefficients.append(dict(design=design.name, load_hp=load, center=model['center'], scale=model['scale'],
                                     b0=model['b'][0], b1=model['b'][1], b2=model['b'][2],
                                     v0=model['v'][0], v1=model['v'][1], raw_v0=model['v_raw'][0],
                                     raw_v1=model['v_raw'][1], information=model['information'], condition_number=model['condition_number']))
            if design.name == 'Primary':
                frame = pd.DataFrame(cal_stats, columns=STATISTICS)
                frame.insert(0, 'start_row', cal_starts + 1); frame.insert(0, 'load_hp', load)
                calibrations.append(frame)
                for channel, values, unit in [('innovations', innovation[model['cal_lo']:model['cal_hi']], 1),
                                               ('micro_scores', z0[np.arange(model['cal_lo'] + design.r, model['cal_hi'] + 1, design.q)], design.q),
                                               ('calibration_counts', cal_stats[:, 0], design.span + design.gap)]:
                    for lag, value in enumerate(acf(values, min(100, len(values) - 1))):
                        acfs.append(dict(load_hp=load, series=channel, lag=lag, raw_lag=lag * unit, acf=float(value)))
                p_hat = cal_stats[:, 0].mean() / design.m
                diagnostic_rows.append(dict(load_hp=load, calibration_mean_count=float(cal_stats[:, 0].mean()),
                                           calibration_count_variance=float(cal_stats[:, 0].var(ddof=1)),
                                           calibration_exceedance_rate=float(p_hat),
                                           binomial_variance_at_empirical_rate=float(design.m*p_hat*(1-p_hat)),
                                           binomial_variance_at_nominal_rate=design.m*.05*.95,
                                           max_abs_score_acf_1_20=float(np.max(np.abs(acf(z0[np.arange(model['cal_lo'] + design.r, model['cal_hi'] + 1, design.q)], 20)[1:])))))
                ref_u = np.random.default_rng(np.random.SeedSequence([SEED, load, 100])).random(cal_stats.shape)
                tie_values = np.empty(6); tie_u = np.empty(6)
                for j in range(6):
                    idx = np.lexsort((ref_u[:, j], cal_stats[:, j]))[model['h_upper'] - 1]
                    tie_values[j], tie_u[j] = cal_stats[idx, j], ref_u[idx, j]
            for item in [x for x in manifest if x['load_hp'] == load]:
                y = all_series[item['file_id']]
                z = z0 if item['cohort'] == 'normal' else score_series(y, model, design)[0]
                first = model['holdout_lo'] if item['cohort'] == 'normal' else 0
                starts = np.arange(first, len(y) - design.span + 1, design.stride)
                s = blocks(z, starts, design)
                counts = (np.abs(s) > model['c0']).sum(axis=1)
                if design.name == 'Primary':
                    vals = statistics(s, model['c0']); d = labels(vals, model)
                    frame = pd.DataFrame(vals, columns=STATISTICS)
                    frame['file_id'] = item['file_id']; frame['start_row'] = starts + 1
                    for j, method in enumerate(METHODS): frame[method + ' label'] = d[:, j]
                    primary_frames.append(frame)
                    u = np.random.default_rng(np.random.SeedSequence([SEED, item['file_id'], 200])).random(vals.shape)
                    reject = (vals > tie_values) | ((vals == tie_values) & (u > tie_u))
                    for j, statistic in enumerate(STATISTICS):
                        matched.append(dict(file_id=item['file_id'], load_hp=load, cohort=item['cohort'],
                                            fault_type=item['fault_type'], diameter_in=item['diameter_in'], outer_position=item['outer_position'],
                                            statistic=statistic, blocks=len(vals), rejected=int(reject[:, j].sum()),
                                            rate=float(reject[:, j].mean()), threshold=float(tie_values[j]), tie_threshold=float(tie_u[j])))
                    for gamma in [1., 1.02, 1.05, 1.10, 1.25]:
                        floor, _ = record_floor(design.m, gamma=gamma)
                        cutoff = max(model['k_direct'], floor)
                        gamma_rows.append(dict(file_id=item['file_id'], load_hp=load, cohort=item['cohort'],
                                               fault_type=item['fault_type'], diameter_in=item['diameter_in'], gamma=gamma,
                                               record_floor=floor, cutoff=cutoff, blocks=len(counts),
                                               high=int((counts >= cutoff).sum()), high_rate=float(np.mean(counts >= cutoff))))
                    for bin_id, ix in enumerate(np.array_split(np.arange(len(starts)), 8), 1):
                        stability.append(dict(file_id=item['file_id'], load_hp=load, cohort=item['cohort'], bin=bin_id,
                                              first_sample=int(starts[ix[0]] + 1), last_sample=int(starts[ix[-1]] + design.span),
                                              blocks=len(ix), mean_count=float(counts[ix].mean()),
                                              direct_high_rate=float(np.mean(d[ix, 0] == 2)),
                                              record_high_rate=float(np.mean(d[ix, 1] == 2))))
                else:
                    d = np.column_stack([np.where(counts >= k, 2, np.where(counts > model['a'], 1, 0))
                                         for k in [model['k_direct'], model['k_record']]])
                assert np.all(d[:, 1] <= d[:, 0])
                assert np.array_equal(d[:, 0] > 0, d[:, 1] > 0)
                metrics.extend(summarize(item, design, d, counts, model))
            print('Completed', design.name, 'load', load, 'cutoffs', model['a'], model['k_direct'], model['k_record'], flush=True)
    pd.DataFrame(metrics).to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(thresholds).to_csv(out / 'thresholds.csv', index=False)
    pd.DataFrame(coefficients).to_csv(out / 'model_coefficients.csv', index=False)
    if primary_frames:
        pd.concat(primary_frames, ignore_index=True).to_csv(out / 'primary_blocks.csv.gz', index=False, compression='gzip')
        pd.concat(calibrations, ignore_index=True).to_csv(out / 'calibration_blocks.csv', index=False)
        pd.DataFrame(matched).to_csv(out / 'common_nominal_rank.csv', index=False)
        pd.DataFrame(gamma_rows).to_csv(out / 'record_benchmarks.csv', index=False)
        pd.DataFrame(acfs).to_csv(out / 'reference_autocorrelations.csv', index=False)
        pd.DataFrame(diagnostic_rows).to_csv(out / 'reference_diagnostics.csv', index=False)
        pd.DataFrame(stability).to_csv(out / 'within_record_stability.csv', index=False)
    versions = {'python':platform.python_version(), 'numpy':np.__version__, 'scipy':scipy.__version__, 'pandas':pd.__version__, 'seed':SEED}
    (out / 'software_versions.json').write_text(json.dumps(versions, indent=2))
    print('Analysis complete:', out)


if __name__ == '__main__':
    main()
