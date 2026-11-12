from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from scipy.io import loadmat
import cwru_analysis as study

ROOT = Path(__file__).resolve().parent


def scalar_score(y, start, model, design):
    x = (y[start:start + design.r + 1] - model['center']) / model['scale']
    total = 0.
    for j in range(design.r):
        lag = x[j]
        mean = model['b'][0] + model['b'][1] * lag + model['b'][2] * lag * math.exp(-lag * lag / 2)
        variance = model['v'][0] + model['v'][1] * lag * lag
        residual = (x[j + 1] - mean) / math.sqrt(variance)
        psi = residual if design.law == 'Gaussian' else 6 * residual / (3 + residual * residual)
        total += psi / math.sqrt(variance)
    return total / math.sqrt(design.r * model['information'])


def main():
    manifest = json.loads((ROOT / 'data/data_manifest.json').read_text())
    metrics = pd.read_csv(ROOT / 'results/metrics.csv')
    blocks = pd.read_csv(ROOT / 'results/primary_blocks.csv.gz')
    checks = []
    for m in [1, 20, 40]:
        integers = [1]
        for n in range(1, m + 1):
            new = [0] * (n + 1)
            for k, value in enumerate(integers):
                new[k] += (n - 1) * value
                new[k + 1] += value
            integers = new
        exact = np.array([x / math.factorial(m) for x in integers])
        k, pmf = study.record_floor(m)
        np.testing.assert_allclose(pmf, exact, atol=2e-16, rtol=1e-14)
        assert exact[k:].sum() <= .1 and (k == 1 or exact[k - 1:].sum() > .1)
    checks.append('Record-count distribution agrees with unsigned Stirling numbers for m=1,20,40.')
    score_error, scan_error, fit_error = 0., 0., 0.
    for item in manifest:
        y = study.load_series(item, ROOT / 'data')
        model = json.loads((ROOT / f"results/model_Primary_{item['load_hp']}.json").read_text())
        d = study.Design()
        z, _ = study.score_series(y, model, d)
        starts = np.linspace(0, len(y) - d.span, 11, dtype=int)
        for start in starts:
            for j in [0, 7, 19]:
                i = start + j * d.q
                expected = scalar_score(y, i, model, d)
                score_error = max(score_error, abs(expected - z[i + d.r]))
        score_blocks = study.blocks(z, starts, d)
        actual = study.statistics(score_blocks, model['c0'])[:, 4]
        expected = []
        for row in score_blocks:
            expected.append(max(abs(sum(row[i:j])) / math.sqrt(j-i)
                                for i in range(d.m) for j in range(i+3, d.m+1)))
        scan_error = max(scan_error, float(np.max(np.abs(actual - expected))))
        b = blocks[blocks.file_id == item['file_id']]
        for method in study.METHODS:
            summary = metrics[(metrics.design == 'Primary') & (metrics.file_id == item['file_id']) & (metrics.method == method)].iloc[0]
            labels = b[method + ' label']
            assert len(b) == summary.blocks
            for k, field in enumerate(['low','intermediate','high']):
                assert int((labels == k).sum()) == summary[field]
        count = b['Count'].to_numpy()
        direct, record = b['Direct count label'].to_numpy(), b['Record count label'].to_numpy()
        changed = (count >= model['k_direct']) & (count < model['k_record'])
        assert np.array_equal(direct != record, changed)
        assert np.all(direct[changed] == 2) and np.all(record[changed] == 1)
        if item['cohort'] == 'normal':
            fit = (y[:2001] - model['center']) / model['scale']
            lag = fit[:-1]
            X = np.column_stack([np.ones(2000), lag, lag*np.exp(-lag**2/2)])
            b_hat = np.linalg.solve(X.T @ X, X.T @ fit[1:])
            fit_error = max(fit_error, float(np.max(np.abs(b_hat - model['b']))))
            assert model['point_lo'] == 2051 and model['point_hi'] == 4051
            assert model['cal_lo'] == 4102 and model['cal_hi'] == 54000
            assert model['holdout_lo'] == 54101
            for design in [study.Design(law='Student'), study.Design(r=5,q=5), study.Design(r=20,q=20)]:
                local_model = study.fit_model(y, design)
                zz = study.score_series(y, local_model, design)[0]
                for start in [0, 1001, 54001]:
                    np.testing.assert_allclose(zz[start+design.r], scalar_score(y,start,local_model,design),atol=1e-12)
    assert score_error < 1e-11 and scan_error < 1e-10 and fit_error < 1e-11
    checks.append('Scalar score formula checked at 33 micro-windows in every recording; Student and alternate-r scores also checked.')
    checks.append('Interval scan agrees with independent interval enumeration.')
    checks.append('Normal-model mean coefficients agree with the manuscript normal-equation fit.')
    checks.append('All primary per-record summaries agree with saved block-level labels; all changed labels are exactly the record-threshold band.')
    checks.append('Fitting, point calibration, block calibration and normal evaluation use the documented disjoint raw supports.')
    expected_spaced = [(756,1,0),(1712,8,8),(1717,12,1),(1719,7,1)]
    for load,(n,d,r) in enumerate(expected_spaced):
        subset = metrics[(metrics.design=='Spaced evaluation') & (metrics.cohort=='normal') & (metrics.load_hp==load)].set_index('method')
        assert int(subset.loc['Direct count','blocks']) == n
        assert int(subset.loc['Direct count','high']) == d
        assert int(subset.loc['Record count','high']) == r
    checks.append('Spaced normal evaluation exactly reproduces all four previous normal-only screening counts.')
    assert len(manifest)==56 and len({x['file_id'] for x in manifest})==56
    assert len(metrics)==56*(7+2*8) and len(blocks)==int(metrics[(metrics.design=='Primary')&(metrics.method=='Direct count')].blocks.sum())
    result = {'status':'passed','checks':checks,'maximum_absolute_score_discrepancy':score_error,
              'maximum_absolute_interval_scan_discrepancy':scan_error,'maximum_mean_coefficient_discrepancy':fit_error,
              'recordings':56,'primary_evaluated_blocks':len(blocks)}
    (ROOT/'results/verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
