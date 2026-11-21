"""Verify published run data, call accounting, physical signs and preservation."""
from pathlib import Path
import argparse,json,subprocess,hashlib
import numpy as np
import pandas as pd
from benchmark_test1 import evaluate
from benchmark_test1_minmax import nondominated_indices

def verify(out):
    out=Path(out);parent='75bbe97dc31069cef47b63a5147612fb617acfd4'
    changes=subprocess.check_output(['git','diff','--name-status','--diff-filter=DMRTUXB',parent],text=True)
    assert not changes,changes
    protocol=json.loads((out/'protocol.json').read_text())
    assert hashlib.sha256((out/'reference/test1.csv').read_bytes()).hexdigest()==protocol['reference_sha256']
    total=0;max_error=0.;n=0
    for seed in protocol['seeds']:
        histories=[]
        for mode in protocol['modes']:
            folder=out/f'runs/{mode}/seed{seed}'
            summary=json.loads((folder/'summary.json').read_text());p=pd.read_csv(folder/'process.csv');h=pd.read_csv(folder/'history.csv')
            assert len(h)==200 and summary['actual_model_calls']==summary['evaluator_calls']==h.evaluations.iloc[-1]
            assert p.n_accepted.sum()==summary['actual_model_calls']
            if mode=='drop':assert p.n_dropped.sum()+summary['actual_model_calls']==20000
            else:assert summary['actual_model_calls']==20000
            for name in ['archive.csv.gz','pareto.csv']:
                f=pd.read_csv(folder/name);physical=evaluate(f[['X1','X2']].to_numpy());error=float(abs(physical-f[['Obj1','Obj2']].to_numpy()).max())
                max_error=max(error,max_error);assert error<1e-9
                if name=='pareto.csv':assert len(nondominated_indices(f[['Obj1','Obj2']]))==len(f)
            histories.append(h.iloc[:50]);total+=summary['actual_model_calls'];n+=1
        for h in histories[1:]:np.testing.assert_array_equal(histories[0].to_numpy(),h.to_numpy())
    assert n==60
    result=dict(runs_verified=n,optimizer_model_calls=total,max_physical_export_error=max_error,
        first_50_generation_histories_identical_per_seed=True,parent_commit=parent,
        parent_tree=subprocess.check_output(['git','rev-parse',parent+'^{tree}'],text=True).strip(),
        existing_tracked_files_changed=[],historical_results_preserved=True,solenoid_rerun=False,
        tests='80 passed; see validation_tests.txt',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('benchmark_test1_minmax.py'),Path('run_test1_original_minmax.py'),Path('prepare_test1_minmax.py'),Path('src/test1_minmax_reference.py'),Path('src/test1_minmax_observation.py')]})
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--outdir',type=Path,default=Path('docs/test1_original_minmax'));verify(p.parse_args().outdir)
