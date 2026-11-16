from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
import pytest
from src.study_reference import test1_reference as make_reference
from run_repeated_study import run_case
from analyze_repeated_study import paired_effect,holm,analyze


def test_paired_confidence_intervals_and_holm():
    result=paired_effect(np.arange(20),np.arange(20)+2)
    assert result['mean_difference']==2
    assert result['ci95_low']==result['ci95_high']==2
    np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
    result=paired_effect(np.arange(20),np.arange(20))
    assert result['p_raw']==1
    assert result['ci95_low']==result['ci95_high']==0


def test_common_harness_and_complete_paired_analysis(tmp_path):
    root=tmp_path/'study';(root/'reference').mkdir(parents=True)
    make_reference(501).to_csv(root/'reference/test1.csv',index=False)
    shutil.copy2('docs/alsf_drop/reference/reference.csv',root/'reference/solenoid.csv')
    protocol=dict(benchmarks=['test1','solenoid'],modes=['baseline','resample','drop'],seeds=[1,2])
    (root/'protocol.json').write_text(json.dumps(protocol))
    for benchmark in protocol['benchmarks']:
        for seed in protocol['seeds']:
            histories=[]
            for mode in protocol['modes']:
                result=run_case(root,benchmark,mode,seed,generations=4,pop_size=8,flag_start=50)
                assert result['actual_model_calls']==32
                assert result['saved_evaluations']==0
                histories.append(pd.read_csv(root/'runs'/benchmark/mode/f'seed{seed}'/'history.csv'))
            pd.testing.assert_frame_equal(histories[0],histories[1]);pd.testing.assert_frame_equal(histories[0],histories[2])
    saved=(root/'runs/test1/drop/seed1/summary.json').read_bytes()
    run_case(root,'test1','drop',1,generations=4,pop_size=8,flag_start=50)
    assert saved==(root/'runs/test1/drop/seed1/summary.json').read_bytes()
    with pytest.raises(ValueError,match='Resume settings differ'):
        run_case(root,'test1','drop',1,generations=5,pop_size=8,flag_start=50)
    out=tmp_path/'analysis';frame,effects=analyze(root,out)
    assert len(frame)==12 and len(effects)==18
    hits=pd.read_csv(out/'target_hv_attainment.csv')
    assert len(hits)==36
    assert set(hits.target_fraction)=={.99,.995,.999}
    assert len(list(out.glob('*.png')))==8
    assert len(list(out.glob('*.pdf')))==8
    assert effects.p_holm.between(0,1).all()
