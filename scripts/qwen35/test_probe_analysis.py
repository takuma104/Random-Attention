import hashlib
import json
import numpy as np
import pytest
from scripts.qwen35.analyze_retrieval_probe import audit,clustered_difference


def test_case_cluster_zero_differences():
    result=clustered_difference([0.,0.,0.])
    assert result['n_cases']==3 and result['ci95']==[0.,0.] and result['exact_sign_p']==1.


def test_snapshot_off_by_one_and_trace_hash_are_rejected(tmp_path):
    (tmp_path/'native').mkdir()
    (tmp_path/'manifest.json').write_text(json.dumps(dict(git_commit='test',settings=dict(source_hashes={},
        args=dict(phase='calibration',cases=1,gaps='0',cells='native')))))
    trace=[7,8]; scores=np.array([[2.,1.,0.,0.]]*2); e=np.exp(scores-2)
    row=dict(case_id='calibration-000',gap_tokens=0,phase='calibration',batch_size=2,policy='native',capacity=0,
        eviction_seeds=[1,2],prompt_length=1,prompt_ids=[11],fact_span=[1,2],value_span=[1,2],
        query_length=1,trace_ids=trace,trace_sha256=hashlib.sha256(json.dumps(trace).encode()).hexdigest(),
        final_logical_length=3,snapshot_logical_length=2,eviction_layer_events_before_last_query=0,
        eviction_layer_events=0,label_logits=scores.tolist(),conditional_probabilities=(e/e.sum(1,keepdims=True)).tolist(),
        predictions=[0,0],target=0,correct=[True,True],label_probability_mass=[.9,.9],
        fact_token_counts=np.ones((2,8,4),int).tolist(),value_token_counts=np.ones((2,8,4),int).tolist(),
        older_generated_head_jaccard_by_layer=np.ones((2,8)).tolist(),elapsed_seconds=1.,
        label_token_ids=[0,1,2,3],options=['a','b','c','d'])
    path=tmp_path/'native'/'case.json'
    path.write_text(json.dumps(row)); audit(tmp_path)
    row['snapshot_logical_length']=3
    path.write_text(json.dumps(row))
    with pytest.raises(AssertionError): audit(tmp_path)
    row['snapshot_logical_length']=2; row['trace_ids'][0]=99
    path.write_text(json.dumps(row))
    with pytest.raises(AssertionError): audit(tmp_path)
