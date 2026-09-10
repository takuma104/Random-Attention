from kvcompress.qwen35 import runtime
import json
from types import SimpleNamespace
import pytest
import torch
from kvcompress.qwen35.batch import generate_batch
from kvcompress.qwen35.test_cache import tiny_config
from scripts.qwen35.run_math_batch import grade_pending_group


@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU sampler')
def test_finished_row_does_not_change_other_row_or_its_metrics():
    class ScriptedModel:
        config=tiny_config()
        generation_config=SimpleNamespace(eos_token_id=[127])
        def __init__(self, scripts):
            self.step=0
            self.scripts=scripts
        def __call__(self, current, past_key_values, **kwargs):
            b=current.shape[0]
            for i in past_key_values.full_indices:
                layer=past_key_values.layers[i]
                x=torch.zeros(b,2,current.shape[1],32,device=current.device,dtype=torch.bfloat16)
                layer.update(x,x)
                layer.evict_if_needed()
            logits=torch.full((b,1,128),float('-inf'),device=current.device)
            for row,script in enumerate(self.scripts):
                logits[row,0,script[min(self.step,len(script)-1)]]=0
            self.step+=1
            return SimpleNamespace(logits=logits)
    short=[10,11,127]
    long=[10,11,12,13,14,15,16,127]
    ids=torch.tensor([[42],[42]],device='cuda')
    args=dict(policy='random_pp',capacity=4,recent=1,max_new_tokens=10,heartbeat=0)
    pair=generate_batch(ScriptedModel([short,long]),ids,generation_seeds=[1,2],eviction_seeds=[3,4],**args)
    alone=generate_batch(ScriptedModel([long]),ids[:1],generation_seeds=[2],eviction_seeds=[4],**args)
    assert pair[0][0]==short and pair[1][0]==alone[0][0]==long
    assert pair[0][1]['logical_cached_tokens']==3
    assert pair[1][1]['logical_cached_tokens']==8
    assert pair[0][1]['eviction_layer_events']==0
    assert pair[1][1]['eviction_layer_events']==alone[0][1]['eviction_layer_events']>0
    assert pair[0][1]['request_latency_seconds']<pair[1][1]['request_latency_seconds']
    assert sum(r[1]['elapsed_seconds'] for r in pair)==pair[0][1]['batch_wall_seconds']


def test_atomic_batch_partial_grading_recovery(tmp_path,monkeypatch):
    calls=[]
    def fake_run(*args,**kwargs):
        calls.append(kwargs['input'])
        return SimpleNamespace(stdout=json.dumps(dict(final_correct=True,grading_errors=[])))
    monkeypatch.setattr('scripts.qwen35.run_math_batch.subprocess.run',fake_run)
    rows=[dict(problem_id='p',run=i,token_ids=[10+i,127],generation_seed=i,eviction_seed=i+10,
               completion=r'</think>\boxed{42}',gold='42',termination='eos',generated_tokens=2,
               request_latency_seconds=1.) for i in range(2)]
    pending=tmp_path/'group.batch-ungraded'
    pending.write_text(json.dumps(rows))
    paths=[tmp_path/f'run{i}.json' for i in range(2)]
    paths[0].write_text(json.dumps(dict(rows[0],status='complete',final_correct=True,grading_errors=[])))
    original=paths[0].read_bytes()
    grade_pending_group(pending,paths)
    assert len(calls)==1
    assert paths[0].read_bytes()==original
    assert not pending.exists()
    assert json.loads(paths[1].read_text())['status']=='complete'
