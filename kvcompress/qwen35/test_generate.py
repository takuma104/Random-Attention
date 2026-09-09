from kvcompress.qwen35 import runtime  # initialize FLA before modeling
import torch
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper
from kvcompress.qwen35.generate import SamplingConfig, candidate_distribution, stable_seed
from kvcompress.qwen35.grade import last_boxed, grade_completion


def test_sampling_matches_transformers():
    torch.manual_seed(12)
    for size in (10, 100, 1000):
        logits = torch.randn(1,size)
        seen = torch.rand(1,size) < .5
        cfg = SamplingConfig()
        ids, probs = candidate_distribution(logits,seen,cfg)
        actual = torch.zeros_like(logits).scatter(-1,ids,probs)
        expected = (logits - seen.float()*cfg.presence_penalty)/cfg.temperature
        expected = TopKLogitsWarper(cfg.top_k)(None,expected)
        expected = TopPLogitsWarper(cfg.top_p)(None,expected).softmax(-1)
        torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)


def test_seeds_and_rng_independence():
    a=stable_seed('id',0,'generation')
    assert a==stable_seed('id',0,'generation')
    assert a!=stable_seed('id',0,'eviction')
    assert a!=stable_seed('id',1,'generation')
    g=torch.Generator().manual_seed(a)
    h=torch.Generator().manual_seed(a)
    e=torch.Generator().manual_seed(stable_seed('id',0,'eviction'))
    for _ in range(10):
        torch.rand(100,generator=e)
        assert torch.equal(torch.multinomial(torch.ones(20),1,generator=g),
                           torch.multinomial(torch.ones(20),1,generator=h))


def test_boxed_grading_and_thinking_boundary():
    assert last_boxed(r'\boxed{\frac{1}{2}}')==r'\frac{1}{2}'
    assert last_boxed(r'\boxed{unfinished') is None
    assert last_boxed(r'\boxed{3} then \boxed{4}')=='4'
    assert grade_completion(r'Check \boxed{42}', '42', 'length')['final_correct'] is False
    assert grade_completion(r'Check \boxed{42}</think>\boxed{41}', '42', 'eos')['final_correct'] is False
    assert grade_completion(r'Check \boxed{41}</think>\boxed{42}', '42', 'eos')['final_correct'] is True
    assert grade_completion(r'</think>\boxed{\frac{1}{2}}', '0.5', 'eos')['final_correct'] is True
    capped=grade_completion(r'</think>\boxed{42}', '42', 'length')
    assert capped['final_correct'] and not capped['final_correct_terminated']
