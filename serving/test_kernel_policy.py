from serving.kernel_policy import DeviceCapabilities,ModelSignature,select_qwen_policy


PROFILES={
    'model_signature':{'layers':28,'hidden_size':1024,'attention_heads':16,'kv_heads':8,'head_dim':128},
    'architectures':{
        'sm89':{'name':'ada','quality_gate_passed':True,'block_size':16,
                'attention_tile':128,'direct_attention_layers':[24,25,26,27]}}
}
SIGNATURE=ModelSignature(28,1024,16,8,128)


def capabilities(major=8,minor=9):
    return DeviceCapabilities('test',major,minor,80,32,16<<30)


def test_known_architecture_uses_only_calibrated_layers():
    policy=select_qwen_policy(capabilities(),SIGNATURE,PROFILES)
    assert policy.calibrated and policy.profile=='ada'
    assert policy.direct_attention_layers==frozenset({24,25,26,27})


def test_unknown_architecture_uses_exact_fallback():
    policy=select_qwen_policy(capabilities(9,0),SIGNATURE,PROFILES)
    assert not policy.calibrated and not policy.direct_attention_layers
    assert 'no accepted' in policy.reason


def test_model_signature_mismatch_uses_exact_fallback():
    policy=select_qwen_policy(capabilities(),ModelSignature(32,1024,16,8,128),PROFILES)
    assert not policy.calibrated and not policy.direct_attention_layers
    assert 'signature' in policy.reason
