from types import SimpleNamespace

import json

from serving.kernel_policy import DeviceCapabilities
from serving.portability import inspect_qwen_host,signature_from_config


def test_signature_from_config_is_stable_without_model_weights():
    config=SimpleNamespace(
        num_hidden_layers=28,hidden_size=1024,num_attention_heads=16,
        num_key_value_heads=8,head_dim=128)
    signature=signature_from_config(config)
    assert (signature.layers,signature.hidden_size)==(28,1024)
    assert (signature.attention_heads,signature.kv_heads,signature.head_dim)==(16,8,128)


def test_host_report_includes_computed_architecture(monkeypatch,tmp_path):
    config=SimpleNamespace(
        num_hidden_layers=28,hidden_size=1024,num_attention_heads=16,
        num_key_value_heads=8,head_dim=128)
    profiles={
        'model_signature':{'layers':28,'hidden_size':1024,'attention_heads':16,
                           'kv_heads':8,'head_dim':128},
        'architectures':{},
    }
    path=tmp_path/'profiles.json'; path.write_text(json.dumps(profiles))
    monkeypatch.setattr('serving.portability.torch.cuda.device_count',lambda:1)
    monkeypatch.setattr(
        'serving.portability.detect_cuda_capabilities',
        lambda device:DeviceCapabilities('test',8,6,48,32,16<<30))
    report=inspect_qwen_host(config,path)
    assert report['devices'][0]['capabilities']['architecture']=='sm86'
    assert report['devices'][0]['policy']['profile']=='fallback'
