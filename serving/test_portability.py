from types import SimpleNamespace

from serving.portability import signature_from_config


def test_signature_from_config_is_stable_without_model_weights():
    config=SimpleNamespace(
        num_hidden_layers=28,hidden_size=1024,num_attention_heads=16,
        num_key_value_heads=8,head_dim=128)
    signature=signature_from_config(config)
    assert (signature.layers,signature.hidden_size)==(28,1024)
    assert (signature.attention_heads,signature.kv_heads,signature.head_dim)==(16,8,128)
