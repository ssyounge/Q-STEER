from pathlib import Path

import yaml

import llava.qsteer as api


def test_public_exports():
    required = {
        "QSTEERBatchMasks", "QSTEERPreparedBatch", "QSTEERTwoStageRunner",
        "QSTEERRuntimeSettings", "QSTEERMOELoraConfig", "QSTEERMOELoraLinear",
        "QSTEERMOELoraModel", "inject_qsteer_moe_lora", "attach_qsteer_runtime",
        "attach_qsteer_runtime_with_settings", "QSTEERContext", "QSTEERController",
        "QSTEERDriftBuffer", "QSTEERExpansionProbe", "QSTEERTaskFinalizeCallback",
        "build_qsteer_expansion_probe", "build_qsteer_callbacks",
        "inspect_qsteer_runtime", "validate_qsteer_runtime",
    }
    assert required <= set(api.__all__)
    for name in api.__all__:
        assert callable(getattr(api, name))


def test_paper_config_defaults_unchanged():
    config = yaml.safe_load((Path(__file__).resolve().parents[1] /
                             "configs/qsteer_vqav2.yaml").read_text())
    defaults = api.QSTEERRuntimeSettings()
    for name, value in config["runtime"].items():
        assert getattr(defaults, name) == value
    assert defaults.expert_num == 32 and defaults.expert_init == 8
    assert defaults.topk_update == 2 and defaults.probe_samples == 128
    assert defaults.beta_thr == 0.5 and defaults.icr_rank == 8
    assert defaults.late_layer_count == 6 and defaults.icr_scale == 0.3
    adapter = api.QSTEERMOELoraConfig()
    assert adapter.expert_rank == config["adapter"]["expert_rank"] == 8
    assert adapter.lora_alpha == 16.0 and adapter.lora_dropout == 0.0
    assert adapter.target_modules == ("q_proj", "v_proj")
