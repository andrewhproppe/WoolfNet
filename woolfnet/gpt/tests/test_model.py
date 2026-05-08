import torch

from woolfnet.gpt.model import GPT, GPTConfig
from woolfnet.paths import DATA_DIR


def test_gpt_model():
    vocab_size = 5000
    config = GPTConfig(vocab_size=vocab_size)
    model = GPT(config)

    x = torch.randint(0, vocab_size, (2, 128))
    logits = model(x)
    print("Logits shape:", logits.shape)


def test_config_from_yaml():
    test_config_path = DATA_DIR / "test_data" / "test_gpt_config.yml"
    config = GPTConfig.from_yaml(test_config_path)
    model = GPT(config)
