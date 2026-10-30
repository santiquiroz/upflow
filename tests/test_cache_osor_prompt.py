import torch

from scripts.build_osor_pack import build_parser
from scripts.cache_osor_prompt import encode_fixed_prompt


class FakeTokenizer:
    model_max_length = 4

    def __init__(self, token_id):
        self.token_id = token_id
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {"input_ids": torch.full((1, kwargs["max_length"]), self.token_id)}


class FakeOutput:
    def __init__(self, pooled, hidden_states):
        self.pooled = pooled
        self.hidden_states = hidden_states

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.pooled


class FakeEncoder:
    def __init__(self, pooled, hidden_states):
        self.pooled = pooled
        self.hidden_states = hidden_states
        self.calls = []

    def __call__(self, input_ids, **kwargs):
        self.calls.append((input_ids, kwargs))
        return FakeOutput(self.pooled, self.hidden_states)


def test_encode_fixed_prompt_concatenates_hidden_states_and_uses_encoder_two_pool():
    tokenizer = FakeTokenizer(1)
    tokenizer_2 = FakeTokenizer(2)
    first_hidden = torch.ones((1, 4, 3))
    second_hidden = torch.full((1, 4, 5), 2.0)
    encoder = FakeEncoder(None, [first_hidden, torch.zeros_like(first_hidden)])
    pooled = torch.full((1, 5), 3.0)
    encoder_2 = FakeEncoder(pooled, [second_hidden, torch.zeros_like(second_hidden)])

    result = encode_fixed_prompt(encoder, encoder_2, tokenizer, tokenizer_2)

    assert set(result) == {"prompt_embeds", "pooled_prompt_embeds", "prompt"}
    assert result["prompt_embeds"].shape == (
        1, 4, first_hidden.shape[-1] + second_hidden.shape[-1]
    )
    assert torch.equal(result["prompt_embeds"][..., :3], first_hidden)
    assert torch.equal(result["prompt_embeds"][..., 3:], second_hidden)
    assert result["pooled_prompt_embeds"] is pooled
    assert result["prompt"] == "Remove the instance of object"
    assert tokenizer.calls[0][0] == result["prompt"]
    assert tokenizer_2.calls[0][0] == result["prompt"]


def test_build_osor_pack_parser_does_not_require_embeds():
    args = build_parser().parse_args(
        ["--base", "base", "--weights", "weights", "--out", "out"]
    )

    assert args.embeds is None
