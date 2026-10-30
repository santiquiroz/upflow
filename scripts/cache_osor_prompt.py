import argparse
from pathlib import Path

import torch


def encode_fixed_prompt(text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                        prompt: str = "Remove the instance of object") -> dict[str, torch.Tensor | str]:
    tokens = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                       truncation=True, return_tensors="pt")
    tokens_2 = tokenizer_2(prompt, padding="max_length", max_length=tokenizer_2.model_max_length,
                           truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = text_encoder(tokens["input_ids"], output_hidden_states=True)
        output_2 = text_encoder_2(tokens_2["input_ids"], output_hidden_states=True)
    prompt_embeds = torch.cat((output.hidden_states[-2], output_2.hidden_states[-2]), dim=-1).float()
    return {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": output_2[0].float(),
        "prompt": prompt,
    }


def load_fixed_prompt_components(base: Path):
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(base / "tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(
        base / "text_encoder", variant="fp16", torch_dtype=torch.float32, local_files_only=True
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(base / "tokenizer_2", local_files_only=True)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        base / "text_encoder_2", variant="fp16", torch_dtype=torch.float32, local_files_only=True
    )
    return text_encoder, text_encoder_2, tokenizer, tokenizer_2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    components = load_fixed_prompt_components(args.base)
    embeds = encode_fixed_prompt(*components)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeds, args.out)


if __name__ == "__main__":
    main()
