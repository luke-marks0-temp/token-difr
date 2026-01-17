import json
import os
import argparse
from dataclasses import asdict
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from token_difr import audit_provider, construct_prompts, list_openrouter_providers

# Audit parameters
N_PROMPTS = 100
MAX_TOKENS = 200
SEED = 42
TOP_K = 50
TOP_P = 0.95
TEMPERATURE = 0.0


def save_results(results: dict, output_file: str) -> None:
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit providers for one or more Hugging Face model names.",
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="One or more Hugging Face model names (e.g. Qwen/Qwen3-235B-A22B-Instruct-2507).",
    )
    return parser.parse_args()


def main(models: list[str]) -> None:
    for HF_MODEL in models:
        try:
            providers = list_openrouter_providers(HF_MODEL)
        except Exception as exc:
            print(f"Failed to list providers for {HF_MODEL}: {exc}")
            continue
        if not providers:
            print(f"No providers listed for {HF_MODEL}")
            continue

        prompts = construct_prompts(
            n_prompts=N_PROMPTS,
            model_name=HF_MODEL,
            system_prompt="You are a helpful assistant.",
        )
        print(f"Constructed {len(prompts)} prompts")

        # Initialize results structure with metadata
        results = {
            "model": HF_MODEL,
            "parameters": {
                "n_prompts": N_PROMPTS,
                "max_tokens": MAX_TOKENS,
                "seed": SEED,
                "top_k": TOP_K,
                "top_p": TOP_P,
                "temperature": TEMPERATURE,
            },
            "providers": {},
        }

        safe_model_name = HF_MODEL.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "audit_results"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{safe_model_name}_audit_results_{timestamp}.json"

        # Write initial file so we can watch progress
        save_results(results, output_file)
        print(f"Results will be saved to {output_file}")

        for provider in providers:
            print(f"\nAuditing provider: {provider}")
            try:
                result = audit_provider(
                    prompts,
                    model=HF_MODEL,
                    provider=provider,
                    max_tokens=MAX_TOKENS,
                    seed=SEED,
                    top_k=TOP_K,
                    top_p=TOP_P,
                    temperature=TEMPERATURE,
                )

                results["providers"][provider] = asdict(result)

                print(f"  Total tokens: {result.total_tokens}")
                print(f"  Exact match rate: {result.exact_match_rate:.2%}")
                print(f"  Avg probability: {result.avg_prob:.4f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                results["providers"][provider] = {"error": str(e)}

            # Save after each provider completes
            save_results(results, output_file)

        print(f"\nAll results saved to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args.models)
