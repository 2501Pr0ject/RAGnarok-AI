"""Example: Medical abbreviation normalization in evaluation.

This example demonstrates how to use medical_mode=True to automatically
normalize medical abbreviations during evaluation, reducing false positives
from abbreviation ambiguity.
"""

import asyncio

from ragnarok_ai.adapters.llm.ollama import OllamaLLM
from ragnarok_ai.evaluators.faithfulness import FaithfulnessEvaluator


async def main() -> None:
    """Demonstrate medical abbreviation normalization."""
    # Initialize LLM
    llm = OllamaLLM(model="llama3.2")

    # Test case: Context uses abbreviation, answer uses full form
    context = "Patient diagnosed with CHF and prescribed furosemide 40mg daily."
    answer = "Patient has congestive heart failure and is taking furosemide."

    print("=" * 70)
    print("MEDICAL ABBREVIATION NORMALIZATION DEMO")
    print("=" * 70)

    print("\nContext (with abbreviation):")
    print(f"  {context}")

    print("\nAnswer (with full form):")
    print(f"  {answer}")

    # Without medical_mode - may score lower due to abbreviation mismatch
    print("\n" + "-" * 70)
    print("WITHOUT medical_mode (standard evaluation):")
    print("-" * 70)

    evaluator_standard = FaithfulnessEvaluator(llm, medical_mode=False)
    score_standard = await evaluator_standard.evaluate(
        response=answer,
        context=context,
    )
    print(f"Faithfulness Score: {score_standard:.2f}")
    print("Note: May score lower because 'CHF' and 'congestive heart failure'")
    print("      are treated as different terms.")

    # With medical_mode - abbreviations normalized before evaluation
    print("\n" + "-" * 70)
    print("WITH medical_mode=True (medical evaluation):")
    print("-" * 70)

    evaluator_medical = FaithfulnessEvaluator(llm, medical_mode=True)
    score_medical = await evaluator_medical.evaluate(
        response=answer,
        context=context,
    )
    print(f"Faithfulness Score: {score_medical:.2f}")
    print("Note: Both 'CHF' and 'congestive heart failure' normalized to same form,")
    print("      resulting in higher score (reduced false positive).")

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Standard evaluation:  {score_standard:.2f}")
    print(f"Medical evaluation:   {score_medical:.2f}")
    print(f"Improvement:          {score_medical - score_standard:+.2f}")
    print("\nConclusion: medical_mode=True reduces false positives from")
    print("            medical abbreviation ambiguity!")


if __name__ == "__main__":
    asyncio.run(main())
