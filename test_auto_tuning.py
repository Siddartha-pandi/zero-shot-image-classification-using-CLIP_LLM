import sys
sys.path.insert(0, 'backend')

from services.prediction_engine import get_prediction_engine
from services.llm_auto_tuner import get_llm_auto_tuner

# Test 1: Verify learned labels retrieval
print("=" * 60)
print("TEST 1: Learned Labels Retrieval")
print("=" * 60)
tuner = get_llm_auto_tuner()
learned_animal = tuner.get_learned_labels('animal')
learned_veg = tuner.get_learned_labels('vegetable')
print(f"Animal domain learned labels: {learned_animal}")
print(f"Vegetable domain learned labels: {learned_veg}")
print(f"Min feedback support threshold: {tuner.min_feedback_support}")

# Test 2: Verify label merging
print("\n" + "=" * 60)
print("TEST 2: Label Merging in Prediction Engine")
print("=" * 60)
engine = get_prediction_engine()
merged_labels = engine._merge_labels('animal')
print(f"Total merged labels for animal: {len(merged_labels)}")
print(f"First 15 labels: {merged_labels[:15]}")
print(f"Contains 'flamingo': {'flamingo' in merged_labels}")
print(f"Contains 'bird': {'bird' in merged_labels}")

# Test 3: Correction boost functionality
print("\n" + "=" * 60)
print("TEST 3: Correction Boost System")
print("=" * 60)
boost = tuner.get_correction_boost('animal', 'bird', 'flamingo')
print(f"Boost for correcting 'bird' to 'flamingo': {boost}")

# Test 4: Auto-tuning configuration
print("\n" + "=" * 60)
print("TEST 4: Auto-Tuning Configuration")
print("=" * 60)
print(f"LLM auto-tuning enabled: {tuner.enabled}")
print(f"Self-learning enabled: {tuner.self_learning_enabled}")
print(f"Auto-reinforce enabled: {tuner.auto_reinforce_enabled}")
print(f"Auto-reinforce threshold: {tuner.auto_reinforce_threshold}")
print(f"Feedback storage path: {tuner.file_path}")

print("\n" + "=" * 60)
print("CONCLUSION: LLM Auto-Tuning Status")
print("=" * 60)
if tuner.self_learning_enabled and learned_animal:
    print("✓ Self-learning is WORKING - labels are being learned from feedback")
if tuner.auto_reinforce_enabled:
    print("✓ Auto-reinforcement is ENABLED - high-confidence predictions are auto-learned")
if 'flamingo' in merged_labels and 'bird' in merged_labels:
    print("✓ Label merging is WORKING - learned labels are included in predictions")
print("\nNote: LLM prompt generation may hit quota limits (Gemini free tier: 20 req/day)")
print("      This is expected and the system gracefully falls back to base prompts")
