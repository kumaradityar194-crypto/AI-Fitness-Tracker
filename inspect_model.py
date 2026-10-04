import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"

model = joblib.load(MODEL_DIR / "exercise_classifier.pkl")

print("=" * 70)
print("MODEL INSPECTION")
print("=" * 70)

print("Model type:", type(model).__name__)

if hasattr(model, "classes_"):
    print("Classes:", list(model.classes_))

if hasattr(model, "n_features_in_"):
    print("Input features:", model.n_features_in_)

if hasattr(model, "coef_"):
    print("Coefficient shape:", model.coef_.shape)

print()
print("Saved model files:")

for file in MODEL_DIR.iterdir():
    if file.is_file():
        print(file.name)

print("=" * 70)
print("MODEL INSPECTION COMPLETED")
print("=" * 70)
