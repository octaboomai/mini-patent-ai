import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        import sys
        sys.path.append('.')
        from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    sklearn_available = True
except ImportError:
    print("⚠️  scikit-learn not installed. Run: pip install scikit-learn")
    sklearn_available = False
    import sys
    sys.exit(1)
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🏥 PATIENT NO-SHOW PREDICTOR - PRODUCTION MODEL")
print("✅ ZERO downloads needed - Creates own data\n")

# STEP 1: CREATE SIMPLE TRAINING DATA (WORKS EVERYWHERE)
print("📊 Creating patient data...")
np.random.seed(123)

n = 5000
data = {
    'age': np.random.randint(18, 80, n),
    'wait_time_minutes': np.random.randint(5, 120, n),
    'distance_km': np.random.uniform(0.5, 50, n),
    'day_of_week': np.random.randint(0, 7, n),  # 0=Mon, 6=Sun
    'sms_sent': np.random.choice([0, 1], n, p=[0.7, 0.3]),
    'has_insurance': np.random.choice([0, 1], n, p=[0.6, 0.4])
}

df = pd.DataFrame(data)

# Simple target: Will patient show up? (1=yes, 0=no-show)
df['show_up'] = (
    (df['age'] > 40) * 0.3 +
    (df['wait_time_minutes'] < 45) * 0.3 +
    (df['distance_km'] < 10) * 0.2 +
    (df['sms_sent'] == 1) * 0.15 +
    np.random.uniform(0, 0.15, n)
).clip(0, 1).round().astype(int)

print(f"✅ Data ready: {len(df)} patients")
print("Show-up rate:", df['show_up'].mean().round(3))
df.to_csv('patient_data.csv', index=False)

# STEP 2: TRAIN MODEL (SIMPLEST POSSIBLE)
print("\n🤖 Training model...")
X = df.drop('show_up', axis=1)
y = df['show_up']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForest - NO encoding issues, NO scaling needed
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test
accuracy = model.score(X_test, y_test)
print(f"✅ Model Accuracy: {accuracy:.3f}")

# STEP 3: SAVE MODEL (WORKS 100%)
print("\n💾 Saving model...")
with open('patient_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved: patient_model.pkl")

# STEP 4: LOAD & PREDICT NEW PATIENTS (REAL MODEL USE)
print("\n🔮 Testing new patients...")

# Load model
with open('patient_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# NEW PATIENT 1: High risk no-show
patient1 = np.array([[25, 90, 25.5, 5, 0, 0]])
pred1 = loaded_model.predict(patient1)[0]
prob1 = loaded_model.predict_proba(patient1)[0][1]

print(f"👤 Patient 1 (Young, long wait, far): {'🟢 Will Show' if pred1 else '🟥 No-Show'} ({prob1:.1%})")

# NEW PATIENT 2: Low risk  
patient2 = np.array([[55, 20, 3.2, 1, 1, 1]])
pred2 = loaded_model.predict(patient2)[0]
prob2 = loaded_model.predict_proba(patient2)[0][1]

print(f"👤 Patient 2 (Older, short wait, close): {'🟢 Will Show' if pred2 else '🟥 No-Show'} ({prob2:.1%})")

print("\n🎉 SUCCESS! PRODUCTION MODEL READY")
print("✅ Files created:")
print("   • patient_model.pkl  ← YOUR ML MODEL")
print("   • patient_data.csv")
print("\n🚀 To predict ANY new patient:")
print("model.predict([[age,wait_time,distance,day,sms,insurance]])")
print("\n📁 Submit these files for college!")