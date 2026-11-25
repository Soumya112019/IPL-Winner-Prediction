import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("matches.csv")

feature_cols = ["city", "team1", "team2", "toss_winner", "toss_decision", "venue"]
target_col = "winner"

X = df[feature_cols].copy()
y = df[target_col].copy()

encoder = LabelEncoder()

for col in feature_cols:
    X[col] = encoder.fit_transform(X[col])

y = encoder.fit_transform(y)

print("Numbers mein badle hue Features (X):")
print(X.head(3))

print("\nNumbers mein badla hua target (y):")
print(y[:10])

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train , y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

print("Asli winners (y_test): ", y_test[:5])
print("Model ke guess (pred): ", predictions[:5])

df = pd.read_csv("matches.csv")

feature_cols = ["city","team1","team2", "toss_winner", "toss_decision", "venue"]
target_col = "winner"

X = df[feature_cols].copy()
y = df[target_col].copy()

encoders = {}

for col in feature_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test,predictions)

print("Accuracy:", accuracy)

sample_match = pd.DataFrame([{
    "city": "Mumbai",
    "team1": "Mumbai Indians",
    "team2": "Royal Challengers Bengaluru",
    "toss_winner": "Mumbai Indians",
    "toss_decision": "field",
    "venue": "Wankhede Stadium"
}])

for col in feature_cols:
    sample_match[col] = encoders[col].transform(sample_match[col])

pred_code = model.predict(sample_match)
pred_team = target_encoder.inverse_transform(pred_code)

print("\nNaye match ka predicted winner:", pred_team[0])