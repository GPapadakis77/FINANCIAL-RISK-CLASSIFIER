# full_pipeline_no_svm.py

# Κελί 1: Εισαγωγή Βιβλιοθηκών
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Κελί 2: Φόρτωση & Καθαρισμός Δεδομένων
try:
    df = pd.read_csv('application_record.csv')
    print("Το αρχείο application_record.csv φορτώθηκε επιτυχώς!")
except FileNotFoundError:
    print("Σφάλμα: Το αρχείο application_record.csv δεν βρέθηκε. Βεβαιωθείτε ότι βρίσκεται στον ίδιο φάκελο.")
    exit()

if 'OCCUPATION_TYPE' in df.columns:
    df['OCCUPATION_TYPE'].fillna('Unknown', inplace=True)

df.drop_duplicates(subset='ID', keep='first', inplace=True)

# Κελί 3: Μηχανική Χαρακτηριστικών
df['AGE_YEARS'] = np.abs(df['DAYS_BIRTH']) / 365.25

df['YEARS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: np.abs(x) / 365.2425 if x != 365243 else 0)

df['INCOME_PER_FAMILY_MEMBER'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df['INCOME_PER_FAMILY_MEMBER'].fillna(0, inplace=True)

df['EMPLOYMENT_AGE_RATIO'] = df.apply(lambda row: row['YEARS_EMPLOYED'] / row['AGE_YEARS'] if row['AGE_YEARS'] > 0 else 0, axis=1)

# Κελί 4: Κωδικοποίηση Κατηγορικών Δεδομένων (Πριν τον ορισμό του X)
# Δημιουργούμε dummy variables για ΟΛΑ τα κατηγορικά αρχικά
categorical_cols_for_encoding = df.select_dtypes(include='object').columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols_for_encoding, drop_first=True)

# Κελί 5: Δημιουργία & Κωδικοποίηση Μεταβλητής Στόχου (RISK_LEVEL)
def assign_risk_level(row):
    income_low_q_threshold = 120000
    income_high_q_threshold = 280000

    # Διασφάλιση ότι οι dummy στήλες υπάρχουν (με τιμή 0 αν όχι), χρησιμοποιώντας .get()
    flag_own_realty_y = row.get('FLAG_OWN_REALTY_Y', 0)
    name_family_status_married = row.get('NAME_FAMILY_STATUS_Married', 0)
    name_education_type_lower_secondary = row.get('NAME_EDUCATION_TYPE_Lower secondary', 0)
    name_income_type_state_servant = row.get('NAME_INCOME_TYPE_State servant', 0)

    if row['AMT_INCOME_TOTAL'] < income_low_q_threshold:
        base_risk = 'Υψηλό Ρίσκο'
    elif row['AMT_INCOME_TOTAL'] > income_high_q_threshold:
        base_risk = 'Χαμηλό Ρίσκο'
    else:
        base_risk = 'Μεσαίο Ρίσκο'

    final_risk = base_risk

    if (row['CNT_CHILDREN'] >= 3 and row['INCOME_PER_FAMILY_MEMBER'] < 40000) or \
       (row['YEARS_EMPLOYED'] < 3 and row['AGE_YEARS'] < 30 and name_education_type_lower_secondary == 1):
        if final_risk == 'Χαμηλό Ρίσκο':
            final_risk = 'Μεσαίο Ρίσκο'
        elif final_risk == 'Μεσαίο Ρίσκο':
            final_risk = 'Υψηλό Ρίσκο'
            
    if (flag_own_realty_y == 1 and row['AMT_INCOME_TOTAL'] > 200000 and name_family_status_married == 1) or \
       (name_income_type_state_servant == 1 and row['YEARS_EMPLOYED'] > 10 and row['EMPLOYMENT_AGE_RATIO'] > 0.5):
        if final_risk == 'Υψηλό Ρίσκο':
            final_risk = 'Μεσαίο Ρίσκο'
        elif final_risk == 'Μεσαίο Ρίσκο':
            final_risk = 'Χαμηλό Ρίσκο'

    return final_risk

df['RISK_LEVEL'] = df.apply(assign_risk_level, axis=1)

risk_mapping = {
    'Χαμηλό Ρίσκο': 0,
    'Μεσαίο Ρίσκο': 1,
    'Υψηλό Ρίσκο': 2
}
df['RISK_LEVEL_ENCODED'] = df['RISK_LEVEL'].map(risk_mapping)

print("Κατανομή κατηγοριών ρίσκου μετά τη δημιουργία (πριν το διαχωρισμό):")
print(df['RISK_LEVEL_ENCODED'].value_counts())

# Κελί 6: Διαχωρισμός σε X και y, και σε Training/Testing Sets
# ΕΠΙΛΟΓΗ ΤΕΛΙΚΩΝ ΧΑΡΑΚΤΗΡΙΣΤΙΚΩΝ ΠΟΥ ΘΑ ΧΡΗΣΙΜΟΠΟΙΗΘΟΥΝ ΓΙΑ ΤΗΝ ΕΚΠΑΙΔΕΥΣΗ
selected_features_for_model_base = [
    'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
    'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
    'CNT_FAM_MEMBERS', 'DAYS_BIRTH', 'DAYS_EMPLOYED', # Οι αρχικές που χρησιμοποιούνται για μηχανική χαρακτηριστικών
    'AGE_YEARS', 'YEARS_EMPLOYED', 'INCOME_PER_FAMILY_MEMBER', 'EMPLOYMENT_AGE_RATIO' # Οι μηχανικές
]

final_X_columns = [col for col in df.columns if col in selected_features_for_model_base or any(col.startswith(f'{base}_') for base in selected_features_for_model_base if f'{base}_' in col)]
final_X_columns = [col for col in final_X_columns if col not in ['ID', 'RISK_LEVEL', 'RISK_LEVEL_ENCODED', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']]

X = df[final_X_columns].copy()

y = df['RISK_LEVEL_ENCODED']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nΚατανομή κατηγοριών ρίσκου στο y_train (εκπαίδευση):")
print(y_train.value_counts())
print("\nΚατανομή κατηγοριών ρίσκου στο y_test (δοκιμή):")
print(y_test.value_counts())

# Κελί 6.5: Κλιμάκωση Δεδομένων (Feature Scaling) - ΤΩΡΑ ΕΔΩ ΓΙΝΕΤΑΙ!
# Ορίζουμε τις ακριβείς στήλες που θα κλιμακωθούν (μόνο οι αριθμητικές που υπάρχουν στο X)
scaled_feature_names_for_scaler = X_train.select_dtypes(include=np.number).columns.tolist()

scaler = StandardScaler()
X_train[scaled_feature_names_for_scaler] = scaler.fit_transform(X_train[scaled_feature_names_for_scaler])
X_test[scaled_feature_names_for_scaler] = scaler.transform(X_test[scaled_feature_names_for_scaler]) # Εφαρμόζουμε τον ίδιο scaler στο test set

# Κελί 7: Εκπαίδευση Random Forest
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
print("\nΟ Random Forest εκπαιδεύτηκε!")

# Κελί 8: Εκπαίδευση Logistic Regression
model_lr = LogisticRegression(random_state=42, solver='lbfgs', multi_class='auto', max_iter=1000)
model_lr.fit(X_train, y_train)
print("Ο Logistic Regression εκπαιδεύτηκε!")

# Κελί 10: Προβλέψεις
y_pred_rf = model_rf.predict(X_test)
y_pred_lr = model_lr.predict(X_test)

# Κελί 11: Αξιολόγηση
risk_names = ['Χαμηλό Ρίσκο', 'Μεσαίο Ρίσκο', 'Υψηλό Ρίσκο']

models_to_evaluate = {
    "Random Forest": y_pred_rf,
    "Logistic Regression": y_pred_lr
}

for model_name, y_pred in models_to_evaluate.items():
    print(f"\n--- Αξιολόγηση για το μοντέλο: {model_name} ---")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Ακρίβεια (Accuracy): {accuracy:.4f}")

    print("\nΑναφορά Ταξινόμησης:")
    print(classification_report(y_test, y_pred, target_names=risk_names, labels=[0, 1, 2], zero_division='warn'))

    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    print("\nΠίνακας Σύγχυσης:")
    print(conf_matrix)

    plt.figure(figsize=(7, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Προβλ. {name}' for name in risk_names],
                yticklabels=[f'Πραγμ. {name}' for name in risk_names])
    plt.xlabel('Προβλεπόμενη Κατηγορία Ρίσκου')
    plt.ylabel('Πραγματική Κατηγορία Ρίσκου')
    plt.title(f'Πίνακας Σύγχυσης για {model_name}')
    plt.show()

print("\n--- Η Αξιολόγηση Όλων των Μοντέλων Ολοκληρώθηκε ---")

# Κελί 12: Σημαντικότητα Χαρακτηριστικών (για Random Forest)
print("\n--- Σημαντικότητα Χαρακτηριστικών (Random Forest) ---")
feature_importances = model_rf.feature_importances_
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})
features_df = features_df.sort_values(by='Importance', ascending=False)
print(features_df.head(15))

plt.figure(figsize=(12, 7))
sns.barplot(x='Importance', y='Feature', data=features_df.head(15), palette='viridis')
plt.title('Top 15 Σημαντικά Χαρακτηριστικά για την Πρόβλεψη Ρίσκου (Random Forest)')
plt.xlabel('Σημαντικότητα')
plt.ylabel('Χαρακτηριστικό')
plt.show()

# Κελί 13: Αποθήκευση Μοντέλου και Προεπεξεργαστών
joblib.dump(model_rf, 'risk_classifier_model.pkl')
print("\nΤο μοντέλο Random Forest αποθηκεύτηκε ως 'risk_classifier_model.pkl'")

joblib.dump(scaler, 'scaler.pkl')
print("Ο StandardScaler αποθηκεύτηκε ως 'scaler.pkl'")

joblib.dump(X.columns.tolist(), 'model_features.pkl')
print("Τα ονόματα των χαρακτηριστικών αποθηκεύτηκαν ως 'model_features.pkl'")

joblib.dump(risk_mapping, 'risk_mapping.pkl')
print("Το risk_mapping αποθηκεύτηκε ως 'risk_mapping.pkl'")

# ΣΗΜΑΝΤΙΚΟ: scaled_feature_names_for_scaler
# Αυτό πλέον δημιουργείται και ορίζεται μετά το X_train/X_test
joblib.dump(scaled_feature_names_for_scaler, 'scaled_feature_names_for_scaler.pkl')
print("Οι ονομασίες των κλιμακωμένων χαρακτηριστικών για τον scaler αποθηκεύτηκαν ως 'scaled_feature_names_for_scaler.pkl'")