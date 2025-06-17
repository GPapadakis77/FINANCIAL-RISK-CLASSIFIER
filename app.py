import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Φόρτωση των αποθηκευμένων αρχείων ---
try:
    model = joblib.load('risk_classifier_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_features = joblib.load('model_features.pkl')
    risk_mapping_loaded = joblib.load('risk_mapping.pkl')
    scaled_feature_names_for_scaler = joblib.load('scaled_feature_names_for_scaler.pkl')
    
    reverse_risk_mapping = {v: k for k, v in risk_mapping_loaded.items()}

except FileNotFoundError:
    st.error("Σφάλμα: Τα αρχεία του μοντέλου (.pkl) δεν βρέθηκαν. Βεβαιωθείτε ότι βρίσκονται στον ίδιο φάκελο με το app.py.")
    st.stop()

# --- 2. Συναρτήσεις Προεπεξεργασίας για Νέα Δεδομένα ---
def preprocess_new_data(input_data):
    df_single = pd.DataFrame([input_data])

    df_single['AGE_YEARS'] = np.abs(df_single['DAYS_BIRTH']) / 365.25
    df_single['YEARS_EMPLOYED'] = df_single['DAYS_EMPLOYED'].apply(lambda x: np.abs(x) / 365.2425 if x != 365243 else 0)
    df_single['INCOME_PER_FAMILY_MEMBER'] = df_single['AMT_INCOME_TOTAL'] / df_single['CNT_FAM_MEMBERS']
    df_single.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_single['INCOME_PER_FAMILY_MEMBER'].fillna(0, inplace=True)

    df_single['EMPLOYMENT_AGE_RATIO'] = df_single.apply(lambda row: row['YEARS_EMPLOYED'] / row['AGE_YEARS'] if row['AGE_YEARS'] > 0 else 0, axis=1)

    # Δημιουργούμε ένα κενό DataFrame με ΟΛΕΣ τις στήλες που περίμενε το μοντέλο (model_features)
    # Θέτουμε αρχικά όλες τις τιμές σε 0
    processed_df = pd.DataFrame(0, index=[0], columns=model_features)

    # Συμπληρώνουμε τις αριθμητικές τιμές και τις μηχανικές στήλες που επιλέχθηκαν
    # (οι FLAG_ στήλες δεν είναι πλέον μέρος του μοντέλου)
    selected_base_features_numeric_app = [
        'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS',
        'DAYS_BIRTH', 'DAYS_EMPLOYED',
        'AGE_YEARS', 'YEARS_EMPLOYED', 'INCOME_PER_FAMILY_MEMBER', 'EMPLOYMENT_AGE_RATIO'
    ]

    for col in selected_base_features_numeric_app:
        if col in df_single.columns and col in processed_df.columns:
            processed_df[col] = df_single[col].iloc[0]

    # Χειρισμός κατηγορικών στηλών (One-Hot Encoding)
    original_categorical_cols = [
        'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE',
        'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'
    ]
    
    for col in original_categorical_cols:
        if col in df_single.columns and df_single[col].iloc[0] is not None:
            dummy_col_name = f'{col}_{df_single[col].iloc[0]}'
            if dummy_col_name in processed_df.columns:
                processed_df[dummy_col_name] = 1

    # Εξασφαλίζουμε ότι η σειρά των στηλών είναι ΑΚΡΙΒΩΣ η ίδια με αυτή που εκπαιδεύτηκε το μοντέλο
    processed_df = processed_df[model_features].copy()
    
    # --- Κλιμάκωση Δεδομένων (Feature Scaling) ---
    # Χρησιμοποιούμε την ακριβή λίστα των στηλών που "είδε" ο scaler κατά την εκπαίδευση
    processed_df[scaled_feature_names_for_scaler] = scaler.transform(processed_df[scaled_feature_names_for_scaler])
    
    return processed_df

# --- 3. Δημιουργία Περιβάλλοντος Streamlit ---
PURPLE_PRIMARY = "#6A0DAD"
PURPLE_ACCENT = "#9370DB"
WHITE_BG = "#FFFFFF"
TEXT_COLOR = "#333333"
LIGHT_GRAY_LINE = "#E0E0E0"

st.set_page_config(layout="wide", page_title="AI Αξιολόγηση Ρίσκου Πελάτη", page_icon="📊")

st.markdown(
    f"""
    <style>
    body {{
        background-color: {WHITE_BG};
        color: {TEXT_COLOR};
        font-family: 'Arial', sans-serif;
    }}
    .main .block-container {{
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
        background-color: {WHITE_BG};
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}
    
    h1 {{
        color: {PURPLE_PRIMARY};
        font-family: 'Arial', sans-serif;
        text-align: center;
        padding-bottom: 10px;
        font-size: 2.5em;
    }}
    h2, h3 {{
        color: {PURPLE_PRIMARY};
        font-family: 'Arial', sans-serif;
        text-align: center;
    }}
    h4, h5, h6 {{
        color: {TEXT_COLOR};
        font-family: 'Arial', sans-serif;
    }}

    .stButton>button {{
        background-color: {PURPLE_ACCENT};
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 25px;
        font-size: 1.1em;
        font-weight: bold;
        transition: background-color 0.3s ease, transform 0.1s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
    .stButton>button:hover {{
        background-color: {PURPLE_PRIMARY};
        transform: translateY(-2px);
    }}

    .stAlert.success {{
        background-color: #e6ffe6;
        color: #006600;
        border-left: 5px solid #4CAF50;
        border-radius: 5px;
    }}
    .stAlert.warning {{
        background-color: #fff9e6;
        color: #cc6600;
        border-left: 5px solid #FFC107;
        border-radius: 5px;
    }}
    .stAlert.error {{
        background-color: #ffe6e6;
        color: #cc0000;
        border-left: 5px solid #F44336;
        border-radius: 5px;
    }}
    
    .stSlider > div > div > div:nth-child(2) > div {{
        background: {PURPLE_ACCENT};
    }}
    .stSlider > div > div > div:nth-child(1) > div {{
        background: {PURPLE_PRIMARY};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Κύριο Περιεχόμενο Εφαρμογής ---

try:
    st.image('grant.png', width=200)
except FileNotFoundError:
    st.warning("Η φωτογραφία 'grant.png' δεν βρέθηκε. Βεβαιωθείτε ότι βρίσκεται στον ίδιο φάκελο.")

st.title("AI-Driven Financial Risk Classifier")
st.markdown("## Έξυπνο μοντέλο αξιολόγησης οικονομικού ρίσκου πελατών για το Onboarding")
st.markdown(f"""
    <div style="text-align: center; color: {TEXT_COLOR}; font-size: 1.1em; margin-bottom: 20px;">
    Αυτό το εργαλείο χρησιμοποιεί τεχνητή νοημοσύνη για να αξιολογήσει γρήγορα και με ακρίβεια το οικονομικό ρίσκο
    των νέων πελατών, συμβάλλοντας στην αποτελεσματικότητα και ασφάλεια των διαδικασιών onboarding.
    </div>
    <br style='line-height:2;'/>
    """, unsafe_allow_html=True)

st.markdown(f"<hr style='border-top: 3px solid {LIGHT_GRAY_LINE};'>", unsafe_allow_html=True)

st.header("Εισάγετε τα στοιχεία του νέου πελάτη:")

with st.expander("Βασικά Στοιχεία Πελάτη", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Δημογραφικά")
        gender = st.selectbox("Φύλο:", ('M', 'F'))
        own_car = st.selectbox("Ιδιοκτήτης Αυτοκινήτου;", ('Y', 'N'))
        own_realty = st.selectbox("Ιδιοκτήτης Ακινήτου;", ('Y', 'N'))
        children = st.number_input("Αριθμός Παιδιών:", min_value=0, max_value=19, value=0)
        
    with col2:
        st.markdown("#### Οικογενειακή Κατάσταση & Ηλικία")
        default_family_members = max(1, children + (1 if children == 0 else 0))
        family_members = st.number_input("Αριθμός Μελών Οικογένειας:", min_value=1, max_value=20, value=default_family_members)
        
        if family_members < children:
            st.warning("Ο αριθμός μελών οικογένειας δεν μπορεί να είναι μικρότερος από τον αριθμό παιδιών. Διορθώθηκε.")
            family_members = children if children > 0 else 1

        age_years = st.slider("Ηλικία (σε χρόνια):", min_value=18, max_value=70, value=30)
        days_birth = int(age_years * -365.25)

with st.expander("Οικονομικά & Εργασιακά Στοιχεία", expanded=True):
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Οικονομικά")
        income_total = st.number_input("Συνολικό Εισόδημα (€):", min_value=0.0, value=150000.0, step=1000.0)
        income_type = st.selectbox("Τύπος Εισοδήματος:", ('Working', 'Commercial associate', 'State servant', 'Pensioner', 'Student', 'Unemployed', 'Businessman'))
        education_type = st.selectbox("Επίπεδο Εκπαίδευσης:", ('Higher education', 'Secondary / secondary special', 'Incomplete higher', 'Lower secondary', 'Academic degree'))
        
    with col4:
        st.markdown("#### Εργασιακά & Κατοικία")
        family_status = st.selectbox("Οικογενειακή Κατάσταση:", ('Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'))
        housing_type = st.selectbox("Τύπος Κατοικίας:", ('House / apartment', 'Rented apartment', 'Municipal apartment', 'With parents', 'Co-op apartment', 'Office apartment'))
        
        years_employed = st.slider("Χρόνια Απασχόλησης:", min_value=0, max_value=50, value=5)
        if years_employed == 0:
            days_employed = 365243
        else:
            days_employed = int(years_employed * -365.25)

        occupation_type = st.selectbox("Επάγγελμα:", ('Unknown', 'Core staff', 'Working', 'Laborers', 'Sales staff', 'Managers', 'Drivers', 'Accountants', 'High skill tech staff', 'Medicine staff', 'Security staff', 'Cooking staff', 'Cleaning staff', 'Private service staff', 'Low-skill Laborers', 'Waiters/barmen staff', 'Secretaries', 'Realty agents', 'HR staff', 'IT staff'))

# --- 4. Κουμπί Πρόβλεψης ---
st.markdown(f"<hr style='border-top: 3px solid {LIGHT_GRAY_LINE};'>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)
if st.button("Προβλεψε Επίπεδο Ρίσκου"):
    st.markdown("</div>", unsafe_allow_html=True)
    input_data = {
        'CODE_GENDER': gender,
        'FLAG_OWN_CAR': own_car,
        'FLAG_OWN_REALTY': own_realty,
        'CNT_CHILDREN': children,
        'AMT_INCOME_TOTAL': income_total,
        'NAME_INCOME_TYPE': income_type,
        'NAME_EDUCATION_TYPE': education_type,
        'NAME_FAMILY_STATUS': family_status,
        'NAME_HOUSING_TYPE': housing_type,
        'DAYS_BIRTH': days_birth,
        'DAYS_EMPLOYED': days_employed,
        'OCCUPATION_TYPE': occupation_type,
        'CNT_FAM_MEMBERS': family_members,
    }

    with st.spinner('Υπολογισμός ρίσκου...'):
        processed_input = preprocess_new_data(input_data)
        
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(processed_input)[0]
            prediction_encoded = model.predict(processed_input)[0]
            confidence = prediction_proba[prediction_encoded] * 100
            confidence_text = f" (με εμπιστοσύνη: {confidence:.2f}%)"
        else:
            prediction_encoded = model.predict(processed_input)[0]
            confidence_text = ""

    predicted_risk_level = reverse_risk_mapping.get(prediction_encoded, "Άγνωστο Ρίσκο")

    st.subheader("Αποτέλεσμα Πρόβλεψης:")
    if predicted_risk_level == 'Υψηλό Ρίσκο':
        st.error(f"Το προβλεπόμενο οικονομικό ρίσκο είναι: **{predicted_risk_level}** 🔴{confidence_text}")
    elif predicted_risk_level == 'Μεσαίο Ρίσκο':
        st.warning(f"Το προβλεπόμενο οικονομικό ρίσκο είναι: **{predicted_risk_level}** 🟠{confidence_text}")
    else:
        st.success(f"Το προβλεπόμενο οικονομικό ρίσκου είναι: **{predicted_risk_level}** 🟢{confidence_text}")

    st.write("---")
    st.markdown("##### Αναλυτικά Δεδομένα Εισόδου (μετά την προεξεργασία):")
    st.write(processed_input)
else:
    st.markdown("</div>", unsafe_allow_html=True)