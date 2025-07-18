# AI-Driven Financial Risk Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://financial-risk-classifier-faviwyfhsquurylawzzxji.streamlit.app/)

---

## Ζωντανή Εφαρμογή

Δοκιμάστε την εφαρμογή μας live εδώ: [https://financial-risk-classifier-faviwyfhsquurylawzzxji.streamlit.app/](https://financial-risk-classifier-faviwyfhsquurylawzzxji.streamlit.app/)

---

## Μεθοδολογία & Πώς Λειτουργεί

Το project αυτό αναπτύχθηκε για να προσφέρει έναν έξυπνο τρόπο αξιολόγησης του οικονομικού ρίσκου των πελατών μας. Η πορεία μας προς το μοντέλο ρίσκου περιλάμβανε τα εξής βήματα:

1.  **Προετοιμασία Δεδομένων:**
    * **Καθάρισμα:** Συγκεντρώσαμε και καθαρίσαμε σχολαστικά τα δεδομένα από το αρχείο `application_record.csv`. Αυτό σήμαινε να διορθώσουμε τυχόν κενά ή λάθη (π.χ., άγνωστο επάγγελμα), και να βεβαιωθούμε ότι κάθε πελάτης εμφανίζεται μία μόνο φορά.
    * **Προσθήκη Νοήματος (Feature Engineering):** Δεν μείναμε μόνο στις αρχικές πληροφορίες. Δημιουργήσαμε νέες, πιο έξυπνες παραμέτρους, όπως η πραγματική ηλικία σε χρόνια, πόσα χρόνια εργάζεται κάποιος, ή πόσο εισόδημα αναλογεί σε κάθε μέλος της οικογένειας. Αυτό βοήθησε το μοντέλο να "καταλάβει" καλύτερα τα στοιχεία.
    * **Μετατροπή για τον Υπολογιστή:** Για να μπορεί ο υπολογιστής μας να επεξεργαστεί τα δεδομένα, μετατρέψαμε τις λέξεις (π.χ., "άνδρας", "γυναίκα") σε αριθμούς. Επίσης, κάναμε όλους τους αριθμούς να έχουν την ίδια "βαρύτητα" (κλιμάκωση), ώστε κανένας αριθμός να μην επηρεάζει άδικα την απόφαση του μοντέλου.

2.  **Ορισμός του "Τι Ψάχνουμε" (Επίπεδο Ρίσκου):**
    * Το ρίσκο δεν ήταν έτοιμο στο αρχείο μας, οπότε το φτιάξαμε εμείς. Ορίσαμε τι σημαίνει Χαμηλό, Μεσαίο και Υψηλό Ρίσκο. Αυτός ο ορισμός δεν βασίστηκε απλά στο εισόδημα, αλλά σε έναν συνδυασμό παραγόντων όπως το συνολικό εισόδημα, ο αριθμός των παιδιών και τα χρόνια απασχόλησης. Έτσι, το μοντέλο έμαθε να λαμβάνει υπόψη περισσότερα στοιχεία.

3.  **Εκπαίδευση & Δοκιμή του Μοντέλου:**
    * **Διαχωρίσαμε τα Δεδομένα:** Δώσαμε το μεγαλύτερο μέρος των δεδομένων στον "εγκέφαλο" (το μοντέλο) για να "διδαχθεί" και κρατήσαμε ένα μικρότερο μέρος για να τον "δοκιμάσουμε" σε πράγματα που δεν έχει ξαναδεί.
    * **Οι "Εγκέφαλοι" που Χρησιμοποιήσαμε:**
        * **Random Forest Classifier:** Αυτός ο "εγκέφαλος" είναι σαν μια ομάδα από 100 μικρά ρομποτάκια που συνεργάζονται για να πάρουν την πιο σωστή απόφαση. Τον επιλέξαμε γιατί είναι πολύ ακριβής και μπορεί να χειριστεί σύνθετα δεδομένα.
        * **Logistic Regression:** Αυτός είναι ένας πιο απλός "εγκέφαλος", που λειτουργεί σαν να σχεδιάζει μια γραμμή για να ξεχωρίσει τις ομάδες. Τον χρησιμοποιήσαμε ως σημείο αναφοράς για να δούμε την απλούστερη προσέγγιση.
    * Η απόδοση των μοντέλων αξιολογήθηκε μέσω μετρικών όπως η ακρίβεια (Accuracy), η αναφορά ταξινόμησης (Precision, Recall, F1-Score) και ο πίνακας σύγχυσης (Confusion Matrix).

### Βασικές Παράμετροι Αξιολόγησης

Το μοντέλο μας δεν κοιτάει απλά τον πελάτη. Μετράει συγκεκριμένες, σημαντικές παραμέμετρους που "μαρτυρούν" πολλά για το οικονομικό του ρίσκο. Εδώ είναι οι πιο βασικές, και γιατί έχουν σημασία:

* **Συνολικό Εισόδημα (AMT_INCOME_TOTAL):** Είναι, φυσικά, η πιο άμεση ένδειξη του πόσα μπορεί να σηκώσει κάποιος. Λίγα χρήματα, μεγαλύτερο ρίσκο, γενικά.
* **Ηλικία (AGE_YEARS):** Όπως κι εμείς, έτσι και το μοντέλο ξέρει πως η ηλικία φέρνει συνήθως μεγαλύτερη οικονομική σταθερότητα και εμπειρία.
* **Χρόνια Απασχόλησης (YEARS_EMPLOYED):** Σταθερή δουλειά σημαίνει συνήθως πιο σταθερά έσοδα. Οπότε, τα χρόνια που κάποιος εργάζεται είναι σημαντικός δείκτης.
* **Εισόδημα ανά Μέλος Οικογένειας (INCOME_PER_FAMILY_MEMBER):** Δεν είναι μόνο το τι μπαίνει στην τσέπη σου, αλλά και πόσα άτομα πρέπει να ταΐσεις. Αυτή η παράμετρος δείχνει την πραγματική οικονομική πίεση.
* **Αριθμός Παιδιών (CNT_CHILDREN):** Περισσότερα παιδιά, περισσότερες υποχρεώσεις. Το μοντέλο το λαμβάνει υπόψη.
* **Ιδιοκτήτης Ακινήτου (FLAG_OWN_REALTY):** Το να έχεις δικό σου σπίτι συνήθως σημαίνει οικονομική ωριμότητα και ένα "μαξιλάρι ασφαλείας".
* **Επίπεδο Εκπαίδευσης (NAME_EDUCATION_TYPE):** Υψηλότερη μόρφωση συχνά οδηγεί σε πιο σταθερές και καλύτερα αμειβόμενες θέσεις εργασίας.
* **Οικογενειακή Κατάσταση (NAME_FAMILY_STATUS):** Μπορεί να επηρεάζει τις κοινές οικονομικές υποχρεώσεις και τη γενικότερη σταθερότητα.
* **Τύπος Εισοδήματος (NAME_INCOME_TYPE):** Το αν είσαι μισθωτός, ελεύθερος επαγγελματίας ή συνταξιούχος λέει πολλά για τη ροή και τη σταθερότητα των χρημάτων σου.

Αυτές οι παράμετροι, μαζί με άλλες όπως ο τύπος κατοικίας και το επάγγελμα, "διδάσκουν" στο μοντέλο μας να αναγνωρίζει τα μοτίβα ρίσκου.

