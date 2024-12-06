import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import base64
import os

# Function to add a background image
def add_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_image}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("Background image not found. Please check the file path.")

# Set the background image
background_image_path = "/Users/akashngowda/Desktop/verdictAI2/static/backround.jpg"  # Ensure this file exists
add_background(background_image_path)

# App title
st.title("VerdictAI: IPC Section Prediction")
st.markdown(
    """
    <style>
    h1 {
        text-align: center;
        font-family: 'Arial', sans-serif;
        font-size: 48px;
        color: #ffffff;
        text-shadow: 2px 2px 4px #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load and preprocess dataset
file_path = '/Users/akashngowda/Desktop/5th sem mini pro/ipc_sections 2.csv'  # Update with the correct file path
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    df = df[['Offense', 'Description']]
    X = df['Offense'].astype(str).values  # Features
    y = df['Description'].astype(str).values  # Target

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Logistic Regression Model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Predict function
    def predict_description(offense_text):
        try:
            offense_tfidf = vectorizer.transform([offense_text])
            predicted_label = model.predict(offense_tfidf)
            return label_encoder.inverse_transform(predicted_label)[0]
        except Exception as e:
            return f"An error occurred: {str(e)}"

    # Predefined keywords for dropdown menu
    predefined_keywords = [
        "Soldier",
        "property taken by a war",
        "voluntarily allowing prisoner",
        "negligently suffering prisoner",
        "Aiding escape",
        "attempting to seduce an officer",
        "assault by An officer",
        "Abetment of such assault, if the assault is committed",
        "Harbouring an officer",
        "Deserter concealed",
        "Abetment of act of insubordination",
        "offences against the State",
        "design to wage War",
        "Assaulting President",
        "Sedition",
        "Committing depredation On the territories",
        "Promoting enmity between classes",
        "Owner or occupier of land not giving information of riot",
        "whose behalf a riot takes place",
        "whose benefit a riot is committed",
        "Harbouring persons",
        "Engaging in a public fight",
        "Joining or continuing in an unlawful assembly",
        "Committing affray",
        "Taking a gratification in order",
        "personal influence",
        "Abetment by public servant of the offences defined",
        "Public servant obtaining any valuable thing, without consideration",
        "Public servant disobeying direction under law",
        "Public servant framing an incorrect document with intent to cause injury",
        "unlawfully engaging in trade",
        "Personating a Public servant",
        "Bribery",
        "Undue influence at an election",
        "Failure to keep election accounts",
        "Absconding to avoid service of summons",
        "Preventing the service",
        "Not obeying a legal order to attend at a certain place in person",
        "Failure to appear at specified place",
        "intentionally omitting to produce a document",
        "Intentionally omitting to Give notice",
        "Rioting armed with deadly Weapon",
        "false information to a public servant",
        "Refusing oat",
        "refusing to answer questions",
        "Refusing to sign statement",
        "lawful power to the injury or annoyance of any person",
        "Bidding",
        "Threatening any person to induce him to refrain",
        "Hiring, engaging or employing persons to Take part in an unlawful assembly",
        "false evidence in a judicial proceeding",
        "Trespassing in place of worship or sepulchr",
        "Threatening any person to give false evidence",
        "Using in a judicial proceeding evidence known to be false or fabricated",
        "Knowingly issuing or signing a false certificate",
        "Using as a true certificate one known to be false in a material point",
        "False statement made in any declaration",
        "Using as true any such declaration known to be false",
        "Causing disappearance of evidence of an offence committed",
        "Giving false information respecting an offence committed",
        "Secreting or destroying any document",
        "False personation for the purpose of any act",
        "Fraudulent removal or concealment, etc.",
        "Claiming property without right",
        "suffering decree to be executed after it has been satisfied",
        "False claim in a court of Justice",
        "Fraudulently obtaining a decree for a sum not due",
        "False charge of offence made with intent to injure",
        "Harbouring an offender",
        "Taking gift, etc., to screen an offender from punishment, If the offence be capital",
        "Offering gift or restoration of property in consideration of screening offender, If the offence be capital",
        "Taking gift to help to recover movable property of which a person has been deprived",
        "Harbouring an offender who has escaped from custody",
        "property from forfeiture",
        "Public servant framing an incorrect record or writing with intent to save person from punishment",
        "Public servant in a judicial proceeding corruptly Making and pronouncing an order",
        "Commitment for trial or confinement by a person Having authority",
        "Intentional omission to apprehend on the part of A public servant bound by law to apprehend person under sentence of a court of Justice",
        "Violation of condition of remission of punishment",
        "Intentional insult or interruption to a public servant sitting in any stage of a judicial proceeding",
        "Disclosure of identity of the victim of certain offences, etc.",
        "Personation of a juror or assessor",
        "Failure by person released on bail or bond to appear in Court",
        "Counterfeiting, or performing any part of the process of counterfeiting coin",
        "Making, buying or selling instrument for the purpose of counterfeiting Indian coin",
        "Possession of instrument or material for the purpose of using the same for counterfeiting coin",
        "Abetting, in India, the counterfeiting, out of India, of coin",
        "Import or export of counterfeit coin",
        "Import or export of counterfeit of Indian coin, knowing the same to be counterfeit",
        "Having any counterfeit coin known to be such when it came into possession",
        "Knowingly delivering to another any counterfeit coin as genuine, which, when first possessed",
        "Possession of counterfeit coin by a person who knew it to be counterfeit when he became possessed thereof",
        "Person employed in a Mint causing coin to be of a different weight",
        "Unlawfully taking from a Mint any coining instrument",
        "Fraudulently diminishing the weight or altering the composition of any coin",
        "Fraudulently diminishing the weight or altering the composition of Indian coin",
        "Altering appearance of any coin with intent that it shall pass as a coin of a different description",
        "Delivery to another of coin possessed with the knowledge that it is altered",
        "Delivery of Indian coin possessed with the knowledge that it is altered",
        "Possession of altered coin by a person who knew it to be altered when he became possessed thereof",
        "Possession of Indian coin by a person who knew it to be altered when he became possessed thereof",
        "Delivery to another of coin as genuine which, when first possessed, the deliverer did not know to be altered"
    ]

    # Input form
    st.subheader("Enter Offense Details:")
    
    # Dropdown menu
    selected_keyword = st.selectbox("Choose a predefined offense keyword:", ["Select a keyword"] + predefined_keywords)

    # Input text box
    offense_input = st.text_input(
        "Or describe the offense:", 
        value=selected_keyword if selected_keyword != "Select a keyword" else "",
        placeholder="Enter details of the offense..."
    )

    # Predict button
    if st.button("Predict"):
        if offense_input:
            prediction = predict_description(offense_input)
            st.success(f"Predicted IPC Section: {prediction}")
        else:
            st.warning("Please enter offense details to predict.")
else:
    st.error("Dataset not found. Please check the file path.")

# Footer
st.markdown(
    """
    <footer style='text-align: center; margin-top: 20px;'>
        <p>by VerdictAI Team | For inquiries, email: akashngowda2004@gmail.com</p>
    </footer>
    """,
    unsafe_allow_html=True
)
