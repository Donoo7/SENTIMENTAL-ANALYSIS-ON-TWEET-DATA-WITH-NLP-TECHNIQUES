import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from streamlit_extras.let_it_rain import rain
import base64

# Load and preprocess the data
df = pd.read_csv("cleanedtwitter.csv")
# Perform any necessary preprocessing steps here
df.dropna(subset=['clean_text'], inplace=True)
# Handle missing values in text data
df['clean_text'].fillna('', inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['sentiment'], test_size=0.2,
                                                     random_state=42)
# Vectorize the text data
Tfidf_vector = TfidfVectorizer(max_features=5000)
X_train_tfidf = Tfidf_vector.fit_transform(X_train)
X_test_tfidf = Tfidf_vector.transform(X_test)

# Train a sentiment analysis model
svm = SVC(kernel = 'linear')
svm.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = svm.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy}")

#save the model
joblib.dump(svm, "svm.pkl")
joblib.dump(Tfidf_vector, "svm_vector.pkl")

# Streamlit app
def main():
    st.title("Sentiment Analysis App")

    # Text input for user to enter new text
    user_input = st.text_input("Enter text to analyze sentiment:")

    # Button to perform sentiment analysis
    if st.button("Analyze"):
        if user_input:
            # Vectorize the user input
            user_input_tfidf = Tfidf_vector.transform([user_input])

            # Predict sentiment using the trained model
            prediction = svm.predict(user_input_tfidf)[0]

            # Display the sentiment prediction
            st.subheader("Predicted Sentiment:")
            st.write(f"{prediction}")

            # Display rain emojis based on predicted sentiment
            if prediction == 'positive':
                rain(
                    emoji = "😊😁😸",
                    font_size = 20, # the size of emoji
                    falling_speed = 3, # speed of raining
                    animation_length = "infinite", # for how much time the animation
                    )
            elif prediction == 'negative':
                rain(
                    emoji = "😒😣😓",
                    font_size = 20, # the size of emoji
                    falling_speed = 3, # speed of raining
                    animation_length = "infinite", # for how much time the animation
                    )
            else:
                rain(
                    emoji = "😉🫡🙂",
                    font_size = 20, # the size of emoji
                    falling_speed = 3, # speed of raining
                    animation_length = "infinite", # for how much time the animation
                    )


            # Display a GIF
            if prediction == 'positive':
                file_ = open("positive.gif", "rb")
                contents = file_.read()
                data_url = base64.b64encode(contents).decode("utf-8")
                file_.close()
                st.markdown(
                    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
                    unsafe_allow_html=True,)
            elif prediction == 'negative':
                file_ = open("negative.gif", "rb")
                contents = file_.read()
                data_url = base64.b64encode(contents).decode("utf-8")
                file_.close()
                st.markdown(
                    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
                    unsafe_allow_html=True,)
            else:
                file_ = open("neutral.gif", "rb")
                contents = file_.read()
                data_url = base64.b64encode(contents).decode("utf-8")
                file_.close()
                st.markdown(
                    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
                    unsafe_allow_html=True,)
if __name__ == "__main__":
    main()
