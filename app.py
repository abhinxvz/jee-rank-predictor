import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="JEE College Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model and preprocessor
@st.cache_resource
def load_model():
    model = joblib.load('college_predictor_model.pkl')
    preprocessor = joblib.load('college_predictor_preprocessor.pkl')
    return model, preprocessor


def predict_college(rank, category, gender, year=2022, quota='AI', institute_type=None):
    input_data = pd.DataFrame({
        'opening_rank': [rank],
        'closing_rank': [rank],
        'rank_difference': [0],
        'rank_average': [rank],
        'rank_ratio': [1.0],
        'year_numeric': [year],
        'program_duration_years': [4.0],
        'is_female': [1 if gender.lower() == 'female' else 0],
        'is_AI_quota': [1 if quota == 'AI' else 0],
        'is_HS_quota': [1 if quota == 'HS' else 0],
        'is_OS_quota': [1 if quota == 'OS' else 0],
        'is_IIT': [1 if institute_type == 'IIT' else (0 if institute_type else 1)],
        'is_NIT': [1 if institute_type == 'NIT' else 0],
        'is_btech': [1],
        'category': [category],
        'is_preparatory': [0]
    })
    
    model, preprocessor = load_model()
    
  
    input_transformed = preprocessor.transform(input_data)

    proba = model.predict_proba(input_transformed)
    top_indices = proba[0].argsort()[::-1][:10] 
    
  
    classes = model.classes_

    return [(classes[i], proba[0][i]) for i in top_indices]

def main():
    st.title("JEE College Predictor 🎓")
    st.write("Predict your potential colleges based on JEE rank and other parameters")
    
    
    with st.container():
        st.subheader("Enter Your Details")
        col1, col2 = st.columns(2)
        with col1:
            rank = st.number_input("JEE Rank", min_value=1, max_value=1000000, value=1000)
            category = st.selectbox("Category", ["GEN", "OBC-NCL", "SC", "ST"])
            gender = st.radio("Gender", ["Male", "Female"])
        with col2:
            quota = st.selectbox("Quota", ["AI", "HS", "OS"], help="AI: All India, HS: Home State, OS: Other State")
            institute_type = st.selectbox("Institute Type", ["All", "IIT", "NIT"])
    
        if st.button("Predict Colleges", type="primary"):
            institute_type_param = None if institute_type == "All" else institute_type
            predictions = predict_college(rank, category, gender, quota=quota, institute_type=institute_type_param)
            
            st.subheader("Top College Predictions")
            
            
            pred_df = pd.DataFrame(predictions, columns=["Institute", "Probability"])
            
          
            pred_df["Probability_num"] = pred_df["Probability"]
            pred_df["Probability"] = pred_df["Probability"].apply(lambda x: f"{x*100:.2f}%")
            
          
            styled_df = pred_df.style.background_gradient(
                subset=["Probability_num"],  
                cmap="YlOrRd",
                vmin=0,
                vmax=1
            )
           
            st.dataframe(
                styled_df,
                use_container_width=True
            )
           
            pred_df.drop(columns=["Probability_num"], inplace=True)
            
            
            fig, ax = plt.subplots(figsize=(10, 6))
            probabilities = [float(p[1]) for p in predictions]
            institutes = [p[0] for p in predictions]
            
            sns.barplot(x=probabilities, y=institutes)
            plt.title("College Prediction Probabilities")
            plt.xlabel("Probability")
            st.pyplot(fig)

    with st.expander("About the Model"):
        st.write("""
        This college prediction model uses machine learning to predict potential colleges based on your JEE rank and other parameters.
        The predictions are based on historical data and should be used as a reference only.
        
        **Note:** The actual college allotment may vary based on various other factors not considered in this model.
        """)

    
    st.markdown("Made by [abhinxvz](https://github.com/abhinxvz)")

if __name__ == "__main__":
    main()
