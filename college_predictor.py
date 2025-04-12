import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("Loading dataset...")
df = pd.read_csv('data.csv')

print("\nDataset Info:")
print(f"Shape: {df.shape}")
print("\nColumns:", df.columns.tolist())
print("\nSample data:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nNumber of unique institutes:", df['institute_short'].nunique())
print("\nTop 10 institutes by frequency:")
print(df['institute_short'].value_counts().head(10))

print("\nCategory distribution:")
print(df['category'].value_counts())

print("\nGender pool distribution:")
print(df['pool'].value_counts())

print("\nEnhanced preprocessing and feature engineering...")

def create_features(df):
    data = df.copy()
    
    data['year_numeric'] = data['year']
    data['rank_difference'] = data['closing_rank'] - data['opening_rank']
    data['rank_average'] = (data['closing_rank'] + data['opening_rank']) / 2
    data['rank_ratio'] = data['closing_rank'] / data['opening_rank'].replace(0, 1)
    data['program_duration_years'] = data['program_duration'].str.extract('(\d+)').astype(float)
    data['is_female'] = data['pool'].apply(lambda x: 1 if 'Female' in str(x) else 0)
    data['is_AI_quota'] = data['quota'].apply(lambda x: 1 if x == 'AI' else 0)
    data['is_HS_quota'] = data['quota'].apply(lambda x: 1 if x == 'HS' else 0)
    data['is_OS_quota'] = data['quota'].apply(lambda x: 1 if x == 'OS' else 0)
    data['is_IIT'] = data['institute_type'].apply(lambda x: 1 if x == 'IIT' else 0)
    data['is_NIT'] = data['institute_type'].apply(lambda x: 1 if x == 'NIT' else 0)
    data['is_btech'] = data['degree_short'].apply(lambda x: 1 if x == 'B.Tech' else 0)
    
    return data

df_engineered = create_features(df)

features = [
    'opening_rank', 'closing_rank', 'rank_difference', 'rank_average', 'rank_ratio',
    'year_numeric', 'program_duration_years', 'is_female', 'is_AI_quota', 'is_HS_quota',
    'is_OS_quota', 'is_IIT', 'is_NIT', 'is_btech', 'category', 'is_preparatory'
]

X = df_engineered[features]
y = df_engineered['institute_short']

categorical_features = ['category']
numerical_features = [col for col in features if col != 'category']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

# Check for class imbalance
print("\nClass distribution in training set:")
print(y_train.value_counts().head(10))

print("\nUsing class weights to handle imbalance instead of SMOTE...")
X_train_preprocessed = preprocessor.fit_transform(X_train)

print("\nTraining optimized Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=RANDOM_SEED,
    n_jobs=-1
)

rf_model.fit(X_train_preprocessed, y_train)

X_test_preprocessed = preprocessor.transform(X_test)
rf_pred = rf_model.predict(X_test_preprocessed)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')

print(f"\nRandom Forest - Accuracy: {rf_accuracy:.4f}, F1 Score: {rf_f1:.4f}")

best_model = rf_model

joblib.dump(best_model, 'college_predictor_model.pkl')
joblib.dump(preprocessor, 'college_predictor_preprocessor.pkl')

feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(['category']))

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance.head(10))

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Feature Importance for College Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')

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
    
    # Load the preprocessor and model
    preprocessor = joblib.load('college_predictor_preprocessor.pkl')
    model = joblib.load('college_predictor_model.pkl')
    
    # Transform the input data
    input_transformed = preprocessor.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_transformed)
    
    # Get prediction probabilities
    proba = model.predict_proba(input_transformed)
    top_indices = np.argsort(proba[0])[::-1][:10]  # Get indices of top 10 predictions
    
    # Get class names
    classes = model.classes_
    
    # Return top predictions with probabilities
    return [(classes[i], proba[0][i]) for i in top_indices]

print("\nExample Prediction:")
example_categories = df['category'].unique()
print(f"Available categories: {example_categories}")

# Test with an example
example_rank = 1000
example_category = 'GEN'
example_gender = 'Male'

# Create a more sophisticated interactive prediction function
def interactive_prediction():
    print("\n=== Advanced College Predictor ===")
    try:
        rank = int(input("Enter your rank: "))
        print(f"Available categories: {example_categories}")
        category = input("Enter your category (e.g., GEN, OBC-NCL, SC, ST): ")
        gender = input("Enter your gender (Male/Female): ")
        
        # Additional parameters (without year input)
        print("\nAdditional parameters (press Enter to use defaults):")
        # Using fixed year value
        year = 2022
        
        quota_input = input("Enter quota (AI/HS/OS) (default: AI): ")
        quota = quota_input.upper() if quota_input else 'AI'
        
        institute_type_input = input("Filter by institute type (IIT/NIT/All) (default: All): ")
        institute_type = None
        if institute_type_input.upper() == 'IIT':
            institute_type = 'IIT'
        elif institute_type_input.upper() == 'NIT':
            institute_type = 'NIT'
        
        if category not in example_categories:
            print(f"Warning: Category '{category}' not found in training data. Results may be inaccurate.")
        
        predictions = predict_college(rank, category, gender, year, quota, institute_type)
        
        print(f"\nTop 10 college predictions for Rank: {rank}, Category: {category}, Gender: {gender}:")
        for i, (college, probability) in enumerate(predictions, 1):
            print(f"{i}. {college}: {probability:.4f} probability")
            
        # Additional insights
        if predictions[0][1] > 0.5:
            print(f"\nHigh confidence prediction for {predictions[0][0]} with {predictions[0][1]:.2f} probability")
        elif predictions[0][1] - predictions[1][1] < 0.05:
            print(f"\nClose competition between {predictions[0][0]} and {predictions[1][0]}")
            
        # Return the prediction results and parameters for visualization
        return predictions, rank, category, gender, year, quota, institute_type
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please try again with valid inputs.")
        return None, 0, "", "", 0, "", None

def visualize_predictions(predictions, title):
    colleges = [p[0] for p in predictions[:5]]
    probabilities = [p[1] for p in predictions[:5]]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(colleges, probabilities, color='skyblue')
    plt.title(title)
    plt.xlabel('College')
    plt.ylabel('Probability')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    
    # Add probability values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    print("\nVisualization saved as 'prediction_results.png'")

if __name__ == "__main__":
    print("\nEnhanced model training completed. You can now use the advanced interactive predictor.")
    try:
        while True:
            try:
                result = interactive_prediction()
                
                if result[0] is not None:
                    predictions, rank, category, gender, year, quota, institute_type = result
                    
                    try:
                        viz_option = input("\nVisualize these results? (y/n): ").lower()
                        if viz_option == 'y':
                            visualize_predictions(predictions, f"College Predictions for Rank {rank} ({category}, {gender})")
                    except KeyboardInterrupt:
                        print("\nVisualization skipped.")
                        continue
                
                try:
                    if input("\nMake another prediction? (y/n): ").lower() != 'y':
                        break
                except KeyboardInterrupt:
                    print("\nExiting program.")
                    break
            except KeyboardInterrupt:
                print("\nPrediction interrupted.")
                if input("\nExit program? (y/n): ").lower() == 'y':
                    break
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\nThank you for using the College Predictor!")
