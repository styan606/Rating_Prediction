import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load your dataset
df = pd.read_excel(r"C:\Users\Sendsteps\Desktop\SMOTENC.xlsx", engine='openpyxl')

# Define feature groups
linear = ["imgRelevance"]
z_score = ["avgWidth", "avgHeight", "avgShapeCoordinates", "shapesPerSlide",
           "promptTokens", "responseTokens", "numSlides",  "randomControl", "backgroundOpacity"]
one_hot = ["presentationStyle", "toneOfVoice", "length"]

# Define ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("z_score", StandardScaler(), z_score),  # Apply z-score scaling
        ("linear", MinMaxScaler(), linear),     # Apply MinMax scaling
        ("one_hot", OneHotEncoder(sparse_output=False), one_hot)  # Apply OneHotEncoder
    ],
    remainder="passthrough"  # Keep any columns not explicitly transformed
)

# Fit and transform the data
transformed_data = preprocessor.fit_transform(df)

# Extract one-hot encoded column names
one_hot_features = preprocessor.named_transformers_["one_hot"].get_feature_names_out(one_hot)

# Combine all column names
all_columns = z_score + linear + list(one_hot_features) + [
    col for col in df.columns if col not in z_score + linear + one_hot
]

# Create a new DataFrame with the transformed data
transformed_df = pd.DataFrame(transformed_data, columns=all_columns)

# Save the processed DataFrame if needed
transformed_df.to_excel(r"C:\Users\Sendsteps\Desktop\afterSCALING.xlsx", index=False, engine='openpyxl')



print(transformed_df.head())
