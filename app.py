import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import traceback

# Set page config
st.set_page_config(layout="wide", page_title="Air Quality Analysis System")

# Custom CSS for better styling
st.markdown("""
<style>
    .prediction-box {
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
    }
    .good { background-color: #55A84F; color: white; }
    .satisfactory { background-color: #A3C853; color: white; }
    .moderate { background-color: #FFF833; color: black; }
    .poor { background-color: #F29C33; color: white; }
    .very-poor { background-color: #E93F33; color: white; }
    .severe { background-color: #AF2D24; color: white; }
    .plot-container {
        border: 1px solid #e1e4e8;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Global error message placeholder
error_placeholder = st.empty()

def show_error(message):
    """Display error message in a consistent location"""
    error_placeholder.error(f"⚠️ {str(message)}")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r'C:\Users\Thaku\OneDrive\Desktop\6th_Sem\AIML_Lab\Project\city_day.csv')
        
        # Validate required columns
        required_columns = ['Date', 'City', 'AQI']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Convert and validate dates
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['Date_Display'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # Check for AQI values
        if df['AQI'].isna().all():
            raise ValueError("No valid AQI values found in dataset")
        df = df.dropna(subset=['AQI'])

        # Handle pollutants
        pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 
                         'Benzene', 'Toluene', 'Xylene']
        existing_pollutants = [col for col in pollutant_cols if col in df.columns]
        df = df.dropna(subset=existing_pollutants, how='all')

        # Fill missing values
        for col in existing_pollutants:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())

        return df
    
    except FileNotFoundError:
        raise FileNotFoundError("Data file not found. Please check the file path.")
    except pd.errors.EmptyDataError:
        raise ValueError("The data file is empty or corrupted.")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def display_prediction(prediction, r2_score):
    """Display prediction with appropriate styling"""
    if prediction <= 50:
        category = "Good"
        css_class = "good"
    elif prediction <= 100:
        category = "Satisfactory"
        css_class = "satisfactory"
    elif prediction <= 200:
        category = "Moderate"
        css_class = "moderate"
    elif prediction <= 300:
        category = "Poor"
        css_class = "poor"
    elif prediction <= 400:
        category = "Very Poor"
        css_class = "very-poor"
    else:
        category = "Severe"
        css_class = "severe"
    
    st.markdown(
        f"""
        <div class='prediction-box {css_class}'>
            <h2>Predicted AQI: {prediction:.2f}</h2>
            <h3>{category}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Show confidence indicator
    st.write(f"Model Confidence (R² score): {r2_score:.1%}")
    st.progress(float(r2_score))

def main():
    try:
        st.title("Air Quality Index Predictive Analysis System")
        
        # Load data with error handling
        try:
            df = load_data()
        except Exception as e:
            show_error(f"Data loading failed: {str(e)}")
            st.stop()
        
        # Overview section
        st.header("Dataset Overview")
        st.write(f"Dataset contains {len(df)} records from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")

        # Show raw data
        if st.checkbox("Show Raw Data"):
            try:
                display_df = df.copy()
                display_df['Date'] = display_df['Date_Display']
                st.dataframe(display_df.drop(columns=['Date_Display']))
            except Exception as e:
                show_error(f"Error displaying data: {str(e)}")

        # Basic statistics
        if st.checkbox("Show Basic Statistics"):
            try:
                st.write(df.describe())
            except Exception as e:
                show_error(f"Error generating statistics: {str(e)}")

        # Visualization section
        st.header("Data Visualization")
        
        # Time series analysis
        st.subheader("Time Series Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            time_resolution = st.selectbox("Time Resolution", ["Daily", "Monthly", "Yearly"])
        
        with col2:
            city_filter = st.multiselect("Select Cities", df['City'].unique(), default=["Ahmedabad"])

        try:
            filtered_df = df[df['City'].isin(city_filter)]
            
            if time_resolution == "Monthly":
                time_df = filtered_df.groupby(['Year', 'Month', 'City'])['AQI'].mean().reset_index()
                time_df['Date'] = pd.to_datetime(time_df[['Year', 'Month']].assign(DAY=1))
                fig = px.line(time_df, x='Date', y='AQI', color='City', 
                             title="Monthly Average AQI Trend")
            elif time_resolution == "Yearly":
                time_df = filtered_df.groupby(['Year', 'City'])['AQI'].mean().reset_index()
                fig = px.line(time_df, x='Year', y='AQI', color='City', 
                             title="Yearly Average AQI Trend")
            else:
                fig = px.line(filtered_df, x='Date', y='AQI', color='City', 
                             title="Daily AQI Trend")
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            show_error(f"Error generating time series plot: {str(e)}")

        # AQI Bucket distribution
        st.subheader("AQI Category Distribution")
        try:
            if 'AQI_Bucket' in df.columns:
                aqi_color_map = {
                    'Good': '#55A84F', 'Satisfactory': '#A3C853', 'Moderate': '#FFF833',
                    'Poor': '#F29C33', 'Very Poor': '#E93F33', 'Severe': '#AF2D24'
                }
                
                aqi_bucket_counts = df['AQI_Bucket'].value_counts().reset_index()
                aqi_bucket_counts.columns = ['AQI Category', 'Count']
                
                category_order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
                aqi_bucket_counts['AQI Category'] = pd.Categorical(
                    aqi_bucket_counts['AQI Category'],
                    categories=category_order,
                    ordered=True
                )
                
                fig = px.bar(
                    aqi_bucket_counts.sort_values('AQI Category'), 
                    x='AQI Category', 
                    y='Count', 
                    color='AQI Category',
                    color_discrete_map=aqi_color_map,
                    title="Distribution of AQI Categories"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("AQI_Bucket column not found in dataset")
        except Exception as e:
            show_error(f"Error generating AQI category distribution: {str(e)}")

        # Pollutant correlation
        st.subheader("Pollutant Correlation with AQI")
        pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
        existing_pollutants = [col for col in pollutant_cols if col in df.columns]
        
        try:
            selected_pollutants = st.multiselect("Select Pollutants", existing_pollutants, 
                                               default=['PM2.5', 'PM10', 'NO2'] if 'PM2.5' in existing_pollutants else existing_pollutants[:3])
            
            if selected_pollutants:
                corr_df = df[selected_pollutants + ['AQI']].corr()
                fig = px.imshow(corr_df, text_auto=True, aspect="auto", 
                               title="Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            show_error(f"Error generating correlation matrix: {str(e)}")

        # Custom 2D/3D visualization
        st.subheader("Custom Visualization")
        try:
            viz_type = st.radio("Visualization Type", ["2D Scatter", "3D Scatter", "Heatmap"])
            
            cols = df.select_dtypes(include=np.number).columns.tolist()
            
            if viz_type == "2D Scatter":
                col1, col2 = st.columns(2)
                with col1:
                    x_axis = st.selectbox("X-axis", cols, index=cols.index('PM2.5') if 'PM2.5' in cols else 0)
                with col2:
                    y_axis = st.selectbox("Y-axis", cols, index=cols.index('AQI') if 'AQI' in cols else 1)
                
                fig = px.scatter(df, x=x_axis, y=y_axis, color='AQI_Bucket' if 'AQI_Bucket' in df.columns else None,
                                 hover_data=['City', 'Date_Display'],
                                 title=f"{x_axis} vs {y_axis}")
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "3D Scatter":
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_axis = st.selectbox("X-axis", cols, index=cols.index('PM2.5') if 'PM2.5' in cols else 0)
                with col2:
                    y_axis = st.selectbox("Y-axis", cols, index=cols.index('PM10') if 'PM10' in cols else 1)
                with col3:
                    z_axis = st.selectbox("Z-axis", cols, index=cols.index('AQI') if 'AQI' in cols else 2)
                
                fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, 
                                    color='AQI_Bucket' if 'AQI_Bucket' in df.columns else None, 
                                    hover_data=['City', 'Date_Display'],
                                    title=f"3D View: {x_axis}, {y_axis}, {z_axis}")
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Heatmap":
                selected_cols = st.multiselect("Select columns for heatmap", cols, 
                                             default=['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI'] if all(x in cols for x in ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI']) else cols[:5])
                if selected_cols:
                    fig = px.imshow(df[selected_cols].corr(), text_auto=True, aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            show_error(f"Error in custom visualization: {str(e)}")

        # Predictive modeling section
        st.header("AQI Prediction")
        
        try:
            # Model training
            st.subheader("Train Prediction Model")
            
            feature_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3']
            existing_features = [col for col in feature_cols if col in df.columns]
            target_col = 'AQI'
            
            if not existing_features:
                raise ValueError("No valid features available for modeling")
            
            X = df[existing_features]
            y = df[target_col]
            
            if 'City' in existing_features:
                le = LabelEncoder()
                X['City'] = le.fit_transform(X['City'])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = DecisionTreeRegressor(max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            
            if hasattr(model, 'feature_names_in_'):
                model.feature_names_in_ = np.array(X_train.columns.tolist())
            
            # Evaluation metrics
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R-squared Score", f"{r2:.3f}")
                st.metric("Mean Absolute Error", f"{mae:.2f}")
            with col2:
                st.metric("Root Mean Squared Error", f"{rmse:.2f}")
                st.metric("Mean Absolute % Error", f"{mape:.2%}")
            with col3:
                st.metric("Mean Squared Error", f"{mse:.2f}")

            # Feature importance
            st.subheader("Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': existing_features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                         title="Feature Importance in AQI Prediction")
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction interface
            st.subheader("Make a Prediction")
            st.write("Enter pollutant levels to predict AQI:")
            
            col1, col2, col3, col4 = st.columns(4)
            inputs = {}
            
            with col1:
                inputs['PM2.5'] = st.number_input("PM2.5", min_value=0.0, value=50.0)
                inputs['PM10'] = st.number_input("PM10", min_value=0.0, value=60.0)
            
            with col2:
                inputs['NO'] = st.number_input("NO", min_value=0.0, value=10.0)
                inputs['NO2'] = st.number_input("NO2", min_value=0.0, value=20.0)
            
            with col3:
                inputs['NOx'] = st.number_input("NOx", min_value=0.0, value=25.0)
                inputs['CO'] = st.number_input("CO", min_value=0.0, value=1.0)
            
            with col4:
                inputs['SO2'] = st.number_input("SO2", min_value=0.0, value=15.0)
                inputs['O3'] = st.number_input("O3", min_value=0.0, value=40.0)
            
            if st.button("Predict AQI"):
                try:
                    input_data = pd.DataFrame([inputs], columns=existing_features)
                    prediction = model.predict(input_data)[0]
                    display_prediction(prediction, r2)
                    
                except Exception as e:
                    show_error(f"Prediction failed: {str(e)}")
                    st.error("Please check your input values and try again.")
        
        except Exception as e:
            show_error(f"Model training error: {str(e)}")

    except Exception as e:
        show_error(f"Application error: {str(e)}")
        st.text(traceback.format_exc())
        st.error("The application has encountered a critical error and cannot continue.")

if __name__ == "__main__":
    main()