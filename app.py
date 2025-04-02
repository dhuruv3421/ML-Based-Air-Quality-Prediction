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
# Update the custom CSS section at the top of the code
st.markdown("""
<style>
    /* Base font size increase */
    html, body, .stApp, .stMarkdown, .stText, .stNumberInput, .stSelectbox, .stSlider {
        font-size: 18px !important;
    }
    
    /* Headers */
    h1 {
        font-size: 36px !important;
    }
    h2 {
        font-size: 30px !important;
    }
    h3 {
        font-size: 26px !important;
    }
    h4 {
        font-size: 22px !important;
    }
    
    /* Tables and charts */
    .stTable {
        font-size: 18px !important;
    }
    
    /* Prediction boxes */
    .prediction-box h2 {
        font-size: 32px !important;
    }
    .prediction-box h3 {
        font-size: 28px !important;
    }
    
    /* AQI Health Impact Chart specific styles */
    .aqi-impact-table {
        font-size: 20px !important;
    }
    .aqi-impact-table th {
        font-size: 22px !important;
        font-weight: bold !important;
    }
    .aqi-impact-table td {
        font-size: 20px !important;
    }
    
    /* Other existing styles remain the same */
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
    .suggestion-box {
        background-color: #f0f7fb;
        border-left: 5px solid #2196F3;
        padding: 15px;
        margin: 10px 0;
        border-radius: 3px;
        font-size: 18px !important;
    }
    .impact-factor {
        background-color: #fff3e0;
        border-left: 5px solid #FF9800;
        padding: 15px;
        margin: 10px 0;
        border-radius: 3px;
        font-size: 18px !important;
    }
</style>
""", unsafe_allow_html=True)

# Global error message placeholder
error_placeholder = st.empty()

def show_error(message):
    """Display error message in a consistent location"""
    error_placeholder.error(f" {str(message)}")

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

def get_pollutant_impact_info(feature_name):
    """Return information about a pollutant's impact and mitigation strategies"""
    pollutant_info = {
        'PM2.5': {
            'description': 'Fine particulate matter less than 2.5 micrometers in diameter that can penetrate deep into the lungs.',
            'sources': 'Vehicle exhaust, industrial emissions, indoor cooking, dust, and wildfires.',
            'health_effects': 'Can cause respiratory problems, decreased lung function, aggravated asthma, and heart disease.',
            'mitigation': 'Use air purifiers with HEPA filters, avoid outdoor activities during high pollution days, proper ventilation when cooking, and reduce use of wood-burning stoves.'
        },
        'PM10': {
            'description': 'Coarse particulate matter between 2.5 and 10 micrometers in diameter.',
            'sources': 'Road dust, construction sites, industrial processes, and agricultural operations.',
            'health_effects': 'Can cause irritation of the eyes, nose, and throat, as well as respiratory issues.',
            'mitigation': 'Reduce dust sources, use dust masks when in construction areas, and keep windows closed during high dust conditions.'
        },
        'NO': {
            'description': 'Nitric oxide, a reactive air pollutant.',
            'sources': 'Combustion processes, particularly in vehicles and power plants.',
            'health_effects': 'Contributes to respiratory problems and can convert to NO2 in the atmosphere.',
            'mitigation': 'Use of catalytic converters, improved combustion technology, and reducing vehicle emissions.'
        },
        'NO2': {
            'description': 'Nitrogen dioxide, a highly reactive gas with a pungent odor.',
            'sources': 'Vehicle exhaust, power plants, and industrial processes.',
            'health_effects': 'Can cause respiratory inflammation, reduced lung function, and increased sensitivity to respiratory infections.',
            'mitigation': 'Promote public transportation, use electric vehicles, and improve industrial emission controls.'
        },
        'NOx': {
            'description': 'A collective term for nitrogen oxides (NO and NO2).',
            'sources': 'Combustion processes in vehicles, power plants, and industrial operations.',
            'health_effects': 'Contributes to respiratory problems, formation of smog, and acid rain.',
            'mitigation': 'Use low-NOx burners, selective catalytic reduction systems, and promote cleaner transportation.'
        },
        'NH3': {
            'description': 'Ammonia, a colorless gas with a pungent odor.',
            'sources': 'Agricultural activities, livestock waste, and fertilizer application.',
            'health_effects': 'Can cause respiratory irritation and contribute to secondary particle formation.',
            'mitigation': 'Improve manure management, use covered manure storage, and implement precise fertilizer application.'
        },
        'CO': {
            'description': 'Carbon monoxide, a colorless, odorless toxic gas.',
            'sources': 'Incomplete combustion in vehicles, residential heating, and industrial processes.',
            'health_effects': 'Reduces oxygen delivery to body organs, causing headaches, dizziness, and at high concentrations, death.',
            'mitigation': 'Regular vehicle maintenance, proper ventilation of combustion sources, and use of carbon monoxide detectors.'
        },
        'SO2': {
            'description': 'Sulfur dioxide, a colorless gas with a strong odor.',
            'sources': 'Burning of fossil fuels containing sulfur, particularly in power plants and industrial processes.',
            'health_effects': 'Can cause respiratory problems, worsen asthma, and contribute to acid rain.',
            'mitigation': 'Use low-sulfur fuels, flue gas desulfurization systems, and promote renewable energy sources.'
        },
        'O3': {
            'description': 'Ground-level ozone, a major component of smog.',
            'sources': 'Formed by chemical reactions between NOx and VOCs in the presence of sunlight.',
            'health_effects': 'Can trigger asthma attacks, cause throat irritation, and reduce lung function.',
            'mitigation': 'Reduce VOC emissions, limit outdoor activities during high ozone days, and improve industrial emission controls.'
        }
    }
    
    return pollutant_info.get(feature_name, {
        'description': 'Information not available for this pollutant.',
        'sources': 'Various industrial and natural sources.',
        'health_effects': 'May cause respiratory and other health issues.',
        'mitigation': 'Follow local air quality guidelines and reduce exposure.'
    })

def get_aqi_category_info(aqi_value):
    """Return information and recommendations based on AQI category"""
    if aqi_value <= 50:
        return {
            'category': 'Good',
            'description': 'Air quality is considered satisfactory, and air pollution poses little or no risk.',
            'recommendations': 'It\'s a great day to be active outside. Enjoy outdoor activities and open windows for fresh air.',
            'health_implications': 'No health implications for the general population.'
        }
    elif aqi_value <= 100:
        return {
            'category': 'Satisfactory',
            'description': 'Air quality is acceptable; however, there may be a moderate health concern for a very small number of people.',
            'recommendations': 'Unusually sensitive people should consider reducing prolonged or heavy exertion.',
            'health_implications': 'May cause minor breathing discomfort to sensitive people.'
        }
    elif aqi_value <= 200:
        return {
            'category': 'Moderate',
            'description': 'Members of sensitive groups may experience health effects.',
            'recommendations': 'People with respiratory or heart disease, the elderly and children should limit prolonged exertion.',
            'health_implications': 'May cause breathing discomfort to people with lung disease, children, and older adults.'
        }
    elif aqi_value <= 300:
        return {
            'category': 'Poor',
            'description': 'Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.',
            'recommendations': 'Active children and adults, and people with respiratory disease should avoid prolonged outdoor exertion.',
            'health_implications': 'May cause respiratory illness on prolonged exposure. Heart disease patients, elderly, and children are at higher risk.'
        }
    elif aqi_value <= 400:
        return {
            'category': 'Very Poor',
            'description': 'Health warnings of emergency conditions. The entire population is more likely to be affected.',
            'recommendations': 'Everyone should avoid outdoor activities. Keep windows and doors closed.',
            'health_implications': 'May cause respiratory impact even on healthy people, and serious health impacts on people with lung/heart disease.'
        }
    else:
        return {
            'category': 'Severe',
            'description': 'Health alert: everyone may experience more serious health effects.',
            'recommendations': 'Everyone should avoid all outdoor exertion. Consider wearing N95 masks outdoors. Use air purifiers indoors.',
            'health_implications': 'May cause respiratory effects even on healthy people. Serious aggravation of heart/lung disease.'
        }

def display_prediction(prediction, r2_score, importance_df, input_values):
    """Display prediction with appropriate styling and improvement suggestions"""
    aqi_info = get_aqi_category_info(prediction)
    
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
    
    # Add AQI category information
    st.markdown(f"### Air Quality Assessment")
    st.markdown(f"**Description**: {aqi_info['description']}")
    st.markdown(f"**Health Implications**: {aqi_info['health_implications']}")
    
    # Add recommendations based on AQI category
    st.markdown(
        f"""
        <div class='suggestion-box'>
            <h4>Recommendations</h4>
            <p>{aqi_info['recommendations']}</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Show top factors affecting the prediction
    st.subheader("Key Factors Affecting Air Quality")
    
    # Get top 3 influential factors
    top_factors = importance_df.head(3)['Feature'].tolist()
    
    for factor in top_factors:
        factor_info = get_pollutant_impact_info(factor)
        factor_value = input_values[factor]
        
        st.markdown(
            f"""
            <div class='impact-factor'>
                <h4>{factor} ({factor_value:.2f})</h4>
                <p><strong>What is it?</strong> {factor_info['description']}</p>
                <p><strong>Main sources:</strong> {factor_info['sources']}</p>
                <p><strong>Health effects:</strong> {factor_info['health_effects']}</p>
                <p><strong>How to reduce:</strong> {factor_info['mitigation']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Community-level improvement suggestions
    st.subheader("Community-Level Improvement Strategies")
    
    if prediction > 100:  # Only show for moderate and worse AQI
        st.markdown("""
        <div class='suggestion-box'>
            <h4>Short-term Actions</h4>
            <ul>
                <li>Implement odd-even vehicle schemes during high pollution periods</li>
                <li>Temporarily restrict construction activities</li>
                <li>Increase public transport frequency</li>
                <li>Issue public health advisories</li>
                <li>Set up emergency air quality monitoring</li>
            </ul>
        </div>
        <div class='suggestion-box'>
            <h4>Long-term Solutions</h4>
            <ul>
                <li>Promote electric vehicles and establish charging infrastructure</li>
                <li>Expand green spaces and urban forests</li>
                <li>Invest in renewable energy sources</li>
                <li>Implement stricter industrial emission standards</li>
                <li>Develop better waste management systems to prevent open burning</li>
                <li>Improve public transportation networks</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

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
                # Create AQI buckets if they don't exist
                df['AQI_Bucket'] = pd.cut(
                    df['AQI'],
                    bins=[0, 50, 100, 200, 300, 400, float('inf')],
                    labels=['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
                )
                
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
                
                # Add pollutant information section
                st.markdown("### Pollutant Information")
                selected_pollutant = st.selectbox("Select a pollutant to learn more", selected_pollutants)
                
                pollutant_info = get_pollutant_impact_info(selected_pollutant)
                st.markdown(
                    f"""
                    <div class='impact-factor'>
                        <h4>{selected_pollutant}</h4>
                        <p><strong>What is it?</strong> {pollutant_info['description']}</p>
                        <p><strong>Main sources:</strong> {pollutant_info['sources']}</p>
                        <p><strong>Health effects:</strong> {pollutant_info['health_effects']}</p>
                        <p><strong>How to reduce:</strong> {pollutant_info['mitigation']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
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
                    display_prediction(prediction, r2, importance_df, inputs)
                    
                except Exception as e:
                    show_error(f"Prediction failed: {str(e)}")
                    st.error("Please check your input values and try again.")

            # Add a "What-if" analysis section
            st.subheader("What-if Analysis")
            st.write("Adjust the top pollutant levels to see how they affect the predicted AQI:")
            
            # Get top pollutants from feature importance
            top_pollutants = importance_df.head(3)['Feature'].tolist()
            
            col1, col2, col3 = st.columns(3)
            what_if_values = inputs.copy()
            
            # Create sliders for top pollutants
            with col1:
                if top_pollutants and len(top_pollutants) > 0:
                    what_if_values[top_pollutants[0]] = st.slider(
                        f"{top_pollutants[0]} Adjustment",
                        min_value=0.0,
                        max_value=float(df[top_pollutants[0]].max()) if top_pollutants[0] in df.columns else 200.0,
                        value=inputs[top_pollutants[0]]
                    )
            
            with col2:
                if top_pollutants and len(top_pollutants) > 1:
                    what_if_values[top_pollutants[1]] = st.slider(
                        f"{top_pollutants[1]} Adjustment",
                        min_value=0.0,
                        max_value=float(df[top_pollutants[1]].max()) if top_pollutants[1] in df.columns else 200.0,
                        value=inputs[top_pollutants[1]]
                    )
            
            with col3:
                if top_pollutants and len(top_pollutants) > 2:
                    what_if_values[top_pollutants[2]] = st.slider(
                        f"{top_pollutants[2]} Adjustment",
                        min_value=0.0,
                        max_value=float(df[top_pollutants[2]].max()) if top_pollutants[2] in df.columns else 200.0,
                        value=inputs[top_pollutants[2]]
                    )
            
            if st.button("Run What-if Analysis"):
                try:
                    # Create a range of values for sensitivity analysis
                    ranges = {}
                    predictions = []
                    
                    # Create sensitivity data for the most important feature
                    if top_pollutants:
                        main_feature = top_pollutants[0]
                        base_value = what_if_values[main_feature]
                        min_val = max(0, base_value * 0.5)
                        max_val = base_value * 1.5
                        ranges[main_feature] = np.linspace(min_val, max_val, 20)
                        
                        for val in ranges[main_feature]:
                            temp_values = what_if_values.copy()
                            temp_values[main_feature] = val
                            input_data = pd.DataFrame([temp_values], columns=existing_features)
                            predictions.append({
                                main_feature: val,
                                'AQI': model.predict(input_data)[0]
                            })
                        
                        # Create dataframe for plotting
                        sensitivity_df = pd.DataFrame(predictions)
                        
                        # Plot sensitivity analysis
                        fig = px.line(
                            sensitivity_df,
                            x=main_feature,
                            y='AQI',
                            title=f"Sensitivity Analysis: Impact of {main_feature} on AQI",
                            markers=True
                        )
                        
                        # Add colored regions for AQI categories
                        fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0)
                        fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, line_width=0)
                        fig.add_hrect(y0=100, y1=200, fillcolor="orange", opacity=0.1, line_width=0)
                        fig.add_hrect(y0=200, y1=300, fillcolor="red", opacity=0.1, line_width=0)
                        fig.add_hrect(y0=300, y1=400, fillcolor="purple", opacity=0.1, line_width=0)
                        fig.add_hrect(y0=400, y1=500, fillcolor="maroon", opacity=0.1, line_width=0)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display what-if prediction
                    input_data = pd.DataFrame([what_if_values], columns=existing_features)
                    what_if_prediction = model.predict(input_data)[0]
                    
                    st.subheader("What-if Prediction Result")
                    display_prediction(what_if_prediction, r2, importance_df, what_if_values)
                    
                    # Add comparison with original prediction
                    original_input_data = pd.DataFrame([inputs], columns=existing_features)
                    original_prediction = model.predict(original_input_data)[0]
                    
                    change = what_if_prediction - original_prediction
                    change_percent = (change / original_prediction) * 100 if original_prediction > 0 else 0
                    
                    st.markdown(f"""
                    <div style="background-color:#f0f7fb; padding:15px; border-radius:5px; margin:10px 0;">
                        <h4>Comparison with Original Values</h4>
                        <p>Original AQI Prediction: <b>{original_prediction:.2f}</b></p>
                        <p>New AQI Prediction: <b>{what_if_prediction:.2f}</b></p>
                        <p>Change: <b>{change:.2f}</b> ({change_percent:.1f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    show_error(f"What-if analysis failed: {str(e)}")
                    st.error("Please check your input values and try again.")

        except Exception as e:
            show_error(f"Error in predictive modeling section: {str(e)}")
            st.error(traceback.format_exc())
        
        # Historical data analysis
        st.header("Historical Trends & Seasonal Analysis")
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                trend_city = st.selectbox("Select City for Trends", df['City'].unique())
            
            with col2:
                trend_year = st.selectbox("Select Year", sorted(df['Year'].unique(), reverse=True))
            
            city_year_df = df[(df['City'] == trend_city) & (df['Year'] == trend_year)]
            
            if not city_year_df.empty:
                # Monthly trends for the selected year and city
                monthly_avg = city_year_df.groupby('Month')['AQI'].mean().reset_index()
                monthly_avg['Month_Name'] = monthly_avg['Month'].apply(lambda x: datetime(2000, x, 1).strftime('%b'))
                
                fig = px.line(
                    monthly_avg, 
                    x='Month_Name', 
                    y='AQI',
                    markers=True,
                    title=f"Monthly AQI Trends for {trend_city} in {trend_year}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Seasonal patterns
                def get_season(month):
                    if month in [12, 1, 2]:
                        return 'Winter'
                    elif month in [3, 4, 5]:
                        return 'Spring'
                    elif month in [6, 7, 8]:
                        return 'Summer'
                    else:
                        return 'Fall'
                
                city_year_df['Season'] = city_year_df['Month'].apply(get_season)
                seasonal_avg = city_year_df.groupby('Season')['AQI'].mean().reset_index()
                
                # Order seasons properly
                season_order = ['Winter', 'Spring', 'Summer', 'Fall']
                seasonal_avg['Season'] = pd.Categorical(seasonal_avg['Season'], categories=season_order, ordered=True)
                seasonal_avg = seasonal_avg.sort_values('Season')
                
                fig = px.bar(
                    seasonal_avg,
                    x='Season',
                    y='AQI',
                    color='Season',
                    title=f"Seasonal AQI Patterns for {trend_city} in {trend_year}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Pollutant composition by season
                if existing_pollutants:
                    seasonal_pollutants = city_year_df.groupby('Season')[existing_pollutants].mean().reset_index()
                    seasonal_pollutants['Season'] = pd.Categorical(
                        seasonal_pollutants['Season'], 
                        categories=season_order, 
                        ordered=True
                    )
                    seasonal_pollutants = seasonal_pollutants.sort_values('Season')
                    
                    # Normalize data for better visualization
                    for col in existing_pollutants:
                        max_val = seasonal_pollutants[col].max()
                        if max_val > 0:
                            seasonal_pollutants[f'{col}_norm'] = seasonal_pollutants[col] / max_val
                    
                    norm_cols = [f'{col}_norm' for col in existing_pollutants]
                    
                    # Create radar chart
                    fig = go.Figure()
                    
                    for i, season in enumerate(season_order):
                        if season in seasonal_pollutants['Season'].values:
                            season_data = seasonal_pollutants[seasonal_pollutants['Season'] == season]
                            fig.add_trace(go.Scatterpolar(
                                r=season_data[norm_cols].values.flatten().tolist(),
                                theta=existing_pollutants,
                                fill='toself',
                                name=season
                            ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        title=f"Seasonal Pollutant Composition for {trend_city} in {trend_year}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data available for {trend_city} in {trend_year}")
        
        except Exception as e:
            show_error(f"Error in historical trend analysis: {str(e)}")
        
        # Air quality health impact section
        # Update the AQI Health Impact Chart section
        st.header("Health Impact Assessment", divider='rainbow')
        st.write("Understand the health impacts of different pollutants and air quality levels.")


        # Custom CSS for the table
        st.markdown("""
        <style>
            .aqi-table {
                width: 100%;
                border-collapse: collapse;
                margin: 1em 0;
                font-size: 18px;
            }
            .aqi-table th {
                background-color: #2c3e50;
                color: white;
                padding: 12px;
                text-align: left;
                font-size: 20px;
            }
            .aqi-table td {
                padding: 12px;
                border-bottom: 1px solid #ddd;
            }
            .aqi-table tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            .good-row { background-color: #55A84F !important; color: white; }
            .satisfactory-row { background-color: #A3C853 !important; color: white; }
            .moderate-row { background-color: #FFF833 !important; color: black; }
            .poor-row { background-color: #F29C33 !important; color: white; }
            .very-poor-row { background-color: #E93F33 !important; color: white; }
            .severe-row { background-color: #AF2D24 !important; color: white; }
            .quick-ref {
                display: flex;
                justify-content: space-between;
                margin-top: 20px;
                flex-wrap: wrap;
                gap: 10px;
            }
            .quick-ref-item {
                padding: 10px 15px;
                border-radius: 5px;
                font-weight: bold;
                text-align: center;
                flex-grow: 1;
                min-width: 120px;
            }
        </style>
        """, unsafe_allow_html=True)

        # AQI Health Impact Table with Colors
        st.markdown("""
        ### AQI Health Impact Reference Guide

        <table class="aqi-table">
        <thead>
            <tr>
            <th>AQI Category</th>
            <th>Health Impact</th>
            </tr>
        </thead>
        <tbody>
            <tr class="good-row">
            <td><strong>Good (0-50)</strong></td>
            <td>Minimal impact. Air quality is satisfactory and poses little or no risk.</td>
            </tr>
            <tr class="satisfactory-row">
            <td><strong>Satisfactory (51-100)</strong></td>
            <td>Minor breathing discomfort to sensitive people. Acceptable air quality.</td>
            </tr>
            <tr class="moderate-row">
            <td><strong>Moderate (101-200)</strong></td>
            <td>Breathing discomfort to people with lung disease, children and older adults.</td>
            </tr>
            <tr class="poor-row">
            <td><strong>Poor (201-300)</strong></td>
            <td>Breathing discomfort to everyone. People with respiratory issues may experience more serious health effects.</td>
            </tr>
            <tr class="very-poor-row">
            <td><strong>Very Poor (301-400)</strong></td>
            <td>Health warnings. Everyone may experience more serious health effects.</td>
            </tr>
            <tr class="severe-row">
            <td><strong>Severe (401+)</strong></td>
            <td>Health alert. The risk of health effects is increased for everyone.</td>
            </tr>
        </tbody>
        </table>

        <div class="quick-ref">
        <div class="quick-ref-item good-row">Good (0-50)</div>
        <div class="quick-ref-item satisfactory-row">Satisfactory (51-100)</div>
        <div class="quick-ref-item moderate-row">Moderate (101-200)</div>
        <div class="quick-ref-item poor-row">Poor (201-300)</div>
        <div class="quick-ref-item very-poor-row">Very Poor (301-400)</div>
        <div class="quick-ref-item severe-row">Severe (401+)</div>
        </div>
        """, unsafe_allow_html=True)

        
        # Vulnerable populations section
        st.subheader("Vulnerable Populations")
        st.markdown("""
        <div class='suggestion-box'>
            <h4>Groups at Higher Risk</h4>
            <ul>
                <li><strong>Children and infants:</strong> Developing lungs, higher breathing rates, more outdoor time</li>
                <li><strong>Older adults:</strong> May have undiagnosed heart or lung conditions</li>
                <li><strong>People with respiratory conditions:</strong> Asthma, COPD, lung disease</li>
                <li><strong>People with cardiovascular disease:</strong> Heart disease, high blood pressure</li>
                <li><strong>Pregnant women:</strong> Exposure can affect fetal development</li>
                <li><strong>Outdoor workers:</strong> Extended exposure during work hours</li>
                <li><strong>Athletes and active individuals:</strong> Increased breathing rates during exercise</li>
            </ul>
        </div>
        <div class='suggestion-box'>    
            <h4>Special Precautions</h4>
            <ul>
                <li>Check local air quality index before outdoor activities</li>
                <li>Limit outdoor exertion during high pollution periods</li>
                <li>Consider using air purifiers with HEPA filters</li>
                <li>Keep medications readily available (for those with asthma or respiratory conditions)</li>
                <li>Stay hydrated to help the body process and remove inhaled pollutants</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Footer with credits
        st.markdown("---")
        st.markdown("© 2025 Air Quality Analysis System | Developed with Streamlit")
        
    except Exception as e:
        show_error(f"Application Error: {str(e)}")
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()