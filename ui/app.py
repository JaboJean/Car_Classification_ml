import streamlit as st
import requests
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Car Classification Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# FastAPI backend URL
API_URL = "http://localhost:8000"

# Title with emoji
st.markdown('<p class="main-header">🚗 Car Classification Dashboard</p>', unsafe_allow_html=True)

# -----------------------------
# SIDEBAR: Model controls & Info
# -----------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/car.png", width=100)
st.sidebar.title("⚙️ Model Controls")

# Model Uptime Section
st.sidebar.subheader("📊 System Status")
if st.sidebar.button("🔄 Refresh Status", use_container_width=True):
    try:
        r = requests.get(f"{API_URL}/uptime", timeout=5)
        if r.status_code == 200:
            data = r.json()
            uptime_seconds = data.get("uptime_seconds", 0)
            hours = int(uptime_seconds // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            seconds = int(uptime_seconds % 60)
            
            st.sidebar.success("✅ API is Online")
            st.sidebar.metric("Uptime", f"{hours}h {minutes}m {seconds}s")
            st.sidebar.info(f"Started: {data.get('start_time', 'N/A')}")
        else:
            st.sidebar.error("❌ API Error")
    except Exception as e:
        st.sidebar.error(f"❌ API Offline: {str(e)[:50]}")

st.sidebar.divider()

# Model Information
st.sidebar.subheader("ℹ️ Model Information")
try:
    r = requests.get(f"{API_URL}/", timeout=5)
    if r.status_code == 200:
        st.sidebar.info("🤖 Model: ResNet50 (Transfer Learning)")
        st.sidebar.info("📊 Classes: 50 Car Brands")
        st.sidebar.info("🖼️ Input Size: 224x224")
except:
    st.sidebar.warning("Unable to fetch model info")

st.sidebar.divider()

# Quick Navigation
st.sidebar.subheader("🧭 Quick Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🔍 Prediction", "🎯 Retraining", "📊 Visualizations"],
    label_visibility="collapsed"
)

# -----------------------------
# PAGE 1: IMAGE PREDICTION
# -----------------------------
if page == "🔍 Prediction":
    st.header("📸 Predict Car Brand from Image")
    
    # Instructions
    with st.expander("📖 How to use", expanded=False):
        st.markdown("""
        1. **Upload an image** of a car (JPG, JPEG, or PNG format)
        2. **Preview** the uploaded image
        3. **Click 'Predict'** to get the car brand prediction
        4. **View results** including:
           - Predicted car brand
           - Confidence percentage
           - Probability score
           - Precision metric
           - Top 5 predictions
           - All class probabilities
        """)
    
    st.divider()
    
    # Two columns for layout
    col_upload, col_preview = st.columns([1, 1])
    
    with col_upload:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a car image",
            type=["png", "jpg", "jpeg"],
            help="Upload an image of a car to classify its brand"
        )
        
        if uploaded_file:
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "File type": uploaded_file.type
            }
            st.json(file_details)
    
    with col_preview:
        if uploaded_file:
            st.subheader("🖼️ Image Preview")
            try:
                img = Image.open(uploaded_file).convert("RGB")
                # FIXED: Changed use_column_width to use_container_width
                st.image(img, caption="Uploaded Image", use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error reading image: {e}")
    
    # Prediction Section
    if uploaded_file:
        st.divider()
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_button = st.button(
                "🔍 Predict Car Brand",
                type="primary",
                use_container_width=True
            )
        
        if predict_button:
            with st.spinner("🤖 Analyzing image... Please wait..."):
                try:
                    # Make prediction request
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    start_time = time.time()
                    r = requests.post(f"{API_URL}/predict-file", files=files, timeout=30)
                    response_time = time.time() - start_time

                    if r.status_code == 200:
                        res = r.json()
                        
                        # Success banner
                        st.balloons()
                        st.success(f"### ✅ Prediction Complete! (Response time: {response_time:.2f}s)")
                        
                        st.divider()
                        
                        # Main prediction result - Large banner
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 2rem; border-radius: 1rem; text-align: center; margin: 1rem 0;'>
                            <h1 style='color: white; margin: 0; font-size: 3rem;'>🎯 {res.get('pred_class')}</h1>
                            <p style='color: white; margin-top: 0.5rem; font-size: 1.2rem;'>Predicted Car Brand</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.divider()
                        
                        # Metrics in columns
                        st.subheader("📊 Prediction Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                label="🎯 Confidence",
                                value=res.get('confidence', 'N/A'),
                                help="Confidence level of the prediction"
                            )
                        with col2:
                            st.metric(
                                label="📈 Probability",
                                value=f"{res.get('probability', 0):.4f}",
                                help="Raw probability score (0-1)"
                            )
                        with col3:
                            st.metric(
                                label="🎲 Precision",
                                value=f"{res.get('precision', 0):.4f}",
                                help="Model precision for this prediction"
                            )
                        with col4:
                            st.metric(
                                label="🏷️ Total Classes",
                                value=res.get('total_classes', 'N/A'),
                                help="Number of car brands in the model"
                            )
                        
                        st.divider()
                        
                        # Top 5 Predictions
                        st.subheader("🏆 Top 5 Predictions")
                        top5 = res.get("top_5_predictions", [])
                        
                        if top5:
                            # Create DataFrame
                            df_top5 = pd.DataFrame(top5)
                            df_top5 = df_top5[['rank', 'class', 'confidence', 'probability']]
                            df_top5.columns = ['Rank', 'Car Brand', 'Confidence', 'Probability']
                            
                            # Display styled table
                            st.dataframe(
                                df_top5,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Rank": st.column_config.NumberColumn(
                                        "Rank",
                                        help="Prediction rank (1-5)",
                                        width="small"
                                    ),
                                    "Car Brand": st.column_config.TextColumn(
                                        "Car Brand",
                                        help="Predicted car brand name",
                                        width="medium"
                                    ),
                                    "Confidence": st.column_config.TextColumn(
                                        "Confidence",
                                        help="Confidence percentage",
                                        width="small"
                                    ),
                                    "Probability": st.column_config.NumberColumn(
                                        "Probability",
                                        help="Raw probability value",
                                        width="small",
                                        format="%.4f"
                                    )
                                }
                            )
                            
                            # Visualization of top 5 with Plotly
                            st.write("#### 📊 Top 5 Confidence Visualization")
                            
                            chart_data = pd.DataFrame({
                                'Car Brand': [p['class'] for p in top5],
                                'Confidence (%)': [float(p['confidence'].rstrip('%')) for p in top5]
                            })
                            
                            fig = px.bar(
                                chart_data,
                                x='Confidence (%)',
                                y='Car Brand',
                                orientation='h',
                                color='Confidence (%)',
                                color_continuous_scale='Blues',
                                text='Confidence (%)'
                            )
                            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                            fig.update_layout(
                                showlegend=False,
                                height=300,
                                yaxis={'categoryorder': 'total ascending'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        else:
                            st.warning("⚠️ No top 5 predictions available.")
                        
                        st.divider()
                        
                        # All Class Probabilities (Expandable)
                        with st.expander("📊 View All Class Probabilities (50 Brands)", expanded=False):
                            all_probs = res.get("all_probabilities", {})
                            
                            if all_probs:
                                # Create DataFrame
                                prob_df = pd.DataFrame(
                                    list(all_probs.items()),
                                    columns=["Car Brand", "Probability"]
                                ).sort_values("Probability", ascending=False)
                                
                                prob_df['Confidence'] = prob_df['Probability'].apply(lambda x: f"{x*100:.2f}%")
                                prob_df['Rank'] = range(1, len(prob_df) + 1)
                                
                                # Reorder columns
                                prob_df = prob_df[['Rank', 'Car Brand', 'Confidence', 'Probability']]
                                
                                # Display table
                                st.dataframe(
                                    prob_df,
                                    use_container_width=True,
                                    hide_index=True,
                                    height=400
                                )
                                
                                # Download button
                                csv = prob_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Full Probability Report (CSV)",
                                    data=csv,
                                    file_name=f"prediction_report_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                                
                                # Visualization of all probabilities
                                st.write("#### 📈 All Class Probabilities Distribution")
                                fig_all = go.Figure(data=[
                                    go.Bar(
                                        x=prob_df['Car Brand'],
                                        y=prob_df['Probability'],
                                        marker_color='lightblue'
                                    )
                                ])
                                fig_all.update_layout(
                                    height=400,
                                    xaxis_tickangle=-45,
                                    showlegend=False
                                )
                                st.plotly_chart(fig_all, use_container_width=True)
                            else:
                                st.warning("⚠️ No probability data available.")
                        
                        # Model Information (Expandable)
                        with st.expander("ℹ️ Prediction Details", expanded=False):
                            st.write("##### Model Information")
                            model_info = res.get("model_info", {})
                            info_col1, info_col2 = st.columns(2)
                            
                            with info_col1:
                                st.write(f"- **Total Classes**: {model_info.get('num_classes', 'N/A')}")
                                st.write(f"- **Device**: {model_info.get('device', 'N/A')}")
                                st.write(f"- **Predicted Index**: {res.get('pred_idx', 'N/A')}")
                            
                            with info_col2:
                                st.write(f"- **Filename**: {res.get('filename', 'N/A')}")
                                st.write(f"- **File Size**: {res.get('file_size', 0) / 1024:.2f} KB")
                                st.write(f"- **Response Time**: {response_time:.3f}s")
                        
                    else:
                        st.error(f"❌ Prediction failed with status code {r.status_code}")
                        st.error(f"Error message: {r.text}")
                        
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Please try again.")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure the FastAPI server is running on http://localhost:8000")
                    st.info("💡 Start the API with: `uvicorn src.api:app --reload --port 8000`")
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

# -----------------------------
# PAGE 2: MODEL RETRAINING
# -----------------------------
elif page == "🎯 Retraining":
    st.header("🔄 Retrain Model With New Data")
    
    # Instructions
    with st.expander("📖 Retraining Instructions", expanded=True):
        st.markdown("""
        ### How to Retrain the Model:
        
        1. **Prepare Your Dataset**:
           - Create a ZIP file containing `train/` and `test/` folders
           - Each folder should have subfolders for each car brand
           - Structure: `dataset.zip/train/BMW/*.jpg`, `dataset.zip/train/Toyota/*.jpg`, etc.
        
        2. **Upload Dataset**:
           - Click "Browse files" and select your ZIP file
           - Wait for the upload to complete
        
        3. **Upload to Server**:
           - Click "Upload Dataset to Server" button
           - Wait for confirmation
        
        4. **Configure Training**:
           - Set number of epochs (recommended: 10-20)
           - Set batch size (recommended: 16-32)
        
        5. **Trigger Retraining**:
           - Click "Start Retraining Now" button
           - Monitor progress (this may take several minutes to hours)
        
        ⚠️ **Warning**: Retraining will replace the current model!
        """)
    
    st.divider()
    
    # Upload Section
    st.subheader("📤 Step 1: Upload Dataset")
    uploaded_zip = st.file_uploader(
        "Upload a ZIP file containing train/ and test/ folders",
        type=["zip"],
        help="The ZIP should contain training and testing data organized by car brand",
        key="zip_uploader"
    )
    
    if uploaded_zip:
        st.success(f"✅ File selected: {uploaded_zip.name} ({uploaded_zip.size / 1024 / 1024:.2f} MB)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Upload Dataset to Server", type="primary", use_container_width=True):
                with st.spinner("⏳ Uploading dataset... This may take a while..."):
                    try:
                        files = {"file": (uploaded_zip.name, uploaded_zip.getvalue())}
                        r = requests.post(f"{API_URL}/upload-retrain", files=files, timeout=300)
                        
                        if r.status_code == 200:
                            result = r.json()
                            st.success("✅ Dataset uploaded successfully!")
                            
                            # Display upload details
                            st.write("**Upload Details:**")
                            st.json(result)
                            
                            # Store in session state
                            st.session_state['dataset_uploaded'] = True
                            st.session_state['upload_result'] = result
                        else:
                            st.error(f"❌ Upload failed with status {r.status_code}")
                            st.error(f"Error: {r.text}")
                    except requests.exceptions.Timeout:
                        st.error("⏱️ Upload timed out. Please try again with a smaller file.")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to API. Make sure the FastAPI server is running.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if 'dataset_uploaded' in st.session_state and st.session_state['dataset_uploaded']:
                st.success("✅ Dataset ready for training")
                if 'upload_result' in st.session_state:
                    result = st.session_state['upload_result']
                    st.metric("Classes Found", result.get('train_classes', 'N/A'))
        
        st.divider()
        
        # Training Configuration
        st.subheader("⚙️ Step 2: Configure Training Parameters")
        
        col_epoch, col_batch = st.columns(2)
        
        with col_epoch:
            epochs = st.number_input(
                "Number of Epochs",
                min_value=1,
                max_value=100,
                value=10,
                help="Number of complete passes through the training dataset"
            )
        
        with col_batch:
            batch_size = st.number_input(
                "Batch Size",
                min_value=4,
                max_value=128,
                value=16,
                help="Number of samples processed before the model is updated"
            )
        
        st.info(f"📊 Training will run for **{epochs} epochs** with batch size **{batch_size}**")
        
        st.divider()
        
        # Trigger Retraining
        st.subheader("🚀 Step 3: Start Retraining")
        
        # Check if dataset is uploaded
        if 'dataset_uploaded' not in st.session_state or not st.session_state.get('dataset_uploaded'):
            st.warning("⚠️ Please upload and confirm the dataset first (Step 1)")
        
        col_retrain1, col_retrain2, col_retrain3 = st.columns([1, 2, 1])
        
        with col_retrain2:
            retrain_disabled = 'dataset_uploaded' not in st.session_state or not st.session_state.get('dataset_uploaded')
            
            if st.button(
                "🎯 Start Retraining Now",
                type="primary",
                use_container_width=True,
                disabled=retrain_disabled
            ):
                with st.spinner(f"🔄 Retraining model... This will take several minutes..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        status_text.text("📤 Sending training request...")
                        progress_bar.progress(10)
                        
                        r = requests.post(
                            f"{API_URL}/trigger-retrain",
                            data={"epochs": epochs, "batch_size": batch_size},
                            timeout=7200  # 2 hour timeout
                        )
                        
                        progress_bar.progress(100)
                        
                        if r.status_code == 200:
                            result = r.json()
                            status_text.empty()
                            progress_bar.empty()
                            
                            st.success("✅ Retraining completed successfully!")
                            st.balloons()
                            
                            # Display results
                            st.write("**Training Results:**")
                            st.json(result)
                            
                            # Display metrics
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Final Train Accuracy", result.get('final_train_acc', 'N/A'))
                            with col_b:
                                st.metric("Final Val Accuracy", result.get('final_val_acc', 'N/A'))
                            with col_c:
                                st.metric("Total Classes", result.get('num_classes', 'N/A'))
                            
                            st.info("💡 The model has been updated. Predictions will now use the new model.")
                            st.info("🔄 Please refresh the page or restart the API for changes to take effect.")
                            
                            # Clear upload state
                            st.session_state['dataset_uploaded'] = False
                        else:
                            progress_bar.empty()
                            status_text.empty()
                            st.error(f"❌ Retraining failed with status {r.status_code}")
                            st.error(f"Error: {r.text}")
                            
                    except requests.exceptions.Timeout:
                        progress_bar.empty()
                        status_text.empty()
                        st.warning("⏱️ Request timed out. Training may still be running in the background.")
                        st.info("💡 Check the API logs to monitor training progress.")
                    except requests.exceptions.ConnectionError:
                        progress_bar.empty()
                        status_text.empty()
                        st.error("❌ Cannot connect to API. Make sure the FastAPI server is running.")
                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Error: {str(e)}")
    else:
        st.info("👆 Please upload a ZIP file to begin the retraining process.")

# -----------------------------
# PAGE 3: VISUALIZATIONS
# -----------------------------
elif page == "📊 Visualizations":
    st.header("📊 Data Visualizations & Analysis")
    
    st.info("💡 This section shows insights about the training data and model performance.")
    
    # Tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["📈 Dataset Distribution", "🎯 Model Performance", "📉 Training History"])
    
    with tab1:
        st.subheader("📈 Dataset Class Distribution")
        
        st.write("This shows the distribution of car brands in your training dataset.")
        
        # Try to load real data from training history or use mock data
        try:
            # Attempt to get real class distribution from API
            r = requests.get(f"{API_URL}/dataset-stats", timeout=5)
            if r.status_code == 200:
                data = r.json()
                brands = data.get('classes', [])
                counts = data.get('counts', [])
            else:
                raise Exception("Using mock data")
        except:
            # Mock data as fallback
            brands = ['BMW', 'Toyota', 'Ferrari', 'Tesla', 'Mercedes-Benz', 
                      'Audi', 'Honda', 'Ford', 'Chevrolet', 'Nissan']
            counts = [150, 140, 100, 95, 130, 120, 145, 135, 110, 125]
            st.warning("⚠️ Displaying sample data. Connect to API for real statistics.")
        
        # Bar chart
        fig = px.bar(
            x=brands,
            y=counts,
            labels={'x': 'Car Brand', 'y': 'Number of Images'},
            title='Training Data Distribution by Car Brand',
            color=counts,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Pie chart
        fig_pie = px.pie(
            values=counts,
            names=brands,
            title='Percentage Distribution of Car Brands'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        st.subheader("🎯 Model Performance Metrics")
        
        st.write("Model evaluation metrics and performance indicators")
        
        col1, col2, col3 = st.columns(3)
        
        # Try to load real metrics or use mock data
        try:
            r = requests.get(f"{API_URL}/model-metrics", timeout=5)
            if r.status_code == 200:
                metrics = r.json()
                accuracy = metrics.get('accuracy', '92.5%')
                precision = metrics.get('precision', '91.8%')
                recall = metrics.get('recall', '90.2%')
            else:
                raise Exception("Using mock data")
        except:
            accuracy = "92.5%"
            precision = "91.8%"
            recall = "90.2%"
            st.warning("⚠️ Displaying sample metrics. Train model to see real performance.")
        
        with col1:
            st.metric("Overall Accuracy", accuracy, "2.3%")
        with col2:
            st.metric("Precision", precision, "1.5%")
        with col3:
            st.metric("Recall", recall, "1.8%")
        
        st.info("💡 These metrics are calculated on the validation dataset.")
    
    with tab3:
        st.subheader("📉 Training History")
        
        st.write("Training and validation metrics over epochs")
        
        # Try to load real training history
        try:
            import json
            import os
            
            if os.path.exists('models/training_history.json'):
                with open('models/training_history.json', 'r') as f:
                    history = json.load(f)
                
                epochs_range = list(range(1, len(history['train_loss']) + 1))
                train_loss = history['train_loss']
                val_loss = history['val_loss']
                train_acc = history['train_acc']
                val_acc = history['val_acc']
                
                st.success("✅ Loaded real training history")
            else:
                raise Exception("No training history found")
        except:
            # Mock training history
            epochs_range = list(range(1, 21))
            train_loss = [2.5 - (i * 0.1) + (i % 3 * 0.05) for i in epochs_range]
            val_loss = [2.6 - (i * 0.09) + (i % 3 * 0.07) for i in epochs_range]
            train_acc = [30 + (i * 3) - (i % 3 * 2) for i in epochs_range]
            val_acc = [28 + (i * 2.8) - (i % 3 * 2.5) for i in epochs_range]
            
            st.warning("⚠️ Displaying sample training history. Train model to see real data.")
        
        # Loss plot
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=epochs_range,
            y=train_loss,
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='#1f77b4')
        ))
        fig_loss.add_trace(go.Scatter(
            x=epochs_range,
            y=val_loss,
            mode='lines+markers',
            name='Validation Loss',
            line=dict(color='#ff7f0e')
        ))
        fig_loss.update_layout(
            title='Training vs Validation Loss',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            height=400
        )
        st.plotly_chart(fig_loss, use_container_width=True)
        
        # Accuracy plot
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=epochs_range,
            y=train_acc,
            mode='lines+markers',
            name='Training Accuracy',
            line=dict(color='#2ca02c')
        ))
        fig_acc.add_trace(go.Scatter(
            x=epochs_range,
            y=val_acc,
            mode='lines+markers',
            name='Validation Accuracy',
            line=dict(color='#d62728')
        ))
        fig_acc.update_layout(
            title='Training vs Validation Accuracy',
            xaxis_title='Epoch',
            yaxis_title='Accuracy (%)',
            height=400
        )
        st.plotly_chart(fig_acc, use_container_width=True)

# -----------------------------
# FOOTER
# -----------------------------
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p>🚗 Car Classification ML Project | Built with Streamlit & FastAPI</p>
    <p>Powered by ResNet50 Transfer Learning</p>
    <p style='font-size: 0.8rem; margin-top: 1rem;'>
        💡 Tip: If you encounter issues, make sure the FastAPI server is running on port 8000
    </p>
</div>
""", unsafe_allow_html=True)