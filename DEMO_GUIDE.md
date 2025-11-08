# 🌦️ RainFall AI - Demo Guide

## 🚀 Quick Start Guide

### How to Launch the App

1. **Using the Launch Script (Recommended):**
   ```bash
   cd /home/rounak/Rainfall
   ./launch_app.sh
   ```

2. **Manual Launch:**
   ```bash
   cd /home/rounak/Rainfall
   streamlit run app/streamlit_app.py
   ```

3. **Access the App:**
   - Open your browser and go to `http://localhost:8501`

## 🎯 How to Use the App

### Step 1: Select Location
- Choose from 36 available Indian regions in the dropdown
- All locations are from the trained dataset (1901-2015)

### Step 2: Set Prediction Parameters
- **Year**: Select year (2024-2030)
- **Month**: Choose the month for prediction
- **Previous Year Rainfall**: Enter rainfall from same month of previous year

### Step 3: Get Prediction
- Click "🚀 Predict Rainfall" button
- View the AI-powered prediction with confidence score
- Check risk assessment and recommendations

## 🌟 App Features

### 🎨 **Modern Black Theme**
- Sleek dark interface
- Professional appearance
- Easy on the eyes

### 📊 **Smart Predictions**
- AI model with 83.15% accuracy
- Historical data integration
- Seasonal adjustments

### 🎯 **User-Friendly Interface**
- Simple dropdown for location selection
- Clear input parameters
- Instant results display

### 💡 **Intelligent Insights**
- Risk assessment (Drought/Normal/Good rainfall)
- Actionable recommendations
- Historical data comparison

## 📱 Available Locations

The app supports predictions for these 36 Indian regions:

**Northern India:**
- Jammu & Kashmir, Himachal Pradesh, Punjab, Haryana Delhi & Chandigarh
- Uttarakhand, East Uttar Pradesh, West Uttar Pradesh

**Western India:**
- West Rajasthan, East Rajasthan, Gujarat Region, Saurashtra & Kutch
- Konkan & Goa, Madhya Maharashtra, Matathwada, Vidarbha

**Central India:**
- West Madhya Pradesh, East Madhya Pradesh, Chhattisgarh

**Eastern India:**
- Sub Himalayan West Bengal & Sikkim, Gangetic West Bengal, Orissa
- Jharkhand, Bihar

**Northeastern India:**
- Arunachal Pradesh, Assam & Meghalaya, Naga Mani Mizo Tripura

**Southern India:**
- Coastal Andhra Pradesh, Telangana, Rayalaseema, Tamil Nadu
- Coastal Karnataka, North Interior Karnataka, South Interior Karnataka, Kerala

**Islands:**
- Andaman & Nicobar Islands, Lakshadweep

## 🔧 Model Information

- **Model Type**: Artificial Neural Network (ANN)
- **Accuracy**: 83.15% (R² Score)
- **Training Data**: 115+ years (1901-2015)
- **Architecture**: 128 → 64 → 1 neurons
- **Features**: Seasonal patterns, lag features, regional encoding

## 🎯 Use Cases

### 🌾 **For Farmers**
- Plan crop planting schedules
- Optimize irrigation timing
- Assess drought/flood risks

### 🏛️ **For Government Agencies**
- Water resource planning
- Disaster preparedness
- Agricultural policy making

### 🎓 **For Researchers**
- Climate pattern analysis
- Weather model validation
- Academic research

### 🏢 **For Businesses**
- Agricultural insurance
- Supply chain planning
- Market analysis

## 🚨 Troubleshooting

### Common Issues:

1. **Model Not Loading:**
   - Ensure `models/ann_rainfall_model.h5` exists
   - Check file permissions

2. **Data Not Loading:**
   - Verify `data/rainfall in india 1901-2015.csv` exists
   - Check CSV file format

3. **App Not Starting:**
   - Install dependencies: `pip install -r app/requirements.txt`
   - Check Python version (3.8+)

## 🎉 Success Indicators

When everything works correctly, you should see:
- ✅ AI Model Loaded Successfully
- ✅ Rainfall Data Loaded
- 🎯 Model Accuracy: 83.15%
- 📊 Data Coverage: 115+ Years

## 📞 Support

For issues or questions:
- Check the console output for error messages
- Verify all files are in correct locations
- Ensure Python dependencies are installed

---

**🌦️ Enjoy predicting rainfall with RainFall AI!** 🎉

