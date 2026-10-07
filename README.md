Nexus Air Quality Monitor



Real-Time Air Quality Monitoring and Next-Hour PM2.5 Prediction



This project was developed as part of the Nexus VIT Chennai Technical Department Recruitment 2026 – Technical Round 1.



Problem Statement



Air pollution can vary significantly depending on location and time. Presenting only a pollution value does not always provide enough context for users to understand current conditions or recent trends.



This project provides an interactive dashboard where users can select a geographical region, view current air-quality conditions, analyze historical pollution trends, estimate the current AQI, and predict the next-hour PM2.5 concentration using Machine Learning.



The application also provides environmental comparisons and general recommendations based on the observed pollution levels.



Why I Chose This Problem



I chose this problem because air quality is directly connected to everyday life, while pollution data is often presented as isolated values without enough context.



The project allowed me to combine real-time data retrieval, data analysis, visualization, and Machine Learning into a single application.



Rather than focusing only on building a prediction model, I wanted to understand the complete process of collecting data, processing it, analyzing it, generating predictions, and presenting the results in a way that a user can understand.



Key Features



1\. Region Selection



Users can enter a city or region. The application uses geographical coordinates obtained through the Open-Meteo Geocoding API and displays the selected location on a map.



2\. Current Air Quality



The dashboard displays available measurements for:



PM2.5

PM10

NO2

SO2

O3



3\. Historical Pollution Trends



Users can analyze pollution over different time periods:



Last 24 Hours

Last 7 Days

Last 30 Days



The dashboard displays pollutant trends along with basic statistics such as average and maximum concentration.



4\. Pollution Status



The application provides simple status indicators for individual pollutants to make the numerical values easier to interpret.



5\. Estimated AQI



The application calculates an estimated AQI using pollutant-specific breakpoint ranges and classifies the result into different pollution categories.



The AQI calculation in this project is intended for demonstration and comparison purposes. It should not be considered an official CPCB monitoring-station AQI because official AQI calculations use specific averaging periods and monitoring procedures.



6\. PM2.5 Prediction



A Random Forest Regression model is used to predict the next-hour PM2.5 concentration.



The model uses current pollutant values, previous-hour pollutant values, hour of the day, and day of the week as input features.



Machine Learning Approach



The objective of the Machine Learning component is to estimate the PM2.5 concentration for the following hour based on recent pollution conditions.



Input Features



PM2.5

PM10

NO2

SO2

O3

Hour of the day

Day of the week

Previous-hour PM2.5

Previous-hour PM10

Previous-hour NO2

Previous-hour SO2

Previous-hour O3



Target



Next-hour PM2.5 concentration



Model



Random Forest Regressor



The dataset is divided chronologically into 80% training data and 20% testing data. A chronological split was used instead of randomly shuffling the data because the problem involves time-dependent observations.



Initial Model Performance



MAE: 1.21 µg/m³

RMSE: 1.88 µg/m³

R²: 0.946



These results were obtained using the available historical data used during development. They should not be interpreted as a guarantee of performance for different locations, seasons, or longer periods.



System Workflow



User enters a location



Location is converted into latitude and longitude using the Geocoding API



Air-quality data is retrieved for the selected coordinates



The data is cleaned and processed using Pandas



Current pollutant levels and historical trends are displayed



An estimated AQI is calculated



Recent data is passed to the Machine Learning model



The model predicts the next-hour PM2.5 concentration



Environmental reference values are used for comparison



General recommendations are presented to the user



Technologies Used



Python



Streamlit



Pandas



NumPy



Scikit-learn



Plotly



Open-Meteo Geocoding API



Open-Meteo Air Quality API



CPCB National Ambient Air Quality Standards



Project Structure



nexus-air-quaity/

&#x20;   app.py

&#x20;   requirements.txt

&#x20;   README.md

&#x20;   .gitignore

&#x20;   venv/



The virtual environment is excluded from the Git repository using .gitignore.



Running the Project Locally



Clone the repository:



git clone https://github.com/bhuvana-mellimpudi/air\_quality\_monitor.git



Move into the project directory:



cd air\_quality\_monitor



Create a virtual environment:



python -m venv venv



Activate the virtual environment on Windows PowerShell:



.\\venv\\Scripts\\Activate.ps1



Install the required dependencies:



pip install -r requirements.txt



Run the application:



python -m streamlit run app.py



The application will then open in the browser.



Data Sources



Open-Meteo is used for geographical location lookup and air-quality data.



The air-quality API provides hourly pollutant information for the selected geographical coordinates.



CPCB standards are used as environmental reference values for the comparison section.



Environmental Comparison and Recommendations



The application compares selected pollutant levels with reference limits and provides general recommendations such as reducing unnecessary vehicle usage, using public transportation or carpooling, avoiding open burning, reducing exposure to construction and road dust, and following appropriate emission-control practices.



These recommendations are general environmental best practices and are not intended to provide medical advice.



Limitations



The project currently has several limitations.



The application uses open-source or modelled air-quality data rather than data from a dedicated physical monitoring station.



The Machine Learning model currently uses approximately 30 days of historical data.



The current prediction is limited to the next hour's PM2.5 concentration.



The model is not trained using long-term location-specific datasets for every geographical region.



The AQI displayed by the application is an estimate and should not be considered an official CPCB AQI.



The availability of individual pollutants may vary depending on the selected location and the data provided by the API.



Future Improvements



Possible improvements include:



Using a larger historical dataset



Adding weather variables such as temperature, humidity, wind speed, and precipitation



Supporting multi-hour and multi-day predictions



Training location-specific models



Exploring more advanced time-series models



Using official monitoring-station data where available



Adding automated pollution alerts



Implementing the complete official AQI averaging methodology



Continuously comparing predictions with future observations



What I Learned



This project helped me understand how different parts of a data-driven application work together.



The development process involved the complete pipeline:



API → Data Processing → Feature Engineering → Visualization → Machine Learning → Prediction → User Interface



Through this project, I gained practical experience with REST APIs, geographical data, data preprocessing, time-based feature engineering, regression models, model evaluation, interactive dashboards, and GitHub-based version control.



Project Information



Project: Nexus Air Quality Monitor



Problem: Real-Time Air Quality Monitoring and Prediction for User-Selected Regions



Developed by: Bhuvana Mellimpudi



Repository: https://github.com/bhuvana-mellimpudi/air\_quality\_monitor



Event: Nexus VIT Chennai Technical Department Recruitment 2026



Disclaimer



This project was developed for educational and demonstration purposes.



The air-quality values and predictions should not be treated as official regulatory measurements or medical guidance. The Machine Learning model demonstrates short-term PM2.5 forecasting and its performance may vary depending on location, available data, and environmental conditions.

