from flask import Flask, request, render_template, jsonify
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from predict_model import predict_rainfall

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/visualize')
def visualize():
    return render_template('visualize.html')

@app.route('/result', methods=['POST'])
def result():
    location = request.form['location']
    year = int(request.form['year'])
    month = request.form['month']

    predicted_rainfall = predict_rainfall(location, year, month)

    return render_template('result.html', location=location, year=year, month=month, predicted_rainfall=predicted_rainfall)

@app.route('/visualization_result', methods=['POST'])
def visualization_result():
    location = request.form['location']
    year = int(request.form['year'])

    # Load the dataset
    df = pd.read_csv('rainfall.csv')
    df['SUBDIVISION'] = df['SUBDIVISION'].str.strip()

    # Filter data for the selected year and subdivision
    filtered_data = df[(df['SUBDIVISION'] == location) & (df['YEAR'] == year)]

    if filtered_data.empty:
        # Predict rainfall for future years if historical data is not available
        predicted_data = {}
        for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']:
            predicted_data[month] = predict_rainfall(location, year, month)

        # Convert the predicted data to a DataFrame for visualization
        filtered_data = pd.DataFrame(predicted_data, index=[0])
    else:
        filtered_data = filtered_data.iloc[0, :]

    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    rainfall_values = filtered_data[months].values.flatten()  # Ensure it is a 1D array

    # Ensure the data is in the correct format for Plotly Express
    df_viz = pd.DataFrame({'Month': months, 'Rainfall': rainfall_values})

    # Line chart
    fig_line = px.line(
        df_viz,
        x='Month',
        y='Rainfall',
        title=f'Rainfall({location}, {year})'
    )

    # Bar chart
    fig_bar = px.bar(
        df_viz,
        x='Month',
        y='Rainfall',
        title=f'Rainfall({location}, {year})'
    )

    # Heatmap
    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=[rainfall_values],
            x=months,
            colorscale='Viridis'
        )
    )
    fig_heatmap.update_layout(title=f'Rainfall({location}, {year})')

    # Pie chart
    fig_pie = px.pie(
        df_viz,
        names='Month',
        values='Rainfall',
        title=f'Rainfall({location}, {year})'
    )

    # Enhanced Density plot
    fig_density = go.Figure()
    fig_density.add_trace(go.Histogram(x=rainfall_values, nbinsx=12, histnorm='probability density'))
    fig_density.add_trace(go.Scatter(x=rainfall_values, y=[value for value in rainfall_values], mode='markers', name='Rainfall Points'))
    fig_density.update_layout(
        title=f'Rainfall({location}, {year})',
        xaxis_title='Rainfall',
        yaxis_title='Density',
        showlegend=True
    )


    return render_template(
        'visualization_result.html',
        line_chart=fig_line.to_html(full_html=False),
        bar_chart=fig_bar.to_html(full_html=False),
        heatmap=fig_heatmap.to_html(full_html=False),
        pie_chart=fig_pie.to_html(full_html=False),
        density_plot=fig_density.to_html(full_html=False)
    )


# JSON API endpoint
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    location = data['location']
    year = int(data['year'])
    month = data['month']

    predicted_rainfall = predict_rainfall(location, year, month)

    return jsonify({
        'location': location,
        'year': year,
        'month': month,
        'predicted_rainfall': predicted_rainfall
    })

if __name__ == '__main__':
    app.run(debug=True)
