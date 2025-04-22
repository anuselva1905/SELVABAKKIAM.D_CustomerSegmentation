from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import json
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_dynamic_qcut(series, q=5, labels=None):
    try:
        bins = pd.qcut(series, q=q, retbins=True, duplicates='drop')[1]
        bin_count = len(bins) - 1
        if labels is None or len(labels) != bin_count:
            labels = list(range(1, bin_count + 1))
        return pd.qcut(series, q=bin_count, labels=labels, duplicates='drop')
    except ValueError:
        ranks = pd.qcut(series.rank(method="first"), q=min(q, series.nunique()), labels=False)
        return ranks + 1

def clean_data(df):
    cleaned_df = df.copy()
    cleaned_df["InvoiceNo"] = cleaned_df["InvoiceNo"].astype("str")
    mask = cleaned_df["InvoiceNo"].str.match("^\\d{6}$") == True
    cleaned_df = cleaned_df[mask]
    cleaned_df["StockCode"] = cleaned_df["StockCode"].astype("str")
    mask = (
        (cleaned_df["StockCode"].str.match("^\\d{5}$") == True) |
        (cleaned_df["StockCode"].str.match("^\\d{5}[a-zA-Z]+$") == True) |
        (cleaned_df["StockCode"].str.match("^gift_0001_40$") == True) |
        (cleaned_df["StockCode"].str.match("^DCGS0070$") == True) |
        (cleaned_df["StockCode"].str.match("^DCGS0003") == True) |
        (cleaned_df["StockCode"].str.match("^DCGS0076$") == True) |
        (cleaned_df["StockCode"].str.match("^DCGS0057$") == True) |
        (cleaned_df["StockCode"].str.match("^DCGS0069$") == True) |
        (cleaned_df["StockCode"].str.match("^DCGS0074$") == True) |
        (cleaned_df["StockCode"].str.match("^DCGS0072$") == True) |
        (cleaned_df["StockCode"].str.match("^DCGS0055$") == True) |
        (cleaned_df["StockCode"].str.match("^DCGS0071$") == True) |
        (cleaned_df["StockCode"].str.match("^DCGS0073$") == True) |
        (cleaned_df["StockCode"].str.match("^DCGS0004$") == True) |
        (cleaned_df["StockCode"].str.match("^PADS$") == True) |
        (cleaned_df["StockCode"].str.match("^gift_0001_10$") == True) |
        (cleaned_df["StockCode"].str.match("^DCGSSGIRL$") == True) |
        (cleaned_df["StockCode"].str.match("^DCGSSBOY$") == True) |
        (cleaned_df["StockCode"].str.match("^DCGS0066P$") == True) |
        (cleaned_df["StockCode"].str.match("^DCGS0067$") == True) |
        (cleaned_df["StockCode"].str.match("^DCGS0068$") == True)
    )
    cleaned_df = cleaned_df[mask]
    cleaned_df.dropna(subset=["CustomerID"], inplace=True)
    return cleaned_df

def perform_rfm_analysis(df):
    # Explicit date parsing with error handling
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format='mixed', errors='coerce')
    df = df.dropna(subset=["InvoiceDate"])
    
    df["SalesLineTotal"] = df["Quantity"] * df["UnitPrice"]

    aggregated_df = df.groupby(by="CustomerID", as_index=False).agg(
        MonetaryValue=("SalesLineTotal", "sum"),
        Frequency=("InvoiceNo", "nunique"),
        LastInvoiceDate=("InvoiceDate", "max")
    )

    max_invoice_date = aggregated_df["LastInvoiceDate"].max()
    aggregated_df["Recency"] = (max_invoice_date - aggregated_df["LastInvoiceDate"]).dt.days

    aggregated_df['R_Quartile'] = get_dynamic_qcut(aggregated_df['Recency'], 5, labels=[5, 4, 3, 2, 1])
    aggregated_df['F_Quartile'] = get_dynamic_qcut(aggregated_df['Frequency'], 5, labels=[1, 2, 3, 4, 5])
    aggregated_df['M_Quartile'] = get_dynamic_qcut(aggregated_df['MonetaryValue'], 5, labels=[1, 2, 3, 4, 5])

    aggregated_df['RFM_Score'] = (
        aggregated_df['R_Quartile'].astype(str) + 
        aggregated_df['F_Quartile'].astype(str) + 
        aggregated_df['M_Quartile'].astype(str)
    )

    segments = {
        r'555|554|544|545|454|455|445': 'Champions',
        r'543|444|435|355|354|345|344|335': 'Loyal Customers',
        r'553|551|552|541|542|533|532|531|452|451|442|441|431|453|433|432|423|353|352|351|342|341|333|323': 'Potential Loyalists',
        r'512|511|422|413|412|411|311': 'New Customers',
        r'525|524|523|522|521|515|514|513|425|424|413|414|415|315|314|313': 'Promising',
        r'331|321|312|221|213|231|241|251': 'Need Attention',
        r'155|154|144|214|215|115|114|113': 'Cannot Lose Them',
        r'134|143|234|243|224|233|244|253|252|225|223|222|132|123|122|212|211': 'At Risk',
        r'111|112|121|131|141|151': 'Hibernating',
        r'332|322|231|241|251|233|234|235|225|224|223|222|132|123|122|212|211': 'About to Sleep',
    }

    aggregated_df['Segment'] = aggregated_df['RFM_Score'].replace(segments, regex=True)
    return aggregated_df

def get_segment_description(segment):
    descriptions = {
        'Champions': 'Your best customers who buy often and spend the most',
        'Loyal Customers': 'Frequent buyers who are loyal to your brand',
        'Potential Loyalists': 'Recent customers who could become loyal with engagement',
        'New Customers': 'Newly acquired customers who need nurturing',
        'Promising': 'Showing potential but need more engagement',
        'Need Attention': 'At risk of churning - need re-engagement',
        'Cannot Lose Them': 'High spenders who haven\'t purchased recently',
        'At Risk': 'Spent big and were frequent but haven\'t returned',
        'Hibernating': 'Inactive for a long time - may be lost',
        'About to Sleep': 'Starting to disengage - need reactivation'
    }
    return descriptions.get(segment, 'No description available')

def create_visualizations(rfm_df):
    """Create base64 encoded visualizations for the dashboard"""
    plots = {}
    
    try:
        # Segment Distribution
        plt.figure(figsize=(10, 6))
        segment_order = rfm_df['Segment'].value_counts().index
        palette = sns.color_palette("husl", len(segment_order))
        sns.countplot(data=rfm_df, x='Segment', order=segment_order, palette=palette, hue='Segment', legend=False)
        plt.title('Customer Segments Distribution', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plots['segment_dist'] = base64.b64encode(buf.getvalue()).decode('ascii')
        plt.close()
        
        # Monetary vs Frequency Scatter
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=rfm_df, x='Frequency', y='MonetaryValue', hue='Segment', 
                       palette="husl", alpha=0.7, s=100)
        plt.title('Monetary Value vs Frequency by Segment', pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plots['monetary_freq'] = base64.b64encode(buf.getvalue()).decode('ascii')
        plt.close()
    except Exception as e:
        app.logger.error(f"Error creating visualizations: {str(e)}")
    
    return plots

def prepare_dashboard_data(rfm_df):
    """Prepare data for the enhanced dashboard view"""
    # First check if DataFrame is empty
    if rfm_df.empty or len(rfm_df) < 3:  # Ensure enough data points
        return {
            'total_customers': len(customers),
            'cluster_count': 0,
            'optimal_clusters': 0,
            'clusters': [],
            'cluster_labels': [],
            'cluster_sizes': [],
            'cluster_colors': [],
            'radar_data': [],
            'customers': customers,
            'segment_colors': segment_colors
        }
    
    # Convert DataFrame to native Python types
    rfm_df = rfm_df.where(pd.notnull(rfm_df), None)
    rfm_df = rfm_df.fillna({
        'Recency': 0,
        'Frequency': 0,
        'MonetaryValue': 0
    })
    
    # Define color palette
    segment_colors = {
        'Champions': '#4361ee',
        'Loyal Customers': '#3a0ca3',
        'Potential Loyalists': '#7209b7',
        'New Customers': '#f72585',
        'Promising': '#4cc9f0',
        'Need Attention': '#4895ef',
        'Cannot Lose Them': '#3f37c9',
        'At Risk': '#560bad',
        'Hibernating': '#480ca8',
        'About to Sleep': '#b5179e'
    }
    
    # Prepare cluster profiles with safety checks
    clusters = []
    for segment, color in segment_colors.items():
        segment_data = rfm_df[rfm_df['Segment'] == segment]
        if not segment_data.empty:
            clusters.append({
                'id': str(segment),
                'size': int(len(segment_data)),
                'color': str(color),
                'avg_recency': float(segment_data['Recency'].mean()),
                'avg_frequency': float(segment_data['Frequency'].mean()),
                'avg_monetary': float(segment_data['MonetaryValue'].mean()),
                'description': str(get_segment_description(segment))
            })
    
    # Prepare radar chart data with guaranteed 3 values
    radar_data = []
    required_metrics = ['Recency', 'Frequency', 'MonetaryValue']
    
    for segment, color in segment_colors.items():
        segment_data = rfm_df[rfm_df['Segment'] == segment]
        if not segment_data.empty:
            values = []
            for metric in required_metrics:
                try:
                    max_val = max(float(rfm_df[metric].max()), 1)  # Prevent division by zero
                    norm_val = float(segment_data[metric].mean()) / max_val * 100
                    if metric == 'Recency':
                        norm_val = 100 - norm_val  # Invert for recency
                    values.append(float(norm_val))
                except:
                    values.append(0.0)  # Fallback value
            
            radar_data.append({
                'id': str(segment),
                'values': values[:3],  # Ensure exactly 3 values
                'color': str(color)
            })
    
    # Prepare customer data with safety checks
    customers = []
    sample_size = min(50, len(rfm_df))
    if sample_size > 0:
        sample_df = rfm_df.sample(sample_size)
        for _, row in sample_df.iterrows():
            customers.append({
                'CustomerID': str(row.get('CustomerID', '')),
                'Segment': str(row.get('Segment', '')),
                'Recency': int(row.get('Recency', 0)),
                'Frequency': int(row.get('Frequency', 0)),
                'MonetaryValue': float(row.get('MonetaryValue', 0)),
                'RFM_Score': str(row.get('RFM_Score', ''))
            })
    
    return {
        'total_customers': int(len(rfm_df)),
        'cluster_count': int(len(segment_colors)),
        'optimal_clusters': int(len([c for c in clusters if c['size'] > 0])),
        'clusters': clusters,
        'cluster_labels': [str(c['id']) for c in clusters],
        'cluster_sizes': [int(c['size']) for c in clusters],
        'cluster_colors': [str(c['color']) for c in clusters],
        'radar_data': radar_data,
        'customers': customers,
        'segment_colors': segment_colors
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = 'customer_data.csv'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                df = pd.read_csv(filepath)
                cleaned_df = clean_data(df)
                rfm_df = perform_rfm_analysis(cleaned_df)
                
                # Save results
                results_file = os.path.join(app.config['UPLOAD_FOLDER'], 'rfm_results.csv')
                rfm_df.to_csv(results_file, index=False)
                
                # Prepare data for both old and new views
                plots = create_visualizations(rfm_df)
                dashboard_data = prepare_dashboard_data(rfm_df)
                
                # For backward compatibility
                results_html = rfm_df.to_html(classes='table table-striped', index=False)
                
                return render_template('result.html',
                                     tables=[results_html],
                                     plots=plots,
                                     segment_counts=rfm_df['Segment'].value_counts().to_dict(),
                                     **dashboard_data)
            except Exception as e:
                app.logger.error(f"Error processing file: {str(e)}", exc_info=True)
                return render_template('error.html', 
                                    error_message=f"Error processing file: {str(e)}")
    return render_template('upload.html')

@app.route('/download')
def download_file():
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'rfm_results.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)