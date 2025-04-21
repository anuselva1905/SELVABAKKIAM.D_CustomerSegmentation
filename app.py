from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

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
        # If qcut fails, fallback to ranking and binning manually
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
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
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

def create_visualizations(rfm_df):
    plots = {}

    plt.figure(figsize=(10, 6))
    sns.countplot(data=rfm_df, x='Segment', order=rfm_df['Segment'].value_counts().index)
    plt.title('Customer Segments Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['segment_dist'] = base64.b64encode(buf.getvalue()).decode('ascii')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rfm_df, x='Frequency', y='MonetaryValue', hue='Segment')
    plt.title('Monetary Value vs Frequency by Segment')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['monetary_freq'] = base64.b64encode(buf.getvalue()).decode('ascii')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=rfm_df, x='Recency', bins=20)
    plt.title('Recency Distribution')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['recency_dist'] = base64.b64encode(buf.getvalue()).decode('ascii')
    plt.close()

    return plots

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
                plots = create_visualizations(rfm_df)
                results_file = os.path.join(app.config['UPLOAD_FOLDER'], 'rfm_results.csv')
                rfm_df.to_csv(results_file, index=False)
                results_html = rfm_df.to_html(classes='table table-striped', index=False)
                return render_template('results.html', 
                                     tables=[results_html],
                                     plots=plots,
                                     segment_counts=rfm_df['Segment'].value_counts().to_dict())
            except Exception as e:
                return f"Error processing file: {str(e)}"
    return render_template('upload.html')

@app.route('/download')
def download_file():
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'rfm_results.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
