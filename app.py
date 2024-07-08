from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

config_file_path = 'config.json'
with open(config_file_path, 'r') as f:
    config = json.load(f)

app.config['UPLOAD_FOLDER'] = config['UPLOAD_FOLDER']
app.config['STATIC_FOLDER'] = config['STATIC_FOLDER']
app.config['ALLOWED_EXTENSIONS'] = set(config['ALLOWED_EXTENSIONS'])

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['STATIC_FOLDER']):
    os.makedirs(app.config['STATIC_FOLDER'])

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def handle_missing_values(df):
    df = df.replace(' ', np.nan)
    df = df.replace('?', np.nan)
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)
        else:
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)
    return df

def clean_data(filepath):
    df = pd.read_csv(filepath)
    df = handle_missing_values(df)
    return df.sample(frac=0.5, random_state=42)

def generate_unique_filename(extension='png'):
    return os.path.join(app.config['STATIC_FOLDER'], f"{uuid.uuid4()}.{extension}")

def generate_bar_plot(pdf, param1, param2):
    plt.figure(figsize=(9, 6))
    sns.countplot(data=pdf, x=param1, hue=param2)
    plt.title(f'Count plot of {param1} vs {param2}')
    graph_path = generate_unique_filename()
    plt.savefig(graph_path)
    plt.close()
    return graph_path

def generate_pie_plot(pdf, param1):
    plt.figure(figsize=(9,6))
    pdf[param1].value_counts().plot.pie(autopct='%1.1f%%', shadow=True)
    plt.title(f'Pie plot of {param1}')
    graph_path = generate_unique_filename()
    plt.savefig(graph_path)
    plt.close()
    return graph_path

def generate_line_plot(pdf, param1, param2):
    plt.figure(figsize=(9, 6))
    sns.lineplot(data=pdf, x=param1, y=param2)
    plt.title(f'Line plot of {param1} vs {param2}')
    graph_path = generate_unique_filename()
    plt.savefig(graph_path)
    plt.close()
    return graph_path

def generate_scatter_plot(pdf, param1, param2):
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=pdf, x=param1, y=param2)
    plt.title(f'Scatter plot of {param1} vs {param2}')
    graph_path = generate_unique_filename()
    plt.savefig(graph_path)
    plt.close()
    return graph_path

def generate_box_plot(pdf, param1, param2):
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=pdf, x=param1, y=param2)
    plt.title(f'Box plot of {param1} vs {param2}')
    graph_path = generate_unique_filename()
    plt.savefig(graph_path)
    plt.close()
    return graph_path

def generate_violin_plot(pdf, param1, param2):
    plt.figure(figsize=(9, 6))
    sns.violinplot(data=pdf, x=param1, y=param2)
    plt.title(f'Violin plot of {param1} vs {param2}')
    graph_path = generate_unique_filename()
    plt.savefig(graph_path)
    plt.close()
    return graph_path

def generate_plot(pdf, param1, param2):
    graph_paths = []
    if pdf[param1].dtype == 'object' and pdf[param2].dtype == 'object':
        graph_paths.append(generate_bar_plot(pdf, param1, param2))
        graph_paths.append(generate_pie_plot(pdf, param1))
    elif pdf[param1].dtype in ['int64', 'float64'] and pdf[param2].dtype in ['int64', 'float64']:
        graph_paths.append(generate_line_plot(pdf, param1, param2))
        graph_paths.append(generate_scatter_plot(pdf, param1, param2))
    elif (pdf[param1].dtype == 'object' and pdf[param2].dtype in ['int64', 'float64']) or (pdf[param2].dtype == 'object' and pdf[param1].dtype in ['int64', 'float64']):
        if pdf[param1].dtype == 'object':
            graph_paths.append(generate_box_plot(pdf, param1, param2))
            graph_paths.append(generate_violin_plot(pdf, param1, param2))
        else:
            graph_paths.append(generate_box_plot(pdf, param2, param1))
            graph_paths.append(generate_violin_plot(pdf, param2, param1))
    return graph_paths

def generate_text(prompt, max_length=150, temperature=0.7, top_k=50, top_p=0.95):
    inputs = gpt2_tokenizer.encode(prompt, return_tensors='pt')
    outputs = gpt2_model.generate(
        inputs, 
        max_length=max_length, 
        num_return_sequences=1, 
        pad_token_id=gpt2_tokenizer.eos_token_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        cleaned_data = clean_data(file_path)
        columns = cleaned_data.columns.tolist()
        return render_template('select_params.html', filename=filename, columns=columns)
    return redirect(request.url)

@app.route('/generate_graphs', methods=['POST'])
def generate_graphs():
    filename = request.form['filename']
    param1 = request.form['param1']
    param2 = request.form['param2']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf = clean_data(file_path)
    graph_paths = generate_plot(pdf, param1, param2)
    prompt = f"Generate insights for the columns {param1} and {param2}."
    insights = [generate_text(prompt)]
    return render_template('display_graphs.html', graph_paths=graph_paths, insights=insights)

app.run()