# Dataset Description Evaluation UI

A user-friendly Streamlit application for evaluating dataset descriptions across multiple quality metrics. This tool helps researchers systematically assess the quality of automatically generated or manually written dataset descriptions.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Setup](#setup)
- [Using the App](#using-the-app)
- [Evaluation Criteria](#evaluation-criteria)
- [File Outputs](#file-outputs)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This UI provides a streamlined interface for evaluating dataset descriptions on four key quality metrics:
- **Completeness** - Coverage of essential dataset aspects
- **Conciseness** - Efficiency in conveying information
- **Readability** - Logical flow and narrative coherence
- **Faithfulness** - Accuracy in representing the actual dataset

**Features:**
- Side-by-side comparison of dataset and description
- Manual scoring on 1-10 scale for each criterion
- Progress tracking and resume capability
- Export results to CSV
- Built-in evaluation criteria and examples

---

## 🔧 Installation

### Install Required Packages

```bash
pip install streamlit pandas
```

---

## 📁 Setup

### 1. Download Sample Data

**Two sample datasets are provided for evaluation:**
- `ntcir_sampled/` - Sample NTCIR datasets (36 tables)
- `ECIR_sampled/` - Sample ECIR datasets (12 tables)

**Your task is to evaluate the descriptions for these two datasets.**

**📥 [Download ntcir and ECIR datasets from Google Drive](https://drive.google.com/drive/folders/16Tyqqj-LPd43n5ofjMV8jnvYo4XLUtXQ?usp=sharing)**

After downloading:
1. Extract the folders
2. Place them in the same directory as `streamlit_app.py`

These are already in the correct format and ready to use. You can start evaluating immediately!

### 2. Folder Structure

Your project should look like this:

```
project_folder/
├── streamlit_app.py         # The Streamlit app file
├── ntcir_sampled/           # Sample NTCIR dataset (36 tables)
├── ECIR_sampled/            # Sample ECIR dataset (12 tables)
├── YourDatasets/            # Your custom datasets folder (optional)
│   ├── dataset_1/
│   │   ├── data.csv         # Your dataset (any .csv file)
│   │   ├── title.txt        # Dataset title (optional)
│   │   ├── data_profiler.txt    # Dataset statistics (optional)
│   │   ├── gpt_ufd.txt      # Description to evaluate
│   │   ├── gpt_sfd.txt      # Another description to evaluate
│   │   └── original.txt     # Another description to evaluate
│   ├── dataset_2/
│   │   └── ...
```

### 3. Required Files Per Dataset

**Each dataset folder needs:**

- **At least one CSV file** (`.csv`) - Your actual dataset
- **At least one description file**:
  - `gpt_ufd.txt` - Description type 1
  - `gpt_sfd.txt` - Description type 2
  - `original.txt` - Description type 3
- **Optional**: `title.txt` - Dataset title
- **Optional**: `data_profiler.txt` - Statistical summary

**Note**: You can have any combination of the three description types.

---

## 🚀 Using the App

### 1. Start the Application

```bash
streamlit run streamlit_app.py
```

Your browser will open to `http://localhost:8501`.

### 2. Load Datasets

1. Open the **left sidebar**
2. Enter the path to your datasets folder:
   - `./ntcir_sampled` (for sample data)
   - `./ECIR_sampled` (for sample data)
   - `./YourDatasets` (for your own data)
3. Click **"🔄 Load/Reload Datasets"**

### 3. Interface Overview

**Left Panel - Reference Data:**
- **Dataset Title**: Display of the dataset title (if available)
- **📊 Table Data**: View the actual CSV dataset (scroll to see all data)
- **📈 Statistics**: View statistical summaries
- **Evaluation Criteria**: Quick reference for scoring
- **Examples**: Sample evaluations

**Right Panel - Evaluation:**
- Description text
- Four scoring sliders (1-10 for each criterion)
- Comment box (optional)
- Save button and navigation

**Sidebar:**
- Progress tracker
- Quick navigation dropdown
- Download evaluations button

### 4. Evaluating Descriptions

1. **Read the instructions** (expandable section at top)
2. **Review the dataset title and data** in the left panel
3. **Read the description** on the right panel
4. **Score each criterion** using the sliders (1-10):
   - **Completeness**: Coverage of essential aspects
   - **Conciseness**: Efficiency without redundancy
   - **Readability**: Logical flow
   - **Faithfulness**: Accuracy without errors
5. **Add comments** (optional): Note any specific issues or observations
6. **Click "💾 Save Scores"** (wait for confirmation)
7. **Navigate** using Next/Previous or sidebar dropdowns

### 5. Navigation

- **Next ➡️** / **⬅️ Previous**: Move between descriptions
- **Sidebar dropdowns**: Jump to any dataset or description type
- **Progress**: Track completion percentage and position (e.g., "5 / 30")
- **Status**: Green "✅ Complete" shows when all criteria scored

### 6. Saving & Resuming

- **Save**: Click "💾 Save Scores" after each evaluation
- **Warning**: "⚠️ Unsaved changes" appears if you haven't saved
- **Resume**: Your progress is saved in `evaluations.csv` - the app loads it automatically on restart

---

## 📊 Evaluation Criteria

### Completeness (1-10)
How thoroughly the description covers scope, statistics, and applications.

- **8-10**: Comprehensive with size, structure, fields, and use cases
- **4-7**: Covers main aspects, missing some details
- **1-3**: Missing critical information

### Conciseness (1-10)
Efficiency in conveying information without redundancy.

- **8-10**: Succinct, using semantic types effectively
- **4-7**: Reasonably concise with some redundancy
- **1-3**: Verbose or repetitive

### Readability (1-10)
Logical flow and coherence of the description.

- **8-10**: Clear logical progression, integrated narrative
- **4-7**: Generally understandable, flow could improve
- **1-3**: Confusing structure or poor flow

### Faithfulness (1-10)
Accuracy in representing the dataset's actual content.

- **8-10**: Correct variables, structure, stats; no hallucinations
- **4-7**: Mostly accurate with minor errors
- **1-3**: Contains hallucinations or significant errors

---

## 📤 File Outputs

### evaluations.csv

**Columns:**
- `dataset_name` - Dataset folder name
- `dataset_path` - Full path to dataset folder
- `description_type` - Which description (gpt_ufd, gpt_sfd, original)
- `completeness` - Score (1-10)
- `conciseness` - Score (1-10)
- `readability` - Score (1-10)
- `faithfulness` - Score (1-10)
- `comment` - Optional comments about the description

**Example:**
```csv
dataset_name,dataset_path,description_type,completeness,conciseness,readability,faithfulness,comment
traffic_safety,./ntcir_sampled/traffic_safety,gpt_ufd,7,9,8,9,"Missing temporal coverage"
traffic_safety,./ntcir_sampled/traffic_safety,gpt_sfd,8,8,9,8,""
```

**Download**: Use the "📥 Download Evaluations" button in sidebar anytime.

---

## 🔍 Troubleshooting

### "No datasets loaded"
- Check path is correct (try absolute path if needed)
- Verify folder structure matches expected format
- Ensure each dataset has at least one .csv and one .txt description file
- Make sure you downloaded and extracted the sample datasets from Google Drive

### Description not showing
- File must be exactly named: `gpt_ufd.txt`, `gpt_sfd.txt`, or `original.txt`
- Names are case-sensitive
- App only shows available descriptions

### Scores not saving
- Always click "💾 Save Scores" button
- Wait for "✅ Saved successfully!" confirmation
- Check that `evaluations.csv` exists in app folder

### Can't see all data
- Use scrollbars (horizontal for columns, vertical for rows)
- Table displays first 1,000 rows for large datasets

### App won't start
```bash
pip install --upgrade streamlit pandas
```

---

## 💡 Tips

1. **Start with provided datasets**: Download and evaluate `ntcir_sampled` and `ECIR_sampled`!
2. **Read instructions first**: Review the expandable instruction box at the top
3. **Use examples**: Check example evaluations to calibrate your scoring
4. **Cross-reference**: Always compare description with actual CSV data and title
5. **Save frequently**: Don't forget to save after each evaluation
6. **Be consistent**: Apply the same standards across all evaluations
7. **Check statistics**: Use data profiler stats to verify accuracy
8. **Use comments**: Note specific issues in the comment box
9. **Take breaks**: Maintain evaluation quality by avoiding fatigue
10. **Backup regularly**: Download CSV periodically