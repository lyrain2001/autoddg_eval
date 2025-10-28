import streamlit as st
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Dataset Description Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563eb;
    }
    .criteria-box {
        background-color: #eff6ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bfdbfe;
        margin-bottom: 1rem;
    }
    .title-box {
        background-color: #e0f2fe;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0284c7;
        margin-bottom: 1rem;
        font-size: 1.1rem;
        color: #0c4a6e;
    }
    .description-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
        margin-top: 0;
    }
    .stButton > button {
        width: 100%;
    }
    .method-selection-box {
        background-color: #f0fdf4;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #86efac;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Constants
ALL_DESCRIPTION_TYPES = ['gpt_ufd', 'gpt_sfd', 'original', 'llm_only']
CRITERIA = {
    'completeness': 'Completeness (1-10): Coverage of scope, statistics, and applications',
    'conciseness': 'Conciseness (1-10): Efficiency without redundancy',
    'readability': 'Readability (1-10): Logical flow and coherent narrative',
    'faithfulness': 'Faithfulness (1-10): Accuracy reflecting dataset content'
}
EVALUATIONS_FILE = 'evaluations.csv'

def load_datasets(datasets_path, selected_methods):
    """Load all datasets from the specified path"""
    datasets = []
    
    if not os.path.exists(datasets_path):
        return datasets
    
    for dataset_folder in sorted(os.listdir(datasets_path)):
        folder_path = os.path.join(datasets_path, dataset_folder)
        
        if not os.path.isdir(folder_path):
            continue
        
        dataset_info = {
            'name': dataset_folder,
            'path': folder_path,
            'csv_files': [],
            'descriptions': {},
            'title': None
        }
        
        # Find CSV files and description files
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            
            if file.endswith('.csv'):
                dataset_info['csv_files'].append(file)
            elif file == 'data_profiler.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    dataset_info['stats'] = f.read()
            elif file == 'title.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    dataset_info['title'] = f.read().strip()
            elif file in [f'{method}.txt' for method in ALL_DESCRIPTION_TYPES]:
                desc_type = file.replace('.txt', '')
                # Only load if this method is selected
                if desc_type in selected_methods:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        dataset_info['descriptions'][desc_type] = f.read()
        
        # Only add datasets that have at least one selected description
        if dataset_info['csv_files'] and len(dataset_info['descriptions']) > 0:
            datasets.append(dataset_info)
    
    return datasets

def load_evaluations():
    """Load saved evaluations from CSV file"""
    if os.path.exists(EVALUATIONS_FILE):
        df = pd.read_csv(EVALUATIONS_FILE)
        # Ensure required columns exist for backwards compatibility
        if 'dataset_path' not in df.columns:
            df['dataset_path'] = ''
        if 'comment' not in df.columns:
            df['comment'] = ''
        else:
            # Restore newlines from placeholders
            df['comment'] = df['comment'].apply(
                lambda x: str(x).replace('\\n', '\n').replace('\\r', '\r') if pd.notna(x) and x != 'nan' else ''
            )
        return df
    else:
        # Create empty DataFrame with proper columns
        return pd.DataFrame(columns=['dataset_name', 'dataset_path', 'description_type', 'completeness', 'conciseness', 'readability', 'faithfulness', 'comment'])

def save_evaluations(evaluations_df):
    """Save evaluations to CSV file"""
    # Replace newlines in comments with a placeholder to preserve CSV structure
    if 'comment' in evaluations_df.columns:
        evaluations_df['comment'] = evaluations_df['comment'].apply(
            lambda x: str(x).replace('\n', '\\n').replace('\r', '\\r') if pd.notna(x) else ''
        )
    evaluations_df.to_csv(EVALUATIONS_FILE, index=False, quoting=1)  # quoting=1 means QUOTE_ALL

def get_evaluation_scores(evaluations_df, dataset_name, description_type):
    """Get scores for a specific dataset and description type"""
    row = evaluations_df[
        (evaluations_df['dataset_name'] == dataset_name) & 
        (evaluations_df['description_type'] == description_type)
    ]
    
    if len(row) > 0:
        return {
            'completeness': row.iloc[0]['completeness'] if pd.notna(row.iloc[0]['completeness']) else None,
            'conciseness': row.iloc[0]['conciseness'] if pd.notna(row.iloc[0]['conciseness']) else None,
            'readability': row.iloc[0]['readability'] if pd.notna(row.iloc[0]['readability']) else None,
            'faithfulness': row.iloc[0]['faithfulness'] if pd.notna(row.iloc[0]['faithfulness']) else None,
            'comment': row.iloc[0]['comment'] if pd.notna(row.iloc[0]['comment']) else ''
        }
    return {'completeness': None, 'conciseness': None, 'readability': None, 'faithfulness': None, 'comment': ''}

def get_current_score_key(dataset_name, description_type, criterion):
    """Generate key for current scores"""
    return f"score_{dataset_name}_{description_type}_{criterion}"

def get_current_comment_key(dataset_name, description_type):
    """Generate key for current comment"""
    return f"comment_{dataset_name}_{description_type}"

def initialize_scores_for_description(dataset_name, description_type, saved_scores):
    """Initialize scores in session state for current description"""
    for criterion in CRITERIA.keys():
        key = get_current_score_key(dataset_name, description_type, criterion)
        if key not in st.session_state:
            # Use saved score if available, otherwise default to 5
            if saved_scores[criterion] is not None:
                st.session_state[key] = int(saved_scores[criterion])
            else:
                st.session_state[key] = 5
    
    # Initialize comment
    comment_key = get_current_comment_key(dataset_name, description_type)
    if comment_key not in st.session_state:
        st.session_state[comment_key] = saved_scores.get('comment', '')

def save_current_scores(evaluations_df, dataset_name, dataset_path, description_type):
    """Save current scores from session state to DataFrame"""
    scores = {}
    for criterion in CRITERIA.keys():
        key = get_current_score_key(dataset_name, description_type, criterion)
        if key in st.session_state:
            scores[criterion] = st.session_state[key]
    
    # Get comment
    comment_key = get_current_comment_key(dataset_name, description_type)
    comment = st.session_state.get(comment_key, '')
    
    # Check if row exists
    mask = (evaluations_df['dataset_name'] == dataset_name) & (evaluations_df['description_type'] == description_type)
    
    if mask.any():
        # Update existing row
        for criterion, score in scores.items():
            evaluations_df.loc[mask, criterion] = score
        evaluations_df.loc[mask, 'dataset_path'] = dataset_path
        evaluations_df.loc[mask, 'comment'] = comment
    else:
        # Create new row
        new_row = {
            'dataset_name': dataset_name,
            'dataset_path': dataset_path,
            'description_type': description_type,
            'completeness': scores.get('completeness', None),
            'conciseness': scores.get('conciseness', None),
            'readability': scores.get('readability', None),
            'faithfulness': scores.get('faithfulness', None),
            'comment': comment
        }
        evaluations_df = pd.concat([evaluations_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return evaluations_df

def calculate_progress(datasets, evaluations_df, selected_methods):
    """Calculate completion percentage"""
    if not datasets:
        return 0
    
    total = len(datasets) * len(selected_methods) * len(CRITERIA)
    completed = 0
    
    for dataset in datasets:
        for desc_type in selected_methods:
            if desc_type in dataset['descriptions']:
                scores = get_evaluation_scores(evaluations_df, dataset['name'], desc_type)
                for criterion in CRITERIA.keys():
                    if scores[criterion] is not None:
                        completed += 1
    
    return int((completed / total) * 100) if total > 0 else 0

def is_description_complete(dataset_name, description_type, evaluations_df):
    """Check if all criteria are rated for a description"""
    scores = get_evaluation_scores(evaluations_df, dataset_name, description_type)
    return all(scores[criterion] is not None for criterion in CRITERIA.keys())

def show_method_selection():
    """Show method selection interface"""
    st.markdown('<div class="main-header">📊 Dataset Description Evaluation</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown('<div class="method-selection-box">', unsafe_allow_html=True)
    st.markdown("### 🎯 Step 1: Select Description Methods to Evaluate")
    st.markdown("Choose which description types you want to evaluate. Only the selected methods will be loaded.")
    
    # Create columns for checkboxes
    cols = st.columns(4)
    method_labels = {
        'gpt_ufd': 'GPT UFD',
        'gpt_sfd': 'GPT SFD',
        'original': 'Original',
        'llm_only': 'LLM Only'
    }
    
    selected_methods = []
    for idx, method in enumerate(ALL_DESCRIPTION_TYPES):
        with cols[idx]:
            if st.checkbox(method_labels[method], value=True, key=f"method_{method}"):
                selected_methods.append(method)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Dataset path configuration
    st.markdown("### 📁 Step 2: Configure Dataset Path")
    datasets_path = st.text_input(
        "Datasets folder path:",
        value=st.session_state.get('datasets_path', './Datasets'),
        help="Path to the folder containing dataset subfolders"
    )
    
    # Start button
    st.markdown("### ▶️ Step 3: Start Evaluation")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Load Datasets and Start", type="primary", use_container_width=True):
            if not selected_methods:
                st.error("⚠️ Please select at least one description method!")
            else:
                st.session_state.datasets_path = datasets_path
                st.session_state.selected_methods = selected_methods
                st.session_state.datasets = load_datasets(datasets_path, selected_methods)
                st.session_state.current_dataset_idx = 0
                st.session_state.current_desc_idx = 0
                st.session_state.evaluation_started = True
                
                if not st.session_state.datasets:
                    st.error(f"⚠️ No datasets found in {datasets_path} with the selected methods!")
                else:
                    st.success(f"✅ Loaded {len(st.session_state.datasets)} datasets with {len(selected_methods)} method(s)")
                    st.rerun()
    
    # Show info box
    st.markdown("---")
    st.info("""
    **Expected Folder Structure:**
    ```
    Datasets/
    ├── dataset_name_1/
    │   ├── file.csv
    │   ├── data_profiler.txt
    │   ├── title.txt
    │   ├── gpt_ufd.txt
    │   ├── gpt_sfd.txt
    │   ├── original.txt
    │   └── llm_only.txt
    ├── dataset_name_2/
    │   └── ...
    ```
    """)

def main():
    # Initialize session state
    if 'datasets_path' not in st.session_state:
        st.session_state.datasets_path = './Datasets'
    if 'datasets' not in st.session_state:
        st.session_state.datasets = []
    if 'current_dataset_idx' not in st.session_state:
        st.session_state.current_dataset_idx = 0
    if 'current_desc_idx' not in st.session_state:
        st.session_state.current_desc_idx = 0
    if 'evaluations' not in st.session_state:
        st.session_state.evaluations = load_evaluations()
    if 'last_saved' not in st.session_state:
        st.session_state.last_saved = None
    if 'unsaved_changes' not in st.session_state:
        st.session_state.unsaved_changes = False
    if 'evaluation_started' not in st.session_state:
        st.session_state.evaluation_started = False
    if 'selected_methods' not in st.session_state:
        st.session_state.selected_methods = ALL_DESCRIPTION_TYPES.copy()
    
    # Show method selection screen if not started
    if not st.session_state.evaluation_started:
        show_method_selection()
        return
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="main-header">📊 Dataset Description Evaluation</div>', unsafe_allow_html=True)
    with col2:
        if st.button("🔄 Restart & Change Methods", type="secondary"):
            st.session_state.evaluation_started = False
            st.rerun()
    
    # Show instructions at the top (collapsible)
    with st.expander("📋 **Evaluation Instructions** - Click to expand", expanded=False):
        st.markdown("""
        You will be given one tabular dataset description. Your task is to rate the description on **4 metrics**.
        Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.
        
        ### Evaluation Criteria:
        
        **1. Completeness (1-10)** - Evaluates how thoroughly the dataset description covers essential aspects such as the scope of data, query workloads, summary statistics, and possible tasks or applications.
        
        A high score indicates that the description provides a comprehensive overview, including details on dataset size, structure, fields, and potential use cases.
        
        **2. Conciseness (1-10)** - Measures the efficiency of the dataset description in conveying necessary information without redundancy.
        
        A high score indicates that the description is succinct, avoiding unnecessary details while employing semantic types (e.g., categories, entities) to streamline communication.
        
        **3. Readability (1-10)** - Evaluates the logical flow and readability of the dataset description.
        
        A high score suggests that the description progresses logically from one section to the next, creating a coherent and integrated narrative that facilitates understanding of the dataset.
        
        **4. Faithfulness (1-10)** - Measures how accurately the description reflects the dataset's actual content and meaning.
        
        A high score indicates that the description correctly represents the dataset's variables, structure, and statistics (e.g., ranges, time spans, units) without hallucinations or factual errors. It should preserve the semantic intent and data characteristics of the original dataset while avoiding exaggeration or omission.
        
        ### Evaluation Steps:
        
        Read the dataset description carefully and identify the main topic and key points. Assign a score for each criteria on a scale of 1 to 10, where 1 is the lowest and 10 is the highest based on the Evaluation Criteria.
        """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Show selected methods
        st.markdown("**Selected Methods:**")
        for method in st.session_state.selected_methods:
            st.markdown(f"✓ {method.upper()}")
        
        st.markdown("---")
        
        if not st.session_state.datasets:
            st.warning("No datasets loaded.")
        else:
            st.success(f"✅ Loaded {len(st.session_state.datasets)} datasets")
            
            # Progress
            progress = calculate_progress(
                st.session_state.datasets, 
                st.session_state.evaluations,
                st.session_state.selected_methods
            )
            st.metric("Progress", f"{progress}%")
            st.progress(progress / 100)
            
            # Quick navigation
            st.markdown("---")
            st.subheader("Quick Navigation")
            
            dataset_names = [d['name'] for d in st.session_state.datasets]
            selected_dataset = st.selectbox(
                "Jump to dataset:",
                options=range(len(dataset_names)),
                format_func=lambda x: f"{x+1}. {dataset_names[x]}",
                index=st.session_state.current_dataset_idx
            )
            
            if selected_dataset != st.session_state.current_dataset_idx:
                st.session_state.current_dataset_idx = selected_dataset
                st.session_state.current_desc_idx = 0
                st.rerun()
            
            selected_desc = st.selectbox(
                "Jump to description:",
                options=range(len(st.session_state.selected_methods)),
                format_func=lambda x: st.session_state.selected_methods[x].upper(),
                index=st.session_state.current_desc_idx
            )
            
            if selected_desc != st.session_state.current_desc_idx:
                st.session_state.current_desc_idx = selected_desc
                st.rerun()
            
            # Export results
            st.markdown("---")
            if st.button("📥 Download Evaluations"):
                csv_data = st.session_state.evaluations.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="evaluations.csv",
                    mime="text/csv"
                )
    
    # Main content
    if not st.session_state.datasets:
        st.info("No datasets loaded with the selected methods.")
        return
    
    # Get current dataset and description
    current_dataset = st.session_state.datasets[st.session_state.current_dataset_idx]
    available_descriptions = [d for d in st.session_state.selected_methods if d in current_dataset['descriptions']]
    
    if not available_descriptions:
        st.error(f"No description files found for {current_dataset['name']}")
        return
    
    if st.session_state.current_desc_idx >= len(available_descriptions):
        st.session_state.current_desc_idx = 0
    
    current_desc_type = available_descriptions[st.session_state.current_desc_idx]
    current_description = current_dataset['descriptions'][current_desc_type]
    
    # Initialize scores for this description
    saved_scores = get_evaluation_scores(
        st.session_state.evaluations,
        current_dataset['name'],
        current_desc_type
    )
    initialize_scores_for_description(current_dataset['name'], current_desc_type, saved_scores)
    
    # Progress indicator
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.markdown(f'<div class="sub-header">Dataset {st.session_state.current_dataset_idx + 1} of {len(st.session_state.datasets)}: <b>{current_dataset["name"]}</b></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="sub-header">Description {st.session_state.current_desc_idx + 1} of {len(available_descriptions)}: <b>{current_desc_type.upper()}</b></div>', unsafe_allow_html=True)
    with col3:
        if is_description_complete(current_dataset['name'], current_desc_type, st.session_state.evaluations):
            st.success("✅ Complete")
        
        # Show save status
        if st.session_state.last_saved:
            st.caption(f"💾 {st.session_state.last_saved}")
    
    # Two column layout
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.subheader("📋 Reference Data")
        
        # Display dataset title if available
        if current_dataset.get('title'):
            st.markdown(f'<div class="title-box">📌 <strong>Dataset Title:</strong> {current_dataset["title"]}</div>', unsafe_allow_html=True)
        
        # View mode tabs
        tab1, tab2 = st.tabs(["📊 Table Data", "📈 Statistics"])
        
        with tab1:
            # Display CSV - limit rows to avoid memory issues
            if current_dataset['csv_files']:
                csv_file = current_dataset['csv_files'][0]
                csv_path = os.path.join(current_dataset['path'], csv_file)
                
                try:
                    # Read only first 5000 rows to avoid MessageSizeError
                    df = pd.read_csv(csv_path, nrows=5000)
                    
                    # Try to count total rows efficiently
                    try:
                        with open(csv_path, 'r') as f:
                            total_rows = sum(1 for _ in f) - 1  # minus header
                    except:
                        total_rows = len(df)
                    
                    st.dataframe(df, use_container_width=True, height=600)
                    
                    if total_rows > 5000:
                        st.caption(f"Showing: {csv_file} (first 5,000 of {total_rows:,} rows × {len(df.columns)} columns)")
                    else:
                        st.caption(f"Showing: {csv_file} ({total_rows:,} rows × {len(df.columns)} columns)")
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")
        
        with tab2:
            # Display statistics
            if 'stats' in current_dataset:
                st.text_area(
                    "Data Profiler Statistics",
                    value=current_dataset['stats'],
                    height=600,
                    disabled=True
                )
            else:
                st.info("No statistics file found (data_profiler.txt)")
        
        # Evaluation Criteria
        st.markdown("---")
        st.markdown('<div class="criteria-box">', unsafe_allow_html=True)
        st.markdown("### 📖 Evaluation Criteria (Quick Reference)")
        st.markdown("**Completeness:** Coverage of scope, statistics, and applications")
        st.markdown("**Conciseness:** Efficiency without redundancy")
        st.markdown("**Readability:** Logical flow and coherent narrative")
        st.markdown("**Faithfulness:** Accuracy reflecting dataset content without errors")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Examples
        with st.expander("📚 View Example Evaluations"):
            st.markdown("""
            **Example 1:**
            
            *Description:* The dataset provides information on alcohol-impaired driving deaths and occupant deaths across various states in the United States. It includes data for 51 states, detailing the number of alcohol-impaired driving deaths and occupant deaths, with values ranging from 0 to 3723 and 0 to 10406, respectively. Each entry also contains the state abbreviation and its geographical coordinates. The dataset is structured with categorical and numerical data types, focusing on traffic safety and casualty statistics. Key attributes include state names, death counts, and location coordinates, making it a valuable resource for analyzing traffic safety trends and issues related to impaired driving.
            
            **Scores:** Completeness: 7, Conciseness: 9, Readability: 9, Faithfulness: 8
            
            ---
            
            **Example 2:**
            
            *Description:* The dataset provides a comprehensive overview of traffic safety statistics across various states in the United States, specifically focusing on alcohol-impaired driving deaths and occupant deaths. It includes data from 51 unique states, represented by their two-letter postal abbreviations, such as MA (Massachusetts), SD (South Dakota), AK (Alaska), MS (Mississippi), and ME (Maine). Each entry in the dataset captures critical information regarding the number of alcohol-impaired driving deaths and the total occupant deaths resulting from traffic incidents.
            
            The column "Alcohol-Impaired Driving Deaths" is represented as an integer, indicating the number of fatalities attributed to alcohol impairment while driving. The dataset reveals a range of values, with the highest recorded number being 2367 deaths in Mississippi, highlighting the severity of the issue in certain regions. In contrast, states like Alaska report significantly lower figures, with only 205 alcohol-impaired driving deaths.
            
            The "Occupant Deaths" column also consists of integer values, representing the total number of deaths among vehicle occupants, regardless of the cause. This data spans from 0 to 10406, with Mississippi again showing the highest number of occupant deaths at 6100, which raises concerns about overall traffic safety in the state.
            
            Additionally, the dataset includes a "Location" column that provides geographical coordinates for each state, enhancing the spatial understanding of the data. The coordinates are formatted as latitude and longitude pairs, allowing for potential mapping and geographical analysis of traffic safety trends.
            
            Overall, this dataset serves as a valuable resource for researchers, policymakers, and public safety advocates aiming to understand and address the impact of alcohol on driving safety across different states. It highlights the need for targeted interventions and policies to reduce alcohol-impaired driving incidents and improve occupant safety on the roads.
            
            **Scores:** Completeness: 8, Conciseness: 7, Readability: 8, Faithfulness: 9
            """)
    
    with right_col:
        st.subheader("📝 Description to Evaluate")
        
        # Display description
        st.markdown(current_description)
        
        # Evaluation form
        st.markdown("### ⭐ Rate this Description")
        
        for criterion, description in CRITERIA.items():
            st.markdown(f"**{description}**")
            
            score_key = get_current_score_key(current_dataset['name'], current_desc_type, criterion)
            
            col_slider, col_num = st.columns([4, 1])
            
            with col_slider:
                st.slider(
                    f"{criterion}_slider",
                    min_value=1,
                    max_value=10,
                    key=score_key,
                    label_visibility="collapsed",
                    on_change=lambda: setattr(st.session_state, 'unsaved_changes', True)
                )
            
            with col_num:
                # Display current value (read-only)
                st.markdown(f"<div style='padding: 8px; text-align: center; font-size: 1.2rem; font-weight: bold; border: 1px solid #ddd; border-radius: 4px;'>{st.session_state[score_key]}</div>", unsafe_allow_html=True)
            
            st.caption("1 (Lowest) ← → 10 (Highest)")
            st.markdown("---")
        
        # Comment box
        st.markdown("### 💬 Comments (Optional)")
        comment_key = get_current_comment_key(current_dataset['name'], current_desc_type)
        st.text_area(
            "Add any comments or notes about this description:",
            value=st.session_state.get(comment_key, ''),
            key=comment_key,
            height=100,
            placeholder="e.g., Missing temporal coverage info, contains factual errors in column names...",
            on_change=lambda: setattr(st.session_state, 'unsaved_changes', True)
        )
        
        # Save button
        col_save1, col_save2 = st.columns([1, 1])
        with col_save1:
            if st.button("💾 Save Scores", type="primary", use_container_width=True):
                # Save current scores to DataFrame
                st.session_state.evaluations = save_current_scores(
                    st.session_state.evaluations,
                    current_dataset['name'],
                    current_dataset['path'],
                    current_desc_type
                )
                # Save to CSV file
                save_evaluations(st.session_state.evaluations)
                st.session_state.last_saved = datetime.now().strftime("%H:%M:%S")
                st.session_state.unsaved_changes = False
                st.success("✅ Saved successfully!")
                st.rerun()
        
        with col_save2:
            if st.session_state.unsaved_changes:
                st.warning("⚠️ Unsaved changes")
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Previous", use_container_width=True):
            if st.session_state.current_desc_idx > 0:
                st.session_state.current_desc_idx -= 1
            elif st.session_state.current_dataset_idx > 0:
                st.session_state.current_dataset_idx -= 1
                st.session_state.current_desc_idx = len(st.session_state.selected_methods) - 1
            st.rerun()
    
    with col2:
        # Show completion status
        total_items = len(st.session_state.datasets) * len(st.session_state.selected_methods)
        current_position = st.session_state.current_dataset_idx * len(st.session_state.selected_methods) + st.session_state.current_desc_idx + 1
        st.info(f"Position: {current_position} / {total_items}")
    
    with col3:
        is_last_dataset = st.session_state.current_dataset_idx == len(st.session_state.datasets) - 1
        is_last_desc = st.session_state.current_desc_idx == len(available_descriptions) - 1
        
        if st.button("Next ➡️", use_container_width=True, disabled=(is_last_dataset and is_last_desc)):
            if st.session_state.current_desc_idx < len(available_descriptions) - 1:
                st.session_state.current_desc_idx += 1
            elif st.session_state.current_dataset_idx < len(st.session_state.datasets) - 1:
                st.session_state.current_dataset_idx += 1
                st.session_state.current_desc_idx = 0
            st.rerun()

if __name__ == "__main__":
    main()