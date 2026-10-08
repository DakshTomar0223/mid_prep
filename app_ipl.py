import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from src.optimizer import select_dream11_team

st.set_page_config(page_title="CynapticsAI - Dream11 Predictor", layout="wide")

# Ensure required directories exist
os.makedirs("src/model_artifacts", exist_ok=True)
os.makedirs("src/data/processed", exist_ok=True)

# ---------------------------------------------------------
# Load Data Helper
# ---------------------------------------------------------
@st.cache_data
def load_processed_data():
    container_path = "./data/processed/player_containers.json"
    if not os.path.exists(container_path):
        return pd.DataFrame()
    with open(container_path, 'r') as f:
        containers = json.load(f)
        
    records = []
    for c in containers:
        rec = {
            "player_name": c["player_name"],
            "match_date": c["match_date"],
            "venue": c.get("venue", "Unknown"),
            "target_points": float(c.get("target_points", 0.0)),
            "team": c.get("team", "Team A"),
            "role": c.get("role", "AR")
        }
        rec.update(c["features"])
        records.append(rec)
    df = pd.DataFrame(records)
    df['match_date'] = pd.to_datetime(df['match_date'])
    return df

df_all = load_processed_data()

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.title("CynapticsAI Dream11")
page = st.sidebar.radio("Select Interface", ["Interface 1: Product UI", "Interface 2: Model UI"])

# =========================================================
# INTERFACE 1: PRODUCT UI – Your Ultimate Team Selection Tool
# =========================================================
if page == "Interface 1: Product UI":
    st.title("Interface 1: Product UI – Ultimate Team Selection Tool")
    st.markdown("Recommend an optimal 11-player squad for fantasy cricket adhering strictly to Dream11 constraints.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        team1 = st.text_input("Team 1 Name", "Colombo Strikers")
    with col2:
        team2 = st.text_input("Team 2 Name", "Kandy Falcons")
    with col3:
        match_date = st.date_input("Match Date (>= 2024-07-01)", pd.to_datetime("2024-07-18"))
        
    st.info("Strict Rule Enforced: Predictions are generated using 'ProductUI_Model' trained strictly on data prior to 2024-06-30.")

    if st.button("Generate Recommended Dream Team"):
        model_path = "src/model_artifacts/ProductUI_Model.pkl"
        
        if not os.path.exists(model_path):
            st.error(f"Pretrained model file not found at `{model_path}`. Run `python3 src/train_product_model.py` first.")
        elif df_all.empty:
            st.error("No player container data found. Preprocessing incomplete.")
        else:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
            feature_cols = [c for c in df_all.columns if c.startswith("avg_") or c.startswith("max_") or c.startswith("std_")]
            
            # Filter historical records prior to 2024-07-01
            squad_df = df_all[df_all['match_date'] < pd.to_datetime("2024-07-01")].groupby('player_name').last().reset_index()
            
            if len(squad_df) < 11:
                st.error("Not enough players available in historical dataset.")
            else:
                squad_df['predicted_points'] = model.predict(squad_df[feature_cols])
                
                # Assign default teams & roles if missing in raw log
                squad_df['team'] = np.where(np.arange(len(squad_df)) % 2 == 0, team1, team2)
                roles_cycle = ['BAT', 'BOWL', 'AR', 'WK']
                squad_df['role'] = [roles_cycle[i % 4] for i in range(len(squad_df))]
                
                # Solve optimal team
                selected_team = select_dream11_team(squad_df, 'predicted_points')
                
                st.subheader("Recommended 11-Player Dream Team")
                st.dataframe(
                    selected_team[['player_name', 'team', 'role', 'predicted_points', 'avg_pts_last_5']],
                    use_container_width=True
                )
                
                st.subheader("Player Contribution Justifications & Performance Trends")
                for idx, row in selected_team.iterrows():
                    st.write(
                        f"• **{row['player_name']}** ({row['role']} | {row['team']}): "
                        f"Predicted Fantasy Points: **{row['predicted_points']:.1f}**. "
                        f"Recent 5-match average: **{row['avg_pts_last_5']:.1f}** pts. "
                        f"Recommended role contribution: Key {row['role']} pick."
                    )

# =========================================================
# INTERFACE 2: MODEL UI – Dive into Model Performance Analysis
# =========================================================
else:
    st.title("Interface 2: Model UI – Model Performance Analysis")
    st.markdown("Assess model performance across specific training and testing date ranges, evaluate MAE, and save retrained model artifacts.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Training Period")
        train_start = st.date_input("Train Start Date", pd.to_datetime("2000-01-01"))
        train_end = st.date_input("Train End Date", pd.to_datetime("2024-06-30"))
        
    with col2:
        st.subheader("2. Testing Period")
        test_start = st.date_input("Test Start Date", pd.to_datetime("2024-07-01"))
        test_end = st.date_input("Test End Date", pd.to_datetime("2024-09-22"))

    if st.button("Train Model & Evaluate Test Set"):
        if df_all.empty:
            st.error("Dataset empty. Ensure `./data/processed/player_containers.json` exists.")
        else:
            # 1. Filter Train Data
            train_mask = (df_all['match_date'] >= pd.to_datetime(train_start)) & (df_all['match_date'] <= pd.to_datetime(train_end))
            test_mask = (df_all['match_date'] >= pd.to_datetime(test_start)) & (df_all['match_date'] <= pd.to_datetime(test_end))
            
            df_train = df_all[train_mask]
            df_test = df_all[test_mask]
            
            if df_train.empty:
                st.error("No training data found within selected training date range.")
            else:
                feature_cols = [c for c in df_train.columns if c.startswith("avg_") or c.startswith("max_") or c.startswith("std_")]
                X_train, y_train = df_train[feature_cols], df_train['target_points']
                
                # 2. Train AdaBoost Regressor
                model = AdaBoostRegressor(
                    estimator=DecisionTreeRegressor(max_depth=4),
                    n_estimators=100,
                    learning_rate=0.05,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # 3. Save Retrained Model & Training CSV
                str_train_end = train_end.strftime("%Y-%m-%d")
                saved_csv_path = f"src/data/processed/training_data_{str_train_end}.csv"
                saved_model_path = f"src/model_artifacts/model_{str_train_end}.pkl"
                
                df_train.to_csv(saved_csv_path, index=False)
                with open(saved_model_path, 'wb') as f:
                    pickle.dump(model, f)
                    
                st.success(f"Saved dataset: `{saved_csv_path}`")
                st.success(f"Saved model artifact: `{saved_model_path}`")
                
                # 4. Evaluate Test Set Matches
                if df_test.empty:
                    st.warning("No test match records found within selected test date range.")
                else:
                    results = []
                    
                    for m_date, m_group in df_test.groupby('match_date'):
                        if len(m_group) < 11:
                            continue
                            
                        m_group = m_group.copy()
                        m_group['predicted_points'] = model.predict(m_group[feature_cols])
                        
                        # Apply default teams and roles if missing in raw data
                        teams = m_group['team'].unique()
                        t1 = str(teams[0]) if len(teams) > 0 else "Team 1"
                        t2 = str(teams[1]) if len(teams) > 1 else "Team 2"
                        
                        roles_cycle = ['BAT', 'BOWL', 'AR', 'WK']
                        m_group['role'] = [roles_cycle[i % 4] for i in range(len(m_group))]
                        
                        # Get Predicted Best 11 & Actual Dream Team
                        pred_team = select_dream11_team(m_group, 'predicted_points')
                        actual_team = select_dream11_team(m_group, 'target_points')
                        
                        # Formats strictly required by problem statement
                        pred_players_str = ", ".join(pred_team['player_name'].tolist())
                        actual_players_str = ", ".join(actual_team['player_name'].tolist())
                        pred_points_str = ", ".join([f"{p:.1f}" for p in pred_team['predicted_points'].tolist()])
                        
                        actual_pts_actual_team = actual_team['target_points'].sum()
                        actual_pts_pred_team = pred_team['target_points'].sum()
                        
                        # MAE = Total Fantasy Points of Actual Dream Team - Total Fantasy Points of Predicted Team
                        mae = actual_pts_actual_team - actual_pts_pred_team
                        
                        results.append({
                            "Match Date": str(m_date.date()),
                            "Name of Team 1": t1,
                            "Name of Team 2": t2,
                            "Predicted Best 11 Players": pred_players_str,
                            "Dream Team (Best) 11 Players": actual_players_str,
                            "Predicted Points of Each Player": pred_points_str,
                            "MAE": round(mae, 2)
                        })
                    
                    df_results = pd.DataFrame(results)
                    st.subheader("Test Evaluation Results")
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Download CSV Button
                    csv_data = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Evaluation Results CSV",
                        data=csv_data,
                        file_name=f"evaluation_{str_train_end}.csv",
                        mime="text/csv"
                    )