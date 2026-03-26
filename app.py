import streamlit as st
import requests

st.set_page_config(
    page_title="Cricket Outcome Predictor",
    page_icon="🏏",
    layout="centered"
)

st.title("🏏 Cricket Outcome Predictor")
st.markdown("Predict IPL match outcome using toss & team data")

# ── Team & Venue mappings ──────────────────────────
TEAMS = {
    "Chennai Super Kings": 0,
    "Deccan Chargers": 1,
    "Delhi Daredevils": 2,
    "Gujarat Lions": 3,
    "Kings XI Punjab": 4,
    "Kochi Tuskers Kerala": 5,
    "Kolkata Knight Riders": 6,
    "Mumbai Indians": 7,
    "Pune Warriors": 8,
    "Rajasthan Royals": 9,
    "Rising Pune Supergiants": 10,
    "Royal Challengers Bangalore": 11,
    "Sunrisers Hyderabad": 12,
}

VENUES = {
    "Wankhede Stadium": 55,
    "Eden Gardens": 10,
    "M Chinnaswamy Stadium": 23,
    "Feroz Shah Kotla": 12,
    "MA Chidambaram Stadium": 24,
    "Rajiv Gandhi Intl Stadium": 40,
    "Punjab Cricket Association Stadium": 38,
    "Sawai Mansingh Stadium": 45,
    "Other": 0,
}

# ── Input Form ─────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("🏏 Team 1", list(TEAMS.keys()))
    venue = st.selectbox("🏟️ Venue", list(VENUES.keys()))
    toss_decision = st.radio("🪙 Toss Decision", ["Bat", "Field"])

with col2:
    team2 = st.selectbox("🏏 Team 2", list(TEAMS.keys()))
    toss_winner = st.selectbox("🏆 Toss Winner", [team1, team2])

st.markdown("---")

# ── Predict Button ─────────────────────────────────
if st.button("🎯 Predict Match Outcome", use_container_width=True):

    if team1 == team2:
        st.error("Team 1 and Team 2 cannot be the same!")
    else:
        payload = {
            "team1":         TEAMS[team1],
            "team2":         TEAMS[team2],
            "venue":         VENUES[venue],
            "toss_winner":   TEAMS[toss_winner],
            "toss_decision": 0 if toss_decision == "Bat" else 1,
        }

        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json=payload,
                timeout=5
            )
            result = response.json()

            prob    = result["probability"]
            pred    = result["toss_win_match_win"]
            interp  = result["interpretation"]

            # ── Result Display ──────────────────────
            if pred == 1:
                st.success(f"✅ {interp}")
            else:
                st.warning(f"⚠️ {interp}")

            # ── Probability Bar ─────────────────────
            st.markdown("### Win Probability")
            st.progress(float(prob))
            st.markdown(f"**{round(prob * 100, 1)}%** confidence")

            # ── Match Summary ───────────────────────
            st.markdown("### Match Summary")
            st.json({
                "team1":         team1,
                "team2":         team2,
                "venue":         venue,
                "toss_winner":   toss_winner,
                "toss_decision": toss_decision,
                "prediction":    "Toss winner WINS" if pred == 1 else "Toss winner LOSES",
                "confidence":    f"{round(prob * 100, 1)}%"
            })

        except Exception as e:
            st.error(f"API Error: {str(e)}")
            st.info("Make sure docker-compose is running!")