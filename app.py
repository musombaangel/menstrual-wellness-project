import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor


col1, col2 = st.columns([1,1])
with col1:
    page = st.radio(" ", ["Mood Prediction", "Model Performance"], horizontal=True)


# Page 1: Mood prediction and recommendations
if page == "Mood Prediction":

    # Gradient title container
    st.markdown("""
        <div style="
            padding:20px;
            border-radius:15px;
            background: linear-gradient(135deg, #ff9ecb33, #d77cfc33);
            text-align:center;">
            <h1 style="color:#ffb6df;">Menstrual Wellness & Mood Predictor 🌸</h1>
            <p style="color:#ffffffcc;">Predict mood & receive personalised wellness recommendations</p>
        </div>
        <br>
    """, unsafe_allow_html=True)

    @st.cache_resource
    def load_model():
        model = CatBoostRegressor()
        model.load_model("catboost_mood_model.cbm")
        return model

    model = load_model()

    # Existing functions (NOT CHANGED)
    def compute_phase(days_since_period, period_len, cycle_length):
        luteal_start = cycle_length - 14
        if days_since_period <= period_len:
            return "Menstrual"
        if days_since_period >= luteal_start:
            return "Luteal"
        if days_since_period >= luteal_start - 5:
            return "Ovulation"
        else:
            return "Follicular"

    #phase-based recommendations
    def get_recommendations(phase):
        if phase == "Menstrual":
            foods = """
            - Iron-rich foods: spinach, kunde, beef, liver, beans, ndengu  
            - Vitamin C foods: citrus, pawpaw, strawberries  
            - Hydrating foods: watermelon, soups, coconut water  
            """
            exercises = """
            - Light stretching  
            - Yoga (hip opening stretches)  
            - Light walks or light dancing
            - Meditation and deep breathing exercises
            """
            why_recommendation = """
            - Iron-rich foods support the body during blood loss, while Vitamin C increases iron absorption. 
            - Hydrating foods help prevent headaches and fatigue common in this phase.
            - Exercises are gentle because energy levels are low and the uterus is contracting, so light movement reduces cramps and improves circulation without strain.
            """

        elif phase == "Follicular":
            foods = """
            - High-energy carbs: sweet potatoes, rice, chapati, oats  
            - Lean proteins: chicken, eggs, beans  
            - B vitamin foods: maize, nduma, njahi, spinach
            - Healthy fats: avocados, nuts, seeds
            """
            exercises = """
            - Cardio workouts: running, swimming, dance
            - Cycling  
            - Strength training (lower weights, higher reps)
            - Pilates
            """
            why_recommendation = """
            - Carbs support rising energy, lean proteins and B vitamins help follicle development, and healthy fats assist hormone production.
            - Exercises are more energetic because estrogen is increasing, improving endurance, mood, and strength tolerance.
            """

        elif phase == "Ovulation":
            foods = """
            - Anti-oxidant foods: berries, greens, ginger, hibiscus tea  
            - Protein-rich foods: fish, chicken, eggs, legumes
            - Zinc-rich foods: omena, seafood  
            - Healthy fats: avocado, nuts, olive oil  
            """
            exercises = """
            - Higher-intensity workouts  
            - HIIT  
            - Strength training (peak performance)
            - Cardio sessions: running, swimming, cycling 
            """
            why_recommendation = """
            - Antioxidants reduce oxidative stress from ovulation, proteins support energy and muscle recovery, zinc aids hormone regulation, and healthy fats prepare the body for progesterone production.
            - Feel free to take up more intense exercise because this is the phase of peak strength, endurance, and coordination driven by high estrogen and LH levels.
            """

        else:  # Luteal Phase
            foods = """
            - Complex carbs: sweet potatoes, ugali, brown rice  
            - Magnesium-rich foods: kunde, njahi, spinach, omena  
            - Vitamin B6 foods: matoke, potatoes, bananas  
            """
            exercises = """
            - Light cardio: walking, light jogging, swimming, light dancing
            - Light yoga  
            - Low-intensity workouts  
            - Stretching and relaxation exercises
            """
            why_recommendation = """
            - Complex carbs help regulate PMS irritability, magnesium reduces cramps, and Vitamin B6 supports serotonin and dopamine levels.
            - Lighter exercises are recommended to accommodate rising progesterone, increased inflammation, and lower energy as the body prepares for menstruation.
            """

        return foods, exercises, why_recommendation


    st.header("📝 Enter Your Tracking Information")

    days_since = st.number_input("Days Since Last Period Started", 0, 40, 10)
    period_length = st.number_input("Period Length (days)", 1, 10, 5)
    cycle_length = st.number_input("Cycle Length (days)", 21, 40, 28)

    sleep = st.selectbox("Average Sleep Hours This Week",
        ["Less than 4 hours", "4-5 hours", "6-7 hours", "6-8 hours", "8-9 hours"]
    )
    workout = st.selectbox("Average Workout Duration This Week",
        ["Less than 2 hours", "2-4 hours", "5-7 hours", "8-10 hours", "More than 10 hours"]
    )
    age = st.selectbox("Your Age Range",
        ["11-20", "21-30", "31-40", "41-50", "51+"]
    )

    # Symptom checkboxes
    st.markdown("### What symptoms are you experiencing?")
    col1, col2, col3 = st.columns(3)
    with col1:
        headaches = st.checkbox("Headaches")
        bloating = st.checkbox("Bloating")
        mood_swings = st.checkbox("Mood Swings")
    with col2:
        sex_drive = st.checkbox("Increased Sex Drive")
        cravings = st.checkbox("Cravings")
        irritability = st.checkbox("Irritability")
    with col3:
        fatigue = st.checkbox("Fatigue")

    # Create input df
    def create_input_df():
        data = {
            "Cycle_length": [cycle_length],
            "Sleep_" + sleep: [1],
            "Workout_" + workout: [1],
            "Age_" + age: [1],

            "Headaches": [1 if headaches else 0],
            "Bloating": [1 if bloating else 0],
            "Mood_swings": [1 if mood_swings else 0],
            "Increased_sex_drive": [1 if sex_drive else 0],
            "Cravings": [1 if cravings else 0],
            "Irritability": [1 if irritability else 0],
            "Fatigue": [1 if fatigue else 0],
        }

        df = pd.DataFrame(data)
        for col in model.feature_names_:
            if col not in df.columns:
                df[col] = 0
        return df[model.feature_names_]

    if st.button("Predict Mood / Energy"):
        input_df = create_input_df()
        predicted_mood = model.predict(input_df)[0]

        st.subheader("🔮 Predicted Mood / Energy Level")
        st.metric("Mood Score", f"{predicted_mood:.1f} / 10")

        if predicted_mood >= 7:
            mood_desc = "High Energy / Positive Mood 😊"
        elif 4 <= predicted_mood < 7:
            mood_desc = "Moderate / Neutral Mood 😐"
        else:
            mood_desc = "Low Energy / Negative Mood 😞"

        st.write(f"You are likely to feel: **{mood_desc}**")

        phase = compute_phase(days_since, period_length, cycle_length)
        st.subheader(f"🩸 Detected Phase: **{phase}**")

        foods, exercises, why = get_recommendations(phase)

        st.header("🍽️ Wellness Recommendations")

        with st.expander("🥗 Recommended Foods"):
            st.write(foods)

        with st.expander("🏃 Recommended Exercises"):
            st.write(exercises)

        with st.expander("💡 Why these recommendations?"):
            st.write(why)



#Page 2: Model performance

elif page == "Model Performance":

    import plotly.graph_objects as go
    st.title("Model Performance Comparison")

    import pandas as pd
    import streamlit as st
    import plotly.graph_objects as go

    #performance data
    data = {
        "Model": [
            "Decision Tree", 
            "Random Forest", 
            "XGBoost", 
            "Ensemble", 
            "CatBoost"
        ],
        "R_squared": [0.637, 0.63, 0.62, 0.59, 0.64],
        "RMSE": [1.60, 1.60, 1.55, 1.70, 1.48]
    }

    df = pd.DataFrame(data)

    st.header("📊 Model Performance Comparison")

    # --- CUTE DARK THEME BAR CHART WITH COLORS ---
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["Model"],
        y=df["R_squared"],
        name="R² Score",
        marker_color="#ff7bbf"   # soft pink
    ))

    fig.add_trace(go.Bar(
        x=df["Model"],
        y=df["RMSE"],
        name="RMSE",
        marker_color="#7bbaff"   # soft pastel blue
    ))

    fig.update_layout(
        barmode='group',
        template='plotly_dark',
        title="Model Performance (R² vs RMSE)",
        xaxis_title="Model",
        yaxis_title="Score",
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)"
        ),
        font=dict(
            family="sans-serif",
            size=14,
            color="white"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show the raw DataFrame too
    st.subheader("📄 Performance Table")
    st.dataframe(df, use_container_width=True)
