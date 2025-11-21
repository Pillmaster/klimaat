import streamlit as st
import pandas as pd
import requests
from io import StringIO
import datetime
import re
import plotly.express as px
import numpy as np
import calendar 

# 1. Configuratie en constanten
# ===============================================================================

# Coördinaten van Malmån (gebruikt in het originele script)
LAT = 62.9977
LON = 17.0811

# Volledige historische periode voor de Open-Meteo ERA5 Archive API
START_DATE_FULL = "1940-01-01"
# BIJGEWERKT: Nieuwe einddatum 30-06-2025
END_DATE_FULL = "2025-06-30" 
TIMEZONE = 'Europe/Stockholm'
STATION_NAME_HISTORICAL = f"Open-Meteo ERA5 ({START_DATE_FULL[:4]}-{END_DATE_FULL[:4]})"

# Kolomnamen voor extreme analyse
TEMP_COLUMNS = ['Temp_High_C', 'Temp_Low_C', 'Temp_Avg_C'] 

# Helper voor maandnamen
MONTH_NAMES = {i: datetime.date(2000, i, 1).strftime('%B') for i in range(1, 13)}

# 2. Functies voor Data (Loading & Processing)
# ===============================================================================

@st.cache_data(ttl=86400, show_spinner=f"Laden historische weerdata ({START_DATE_FULL[:4]} - {END_DATE_FULL[:4]}). Dit kan even duren...") 
def load_full_history_data(start_date_str, end_date_str):
    """Haalt de volledige reeks dagelijkse historische data op via de Open-Meteo ERA5 Archive API."""
    api_url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date_str, 
        "end_date": end_date_str,     
        # OPGELOST: snow_depth verwijderd omdat dit een API-fout veroorzaakte.
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,snowfall_sum", 
        "timezone": TIMEZONE,
        "format": "csv"
    }
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status() # Roept HTTPError op voor 4xx/5xx statuscodes
        
        csv_data = response.text.split('\n', 3)[3] 
        df_hist = pd.read_csv(StringIO(csv_data))
        
        df_hist.columns = df_hist.columns.str.strip()
        
        df_hist = df_hist.rename(columns={
            'time': 'Date',
            'temperature_2m_max (°C)': 'Temp_High_C', 
            'temperature_2m_min (°C)': 'Temp_Low_C',  
            'temperature_2m_mean (°C)': 'Temp_Avg_C',
            'precipitation_sum (mm)': 'Precip_Sum_mm',
            # snow_depth (m) is niet beschikbaar in deze werkende call
            'snowfall_sum (cm)': 'Snowfall_Sum_cm'  # Sneeuwval in cm
        })
        
        df_hist['Date'] = pd.to_datetime(df_hist['Date'])
        df_hist['Station Naam'] = STATION_NAME_HISTORICAL
        df_hist = df_hist.set_index('Date')
        
        success_message = f"✅ Volledige historische data ({len(df_hist)} dagen) succesvol geladen via Open-Meteo (ERA5)."
        return df_hist, ('success', success_message)
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
             error_message = f"❌ Kon historische data niet ophalen (API-fout: Te veel verzoeken (429)). Probeer het over een minuut opnieuw. De API-limiet is bereikt."
        else:
             error_message = f"❌ Kon historische data niet ophalen (API-fout: {e})."
        return pd.DataFrame(), ('error', error_message)
    except Exception as e:
        error_message = f"⚠️ Algemene fout bij het verwerken van data. Fout: {e}"
        return pd.DataFrame(), ('warning', error_message)

def safe_format_temp(x):
    """Formats numeric value x to 'X.X °C', returns empty string for NaN or non-numeric types."""
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.1f} °C"
    except (ValueError, TypeError):
        return "" 
        
def safe_format_precip(x):
    """Formats numeric value x to 'X.X mm', returns empty string for NaN or non-numeric types."""
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.1f} mm"
    except (ValueError, TypeError):
        return "" 

def safe_format_snowfall_cm(x):
    """Formats numeric value x (in centimeters) to 'X.X cm', returns empty string for NaN or non-numeric types."""
    if pd.isna(x):
        return ""
    try:
        # Dit is al in centimeters, direct formatteren
        return f"{float(x):.1f} cm"
    except (ValueError, TypeError):
        return "" 

def find_consecutive_periods(df_filtered, min_days, value_column):
    """
    Vindt en retourneert aaneengesloten periodes op basis van een filter.
    """
    if df_filtered.empty:
        return pd.DataFrame(), 0

    df_groups = df_filtered.copy()
    
    if not isinstance(df_groups.index, pd.DatetimeIndex):
        return pd.DataFrame(), 0 
    
    df_groups = df_groups.reset_index().sort_values('Date') 

    df_groups['new_period'] = (df_groups['Date'].diff().dt.days.fillna(0) > 1).astype(int) 
    df_groups['group_id'] = df_groups['new_period'].cumsum()
    
    periods = df_groups.groupby('group_id').agg(
        StartDatum=('Date', 'min'),
        EindDatum=('Date', 'max'),
        Duur=('Date', 'size'),
        Gemiddelde_Waarde_Periode=(value_column, 'mean')
    ).reset_index(drop=True)
        
    periods['Station Naam'] = STATION_NAME_HISTORICAL
        
    periods = periods[periods['Duur'] >= min_days]

    if periods.empty:
        return pd.DataFrame(), 0

    periods['StartDatum'] = periods['StartDatum'].dt.strftime('%d-%m-%Y')
    periods['EindDatum'] = periods['EindDatum'].dt.strftime('%d-%m-%Y')
    
    if value_column in TEMP_COLUMNS:
        periods['Gemiddelde_Waarde_Periode'] = periods['Gemiddelde_Waarde_Periode'].map(safe_format_temp)
        periods = periods.rename(columns={'Gemiddelde_Waarde_Periode': 'Gemiddelde Temp Periode'})
    elif value_column == 'Precip_Sum_mm':
        periods['Gemiddelde_Waarde_Periode'] = periods['Gemiddelde_Waarde_Periode'].map(safe_format_precip)
        periods = periods.rename(columns={'Gemiddelde_Waarde_Periode': 'Gemiddelde Neerslag Periode'})

    total_periods = len(periods)
    
    return periods.reset_index(drop=True), total_periods

def find_extreme_days(df_daily_summary, top_n=10):
    """
    Vindt de warmste, koudste, natste en meest extreme dagen uit de dagelijkse samenvatting (voor Tab Extremen).
    """
    if df_daily_summary.empty:
        return {}

    df_analysis = df_daily_summary.copy()
    df_analysis['Temp_Range_C'] = df_analysis['Temp_High_C'] - df_analysis['Temp_Low_C']

    results = {}

    # Configuratie voor de Algemene Extremen tab (Snow_Depth_m is verwijderd)
    extremes_config = {
        'hoogste_max_temp': ('Temp_High_C', False, 'Max Temp (°C)', 'Warmste Dagen (Max Temp)'), 
        'laagste_min_temp': ('Temp_Low_C', True, 'Min Temp (°C)', 'Koudste Nachten (Min Temp)'),    
        'hoogste_gem_temp': ('Temp_Avg_C', False, 'Gem Temp (°C)', 'Warmste Dagen (Gem. Temp)'),
        'laagste_max_temp': ('Temp_High_C', True, 'Max Temp (°C)', 'Koudste Dagen (Max Temp)'),   
        'hoogste_neerslag': ('Precip_Sum_mm', False, 'Neerslag (mm)', 'Natste Dagen (Neerslag)'), 
        'grootste_range': ('Temp_Range_C', False, 'Range (°C)', 'Grootste Dagelijkse Range'),
        'hoogste_sneeuwval': ('Snowfall_Sum_cm', False, 'Sneeuwval (cm)', 'Dagen met Hoogste Sneeuwval') 
    }

    for key, (column, ascending, display_col, title) in extremes_config.items():
        
        top_days = df_analysis.sort_values(by=column, ascending=ascending).head(top_n)

        df_display = top_days.reset_index().copy()

        if 'Date' in df_display.columns:
             df_display['Datum'] = df_display['Date'].dt.strftime('%d-%m-%Y')
             df_display = df_display.drop(columns=['Date', 'Station Naam']) 
        
        rename_dict = {
            'Temp_High_C': 'Max Temp (°C)', 
            'Temp_Low_C': 'Min Temp (°C)',
            'Temp_Avg_C': 'Gem Temp (°C)',
            'Temp_Range_C': 'Range (°C)',
            'Precip_Sum_mm': 'Neerslag (mm)',
            'Snowfall_Sum_cm': 'Sneeuwval (cm)' 
        }
        
        df_display = df_display.rename(columns=rename_dict)
        
        temp_display_cols = ['Max Temp (°C)', 'Min Temp (°C)', 'Gem Temp (°C)', 'Range (°C)']
        for col_name in temp_display_cols:
            if col_name in df_display.columns:
                 df_display[col_name] = df_display[col_name].map(safe_format_temp)
        
        if 'Neerslag (mm)' in df_display.columns:
             df_display['Neerslag (mm)'] = df_display['Neerslag (mm)'].map(safe_format_precip)
            
        if 'Sneeuwval (cm)' in df_display.columns: 
             df_display['Sneeuwval (cm)'] = df_display['Sneeuwval (cm)'].map(safe_format_snowfall_cm)

        # Volgorde aangepast, Sneeuwdek is verwijderd
        final_cols_order = ['Datum', 'Max Temp (°C)', 'Min Temp (°C)', 'Gem Temp (°C)', 'Neerslag (mm)', 'Sneeuwval (cm)', 'Range (°C)'] 
        
        df_final_display = df_display[[c for c in final_cols_order if c in df_display.columns]].copy()
        
        results[key] = (title, df_final_display.set_index('Datum'))

    return results

def calculate_seasonal_hellmann(df):
    """
    Berekent het Hellmann Getal per seizoen (Juli t/m Juni).
    """
    df_hellmann = df[df['Temp_Avg_C'] <= 0.0].copy()
    
    if df_hellmann.empty:
        return pd.DataFrame()
        
    df_hellmann['Hellmann_Bijdrage'] = df_hellmann['Temp_Avg_C'].abs()

    df_hellmann['Datum'] = df_hellmann.index
    
    df_hellmann['Seizoen Jaar'] = np.where(
        df_hellmann['Datum'].dt.month >= 7,
        df_hellmann['Datum'].dt.year + 1,
        df_hellmann['Datum'].dt.year
    )
    
    df_seasonal = df_hellmann.groupby('Seizoen Jaar')['Hellmann_Bijdrage'].sum().reset_index()
    df_seasonal = df_seasonal.rename(columns={'Hellmann_Bijdrage': 'Hellmann Getal'})
    
    # BELANGRIJK: Laat 'Hellmann Getal' als float voor sorteerbaarheid
    df_seasonal['Seizoen Jaar'] = df_seasonal['Seizoen Jaar'].astype(str)
    
    return df_seasonal
    
def find_daily_extremes(df_full, target_month, target_day, top_n=10):
    """
    Vindt de warmste, koudste, natste dagen voor een specifieke dag/maand over alle jaren (voor Tab Historische Dag).
    """
    # Filteren op dag en maand
    if target_month == 2 and target_day == 29:
        df_filtered = df_full[
            (df_full.index.month == 2) & 
            (df_full.index.day == 29)
        ].copy()
    else:
        df_filtered = df_full[
            (df_full.index.month == target_month) & 
            (df_full.index.day == target_day)
        ].copy()
    
    if df_filtered.empty:
        return {}

    # Bereken de dagelijkse gang (range)
    df_filtered['Temp_Range_C'] = df_filtered['Temp_High_C'] - df_filtered['Temp_Low_C']
    
    results = {}

    # Configuratie van de Extremen (Snow_Depth_m is verwijderd, nu 9 Extremen)
    extremes_config = {
        # Warmte Extremen
        'hoogste_max_temp': ('Temp_High_C', False, 'Max Temp (°C)', '1. Warmste Dag (Max Temp)'), 
        'hoogste_min_temp': ('Temp_Low_C', False, 'Min Temp (°C)', '2. Warmste Nacht (Min Temp)'),    
        'hoogste_gem_temp': ('Temp_Avg_C', False, 'Gem Temp (°C)', '3. Hoogste Gemiddelde Temp'),
        
        # Koude Extremen
        'laagste_min_temp': ('Temp_Low_C', True, 'Min Temp (°C)', '4. Koudste Nacht (Min Temp)'),    
        'laagste_max_temp': ('Temp_High_C', True, 'Max Temp (°C)', '5. Koudste Dag (Max Temp)'),
        'laagste_gem_temp': ('Temp_Avg_C', True, 'Gem Temp (°C)', '6. Laagste Gemiddelde Temp'),
        
        # Overige Extremen
        'grootste_range': ('Temp_Range_C', False, 'Dagelijkse Gang (°C)', '7. Grootste Dagelijkse Gang'),
        'hoogste_neerslag': ('Precip_Sum_mm', False, 'Neerslag (mm)', '8. Hoogste Neerslag'),
        'hoogste_sneeuwval': ('Snowfall_Sum_cm', False, 'Sneeuwval (cm)', '9. Hoogste Sneeuwval')
    }

    for key, (column, ascending, display_col, title) in extremes_config.items():
        
        # Sorteer op de geselecteerde kolom en neem de top N
        top_days = df_filtered.sort_values(by=column, ascending=ascending).head(top_n)

        df_display = top_days.reset_index().copy()

        if 'Date' in df_display.columns:
             # Toon alleen het jaar van de extreme dag
             df_display['Jaar'] = df_display['Date'].dt.year
             df_display = df_display.drop(columns=['Date', 'Station Naam']) 
        
        rename_dict = {
            'Temp_High_C': 'Max Temp (°C)', 
            'Temp_Low_C': 'Min Temp (°C)',
            'Temp_Avg_C': 'Gem Temp (°C)',
            'Temp_Range_C': 'Dagelijkse Gang (°C)', 
            'Precip_Sum_mm': 'Neerslag (mm)',
            'Snowfall_Sum_cm': 'Sneeuwval (cm)' 
        }
        
        df_display = df_display.rename(columns=rename_dict)
        
        # Formatteren met de helper functie
        temp_display_cols = ['Max Temp (°C)', 'Min Temp (°C)', 'Gem Temp (°C)', 'Dagelijkse Gang (°C)']
        for col_name in temp_display_cols:
            if col_name in df_display.columns:
                 df_display[col_name] = df_display[col_name].map(safe_format_temp)
        
        if 'Neerslag (mm)' in df_display.columns:
             df_display['Neerslag (mm)'] = df_display['Neerslag (mm)'].map(safe_format_precip)
            
        if 'Sneeuwval (cm)' in df_display.columns: 
             df_display['Sneeuwval (cm)'] = df_display['Sneeuwval (cm)'].map(safe_format_snowfall_cm)

        # Bepaal de definitieve kolomvolgorde
        final_cols_order = ['Jaar'] + [
            'Max Temp (°C)', 
            'Min Temp (°C)', 
            'Gem Temp (°C)', 
            'Dagelijkse Gang (°C)', 
            'Neerslag (mm)',
            'Sneeuwval (cm)'
        ]
        
        df_final_display = df_display[[c for c in final_cols_order if c in df_display.columns]].copy()
        
        results[key] = (title, df_final_display.set_index('Jaar'))

    return results

def get_last_year_data(df_full):
    """
    Haalt de data van 1 jaar geleden op en het historisch gemiddelde voor die kalenderdag.
    Retourneert (datum_vorig_jaar, data_vorig_jaar_df, historisch_gem_df).
    
    BIJGEWERKTE LOGICA: Gebruikt de huidige datum (today) - 1 jaar als referentie.
    """
    today = datetime.date.today()
    
    # 1. Bepaal de 'Vorig Jaar' datum: Huidige kalenderdag - 1 jaar.
    try:
        date_last_year = today.replace(year=today.year - 1)
    except ValueError:
        # Dit gebeurt alleen op 29 februari in een niet-schrikkeljaar. Val terug op 28 februari.
        if today.month == 2 and today.day == 29:
             date_last_year = today.replace(year=today.year - 1, day=28)
        else:
            # Onwaarschijnlijk, maar veiligheidshalve
            raise

    # 2. Controleer of deze datum in de dataset is 
    # De data index is een DatetimeIndex, dus we converteren de datum naar een string/timestamp
    if str(date_last_year) not in df_full.index:
         # Als de data van een jaar geleden niet in de dataset zit.
         return date_last_year, pd.DataFrame(), pd.DataFrame() 

    # Vorig Jaar Data
    df_last_year = df_full.loc[[str(date_last_year)]].copy()

    # Historisch Gemiddelde
    
    # Voor 29 februari: alleen 29 feb data gebruiken. Anders: alle jaren (behalve vorig jaar)
    # Let op: de analyse voor het historisch gemiddelde moet de oorspronkelijke maand/dag gebruiken
    # om de volledige geschiedenis voor die kalenderdag te omvatten.
    
    # Als de oorspronkelijke datum 29-02 was, maar we terugvielen op 28-02 (in de 'vorig jaar' data):
    # dan moeten we hier de oorspronkelijke datum van de gebruiker (huidige maand/dag) gebruiken voor het historisch gemiddelde.
    # We gebruiken altijd de maand/dag van de huidige datum voor het historisch gemiddelde.
    
    target_month_hist = today.month
    target_day_hist = today.day
    
    if target_month_hist == 2 and target_day_hist == 29:
        df_hist_day = df_full[
            (df_full.index.month == 2) & 
            (df_full.index.day == 29)
        ]
    else:
        df_hist_day = df_full[
            (df_full.index.month == target_month_hist) & 
            (df_full.index.day == target_day_hist)
        ]
    
    # Exclusief het jaar van vorig jaar uit het gemiddelde, indien aanwezig
    df_hist_day = df_hist_day[df_hist_day.index.year != date_last_year.year]
    
    if df_hist_day.empty:
         return date_last_year, df_last_year, pd.DataFrame() 
         
    # Bereken de gemiddelden
    historical_avg = df_hist_day[[
        'Temp_High_C', 
        'Temp_Low_C', 
        'Temp_Avg_C', 
        'Precip_Sum_mm', 
        'Snowfall_Sum_cm'
    ]].mean().to_frame().T
    
    historical_avg.index = ['Historisch Gemiddelde']

    return date_last_year, df_last_year, historical_avg

# 3. Streamlit Applicatie Hoofdsectie
# ===============================================================================

st.set_page_config(
    page_title="Historische Weerdata Zoeker (Open-Meteo 1940-2025)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(f"🌍 Historische Weerdata Zoeker")
st.markdown(f"Analyse van Open-Meteo ERA5 data voor **{LAT}°N, {LON}°E** van **{START_DATE_FULL[:4]}** tot **{END_DATE_FULL[:4]}**.")
st.markdown("---")

# --- Data Laden ---
df_full_history, status = load_full_history_data(START_DATE_FULL, END_DATE_FULL)

status_type, message = status
if status_type == 'success':
    st.sidebar.success(message, icon="📈")
elif status_type == 'error':
    st.sidebar.error(message, icon="❌")
    
if df_full_history.empty:
    st.error("Kan de historische data niet laden. Controleer de sidebar voor de exacte fout. Probeer de app over een minuut opnieuw te laden.")
    st.stop()
    
min_date_hist = df_full_history.index.min().date()
max_date_hist_dt = df_full_history.index.max().date()
max_data_year = max_date_hist_dt.year

# Bepaal het meest recente COMPLETE jaar voor vergelijking (huidig jaar - 1)
MAX_COMPARISON_YEAR = datetime.date.today().year - 1 
if MAX_COMPARISON_YEAR >= max_data_year:
    MAX_COMPARISON_YEAR = max_data_year - 1

# --- Hoofdnavigatie ---
# BIJGEWERKT: Nieuw tabblad 'tab_charts' toegevoegd
tab_last_year, tab_history, tab_extremes, tab_hellmann, tab_daily_extremes, tab_charts = st.tabs([
    "🕰️ Vorig Jaar", 
    "🔍 Historische Zoeker", 
    "🏆 Extremen (Top 10)", 
    "🧊 Hellmann Getal (Seizoenen)", 
    "📆 Historische Dag",
    "📈 Grafieken (Jaar/Maand)" # <-- Nieuwe tab-titel
]) 


# -------------------------------------------------------------------
# --- NIEUW: Tab 0: Vorig Jaar ---
# -------------------------------------------------------------------

with tab_last_year:
    # GEWIJZIGD: Nieuwe titel
    st.header("🕰️ Het weer precies een jaar geleden")
    # Aangepaste info-tekst omdat er geen vergelijking meer is
    st.info("Toont de weersdata van de huidige kalenderdag - 1 jaar, indien beschikbaar in de dataset.")
    
    # df_historical_avg wordt nog steeds berekend, maar wordt niet gebruikt
    date_last_year, df_last_year, df_historical_avg = get_last_year_data(df_full_history)

    st.subheader(f"Geselecteerde Datum: **{date_last_year.strftime('%d %B %Y')}**")
    st.markdown("---")
    
    if df_last_year.empty:
        st.warning(f"Geen data beschikbaar voor **{date_last_year.strftime('%d %B %Y')}**. Dit komt doordat de data van één jaar geleden nog niet in de dataset (t/m {max_date_hist_dt.strftime('%d-%m-%Y')}) is opgenomen, of door een schrikkeldag (29 februari) waarbij de data ontbreekt.")
    else:
        
        # Gegevens van vorig jaar
        last_year_data = df_last_year.iloc[0]
        
        # Presentatie in kolommen (Metric stijl)
        
        st.subheader("Overzicht in Cijfers")
        
        col1, col2, col3 = st.columns(3)
        col4, col5, col_empty = st.columns(3)

        # 1. Gemiddelde Temperatuur
        with col1:
            ly_temp_avg = last_year_data['Temp_Avg_C']
            st.metric(
                label="Gemiddelde Temperatuur",
                # GEWIJZIGD: Emoticon toegevoegd, delta verwijderd
                value=f"🌡️ {safe_format_temp(ly_temp_avg)}"
            )

        # 2. Maximale Temperatuur
        with col2:
            ly_temp_high = last_year_data['Temp_High_C']
            st.metric(
                label="Maximale Temperatuur",
                # GEWIJZIGD: Emoticon toegevoegd, delta verwijderd
                value=f"☀️ {safe_format_temp(ly_temp_high)}"
            )

        # 3. Minimale Temperatuur
        with col3:
            ly_temp_low = last_year_data['Temp_Low_C']
            st.metric(
                label="Minimale Temperatuur",
                # GEWIJZIGD: Emoticon toegevoegd, delta verwijderd
                value=f"🥶 {safe_format_temp(ly_temp_low)}"
            )
            
        # 4. Neerslag
        with col4:
            ly_precip = last_year_data['Precip_Sum_mm']
            st.metric(
                label="Neerslag",
                # GEWIJZIGD: Emoticon toegevoegd, delta verwijderd
                value=f"🌧️ {safe_format_precip(ly_precip)}"
            )
            
        # 5. Sneeuwval
        with col5:
            ly_snow = last_year_data['Snowfall_Sum_cm']
            st.metric(
                label="Sneeuwval",
                # GEWIJZIGD: Emoticon toegevoegd, delta verwijderd
                value=f"❄️ {safe_format_snowfall_cm(ly_snow)}"
            )
        
        st.markdown("---")
        
        # VISUALISATIE BLOK VERWIJDERD


# -------------------------------------------------------------------
# --- Tab 1: Historische Zoeker ---
# -------------------------------------------------------------------

with tab_history:
    st.header("Historische Zoeker")
    st.info(f"Doorzoek de volledige dataset van {len(df_full_history)} dagen op basis van datum of waarde.")

    # 1. Filtermodus en Drempel 
    st.markdown(f"**1. Filtertype & Drempel**")
    
    filter_mode = st.radio(
        "Zoeken op:", 
        ["Losse Dagen (Waarde)", "Aaneengesloten Periode", "Overzicht per Jaar/Maand", "Hellmann Getal Berekenen"], 
        key="filter_mode_hist_new",
        horizontal=True
    )

    # Snow_Depth_m is verwijderd
    temp_type_options = ["Max Temp (Temp_High_C)", "Min Temp (Temp_Low_C)", "Gemiddelde Temp (Temp_Avg_C)", "Neerslag (Precip_Sum_mm)", "Sneeuwval (Snowfall_Sum_cm)"] 
    selected_temp_type = st.selectbox(
        "Kies de te analyseren variabele:",
        options=temp_type_options,
        key="temp_type_hist_new"
    )
    
    value_column = selected_temp_type.split("(")[-1].replace(")", "")
    
    # Bepaal de eenheid
    if 'Temp' in value_column:
        value_unit = "°C"
        default_temp = 25.0 if 'Temp_High' in value_column else 0.0
        step_val = 0.1
    elif 'Precip' in value_column:
        value_unit = "mm"
        default_temp = 1.0
        step_val = 0.1
    elif 'Snowfall_Sum' in value_column: 
        value_unit = "cm"
        default_temp = 5.0
        step_val = 0.1
        
    comparison = "Lager dan (<=)" if 'Temp_Low' in value_column else "Hoger dan (>=)"
    min_consecutive_days = 3
    group_by_period = "Jaar" 

    if filter_mode != "Hellmann Getal Berekenen":
        col_comp, col_thres, col_days_or_group = st.columns(3)

        with col_comp:
            comparison = st.selectbox(
                "Vergelijking:",
                ["Hoger dan (>=)", "Lager dan (<=)"],
                key="comparison_hist_new"
            )
        
        with col_thres:
            
            # logica voor snow_depth is nu verwijderd, alleen de standaard drempel is nodig
            temp_threshold = st.number_input(
                f"Drempel ({value_unit}):", 
                value=default_temp, 
                step=step_val, 
                key="threshold_hist_new"
            )
            display_threshold = temp_threshold # Gebruik de echte drempel voor weergave
        
        with col_days_or_group:
            if filter_mode == "Aaneengesloten Periode":
                min_consecutive_days = st.number_input(
                    "Min. aaneengesloten dagen:", min_value=2, value=3, step=1, key="min_days_period_hist_new"
                )
            elif filter_mode == "Overzicht per Jaar/Maand":
                group_by_period = st.selectbox(
                    "Groeperen op:", 
                    ["Jaar", "Maand"], 
                    key="group_by_period_hist_new"
                )
    
    # 2. Optionele Periode Groepering/Filter
    df_filtered_time = df_full_history.copy()
    
    if filter_mode == "Overzicht per Jaar/Maand":
        st.markdown(f"**2. Optionele Periodefilters (voor Groepering)**")
        
        if group_by_period == "Maand":
            st.warning(f"Let op: De analyse gaat tot en met het laatste complete jaar ({MAX_COMPARISON_YEAR}).")
            
            month_options = list(MONTH_NAMES.keys())
            month_filter_key = st.selectbox(
                "Kies de Maand:", 
                options=month_options, 
                format_func=lambda x: MONTH_NAMES[x], 
                index=datetime.datetime.now().month - 1, 
                key="month_filter_hist_new"
            )
            
            # Filter op maand én tot max vergelijkingsjaar
            df_filtered_time = df_full_history[
                (df_full_history.index.month == month_filter_key) &
                (df_full_history.index.year <= MAX_COMPARISON_YEAR)
            ]
        
        elif group_by_period == "Jaar":
            # Maximale jaar is nu MAX_COMPARISON_YEAR
            st.warning(f"Let op: De jaaranalyse gaat tot en met het laatste complete jaar **{MAX_COMPARISON_YEAR}**.")
            year_filter_start = st.number_input(
                "Filter op Startjaar (optioneel):", 
                min_value=df_full_history.index.min().year, 
                max_value=MAX_COMPARISON_YEAR,
                value=df_full_history.index.min().year, 
                step=1,
                key="year_filter_start_hist_new"
            )
            # Filter op startjaar én tot max vergelijkingsjaar
            df_filtered_time = df_full_history[
                (df_full_history.index.year >= year_filter_start) &
                (df_full_history.index.year <= MAX_COMPARISON_YEAR)
            ]
        
        if df_filtered_time.empty:
            st.warning("Geen data gevonden voor de geselecteerde criteria in de historische vergelijkingsperiode.")
            st.stop()
            
        start_date_display = df_filtered_time.index.min().strftime('%d-%m-%Y')
        end_date_display = df_filtered_time.index.max().strftime('%d-%m-%Y')
        st.markdown(f"**Analysebereik:** {start_date_display} t/m {end_date_display} ({len(df_filtered_time)} dagen)")

    elif filter_mode in ["Losse Dagen (Waarde)", "Aaneengesloten Periode", "Hellmann Getal Berekenen"]:
         # Vereenvoudigde periode selectie voor de andere modi
        st.markdown(f"**2. Tijdsbereik**")
        if 'hist_dates' not in st.session_state:
            st.session_state.hist_dates = (min_date_hist, max_date_hist_dt)
        
        date_range_hist = st.date_input(
            "Kies Start- en Einddatum:", 
            value=st.session_state.hist_dates,
            min_value=min_date_hist,
            max_value=max_date_hist_dt,
            key="hist_dates_input"
        )
        st.session_state.hist_dates = date_range_hist
        
        if len(date_range_hist) == 2:
            df_filtered_time = df_full_history.loc[str(date_range_hist[0]):str(date_range_hist[1])]
        elif len(date_range_hist) == 1:
            df_filtered_time = df_full_history.loc[str(date_range_hist[0]):str(date_range_hist[0])]
        else:
            df_filtered_time = df_full_history
        
        start_date_display = df_filtered_time.index.min().strftime('%d-%m-%Y')
        end_date_display = df_filtered_time.index.max().strftime('%d-%m-%Y')
        st.markdown(f"**Gefilterde dagen:** {len(df_filtered_time)} ({start_date_display} t/m {end_date_display})")
    
    
    st.markdown("---")

    # 3. Uitvoeren van de Zoekactie
    
    if df_filtered_time.empty:
        st.warning("Geen data gevonden voor de geselecteerde filter(s).")
        st.stop()
    
    # Filter de data op basis van drempel
    if filter_mode != "Hellmann Getal Berekenen" and filter_mode != "Overzicht per Jaar/Maand":
        if comparison == "Hoger dan (>=)":
            filtered_data = df_filtered_time[df_filtered_time[value_column] >= temp_threshold]
        else:
            filtered_data = df_filtered_time[df_filtered_time[value_column] <= temp_threshold]
            
        if filtered_data.empty:
            st.info(f"Geen dagen gevonden die voldoen aan de criteria: {selected_temp_type.split(' (')[0]} {comparison} {display_threshold}{value_unit}.") # Gebruik display_threshold
            st.stop()
    elif filter_mode == "Overzicht per Jaar/Maand":
         # Voor Overzicht per Jaar/Maand moet er eerst gefilterd worden voor de count, maar de filtered_data is nog niet nodig voor de 'empty' check hierboven.
         if comparison == "Hoger dan (>=)":
             filtered_data = df_filtered_time[df_filtered_time[value_column] >= temp_threshold]
         else:
             filtered_data = df_filtered_time[df_filtered_time[value_column] <= temp_threshold]
             
         if filtered_data.empty:
              st.info(f"Geen dagen gevonden die voldoen aan de criteria: {selected_temp_type.split(' (')[0]} {comparison} {display_threshold}{value_unit}.")
              st.stop()
    
    
    
    if filter_mode == "Overzicht per Jaar/Maand":
        
        df_grouped = filtered_data.index.year.value_counts().sort_index().reset_index()
        df_grouped.columns = ['Jaar', 'Aantal Dagen']
        df_grouped['Jaar'] = df_grouped['Jaar'].astype(str) 
        x_col, title_ext = 'Jaar', 'Jaar'
        
        display_name = selected_temp_type.split(" (")[0]
        if group_by_period == "Maand":
            month_name_display = MONTH_NAMES.get(month_filter_key, 'Onbekende Maand')
            st.subheader(f"📊 Samenvatting: Aantal dagen {comparison} {display_threshold}{value_unit} in **{month_name_display}** per Jaar")
            plot_title = f'Aantal dagen met {display_name} {comparison} {display_threshold}{value_unit} in {month_name_display} per Jaar (t/m {MAX_COMPARISON_YEAR})'
        else:
             st.subheader(f"📊 Samenvatting: Aantal dagen {comparison} {display_threshold}{value_unit} per Jaar (t/m {MAX_COMPARISON_YEAR})")
             plot_title = f'Aantal dagen met {display_name} {comparison} {display_threshold}{value_unit} per Jaar (t/m {MAX_COMPARISON_YEAR})'

        
        # --- Tabelweergave ---
        st.markdown("##### Tabellarisch Overzicht")
        # CORRECTIE: use_container_width=True -> width='stretch'
        st.dataframe(df_grouped.set_index(x_col), width='stretch')
        
        # --- Staafgrafiek ---
        st.markdown("##### Visualisatie")
        
        fig = px.bar(
            df_grouped, 
            x=x_col, 
            y='Aantal Dagen', 
            title=plot_title,
            labels={x_col: 'Jaar', 'Aantal Dagen': 'Aantal Dagen'},
            color_discrete_sequence=px.colors.qualitative.D3
        )
        
        avg_days = df_grouped['Aantal Dagen'].mean()
        fig.add_hline(
            y=avg_days, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"Gemiddelde: {avg_days:.1f}", 
            annotation_position="top left"
        )

        fig.update_layout(xaxis_title=None)
        # CORRECTIE: use_container_width=True -> width='stretch'
        st.plotly_chart(fig, width='stretch')


    elif filter_mode == "Hellmann Getal Berekenen":
        st.subheader("❄️ Hellmann Getal Berekening")
        st.info("Het Hellmann Getal is de absolute som van de Gemiddelde Dagtemperatuur op dagen met een Gemiddelde Temp $\le 0.0$ °C.")

        df_hellmann_days = df_filtered_time[df_filtered_time['Temp_Avg_C'] <= 0.0]
        
        if not df_hellmann_days.empty:
            df_hellmann_days['Hellmann_Bijdrage'] = df_hellmann_days['Temp_Avg_C'].abs()
            hellmann_score = df_hellmann_days['Hellmann_Bijdrage'].sum()
            hellmann_results = pd.DataFrame({
                'Periode': [f"{df_filtered_time.index.min().strftime('%d-%m-%Y')} t/m {df_filtered_time.index.max().strftime('%d-%m-%Y')}"],
                'Aantal Dagen $\le 0.0$ °C': [len(df_hellmann_days)],
                'Hellmann Getal': [f"{hellmann_score:.1f}"]
            }).set_index('Periode')
            
            # CORRECTIE: use_container_width=True -> width='stretch'
            st.dataframe(hellmann_results, width='stretch')
            
        else:
            st.info("Geen dagen met Gemiddelde Temp $\le 0.0$ °C gevonden in de geselecteerde periode. Hellmann Getal: 0.0")

        
    elif filter_mode == "Losse Dagen (Waarde)":
        st.subheader(f"Zoekresultaten: Losse Dagen ({len(filtered_data)} dagen)")
        
        if not filtered_data.empty:
            filtered_data_display = filtered_data.reset_index()
            
            for col in TEMP_COLUMNS:
                filtered_data_display[col] = filtered_data_display[col].map(safe_format_temp)
            filtered_data_display['Neerslag (mm)'] = filtered_data_display['Precip_Sum_mm'].map(safe_format_precip)
            filtered_data_display['Sneeuwval (cm)'] = filtered_data_display['Snowfall_Sum_cm'].map(safe_format_snowfall_cm) 
            
            df_final = filtered_data_display.rename(columns={'Date': 'Datum'})
            
            # Sneeuwdek (cm) is verwijderd
            cols_to_show = ['Datum'] + [c for c in df_final.columns if c.startswith('Temp') or c == 'Neerslag (mm)' or c == 'Sneeuwval (cm)']
            
            # CORRECTIE: use_container_width=True -> width='stretch'
            st.dataframe(df_final[cols_to_show].set_index('Datum'), width='stretch')

    elif filter_mode == "Aaneengesloten Periode":
        
        display_col_name = selected_temp_type.split(" (")[0]
        comparison_char = "≥" if comparison == "Hoger dan (>=)" else "≤"
        unit_display = f"{display_threshold}{value_unit}" # Gebruik display_threshold
        
        st.subheader(f"🔥 Zoekresultaten: Aaneengesloten Periodes")
        st.info(f"""
        **Criteria:** {min_consecutive_days}+ aaneengesloten dagen waarbij **{display_col_name}** {comparison_char} **{unit_display}**
        """)

        periods_df, total_periods = find_consecutive_periods(filtered_data, min_consecutive_days, value_column)
        
        if total_periods > 0:
            st.success(f"✅ **{total_periods}** aaneengesloten periodes van {min_consecutive_days} dagen of meer gevonden.")
            
            cols_to_show = ['StartDatum', 'EindDatum', 'Duur']
            if 'Gemiddelde Temp Periode' in periods_df.columns:
                cols_to_show.append('Gemiddelde Temp Periode')
            if 'Gemiddelde Neerslag Periode' in periods_df.columns:
                cols_to_show.append('Gemiddelde Neerslag Periode')

            # CORRECTIE: use_container_width=True -> width='stretch'
            st.dataframe(periods_df[cols_to_show].set_index('StartDatum'), width='stretch')
            
        else:
            st.warning("Geen aaneengesloten periodes gevonden die voldoen aan de criteria.")


# -------------------------------------------------------------------
# --- Tab 2: Extremen (Top 10 - Volledige Periode) ---
# -------------------------------------------------------------------

with tab_extremes:
    st.header(f"🏆 Top Extremen (Alle Jaren)")
    st.info(f"Dit overzicht toont de Top 10 dagen voor verschillende extreme categorieën over de gehele periode van **{START_DATE_FULL[:4]}** tot **{END_DATE_FULL[:4]}**.")

    extreme_results_full = find_extreme_days(df_full_history, top_n=10)
    
    tab_titles = [result[0] for result in extreme_results_full.values()]
    
    extreme_tabs = st.tabs(tab_titles)
    
    for i, (key, (title, df_result)) in enumerate(extreme_results_full.items()):
        with extreme_tabs[i]:
            st.subheader(f"{title} (Top 10)")
            
            if df_result.empty:
                st.warning("Geen data gevonden voor deze categorie.")
            else:
                # CORRECTIE: use_container_width=True -> width='stretch'
                st.dataframe(df_result, width='stretch')


# -------------------------------------------------------------------
# --- Tab 3: Hellmann Getal (Seizoenen) ---
# -------------------------------------------------------------------

with tab_hellmann:
    st.header("🧊 Hellmann Getal per Seizoen (Juli - Juni)")
    st.info("Analyse van het Hellmann Getal (som van absolute dagelijkse gemiddelde temperaturen $\\le 0.0$ °C) per seizoen. Een seizoen 'YYYY' loopt van **Juli YYYY-1** t/m **Juni YYYY**.")

    df_hellmann_seasonal = calculate_seasonal_hellmann(df_full_history)

    if df_hellmann_seasonal.empty:
        st.warning("Geen dagen onder 0.0°C gevonden in de historische data.")
    else:
        # 1. Bereken statistieken
        df_seasonal_comparison = df_hellmann_seasonal[
            df_hellmann_seasonal['Seizoen Jaar'].astype(int) <= MAX_COMPARISON_YEAR + 1
        ]
        
        max_hellmann = df_seasonal_comparison['Hellmann Getal'].max()
        jaar_max = df_seasonal_comparison.loc[df_seasonal_comparison['Hellmann Getal'].idxmax(), 'Seizoen Jaar']
        avg_hellmann = df_seasonal_comparison['Hellmann Getal'].mean()
        
        st.metric(
            label=f"Gemiddeld Hellmann Getal (1940 t/m Seizoen {MAX_COMPARISON_YEAR+1})", 
            value=f"{avg_hellmann:.1f}", 
            delta=f"Hoogste: {max_hellmann:.1f} (Seizoen {jaar_max})"
        )
        
        # 2. Plotly Staafgrafiek
        fig = px.bar(
            df_hellmann_seasonal, 
            x='Seizoen Jaar', 
            y='Hellmann Getal', 
            title='Hellmann Getal per Seizoen (Juli t/m Juni)',
            labels={'Seizoen Jaar': 'Seizoen (Eindjaar)', 'Hellmann Getal': 'Hellmann Getal'},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        fig.add_hline(
            y=avg_hellmann, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"Gemiddelde: {avg_hellmann:.1f}", 
            annotation_position="top right"
        )
        
        fig.update_xaxes(nticks=20, tickangle=45, showgrid=True)
        fig.update_layout(
            hovermode="x unified",
            xaxis_title=None, 
        )

        # CORRECTIE: use_container_width=True -> width='stretch'
        st.plotly_chart(fig, width='stretch')
        
        # 3. Toon de data in een tabel (Sorteerbaar)
        st.subheader("Data: Hellmann Getal per Seizoen (Sorteerbaar)")
        
        column_config = {
            "Hellmann Getal": st.column_config.NumberColumn("Hellmann Getal", format="%.1f")
        }
        # CORRECTIE: use_container_width=True -> width='stretch'
        st.dataframe(
            df_hellmann_seasonal.set_index('Seizoen Jaar'), 
            width='stretch',
            column_config=column_config
        )


# -------------------------------------------------------------------
# --- Tab 4: Historische Dag ---
# -------------------------------------------------------------------

with tab_daily_extremes:
    st.header("📆 Historische Dag - Extremen over Alle Jaren")
    st.info("Kies een dag en maand om de top 10 warmste, koudste, natste, en meest sneeuwval-rijke jaren voor die specifieke kalenderdag te zien.")

    # Datum berekening voor standaardwaarden
    today = datetime.date.today()
    current_month = today.month
    current_day = today.day

    col_month, col_day = st.columns(2)

    with col_month:
        # 1. Maand selectie
        selected_month_int = st.selectbox(
            "Kies een Maand:",
            options=list(MONTH_NAMES.keys()),
            format_func=lambda x: MONTH_NAMES[x],
            index=current_month - 1,
            key="daily_month_selector"
        )

    # 2. Dag selectie (dynamisch op basis van de gekozen maand)
    
    # Gebruik 2024 (schrikkeljaar) om de maximale dag in februari correct te bepalen (29)
    try:
        max_days_in_month = calendar.monthrange(2024, selected_month_int)[1] 
    except ValueError:
        max_days_in_month = 31 # Fallback
        
    day_options = list(range(1, max_days_in_month + 1))
    
    # Bepaal de standaardindex: ofwel de huidige dag, of 1 (als de huidige dag te groot is voor de maand)
    default_day_index = current_day - 1 
    if default_day_index >= len(day_options):
        default_day_index = 0 
    
    with col_day:
        selected_day_int = st.selectbox(
            "Kies een Dag:",
            options=day_options,
            index=default_day_index,
            key="daily_day_selector"
        )
        
    st.markdown("---")
    
    # Target dag en maand voor de analyse
    target_month = selected_month_int
    target_day = selected_day_int
    
    day_display = f"{target_day} {MONTH_NAMES[target_month]}"
    st.subheader(f"Top Extremen voor **{day_display}** ({START_DATE_FULL[:4]} - {max_data_year})")
    
    # Gebruik de volledige dataset voor de dagelijkse extreme analyse
    daily_extreme_results = find_daily_extremes(df_full_history, target_month, target_day, top_n=10)
    
    if not daily_extreme_results:
        st.warning(f"Geen historische data gevonden voor {day_display}.")
    else:
        # Haal de titels van de resultaten op voor de tabs
        tab_titles_daily = [result[0] for result in daily_extreme_results.values()]
        daily_tabs = st.tabs(tab_titles_daily)
        
        for i, (key, (title, df_result)) in enumerate(daily_extreme_results.items()):
            with daily_tabs[i]:
                st.markdown(f"##### {title} (Top 10 Jaren)")
                
                if df_result.empty:
                    st.info("Geen data beschikbaar voor deze extreme categorie op deze dag.")
                else:
                    # CORRECTIE: use_container_width=True -> width='stretch'
                    st.dataframe(df_result, width='stretch')

# -------------------------------------------------------------------
# --- BIJGEWERKT: Tab 5: Grafieken (Jaar/Maand) ---
# -------------------------------------------------------------------

with tab_charts:
    st.header("📈 Dagelijkse Weervisualisatie (Jaar/Maand)")
    st.info(f"Kies een **volledig jaar** of een **specifieke maand** in een jaar om de dagelijkse temperatuur, neerslag en sneeuwval te visualiseren. Dataset loopt van {START_DATE_FULL[:4]} t/m {max_data_year}.")

    # Bepaal het bereik van jaren
    min_year = df_full_history.index.min().year
    max_year = df_full_history.index.max().year

    # 1. Selectie Jaar/Maand modus
    chart_mode = st.radio(
        "Kies de te visualiseren periode:", 
        ["Volledig Jaar", "Specifieke Maand"], 
        key="chart_mode_select",
        horizontal=True
    )
    
    col_year_sel, col_month_sel = st.columns(2)

    with col_year_sel:
        # 1. Jaar selectie
        selected_year = st.selectbox(
            "Kies een Jaar:",
            options=list(range(min_year, max_year + 1)),
            index=len(range(min_year, max_year + 1)) - 1, # Standaard het meest recente jaar
            key="chart_year_selector"
        )
    
    selected_month_int = None
    if chart_mode == "Specifieke Maand":
        with col_month_sel:
            # 2. Maand selectie
            selected_month_int = st.selectbox(
                "Kies een Maand:",
                options=list(MONTH_NAMES.keys()),
                format_func=lambda x: MONTH_NAMES[x],
                index=datetime.datetime.now().month - 1,
                key="chart_month_selector"
            )

    # Filter de data
    if selected_year is not None:
        
        # Filter eerst op jaar
        df_chart_data = df_full_history[df_full_history.index.year == selected_year].copy()
        
        # Logica om de X-as en titel correct in te stellen
        if chart_mode == "Specifieke Maand" and selected_month_int is not None:
             df_chart_data = df_chart_data[df_chart_data.index.month == selected_month_int]
             period_title = f"{MONTH_NAMES[selected_month_int]} {selected_year}"
             # Creëer een simpele X-as voor de dag van de maand
             df_chart_data['X_Axis_Label'] = df_chart_data.index.day # Gebruik dagnummer
             x_col = 'X_Axis_Label'
             x_title = 'Dag van de Maand'
        else:
             period_title = f"Volledig Jaar {selected_year}"
             # BIJGEWERKT: Gebruik de Datum als X-as voor Plotly.
             # Plotly zal de datums automatisch groeperen op maand/jaar, wat de gewenste weergave oplevert.
             df_chart_data['X_Axis_Label'] = df_chart_data.index 
             x_col = 'X_Axis_Label'
             x_title = 'Datum' # Dit wordt later in Plotly overruled door de tijd-as

        if df_chart_data.empty:
            st.warning(f"Geen data beschikbaar voor {period_title}.")
        else:
            
            st.subheader(f"Overzicht voor **{period_title}** ({len(df_chart_data)} dagen)")
            st.markdown("---")

            # --- 3. Plot Temperatuur ---
            st.markdown("#### 🌡️ Dagelijkse Temperatuur (°C)")
            
            df_temp_plot = df_chart_data[['Temp_High_C', 'Temp_Avg_C', 'Temp_Low_C', x_col]].copy()
            
            df_temp_long = pd.melt(
                df_temp_plot, 
                id_vars=[x_col], 
                value_vars=['Temp_High_C', 'Temp_Avg_C', 'Temp_Low_C'],
                var_name='Temperatuur Type', 
                value_name='Temperatuur (°C)'
            )
            
            temp_labels = {
                'Temp_High_C': 'Max Temp',
                'Temp_Avg_C': 'Gemiddelde Temp',
                'Temp_Low_C': 'Min Temp'
            }
            df_temp_long['Temperatuur Type'] = df_temp_long['Temperatuur Type'].map(temp_labels)

            fig_temp = px.line(
                df_temp_long,
                x=x_col,
                y='Temperatuur (°C)',
                color='Temperatuur Type',
                title=f"Dagelijkse Temperaturen in {period_title}",
                labels={x_col: x_title, 'Temperatuur (°C)': 'Temperatuur (°C)'},
                color_discrete_map={
                    'Max Temp': 'red',
                    'Gemiddelde Temp': 'blue',
                    'Min Temp': 'lightblue'
                }
            )
            
            if chart_mode == "Volledig Jaar":
                 # Voor Volledig Jaar: gebruik de datum in de tooltip en stel de as type in
                 fig_temp.update_xaxes(
                     tickformat="%b", # Toont de afgekorte maandnaam
                     tickmode='auto',
                     title_text="Maand"
                 )
                 fig_temp.update_traces(hovertemplate='%{x|%d %b}<br>Temperatuur Type: %{full_text}<br>Temperatuur: %{y:.1f} °C<extra></extra>')
            else:
                 # Voor Specifieke Maand: toon de dag van de maand
                 fig_temp.update_xaxes(title_text=x_title)
                 fig_temp.update_traces(hovertemplate='Dag %{x}<br>Temperatuur Type: %{full_text}<br>Temperatuur: %{y:.1f} °C<extra></extra>')
            
            fig_temp.update_layout(hovermode="x unified")
            # CORRECTIE: use_container_width=True -> width='stretch'
            st.plotly_chart(fig_temp, width='stretch')

            # --- 4. Plot Neerslag & Sneeuwval ---
            st.markdown("#### 🌧️ Neerslag (mm)")

            # Neerslag plot (Bar chart)
            fig_precip = px.bar(
                df_chart_data,
                x=x_col,
                y='Precip_Sum_mm',
                title=f"Dagelijkse Neerslag in {period_title}",
                labels={x_col: x_title, 'Precip_Sum_mm': 'Neerslag (mm)'},
                color_discrete_sequence=['#4682B4']
            )
            
            if chart_mode == "Volledig Jaar":
                 # Voor Volledig Jaar: gebruik de datum in de tooltip en stel de as type in
                 fig_precip.update_xaxes(
                     tickformat="%b", # Toont de afgekorte maandnaam
                     tickmode='auto',
                     title_text="Maand"
                 )
                 fig_precip.update_traces(hovertemplate='%{x|%d %b}<br>Neerslag: %{y:.1f} mm<extra></extra>')
            else:
                 # Voor Specifieke Maand: toon de dag van de maand
                 fig_precip.update_xaxes(title_text=x_title)
                 fig_precip.update_traces(hovertemplate='Dag %{x}<br>Neerslag: %{y:.1f} mm<extra></extra>')
                 
            fig_precip.update_layout(hovermode="x unified")
            # CORRECTIE: use_container_width=True -> width='stretch'
            st.plotly_chart(fig_precip, width='stretch')
            
            # Sneeuwval plot (Bar chart)
            st.markdown("#### ❄️ Sneeuwval (cm)")
            fig_snowfall = px.bar(
                df_chart_data,
                x=x_col,
                y='Snowfall_Sum_cm',
                title=f"Dagelijkse Sneeuwval in {period_title}",
                labels={x_col: x_title, 'Snowfall_Sum_cm': 'Sneeuwval (cm)'},
                color_discrete_sequence=['#A8DDE4']
            )

            if chart_mode == "Volledig Jaar":
                 # Voor Volledig Jaar: gebruik de datum in de tooltip en stel de as type in
                 fig_snowfall.update_xaxes(
                     tickformat="%b", # Toont de afgekorte maandnaam
                     tickmode='auto',
                     title_text="Maand"
                 )
                 fig_snowfall.update_traces(hovertemplate='%{x|%d %b}<br>Sneeuwval: %{y:.1f} cm<extra></extra>')
            else:
                 # Voor Specifieke Maand: toon de dag van de maand
                 fig_snowfall.update_xaxes(title_text=x_title)
                 fig_snowfall.update_traces(hovertemplate='Dag %{x}<br>Sneeuwval: %{y:.1f} cm<extra></extra>')

            fig_snowfall.update_layout(hovermode="x unified")
            # CORRECTIE: use_container_width=True -> width='stretch'
            st.plotly_chart(fig_snowfall, width='stretch')
