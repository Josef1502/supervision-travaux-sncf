"""
app.py
======
Application Streamlit — Supervision des Travaux Ferroviaires
SNCF Voyageurs — Axe Sud-Est

Pages :
  1. Carte interactive des travaux
  2. Tableau de bord & statistiques
  3. Prédiction ML de la priorité
  4. FAQ Travaux (assistant IA)

Usage :
    streamlit run app.py
"""

import sqlite3
import warnings
import joblib
import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    from groq import Groq
    GROQ_DISPONIBLE = True
except ImportError:
    GROQ_DISPONIBLE = False

# Charger la clé depuis st.secrets
try:
    GROQ_API_KEY_ENV = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY_ENV = ""

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG GÉNÉRALE
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Supervision Travaux — SNCF Axe Sud-Est",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THÈME VISUEL — signalétique ferroviaire
# ─────────────────────────────────────────────

def appliquer_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Archivo+Expanded:wght@600;700;800&family=Inter:wght@400;500;600&family=Roboto+Mono:wght@500&display=swap');

        :root {
            --rouge-sncf: #C0272D;
            --noir-quai:  #1A1A2E;
            --fond-gare:  #F8F7F4;
            --ardoise:    #4A5568;
            --jaune-travaux: #FFC000;
            --vert-ok:    #1E8E5A;
        }

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        /* Fond général façon hall de gare */
        .stApp {
            background-color: var(--fond-gare);
        }

        /* Titres : typographie condensée façon panneau de gare */
        h1, h2, h3 {
            font-family: 'Archivo Expanded', sans-serif !important;
            color: var(--noir-quai) !important;
            letter-spacing: -0.01em;
        }

        h1 {
            font-weight: 800 !important;
            border-bottom: 4px solid var(--rouge-sncf);
            padding-bottom: 0.4rem;
            display: inline-block;
        }

        h2, h3 {
            font-weight: 700 !important;
        }

        /* Sidebar façon quai sombre */
        section[data-testid="stSidebar"] {
            background-color: var(--noir-quai);
        }
        section[data-testid="stSidebar"] * {
            color: #F0F0F0 !important;
        }
        section[data-testid="stSidebar"] .stMultiSelect span[data-baseweb="tag"] {
            background-color: var(--rouge-sncf) !important;
        }

        /* Barre de navigation horizontale collée en haut */
        div[data-testid="stHorizontalBlock"]:has(div.barre-nav-marker) {
            position: sticky;
            top: 0;
            z-index: 999;
            background-color: var(--fond-gare);
            padding: 0.75rem 0;
            border-bottom: 3px solid var(--rouge-sncf);
        }

        /* Boutons radio navigation -> style "billets/onglets" */
        div[role="radiogroup"] label {
            background: white;
            border: 1.5px solid #DDD8D0;
            border-radius: 8px;
            padding: 6px 14px !important;
            margin-right: 6px;
            transition: all 0.15s ease;
        }
        div[role="radiogroup"] label:hover {
            border-color: var(--rouge-sncf);
        }

        /* Métriques façon panneau d'affichage gare */
        div[data-testid="stMetric"] {
            background: var(--noir-quai);
            border-radius: 10px;
            padding: 14px 16px 10px 16px;
            border-left: 5px solid var(--jaune-travaux);
        }
        div[data-testid="stMetric"] label {
            color: #C9C9D6 !important;
            font-family: 'Roboto Mono', monospace !important;
            font-size: 0.72rem !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            font-family: 'Roboto Mono', monospace !important;
            font-weight: 600 !important;
        }

        /* Boutons primaires */
        .stButton button[kind="primary"] {
            background-color: var(--rouge-sncf);
            border: none;
            font-weight: 600;
            border-radius: 8px;
        }
        .stButton button[kind="primary"]:hover {
            background-color: #9E1F24;
        }
        .stButton button[kind="secondary"] {
            border-radius: 8px;
        }

        /* Cartes / conteneurs (info, success, etc.) */
        div[data-testid="stAlert"] {
            border-radius: 8px;
            border-left: 4px solid var(--rouge-sncf);
        }

        /* Dataframe : en-têtes façon tableau d'affichage */
        div[data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
        }

        /* Champ de saisie texte (mot de passe, chat) */
        .stTextInput input, .stChatInput textarea {
            border-radius: 8px !important;
        }

        /* Expander filtres */
        div[data-testid="stExpander"] {
            border-radius: 8px;
            border: 1px solid #E2DDD3;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

appliquer_theme()

# ─────────────────────────────────────────────
# AUTHENTIFICATION
# ─────────────────────────────────────────────

def verifier_mot_de_passe():
    """
    Affiche un écran de connexion par mot de passe.
    Retourne True si l'utilisateur est authentifié, False sinon.
    """
    # Récupérer le mot de passe attendu depuis st.secrets
    try:
        mdp_attendu = st.secrets["APP_PASSWORD"]
    except Exception:
        mdp_attendu = "sncf2026"  # mot de passe par défaut si secrets.toml absent

    if st.session_state.get("authentifie", False):
        return True

    # ── Écran de connexion façon panneau de gare ──
    st.markdown(
        """
        <style>
        /* Image de fond sur toute la page de connexion */
        .stApp {
            background-image: url("https://www.sncf-voyageurs.com/medias-publics/styles/original/public/2023-08/tgv-mer-header.jpg.webp?VersionId=kIICW3n96pY1B7TzwgRQuXzn4o1plkNq&itok=bBCe8GnP");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        /* Overlay sombre pour lisibilité du formulaire */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(10, 10, 30, 0.55);
            z-index: 0;
            pointer-events: none;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(div.marqueur-connexion) {
            background: rgba(26, 26, 46, 0.92);
            border-radius: 14px;
            box-shadow: 0 12px 40px rgba(0,0,0,0.4);
            backdrop-filter: blur(6px);
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(div.marqueur-connexion) p,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(div.marqueur-connexion) label {
            color: #F0F0F0 !important;
        }
        .badge-quai {
            display: inline-block;
            background: #C0272D;
            color: white;
            font-family: 'Roboto Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.08em;
            padding: 3px 10px;
            border-radius: 4px;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.3, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="marqueur-connexion"></div>', unsafe_allow_html=True)

            st.image(
                "https://upload.wikimedia.org/wikipedia/fr/thumb/5/52/Logotype_SNCF_Voyageurs_2020.svg/1280px-Logotype_SNCF_Voyageurs_2020.svg.png?_=20221023111731",
                width=200,
            )
            st.markdown(
                "<span class='badge-quai'>AXE SUD-EST · ACCÈS RESTREINT</span><br>"
                "<h2 style='color:white; margin-top:0.2rem; margin-bottom:0.2rem; "
                "text-shadow: 0 2px 8px rgba(0,0,0,0.8);'>Supervision Travaux</h2>"
                "<p style='color:#FFFFFF; margin-bottom:1.4rem; "
                "text-shadow: 0 1px 6px rgba(0,0,0,0.9);'>Plateforme de pilotage des chantiers ferroviaires</p>",
                unsafe_allow_html=True
            )

            with st.form("login_form"):
                mdp_saisi = st.text_input(
                    "Mot de passe",
                    type="password",
                    placeholder="Entrez le mot de passe d'accès",
                )
                submit = st.form_submit_button("Accéder à la plateforme", use_container_width=True, type="primary")

                if submit:
                    if mdp_saisi == mdp_attendu:
                        st.session_state.authentifie = True
                        st.rerun()
                    else:
                        st.error("Mot de passe incorrect.")

            st.markdown(
                "<p style='color:#E0E0E0; font-size:0.78rem; text-align:center; margin-top:0.8rem; "
                "text-shadow: 0 1px 4px rgba(0,0,0,0.9);'>"
                "Accès réservé aux équipes SNCF Voyageurs — Axe Sud-Est"
                "</p>",
                unsafe_allow_html=True
            )

    return False

COULEURS_PRIORITE = {
    "Sévérisé fortifié": "#7030A0",
    "Sévérisé":          "#FF0000",
    "Chantier Local":    "#FFC000",
    "Autre":             "#A6A6A6",
    "Jaune":             "#FFFF00",
}

PRIORITES_ML = [
    "Sévérisé fortifié",
    "Sévérisé",
    "Chantier Local",
    "Autre",
]

DB_PATH    = "travaux_sncf.db"
MODEL_PATH = "model.pkl"


def cle_tri_semaine(s):
    """
    Trie une clé semaine 'S08_2026' par ordre chronologique réel
    (année, puis numéro de semaine), et non alphabétiquement.
    """
    try:
        num = int(s[1:3])
        an = int(s[4:8])
        return (an, num)
    except (ValueError, IndexError, TypeError):
        return (9999, 99)


def trier_semaines(liste_semaines):
    """Retourne la liste de semaines triée chronologiquement."""
    return sorted(liste_semaines, key=cle_tri_semaine)


# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────

@st.cache_data
def charger_donnees():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM travaux", conn)
    conn.close()
    df["semaine_num"] = df["semaine"].str.extract(r"(\d+)").astype(float)
    df["PK_longueur"] = df["PK_FIN"] - df["PK_DE"]
    return df


@st.cache_resource
def charger_modele():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def sidebar(df):
    # ── Ligne 1 : logo + titre + déconnexion ──
    col_logo, col_titre, col_deco = st.columns([1.2, 5, 1])

    with col_logo:
        st.image(
            "https://upload.wikimedia.org/wikipedia/fr/thumb/5/52/Logotype_SNCF_Voyageurs_2020.svg/1280px-Logotype_SNCF_Voyageurs_2020.svg.png?_=20221023111731",
            width=140,
        )

    with col_titre:
        st.markdown(
            "<div style='padding-top:10px;'>"
            "<span style='font-family:Archivo Expanded, sans-serif; font-size:21px; "
            "font-weight:800; color:#1A1A2E;'>SUPERVISION TRAVAUX</span><br>"
            "<span style='font-family:Roboto Mono, monospace; font-size:12px; "
            "color:#C0272D; letter-spacing:0.05em;'>SNCF VOYAGEURS · AXE SUD-EST</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    with col_deco:
        st.markdown("<div style='padding-top:18px;'></div>", unsafe_allow_html=True)
        if st.button("Déconnexion", use_container_width=True):
            st.session_state.authentifie = False
            st.rerun()

    # ── Ligne 2 : navigation horizontale ──────
    st.markdown('<div class="barre-nav-marker"></div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Carte des travaux", "Tableau de bord", "Prédire la criticité d'un chantier", "FAQ Travaux"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ── Ligne 3 : filtres dans la sidebar latérale ──
    st.sidebar.markdown(
        "<span style='font-family:Roboto Mono, monospace; font-size:12px; "
        "letter-spacing:0.08em; color:#FFC000;'>🔍 FILTRES</span>",
        unsafe_allow_html=True,
    )

    infrapoles = sorted(df["InfPôle"].dropna().unique().tolist())
    infrapoles_sel = st.sidebar.multiselect("Infrapôle", infrapoles, default=infrapoles)

    priorites = [p for p in COULEURS_PRIORITE.keys() if p in df["Priorité"].unique()]
    priorites_sel = st.sidebar.multiselect("Priorité", priorites, default=priorites)

    # Filtre Semaine : affichage du label lisible (dates), valeur interne = code semaine
    semaines = trier_semaines(df["semaine"].unique().tolist())
    mapping_label = (
        df.drop_duplicates(subset=["semaine"])
        .set_index("semaine")["semaine_label"]
        .to_dict()
        if "semaine_label" in df.columns else {s: s for s in semaines}
    )
    semaine_sel = st.sidebar.multiselect(
        "Semaine",
        semaines,
        default=semaines,
        format_func=lambda code: mapping_label.get(code, code),
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        f"<span style='font-family:Roboto Mono, monospace; font-size:13px;'>"
        f"📦 {len(df):,} chantiers · {df['semaine'].nunique()} semaines</span>",
        unsafe_allow_html=True,
    )

    return page, infrapoles_sel, priorites_sel, semaine_sel


def filtrer(df, infrapoles, priorites, semaines):
    mask = (
        df["InfPôle"].isin(infrapoles) &
        df["Priorité"].isin(priorites) &
        df["semaine"].isin(semaines)
    )
    return df[mask].copy()


# ─────────────────────────────────────────────
# PAGE 1 — CARTE INTERACTIVE
# ─────────────────────────────────────────────

COORDS_INFRAPOLE = {
    "ALP":      (45.18,  5.72),
    "BFC":      (47.32,  5.04),
    "ITIF PSE": (48.85,  2.35),
    "LGV AT":   (47.30,  0.68),
    "LGV Nord": (50.40,  2.50),
    "LGV RR":   (47.75,  7.34),
    "LGV SEE":  (46.50,  4.85),
    "LORraine": (48.69,  6.18),
    "LR":       (43.60,  3.87),
    "Lorraine": (48.69,  6.18),
    "MPY":      (43.60,  1.44),
    "Normandie":(49.44, -0.36),
    "PACA":     (43.30,  5.37),
    "PDL":      (47.22, -1.55),
    "PSE":      (48.85,  2.35),
    "Rhodanien":(45.75,  4.83),
    "Rhénan":   (48.58,  7.75),
    "Suisse":   (46.52,  6.63),
}


@st.cache_data
def charger_lignes_pk():
    csv_path = "lignes_axe_sudest.csv"
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return df.sort_values(["LIGNE RFN", "PK"])


def page_carte(df_filtre):
    st.title("Supervision des travaux — Axe Sud-Est")
    st.markdown(
        "> 🎯 Visualiser les chantiers directement sur les voies ferrées "
        "grâce aux points kilométriques."
    )

    # ── Boutons jours ─────────────────────────
    JOURS = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]

    if "jour_sel" not in st.session_state:
        st.session_state.jour_sel = None

    # 8 colonnes : Tous + 7 jours
    cols_j = st.columns(8)
    with cols_j[0]:
        if st.button(
            "Tous",
            use_container_width=True,
            type="primary" if st.session_state.jour_sel is None else "secondary"
        ):
            st.session_state.jour_sel = None
            st.rerun()

    for i, jour in enumerate(JOURS):
        with cols_j[i + 1]:
            actif = st.session_state.jour_sel == jour
            if st.button(
                jour,                          # nom complet : Lundi, Mardi...
                use_container_width=True,
                type="primary" if actif else "secondary",
            ):
                st.session_state.jour_sel = None if actif else jour
                st.rerun()

    jour_actif = st.session_state.jour_sel
    if jour_actif:
        st.markdown(f"Filtré sur : **{jour_actif}**")
        df_carte = df_filtre[df_filtre["Jour"] == jour_actif].copy()
    else:
        df_carte = df_filtre.copy()

    # ── Jauge horaire ──────────────────────────
    st.divider()
    col_h1, col_h2 = st.columns([4, 1])

    # Générer toutes les heures de 00:00 à 23:45 par pas de 15 min
    horaires_dispo = [
        f"{h:02d}:{m:02d}"
        for h in range(24)
        for m in range(0, 60, 15)
    ]

    with col_h1:
        heure_sel = st.select_slider(
            "Chantiers démarrant à partir de :",
            options=horaires_dispo,
            value="00:00",
        )
    with col_h2:
        st.markdown(f"### {heure_sel}")
        st.caption("heure de début")

    # Convertir en minutes pour comparaison
    h_sel, m_sel = int(heure_sel[:2]), int(heure_sel[3:])
    minutes_sel = h_sel * 60 + m_sel

    # Fonction de conversion horaire → minutes
    def horaire_to_minutes(h_str):
        try:
            if isinstance(h_str, str) and ":" in h_str:
                parts = h_str.split(":")
                return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            pass
        return -1

    # Appliquer filtre horaire si > 00:00
    if minutes_sel > 0 and "Horaire" in df_carte.columns:
        df_filtre_horaire = df_carte[
            df_carte["Horaire"].apply(horaire_to_minutes) >= minutes_sel
        ].copy()
        if not df_filtre_horaire.empty:
            df_carte = df_filtre_horaire
        else:
            st.info(
                f"Aucun chantier ne démarre à partir de **{heure_sel}**. "
                "Affichage de tous les chantiers du filtre jour."
            )

    # ── Métriques ─────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Chantiers", f"{len(df_carte):,}")
    col2.metric("Infrapôles", df_carte["InfPôle"].nunique())
    col3.metric("Sévérisé fortifié",
                len(df_carte[df_carte["Priorité"] == "Sévérisé fortifié"]))
    col4.metric("Sévérisé",
                len(df_carte[df_carte["Priorité"] == "Sévérisé"]))

    if df_carte.empty:
        st.warning("Aucun chantier pour ces filtres.")
        return

    # ── Recherche train + Zoom chantier côte à côte ──
    st.divider()

    # Construire df_unique ici pour pouvoir l'utiliser dans la colonne droite
    df_unique = df_carte.dropna(subset=["PK_DE", "PK_FIN", "CODE_LIGNE"]).copy()
    df_unique["_cle_dedup"] = (
        df_unique["Intitulé Chantier"].astype(str).str.strip().str.lower()
        + "|" + df_unique["CODE_LIGNE"].astype(str)
        + "|" + df_unique["PK_DE"].astype(str)
        + "|" + df_unique["PK_FIN"].astype(str)
        + "|" + df_unique["Priorité"].astype(str)
    )
    df_unique = df_unique.drop_duplicates(subset=["_cle_dedup"]).drop(columns=["_cle_dedup"])
    noms_chantiers = ["— Tous les chantiers —"] + sorted(df_unique["Intitulé Chantier"].unique().tolist())

    col_search, col_zoom = st.columns(2)

    with col_search:
        recherche_train = st.text_input(
            "🔎 Rechercher un train (numéro de sillon)",
            placeholder="Ex : 9269",
            help="Filtre les chantiers dont le champ Sillons Ouvrants contient ce numéro",
        )

    with col_zoom:
        chantier_sel = st.selectbox(
            f"🔍 Zoomer sur un chantier ({len(noms_chantiers)-1} disponibles)",
            noms_chantiers,
            index=0,
        )

    # Appliquer le filtre train si renseigné
    if recherche_train.strip():
        terme = recherche_train.strip()
        if "Sillons Ouvrants" in df_carte.columns:
            df_recherche = df_carte[
                df_carte["Sillons Ouvrants"]
                .fillna("")
                .astype(str)
                .str.contains(terme, case=False, na=False, regex=False)
            ].copy()
            if not df_recherche.empty:
                df_carte = df_recherche
                # Reconstruire df_unique après filtre train
                df_unique = df_carte.dropna(subset=["PK_DE", "PK_FIN", "CODE_LIGNE"]).copy()
                df_unique["_cle_dedup"] = (
                    df_unique["Intitulé Chantier"].astype(str).str.strip().str.lower()
                    + "|" + df_unique["CODE_LIGNE"].astype(str)
                    + "|" + df_unique["PK_DE"].astype(str)
                    + "|" + df_unique["PK_FIN"].astype(str)
                    + "|" + df_unique["Priorité"].astype(str)
                )
                df_unique = df_unique.drop_duplicates(subset=["_cle_dedup"]).drop(columns=["_cle_dedup"])
                st.success(
                    f"🚆 **{df_carte['Intitulé Chantier'].nunique()} chantier(s)** "
                    f"impactent le train **{terme}**"
                )
            else:
                st.warning(f"Aucun chantier trouvé pour « {terme} ». Affichage de tous les chantiers.")
        else:
            st.error("Colonne 'Sillons Ouvrants' introuvable dans la base.")

    # Paramètres de centrage — France entière par défaut
    centre_lat, centre_lon, zoom = 46.8, 2.3, 6
    chantier_row = None

    df_pk = charger_lignes_pk()
    if df_pk is None:
        st.error("Fichier `lignes_axe_sudest.csv` introuvable.")
        return

    if chantier_sel != noms_chantiers[0]:
        # Récupération par nom (plus robuste que l'index positionnel)
        correspondances = df_unique[df_unique["Intitulé Chantier"] == chantier_sel]
        chantier_row = correspondances.iloc[0]
        code  = int(chantier_row["CODE_LIGNE"])
        pk_de = chantier_row["PK_DE"]
        pk_fin = chantier_row["PK_FIN"]
        seg = df_pk[
            (df_pk["LIGNE RFN"] == code) &
            (df_pk["PK"] >= pk_de - 0.1) &
            (df_pk["PK"] <= pk_fin + 0.1)
        ]
        if len(seg) >= 1:
            centre_lat = seg["LATITUDE"].mean()
            centre_lon = seg["LONGITUDE"].mean()
            longueur = pk_fin - pk_de
            zoom = 13 if longueur < 2 else (11 if longueur < 10 else 9)

    # ── Construction de la carte ───────────────
    fig = go.Figure()

    # 1. LIGNES FERROVIAIRES — noir
    for code_ligne, grp in df_pk.groupby("LIGNE RFN"):
        grp = grp.dropna(subset=["LATITUDE","LONGITUDE"])
        if len(grp) < 2:
            continue
        fig.add_trace(go.Scattermapbox(
            lat=grp["LATITUDE"].tolist(),
            lon=grp["LONGITUDE"].tolist(),
            mode="lines",
            line=dict(width=2, color="#1A1A1A"),
            opacity=0.7,
            hoverinfo="skip",
            showlegend=False,
        ))

    # 2. SEGMENTS TRAVAUX
    COULEURS_SEG = {
        "Sévérisé fortifié": "#7030A0",
        "Sévérisé":          "#FF0000",
        "Chantier Local":    "#FFC000",
        "Autre":             "#4472C4",
    }
    LARGEURS = {
        "Sévérisé fortifié": 7,
        "Sévérisé":          6,
        "Chantier Local":    5,
        "Autre":             5,
    }

    nb_traces = 0
    for priorite, couleur in COULEURS_SEG.items():
        df_prio = df_unique[df_unique["Priorité"] == priorite]
        if df_prio.empty:
            continue

        lats_all, lons_all, texts_all = [], [], []
        for _, row in df_prio.iterrows():
            code  = row.get("CODE_LIGNE")
            pk_de = row.get("PK_DE")
            pk_fin = row.get("PK_FIN")
            if pd.isna(code) or pd.isna(pk_de) or pd.isna(pk_fin):
                continue
            seg = df_pk[
                (df_pk["LIGNE RFN"] == int(code)) &
                (df_pk["PK"] >= pk_de - 0.1) &
                (df_pk["PK"] <= pk_fin + 0.1)
            ].sort_values("PK")
            if len(seg) < 2:
                continue
            tooltip = (
                f"<b>{row.get('Intitulé Chantier','N/A')}</b><br>"
                f"Priorité : <b>{priorite}</b><br>"
                f"Sillons Ouvrants : {row.get('Sillons Ouvrants','N/A')}"
            )
            lats_all  += seg["LATITUDE"].tolist()  + [None]
            lons_all  += seg["LONGITUDE"].tolist() + [None]
            texts_all += [tooltip] * len(seg)      + [None]
            nb_traces += 1

        if lats_all:
            fig.add_trace(go.Scattermapbox(
                lat=lats_all, lon=lons_all,
                mode="lines",
                line=dict(width=LARGEURS.get(priorite,5), color=couleur),
                opacity=0.9,
                hovertemplate="%{text}<extra></extra>",
                text=texts_all,
                name=priorite,
                showlegend=True,
            ))

    # 3. MARQUEUR chantier sélectionné
    if chantier_row is not None:
        fig.add_trace(go.Scattermapbox(
            lat=[centre_lat], lon=[centre_lon],
            mode="markers+text",
            marker=dict(size=20, color="#FF6B00"),
            text=["📍"],
            textposition="top center",
            hovertemplate=(
                f"<b>📍 {chantier_row['Intitulé Chantier']}</b><br>"
                f"Ligne {int(chantier_row['CODE_LIGNE'])} | "
                f"PK {chantier_row['PK_DE']}→{chantier_row['PK_FIN']}<br>"
                f"Priorité : {chantier_row['Priorité']}"
                "<extra></extra>"
            ),
            name="📍 Chantier sélectionné",
            showlegend=True,
        ))

    # ── Mise en page ──────────────────────────
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center={"lat": centre_lat, "lon": centre_lon},
            zoom=zoom,
            bounds={"west": -5.5, "east": 9.5, "south": 41.0, "north": 51.5},
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        height=640,
        legend=dict(
            title="<b>Légende</b>",
            orientation="v",
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#CCCCCC",
            borderwidth=1,
            font=dict(size=13),
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True},
    )

    col_a, col_b = st.columns(2)
    col_a.success(f"🛤️ **{df_pk['LIGNE RFN'].nunique()} lignes** Axe Sud-Est")
    col_b.info(f"🚧 **{nb_traces} chantiers** tracés sur les voies")

    # ── Tableau filtré selon jour + recherche + horaire ───
    st.divider()
    st.subheader("📋 Liste des chantiers")

    # Info filtres actifs
    filtres_actifs = []
    if jour_actif:
        filtres_actifs.append(f"📅 **{jour_actif}**")
    if recherche_train.strip():
        filtres_actifs.append(f"🚆 train **{recherche_train.strip()}**")
    if minutes_sel > 0:
        filtres_actifs.append(f"⏰ à partir de **{heure_sel}**")
    if filtres_actifs:
        st.caption("Filtres actifs : " + " | ".join(filtres_actifs))

    cols = ["semaine_label", "InfPôle", "Priorité", "Intitulé Chantier",
            "CODE_LIGNE", "PK_DE", "PK_FIN", "Jour", "Horaire", "Horaire_fin"]
    cols_dispo = [c for c in cols if c in df_carte.columns]
    st.dataframe(
        df_carte[cols_dispo].reset_index(drop=True).rename(
            columns={"semaine_label": "Semaine"}
        ),
        use_container_width=True,
        height=280,
    )


# ─────────────────────────────────────────────
# PAGE 2 — TABLEAU DE BORD
# ─────────────────────────────────────────────

def page_dashboard(df_filtre, df_total):
    st.title("Tableau de bord")
    st.markdown(
        "> 🎯 **Objectif** : Exploiter les données historiques (S02→S23) "
        "pour identifier les tendances, les lignes les plus impactées "
        "et orienter les décisions opérationnelles."
    )
    st.caption("Analyse statistique des travaux ferroviaires — Axe Sud-Est")

    # ── KPI ──────────────────────────────────
    st.subheader("📈 Indicateurs clés")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total chantiers", f"{len(df_filtre):,}", f"{len(df_filtre) - len(df_total):,}")
    col2.metric("Semaines", df_filtre["semaine"].nunique())
    col3.metric("Infrapôles", df_filtre["InfPôle"].nunique())
    col4.metric("Lignes RFN", int(df_filtre["CODE_LIGNE"].nunique()))

    pct_sev = len(df_filtre[df_filtre["Priorité"].isin(["Sévérisé", "Sévérisé fortifié"])]) / max(len(df_filtre), 1) * 100
    col5.metric("% Sévérisé", f"{pct_sev:.1f}%")

    st.divider()

    # ── Ligne 1 : Répartition priorité + Évolution semaine ──
    col_g, col_d = st.columns(2)

    with col_g:
        st.subheader("Répartition par priorité")
        counts = df_filtre["Priorité"].value_counts().reset_index()
        counts.columns = ["Priorité", "Nombre"]
        fig_pie = px.pie(
            counts,
            names="Priorité",
            values="Nombre",
            color="Priorité",
            color_discrete_map=COULEURS_PRIORITE,
            hole=0.4,
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            textfont_size=12,
        )
        fig_pie.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            margin=dict(t=20, b=60),
            height=350,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_d:
        st.subheader("Évolution hebdomadaire")
        evol = (
            df_filtre.groupby(["semaine_num", "Priorité"])
            .size()
            .reset_index(name="Nombre")
        )
        fig_line = px.line(
            evol,
            x="semaine_num",
            y="Nombre",
            color="Priorité",
            color_discrete_map=COULEURS_PRIORITE,
            markers=True,
            labels={"semaine_num": "Semaine", "Nombre": "Nb chantiers"},
        )
        fig_line.update_layout(
            height=350,
            margin=dict(t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.4),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # ── Ligne 2 : Par infrapôle + Par jour ──
    col_g2, col_d2 = st.columns(2)

    with col_g2:
        st.subheader("Chantiers par infrapôle")
        by_infra = (
            df_filtre.groupby(["InfPôle", "Priorité"])
            .size()
            .reset_index(name="Nombre")
            .sort_values("Nombre", ascending=False)
        )
        fig_bar = px.bar(
            by_infra,
            x="InfPôle",
            y="Nombre",
            color="Priorité",
            color_discrete_map=COULEURS_PRIORITE,
            barmode="stack",
        )
        fig_bar.update_layout(
            height=380,
            margin=dict(t=20, b=80),
            xaxis_tickangle=-35,
            legend=dict(orientation="h", yanchor="bottom", y=-0.5),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_d2:
        st.subheader("Répartition par jour de la semaine")
        jours_ordre = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
        by_jour = (
            df_filtre[df_filtre["Jour"].notna()]
            .groupby(["Jour", "Priorité"])
            .size()
            .reset_index(name="Nombre")
        )
        by_jour["Jour"] = pd.Categorical(by_jour["Jour"], categories=jours_ordre, ordered=True)
        by_jour = by_jour.sort_values("Jour")

        fig_jour = px.bar(
            by_jour,
            x="Jour",
            y="Nombre",
            color="Priorité",
            color_discrete_map=COULEURS_PRIORITE,
            barmode="group",
        )
        fig_jour.update_layout(
            height=380,
            margin=dict(t=20, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=-0.5),
        )
        st.plotly_chart(fig_jour, use_container_width=True)

    # ── Ligne 3 : Top lignes + chantiers les plus critiques ──
    col_g3, col_d3 = st.columns(2)

    with col_g3:
        st.subheader("Top 5 trains les plus impactés")
        st.caption("Numéros de sillon à 4 chiffres — colonne Sillons Ouvrants")

        import re as _re
        from collections import Counter as _Counter

        tous_trains = []
        for val in df_filtre["Sillons Ouvrants"].dropna():
            tous_trains.extend(_re.findall(r'\b\d{4}\b', str(val)))

        top_trains = _Counter(tous_trains).most_common(5)

        if top_trains:
            for i, (train, nb) in enumerate(top_trains, start=1):
                st.markdown(
                    f"<div style='display:flex; align-items:center; gap:14px; "
                    f"padding:10px 14px; margin-bottom:6px; background:white; "
                    f"border-radius:8px; border-left:4px solid #C0272D;'>"
                    f"<span style='font-family:Roboto Mono, monospace; font-size:18px; "
                    f"font-weight:600; color:#C0272D; min-width:24px;'>#{i}</span>"
                    f"<span style='font-family:Roboto Mono, monospace; font-size:16px; "
                    f"font-weight:600; flex:1;'>Train {train}</span>"
                    f"<span style='font-size:14px; color:#4A5568;'>{nb} mention(s)</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Aucun numéro de sillon trouvé pour ces filtres.")

    with col_d3:
        st.subheader("Top 5 infrapôles — chantiers critiques")
        st.caption("Sévérisé fortifié + Sévérisé")
        top_infra_critique = (
            df_filtre[df_filtre["Priorité"].isin(["Sévérisé fortifié", "Sévérisé"])]
            .groupby("InfPôle")
            .size()
            .reset_index(name="Nombre")
            .sort_values("Nombre", ascending=False)
            .head(5)
        )

        for i, row in enumerate(top_infra_critique.itertuples(), start=1):
            nb = row.Nombre
            total_infra = len(df_filtre[df_filtre["InfPôle"] == row.InfPôle])
            pct = nb / total_infra * 100 if total_infra else 0
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:14px; "
                f"padding:10px 14px; margin-bottom:6px; background:white; "
                f"border-radius:8px; border-left:4px solid #7030A0;'>"
                f"<span style='font-family:Roboto Mono, monospace; font-size:18px; "
                f"font-weight:600; color:#7030A0; min-width:24px;'>#{i}</span>"
                f"<span style='font-family:Roboto Mono, monospace; font-size:16px; "
                f"font-weight:600; flex:1;'>{row.InfPôle}</span>"
                f"<span style='font-size:14px; color:#4A5568;'>{nb} chantiers ({pct:.0f}% du total)</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Tableau récapitulatif ──
    st.divider()
    st.subheader("📋 Résumé par infrapôle et priorité")
    pivot = (
        df_filtre.groupby(["InfPôle", "Priorité"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    pivot["TOTAL"] = pivot.select_dtypes(include="number").sum(axis=1)
    pivot = pivot.sort_values("TOTAL", ascending=False)
    st.dataframe(pivot, use_container_width=True, height=350)


# ─────────────────────────────────────────────
# PAGE 3 — PRÉDICTION ML
# ─────────────────────────────────────────────

def page_prediction(df, package):
    st.title("Prédire la criticité d'un chantier")
    st.divider()

    # ── Formulaire de prédiction ──────────────
    st.subheader("Saisir les caractéristiques du chantier")

    col1, col2, col3 = st.columns(3)

    with col1:
        infrapole = st.selectbox(
            "Infrapôle",
            sorted(df["InfPôle"].dropna().unique().tolist()),
        )
        jour = st.selectbox(
            "Jour de début",
            ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"],
        )

    with col2:
        code_ligne = st.number_input(
            "Code ligne (ex: 830000)",
            min_value=100000,
            max_value=999999,
            value=830000,
            step=1000,
        )

    with col3:
        pk_de = st.number_input(
            "PK début (km)",
            min_value=0.0,
            max_value=1500.0,
            value=380.0,
            step=0.5,
        )
        pk_fin = st.number_input(
            "PK fin (km)",
            min_value=0.0,
            max_value=1500.0,
            value=390.0,
            step=0.5,
        )

    pk_longueur = pk_fin - pk_de
    st.info(f"📏 Longueur du chantier calculée : **{pk_longueur:.1f} km**")

    # Ligne 2 : horaire
    col4, col5, col6 = st.columns(3)
    with col4:
        horaire_debut = st.select_slider(
            "⏰ Heure de début",
            options=[f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 15)],
            value="22:00",
        )
    with col5:
        horaire_fin = st.select_slider(
            "⏰ Heure de fin",
            options=[f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 15)],
            value="05:00",
        )
    with col6:
        # Calcul automatique de la durée
        h_deb = int(horaire_debut[:2]) * 60 + int(horaire_debut[3:])
        h_fin = int(horaire_fin[:2]) * 60 + int(horaire_fin[3:])
        duree_calc = h_fin - h_deb if h_fin > h_deb else h_fin - h_deb + 1440
        st.metric("⏱️ Durée calculée", f"{duree_calc} min ({duree_calc/60:.1f}h)")

    # Features dérivées
    horaire_dec = int(horaire_debut[:2]) + int(horaire_debut[3:]) / 60
    travaux_nuit = 1 if (int(horaire_debut[:2]) >= 20 or int(horaire_debut[:2]) < 6) else 0

    st.divider()

    # ── Bouton de prédiction ──────────────────
    if st.button("🔮 Prédire la priorité", type="primary", use_container_width=True):

        if pk_fin <= pk_de:
            st.error("Le PK de fin doit être supérieur au PK de début.")
            return

        le_infrapole = package["encoder_infrapole"]
        le_jour      = package["encoder_jour"]
        modele       = package["modele"]

        try:
            infrapole_enc = le_infrapole.transform([infrapole])[0]
        except ValueError:
            infrapole_enc = 0

        try:
            jour_enc = le_jour.transform([jour])[0]
        except ValueError:
            jour_enc = 0

        X_pred = pd.DataFrame([{
            "CODE_LIGNE":        code_ligne,
            "PK_DE":             pk_de,
            "PK_FIN":            pk_fin,
            "PK_longueur":       pk_longueur,
            "InfPôle_enc":       infrapole_enc,
            "Jour_enc":          jour_enc,
            "horaire_dec":       horaire_dec,
            "duree_horaire_min": duree_calc,
            "travaux_nuit":      travaux_nuit,
        }])

        # Prédiction
        prediction   = modele.predict(X_pred)[0]
        probabilites = modele.predict_proba(X_pred)[0]
        classes      = modele.classes_

        # ── Affichage du résultat ──────────────
        couleur = COULEURS_PRIORITE.get(prediction, "#888888")

        st.markdown("---")
        st.subheader("🎯 Résultat de la prédiction")

        col_res, col_prob = st.columns([1, 2])

        with col_res:
            st.markdown(
                f"""
                <div style="
                    background-color: {couleur}22;
                    border-left: 6px solid {couleur};
                    border-radius: 8px;
                    padding: 24px;
                    text-align: center;
                ">
                    <p style="font-size: 14px; color: #666; margin: 0;">Priorité prédite</p>
                    <p style="font-size: 28px; font-weight: bold; color: {couleur}; margin: 8px 0;">
                        {prediction}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_prob:
            st.markdown("**Probabilités par classe :**")
            prob_df = pd.DataFrame({
                "Priorité":    classes,
                "Probabilité": probabilites,
            }).sort_values("Probabilité", ascending=False)

            fig_prob = px.bar(
                prob_df,
                x="Probabilité",
                y="Priorité",
                orientation="h",
                color="Priorité",
                color_discrete_map=COULEURS_PRIORITE,
                text=prob_df["Probabilité"].apply(lambda x: f"{x*100:.1f}%"),
            )
            fig_prob.update_traces(textposition="outside")
            fig_prob.update_layout(
                height=220,
                margin=dict(t=10, b=10, l=10, r=80),
                showlegend=False,
                xaxis=dict(range=[0, 1.1], tickformat=".0%"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_prob, use_container_width=True)


def page_agent_ia(df):
    st.title("FAQ Travaux")
    st.markdown(
        "> 🎯 Posez vos questions sur les travaux ferroviaires en langage naturel. "
        "L'assistant analyse la base de données et vous répond instantanément."
    )

    # ── Vidéo explicative ─────────────────────
    with st.expander("🎬 Voir la vidéo de présentation de l'application"):
        # Remplace l'URL ci-dessous par le lien YouTube, Vimeo ou le chemin local de ta vidéo
        # Exemples :
        #   st.video("https://www.youtube.com/watch?v=TON_ID")
        #   st.video("https://vimeo.com/TON_ID")
        #   st.video("demo_app.mp4")  # si fichier local à la racine du projet
        VIDEO_URL = "https://www.youtube.com/watch?v=TON_ID"
        st.video(VIDEO_URL)
        st.caption(
            "Présentation de l'application de supervision des travaux ferroviaires "
            "SNCF Voyageurs — Axe Sud-Est"
        )

    st.divider()

    # ── Clé API (transparente pour l'utilisateur métier) ──
    if GROQ_API_KEY_ENV:
        api_key = GROQ_API_KEY_ENV
    else:
        st.warning(
            "⚠️ Aucune clé API Groq configurée côté serveur. "
            "Contacte l'administrateur de l'application pour activer l'assistant."
        )
        with st.expander("🔧 Configuration développeur"):
            api_key = st.text_input(
                "Clé API Groq (temporaire, session uniquement)",
                type="password",
                placeholder="gsk_...",
                help="À configurer définitivement dans .streamlit/secrets.toml"
            )
        if not api_key:
            return

    if not GROQ_DISPONIBLE:
        st.error("La librairie groq n'est pas installée. Lance : `pip install groq`")
        return

    # ── Contexte données ──────────────────────
    @st.cache_data
    def preparer_contexte(df):
        priorites  = df["Priorité"].value_counts().to_dict()
        infrapoles = df["InfPôle"].value_counts().head(8).to_dict()
        semaines   = trier_semaines(df["semaine"].unique().tolist())
        jours      = df["Jour"].value_counts().to_dict()
        nb_nuit    = int(df["travaux_nuit"].sum()) if "travaux_nuit" in df.columns else 0

        # Répartition par mois (si la colonne existe dans la base)
        mois_txt = ""
        if "mois_nom" in df.columns:
            mois_counts = (
                df.dropna(subset=["mois"])
                .sort_values("mois")
                .groupby(["mois", "mois_nom"])
                .size()
                .reset_index(name="nb")
            )
            mois_txt = "\n".join(
                [f"  • {row['mois_nom']} : {row['nb']} chantiers" for _, row in mois_counts.iterrows()]
            )

        return f"""
Tu es un assistant expert en supervision des travaux ferroviaires de SNCF Voyageurs (Axe Sud-Est).
Tu réponds en français, de façon concise et professionnelle.

BASE DE DONNÉES :
- {len(df)} chantiers au total sur {df['semaine'].nunique()} semaines
- Semaines : de {semaines[0]} à {semaines[-1]}
- {df['CODE_LIGNE'].nunique()} lignes ferroviaires

RÉPARTITION PAR PRIORITÉ :
{chr(10).join([f"  • {k} : {v} chantiers ({v/len(df)*100:.1f}%)" for k,v in priorites.items()])}

RÉPARTITION PAR MOIS :
{mois_txt if mois_txt else "  (non disponible — relancer le pipeline ETL mis à jour)"}

TOP INFRAPÔLES :
{chr(10).join([f"  • {k} : {v} chantiers" for k,v in infrapoles.items()])}

PAR JOUR :
{chr(10).join([f"  • {k} : {v} chantiers" for k,v in jours.items() if k])}

AUTRES STATS :
  • Travaux de nuit (début ≥ 20h ou < 6h) : {nb_nuit}

COLONNES DISPONIBLES : Intitulé Chantier, Priorité, InfPôle, CODE_LIGNE,
PK_DE, PK_FIN, Jour, Horaire, Horaire_fin, semaine, date_debut_semaine,
date_fin_semaine, mois, mois_nom, duree_horaire_min, travaux_nuit,
Sillons Ouvrants (contient les numéros de train à 4 chiffres impactés)

Réponds uniquement à partir de ces données. Sois précis avec les chiffres.
Si on te demande "quel mois a eu le plus de travaux", base-toi sur la section RÉPARTITION PAR MOIS ci-dessus.
Si on te demande "quel train est le plus impacté", base-toi UNIQUEMENT sur la section
"Top 10 trains" fournie dans le message (numéros à 4 chiffres extraits de Sillons Ouvrants),
ne confonds jamais un numéro de train avec un CODE_LIGNE (qui a 5-6 chiffres).

RÈGLE IMPORTANTE SUR LES DATES :
- Chaque semaine SNCF va toujours du LUNDI au DIMANCHE (semaine ISO).
- Ne calcule JAMAIS une date toi-même à partir du numéro de semaine.
- Si on te demande la date d'une semaine précise (ex: "quelle date pour S02 ?"),
  utilise UNIQUEMENT la correspondance exacte fournie dans le message
  (section "CORRESPONDANCE EXACTE SEMAINE -> DATES"). Si elle n'est pas fournie,
  dis que tu n'as pas l'information plutôt que d'estimer.
"""

    contexte = preparer_contexte(df)

    # ── Suggestions ───────────────────────────
    st.subheader("💡 Exemples de questions")
    exemples = [
        "Combien de chantiers Sévérisé fortifié ?",
        "Quelle infrapôle a le plus de travaux ?",
        "Combien de travaux de nuit ?",
        "Quel train est le plus impacté par les travaux ?",
        "Répartition des priorités ?",
        "Quel mois a eu le plus de travaux ?",
        "Quelle est la ligne la plus impactée ?",
        "Différence entre Sévérisé et Sévérisé fortifié ?",
    ]

    cols = st.columns(4)
    for i, ex in enumerate(exemples):
        with cols[i % 4]:
            if st.button(ex, use_container_width=True, key=f"ex_{i}"):
                st.session_state.messages_ia = st.session_state.get("messages_ia", [])
                st.session_state.messages_ia.append({"role": "user", "content": ex})
                st.session_state["question_en_attente"] = ex

    st.divider()

    # ── Historique ────────────────────────────
    if "messages_ia" not in st.session_state:
        st.session_state.messages_ia = []

    for msg in st.session_state.messages_ia:
        avatar = "🧑" if msg["role"] == "user" else "🚂"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # ── Input ─────────────────────────────────
    question = st.chat_input("Posez votre question sur les travaux ferroviaires...")

    # Récupérer question depuis bouton OU depuis chat_input
    question_a_traiter = question or st.session_state.pop("question_en_attente", None)

    if question_a_traiter:
        # Si vient du chat_input, l'ajouter à l'historique
        if question:
            st.session_state.messages_ia.append({"role": "user", "content": question_a_traiter})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(question_a_traiter)

        with st.chat_message("assistant", avatar="🚂"):
            with st.spinner("Analyse en cours..."):
                try:
                    client = Groq(api_key=api_key)

                    # Enrichir avec stats précises selon la question
                    stats_extra = ""
                    q = question_a_traiter.lower()
                    if any(m in q for m in ["nuit", "nocturne"]):
                        stats_extra += f"\nTravaux de nuit : {int(df['travaux_nuit'].sum())}"
                    if any(m in q for m in ["ligne", "rfn", "code"]):
                        top = df["CODE_LIGNE"].value_counts().head(10).to_dict()
                        stats_extra += f"\nTop 10 lignes : {top}"
                    if any(m in q for m in ["train", "sillon"]):
                        if "Sillons Ouvrants" in df.columns:
                            import re as _re
                            from collections import Counter as _Counter
                            tous_trains = []
                            for val in df["Sillons Ouvrants"].dropna():
                                tous_trains.extend(_re.findall(r'\b\d{4}\b', str(val)))
                            top_trains = _Counter(tous_trains).most_common(10)
                            stats_extra += (
                                "\nTop 10 trains (numéro de sillon à 4 chiffres) les plus "
                                "souvent concernés par des travaux, par nombre de mentions : "
                                f"{top_trains}"
                            )
                    if any(m in q for m in ["semaine", "période", "historique", "date", "s0", "s1", "s2"]):
                        if "date_debut_semaine" in df.columns:
                            mapping_semaines = (
                                df.dropna(subset=["date_debut_semaine", "date_fin_semaine"])
                                .drop_duplicates(subset=["semaine"])
                                .sort_values("date_debut_semaine")
                            )
                            lignes_mapping = [
                                f"{row['semaine']} : du {pd.to_datetime(row['date_debut_semaine']).strftime('%d.%m.%y')} "
                                f"(lundi) au {pd.to_datetime(row['date_fin_semaine']).strftime('%d.%m.%y')} (dimanche)"
                                for _, row in mapping_semaines.iterrows()
                            ]
                            stats_extra += (
                                "\nCORRESPONDANCE EXACTE SEMAINE -> DATES (chaque semaine commence "
                                "un lundi et finit un dimanche) :\n" + "\n".join(lignes_mapping)
                            )
                        else:
                            stats_extra += f"\nToutes les semaines : {trier_semaines(df['semaine'].unique().tolist())}"
                    if any(m in q for m in ["durée", "duree", "minutes"]):
                        moy = df["duree_horaire_min"].mean()
                        stats_extra += f"\nDurée moyenne chantier : {moy:.0f} min"
                    if any(m in q for m in ["mois", "janvier", "février", "fevrier", "mars",
                                              "avril", "mai", "juin", "juillet", "août", "aout",
                                              "septembre", "octobre", "novembre", "décembre", "decembre"]):
                        if "mois_nom" in df.columns:
                            mois_stats = (
                                df.dropna(subset=["mois"])
                                .groupby("mois_nom")
                                .size()
                                .sort_values(ascending=False)
                                .to_dict()
                            )
                            stats_extra += f"\nNombre exact de chantiers par mois : {mois_stats}"

                    messages_groq = [
                        {"role": "system", "content": contexte + stats_extra}
                    ] + st.session_state.messages_ia[-8:]

                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=messages_groq,
                        max_tokens=600,
                        temperature=0.2,
                    )

                    reponse = response.choices[0].message.content
                    st.markdown(reponse)
                    st.session_state.messages_ia.append(
                        {"role": "assistant", "content": reponse}
                    )

                except Exception as e:
                    msg_err = f"Erreur Groq : {e}"
                    st.error(msg_err)

    # ── Reset ─────────────────────────────────
    if st.session_state.get("messages_ia"):
        st.divider()
        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("🗑️ Effacer", use_container_width=True):
                st.session_state.messages_ia = []
                st.rerun()

    # ── À propos ──────────────────────────────
    with st.expander("ℹ️ À propos de l'assistant"):
        st.markdown("""
        **Modèle** : Llama 3.1 8B via Groq API (gratuit)
        **Données** : Base SQLite des travaux SNCF Voyageurs Axe Sud-Est
        **Fonctionnement** : L'assistant reçoit un résumé statistique des données
        et répond aux questions en langage naturel.
        **Limites** : Les réponses sont basées sur les statistiques agrégées,
        pas sur les lignes individuelles.
        """)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    df       = charger_donnees()
    package  = charger_modele()

    page, infrapoles_sel, priorites_sel, semaines_sel = sidebar(df)

    df_filtre = filtrer(df, infrapoles_sel, priorites_sel, semaines_sel)

    if page == "Carte des travaux":
        page_carte(df_filtre)

    elif page == "Tableau de bord":
        page_dashboard(df_filtre, df)

    elif page == "Prédire la criticité d'un chantier":
        if package is None:
            st.error("Modèle introuvable. Lance d'abord `python ml_model.py`.")
        else:
            page_prediction(df, package)

    elif page == "FAQ Travaux":
        page_agent_ia(df)


if __name__ == "__main__":
    if verifier_mot_de_passe():
        main()