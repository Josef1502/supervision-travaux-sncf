"""
pipeline_etl.py
===============
Pipeline ETL — Supervision des Travaux Ferroviaires
SNCF Voyageurs — Axe Sud-Est

Ce script fusionne tous les fichiers Excel hebdomadaires (.xlsx / .xlsm)
d'un dossier et les charge dans une base SQLite unique.

Usage :
    python pipeline_etl.py --dossier ./data --db travaux_sncf.db

Auteur : Alternant Data Analyst — Direction Axe Sud-Est
"""

import re
import sqlite3
import argparse
import logging
import datetime
from pathlib import Path

import pandas as pd

# ── Configuration ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

JOURS_VALIDES = {
    "Lundi", "Mardi", "Mercredi", "Jeudi",
    "Vendredi", "Samedi", "Dimanche"
}

PRIORITES_VALIDES = {
    "Sévérisé fortifié", "Sévérisé", "Chantier Local", "Autre"
}


# ── Fonctions de nettoyage ─────────────────────────────────────

def extraire_semaine(nom_fichier: str) -> str:
    """
    Extrait la clé unique 'S08_2026' depuis le nom du fichier.
    Gère :
      - Travaux_..._S08_du..._26  → S08_2026
      - Travaux_..._S08_du..._25  → S08_2025
      - Travaux_..._S08_du..._261 → S08_2026 (doublon S02)
    """
    nom = nom_fichier

    # Extraire le numéro de semaine
    m_sem = re.search(r'_S(\d{2})_', nom, re.IGNORECASE)
    sem = f"S{m_sem.group(1)}" if m_sem else "SINCONNUE"

    # Extraire l'année (2 derniers chiffres avant l'extension ou en fin de nom)
    # Cherche _25, _26, _251, _261 etc. en fin de nom
    m_an = re.search(r'_(\d{2})(?:\d*)(?:\.xlsx?|\.xlsm)?$', nom, re.IGNORECASE)
    if m_an:
        an_court = m_an.group(1)
        annee = f"20{an_court}"
    else:
        annee = "2026"  # par défaut

    return f"{sem}_{annee}"


def nettoyer_code_ligne(val) -> int | None:
    """
    Normalise CODE_LIGNE.
    Gère : entier, float, chaîne avec \\n, \\xa0, virgule, vide.
    """
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    s = (str(val)
         .strip()
         .replace('\n', '')
         .replace('\xa0', '')
         .replace(' ', '')
         .replace(',', '')
         .split('.')[0])          # retire la partie décimale si float
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def nettoyer_pk(val) -> float | None:
    """
    Normalise PK_DE et PK_FIN.
    Gère : virgule décimale, espaces, sauts de ligne.
    """
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(
            str(val).strip()
            .replace(',', '.')
            .replace(' ', '')
            .split('\n')[0]
        )
    except ValueError:
        return None


def nettoyer_horaire(val):
    """
    Convertit une valeur horaire en chaîne 'HH:MM'.
    Gère : datetime.time, fraction décimale Excel, texte '14:20', texte "'14:20'".
    """
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if val is None:
        return None
    # Objet time (openpyxl le lit directement)
    if hasattr(val, 'hour'):
        return f"{val.hour:02d}:{val.minute:02d}"
    # Texte : '14:20' ou "'14:20'" ou "14:20"
    if isinstance(val, str):
        clean = val.strip().strip("'\"")   # retire les apostrophes/guillemets
        if ":" in clean:
            parts = clean.split(":")
            try:
                h = int(parts[0])
                m = int(parts[1][:2])
                if 0 <= h <= 23 and 0 <= m <= 59:
                    return f"{h:02d}:{m:02d}"
            except (ValueError, IndexError):
                pass
        return None
    # Fraction décimale Excel (ex: 0.979 = 23h30)
    try:
        total = round(float(val) * 24 * 60)
        return f"{total // 60:02d}:{total % 60:02d}"
    except (ValueError, TypeError):
        return None


def nettoyer_jour(val) -> str | None:
    """
    Normalise les colonnes Jour et Jour_fin.
    Ex : 'Lundi_fin' → 'Lundi', 'MARDI' → 'Mardi'.
    """
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    jour = str(val).strip().replace('_fin', '').capitalize()
    return jour if jour in JOURS_VALIDES else None


def calculer_duree(h_debut: str, h_fin: str) -> int | None:
    """
    Calcule la durée en minutes entre deux horaires 'HH:MM'.
    Gère les travaux de nuit (ex. 23h00 → 04h00 = 300 min).
    """
    try:
        if not isinstance(h_debut, str) or not isinstance(h_fin, str):
            return None
        hd = int(h_debut[:2]) * 60 + int(h_debut[3:])
        hf = int(h_fin[:2]) * 60 + int(h_fin[3:])
        diff = hf - hd
        return diff + 1440 if diff < 0 else diff
    except (ValueError, IndexError):
        return None


def est_travaux_nuit(horaire: str) -> int | None:
    """Retourne 1 si le chantier débute entre 20h et 06h, sinon 0."""
    if not isinstance(horaire, str):
        return None
    try:
        h = int(horaire[:2])
        return 1 if h >= 20 or h < 6 else 0
    except ValueError:
        return None


MOIS_FR = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre",
}


def semaine_vers_dates(semaine_str: str):
    """
    Convertit une clé semaine 'S08_2026' en (date_debut, date_fin)
    de la semaine ISO correspondante (lundi -> dimanche).
    Retourne (None, None) si le format est invalide.
    """
    try:
        sem_num = int(semaine_str[1:3])
        annee = int(semaine_str[4:8])
        date_debut = datetime.date.fromisocalendar(annee, sem_num, 1)  # lundi
        date_fin = datetime.date.fromisocalendar(annee, sem_num, 7)    # dimanche
        return date_debut, date_fin
    except (ValueError, IndexError):
        return None, None


# ── Lecture et nettoyage d'un fichier ─────────────────────────

def lire_fichier(path: Path) -> pd.DataFrame:
    """
    Lit un fichier Excel et nettoie toutes les colonnes.
    Essaie d'abord l'onglet 'Travaux', sinon le premier onglet.
    """
    try:
        df = pd.read_excel(path, sheet_name="Travaux", engine="openpyxl")
    except Exception:
        df = pd.read_excel(path, sheet_name=0, engine="openpyxl")

    # Renommer la colonne au nom long
    df.columns = [
        "Type_Interception" if "Type Interception" in str(c) else c
        for c in df.columns
    ]

    semaine = extraire_semaine(path.name)
    df["semaine"] = semaine
    df["fichier_source"] = path.name

    # Dates réelles de la semaine (lundi -> dimanche) + mois associé
    date_debut_sem, date_fin_sem = semaine_vers_dates(semaine)
    df["date_debut_semaine"] = date_debut_sem
    df["date_fin_semaine"]   = date_fin_sem
    if date_debut_sem is not None:
        df["mois"] = date_debut_sem.strftime("%Y-%m")          # ex: "2026-01"
        df["mois_nom"] = f"{MOIS_FR[date_debut_sem.month]} {date_debut_sem.year}"  # ex: "Janvier 2026"
        # Label lisible pour l'affichage : "05.01_au_11.01_2026"
        df["semaine_label"] = (
            f"{date_debut_sem.strftime('%d.%m')}_au_"
            f"{date_fin_sem.strftime('%d.%m')}_{date_debut_sem.year}"
        )
    else:
        df["mois"] = None
        df["mois_nom"] = None
        df["semaine_label"] = semaine  # repli sur le code brut si dates indisponibles

    # Nettoyages
    df["CODE_LIGNE"]   = df["CODE_LIGNE"].apply(nettoyer_code_ligne)
    df["PK_DE"]        = df["PK_DE"].apply(nettoyer_pk)
    df["PK_FIN"]       = df["PK_FIN"].apply(nettoyer_pk)
    df["Horaire"]      = df["Horaire"].apply(nettoyer_horaire)
    df["Horaire_fin"]  = df["Horaire_fin"].apply(nettoyer_horaire)
    df["Jour"]         = df["Jour"].apply(nettoyer_jour)
    df["Jour_fin"]     = df["Jour_fin"].apply(nettoyer_jour)

    # Colonnes texte
    for col in ["Intitulé Chantier", "InfPôle", "Priorité", "Infos Chantier"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace("nan", None)

    # Dates
    for col in ["Date Début", "Date de fin"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce",
                                     dayfirst=True).dt.date

    # Colonnes calculées
    df["duree_horaire_min"] = df.apply(
        lambda r: calculer_duree(r["Horaire"], r["Horaire_fin"]), axis=1
    )
    df["travaux_nuit"] = df["Horaire"].apply(est_travaux_nuit)

    # Supprimer les lignes vides (sans CODE_LIGNE)
    df = df[df["CODE_LIGNE"].notna()].copy()

    return df


# ── Rapport qualité ────────────────────────────────────────────

def rapport_qualite(df: pd.DataFrame, semaine: str) -> dict:
    return {
        "semaine": semaine,
        "lignes": len(df),
        "pk_manquants": int(df["PK_DE"].isna().sum()),
        "priorite_inconnue": int(
            (~df["Priorité"].isin(PRIORITES_VALIDES)).sum()
        ),
        "travaux_nuit": int(df["travaux_nuit"].sum())
        if "travaux_nuit" in df else 0,
    }


# ── Pipeline principal ─────────────────────────────────────────

def run(dossier: str, db_path: str):
    dossier = Path(dossier)

    # Lister les fichiers (xlsx + xlsm)
    fichiers = sorted(dossier.glob("*.xlsx")) + sorted(dossier.glob("*.xlsm"))
    if not fichiers:
        log.warning(f"Aucun fichier trouvé dans {dossier}")
        return

    # Déduplique : garde un seul fichier par (semaine + année)
    # Ex : S08_2025 et S08_2026 sont deux entrées distinctes
    seen: dict[str, Path] = {}
    for f in fichiers:
        cle = extraire_semaine(f.name)
        if cle not in seen or f.stat().st_size > seen[cle].stat().st_size:
            seen[cle] = f
    fichiers_uniques = sorted(seen.values(),
                              key=lambda f: extraire_semaine(f.name))

    log.info(f"Fichiers à traiter : {len(fichiers_uniques)}")

    # Lecture de tous les fichiers
    dfs, rapports = [], []
    for f in fichiers_uniques:
        cle = extraire_semaine(f.name)
        try:
            df = lire_fichier(f)
            dfs.append(df)
            rapports.append(rapport_qualite(df, cle))
            log.info(f"  ✅ {cle} — {len(df)} lignes ({f.name})")
        except Exception as e:
            log.error(f"  ❌ {cle} — {f.name} : {e}")

    if not dfs:
        log.error("Aucun fichier chargé avec succès.")
        return

    # Fusion
    df_all = pd.concat(dfs, ignore_index=True)
    log.info(f"\nTotal fusionné : {len(df_all)} lignes | "
             f"{df_all['semaine'].nunique()} semaines")

    # Chargement SQLite
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS travaux")
    df_all.to_sql("travaux", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_semaine  ON travaux(semaine)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_priorite ON travaux(Priorité)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ligne    ON travaux(CODE_LIGNE)")
    conn.commit()
    conn.close()

    log.info(f"✅ Base SQLite créée : {db_path}")

    # Affichage du rapport
    print("\n" + "=" * 55)
    print("RAPPORT QUALITÉ PAR SEMAINE")
    print("=" * 55)
    print(f"{'Semaine':<10} {'Lignes':>7} {'PK manq.':>10} "
          f"{'Priorité ?':>12} {'Nuit':>6}")
    print("-" * 55)
    for r in rapports:
        print(f"{r['semaine']:<10} {r['lignes']:>7} "
              f"{r['pk_manquants']:>10} "
              f"{r['priorite_inconnue']:>12} "
              f"{r['travaux_nuit']:>6}")
    print("=" * 55)

    total = sum(r["lignes"] for r in rapports)
    print(f"\nTOTAL : {total} lignes chargées depuis "
          f"{len(rapports)} semaines\n")

    print("Répartition par Priorité :")
    print(df_all["Priorité"].value_counts().to_string())
    print("\nRépartition par InfPôle :")
    print(df_all["InfPôle"].value_counts().to_string())


# ── Point d'entrée ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline ETL — Travaux ferroviaires SNCF → SQLite"
    )
    parser.add_argument(
        "--dossier", default="./data",
        help="Dossier contenant les fichiers Excel (défaut: ./data)"
    )
    parser.add_argument(
        "--db", default="travaux_sncf.db",
        help="Chemin de la base SQLite (défaut: travaux_sncf.db)"
    )
    args = parser.parse_args()
    run(args.dossier, args.db)