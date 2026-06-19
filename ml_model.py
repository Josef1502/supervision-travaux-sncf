"""
ml_model.py
===========
Machine Learning supervisé — Classification de la Priorité des chantiers
SNCF Voyageurs — Axe Sud-Est

Problématique : Comment améliorer la supervision des travaux ferroviaires
grâce à l'exploitation des données historiques et à l'intelligence artificielle ?

Ce script :
1. Charge les données depuis la base SQLite
2. Prépare les features (encodage, nettoyage)
3. Entraîne et compare 4 algorithmes de classification
4. Affiche les métriques (Accuracy, F1, matrice de confusion)
5. Sauvegarde le meilleur modèle (model.pkl)

Usage :
    python ml_model.py
"""

import sqlite3
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DB_PATH    = "travaux_sncf.db"
MODEL_PATH = "model.pkl"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

PRIORITES_VALIDES = [
    "Sévérisé fortifié",
    "Sévérisé",
    "Chantier Local",
    "Autre"
]

# ─────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────

def charger_donnees(db_path: str) -> pd.DataFrame:
    """Charge la table travaux depuis la base SQLite."""
    print("=" * 55)
    print("ÉTAPE 1 — Chargement des données")
    print("=" * 55)

    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM travaux", conn)
    conn.close()

    print(f"  Lignes totales       : {len(df)}")
    print(f"  Semaines             : {df['semaine'].nunique()}")
    print(f"  Colonnes             : {len(df.columns)}")

    return df


# ─────────────────────────────────────────────
# 2. PRÉPARATION DES DONNÉES (FEATURE ENGINEERING)
# ─────────────────────────────────────────────

def preparer_features(df: pd.DataFrame):
    """
    Nettoie les données et crée les features pour le ML.

    Features utilisées :
    - CODE_LIGNE        : identifiant numérique de la ligne ferroviaire
    - PK_DE             : point kilométrique de début du chantier
    - PK_FIN            : point kilométrique de fin du chantier
    - PK_longueur       : longueur du chantier en km (PK_FIN - PK_DE)
    - InfPôle_enc       : infrapôle encodé numériquement
    - Jour_enc          : jour de la semaine encodé numériquement
    - horaire_dec       : heure de début en décimal (ex: 22.5 = 22h30)
    - duree_horaire_min : durée du chantier en minutes
    - travaux_nuit      : 1 si chantier de nuit (≥20h ou <6h), sinon 0

    Variable cible : Priorité (4 classes)
    """
    print("\n" + "=" * 55)
    print("ÉTAPE 2 — Préparation des features")
    print("=" * 55)

    df = df[df["Priorité"].isin(PRIORITES_VALIDES)].copy()
    print(f"  Lignes après filtre priorité : {len(df)}")

    # Feature engineering
    df["PK_longueur"]  = df["PK_FIN"] - df["PK_DE"]
    df["semaine_num"]  = df["semaine"].str.extract(r"(\d+)").astype(float)

    # Horaire → décimal (ex: "22:30" → 22.5)
    def h_to_dec(h):
        try:
            if isinstance(h, str) and ":" in h:
                return int(h[:2]) + int(h[3:]) / 60
        except Exception:
            pass
        return None

    df["horaire_dec"] = df["Horaire"].apply(h_to_dec)

    # Encodage des variables catégorielles
    le_infrapole = LabelEncoder()
    le_jour      = LabelEncoder()

    df["InfPôle_enc"] = le_infrapole.fit_transform(df["InfPôle"].fillna("Inconnu"))
    df["Jour_enc"]    = le_jour.fit_transform(df["Jour"].fillna("Inconnu"))

    joblib.dump(le_infrapole, "encoder_infrapole.pkl")
    joblib.dump(le_jour,      "encoder_jour.pkl")

    print(f"  InfPôles encodés : {list(le_infrapole.classes_)}")

    features = [
        "CODE_LIGNE",
        "PK_DE",
        "PK_FIN",
        "PK_longueur",
        "InfPôle_enc",
        "Jour_enc",
        "horaire_dec",
        "duree_horaire_min",
        "travaux_nuit",
    ]

    df_ml = df[features + ["Priorité"]].dropna()
    print(f"  Lignes utilisables   : {len(df_ml)}")
    print(f"  Lignes supprimées    : {len(df) - len(df_ml)} (valeurs manquantes)")

    print("\n  Répartition de la variable cible (Priorité) :")
    for priorite, nb in df_ml["Priorité"].value_counts().items():
        pct = nb / len(df_ml) * 100
        print(f"    {priorite:<25s} : {nb:4d} ({pct:.1f}%)")

    X = df_ml[features]
    y = df_ml["Priorité"]

    return X, y, features, le_infrapole, le_jour


# ─────────────────────────────────────────────
# 3. SPLIT TRAIN / TEST
# ─────────────────────────────────────────────

def splitter(X, y):
    """Découpe les données en jeu d'entraînement (80%) et de test (20%)."""
    print("\n" + "=" * 55)
    print("ÉTAPE 3 — Split Train / Test")
    print("=" * 55)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y          # respecte la répartition des classes
    )

    print(f"  Jeu d'entraînement   : {len(X_train)} lignes ({100-TEST_SIZE*100:.0f}%)")
    print(f"  Jeu de test          : {len(X_test)} lignes ({TEST_SIZE*100:.0f}%)")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 4. ENTRAÎNEMENT ET COMPARAISON DES MODÈLES
# ─────────────────────────────────────────────

def entrainer_modeles(X_train, X_test, y_train, y_test):
    """
    Entraîne et compare 4 algorithmes de classification supervisée :
    - Decision Tree
    - Random Forest
    - K-Nearest Neighbours (KNN)
    - Logistic Regression
    """
    print("\n" + "=" * 55)
    print("ÉTAPE 4 — Entraînement et comparaison des modèles")
    print("=" * 55)

    modeles = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=15
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
    }

    resultats = {}

    print(f"\n  {'Algorithme':<25s} {'Accuracy':>10s} {'F1-score':>10s} {'Temps CV':>10s}")
    print(f"  {'-'*55}")

    for nom, modele in modeles.items():
        import time

        # Entraînement
        t0 = time.time()
        modele.fit(X_train, y_train)
        temps_entrainement = time.time() - t0

        # Prédictions sur le jeu de test
        y_pred = modele.predict(X_test)

        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1       = f1_score(y_test, y_pred, average="weighted")

        # Validation croisée (5 folds) sur le jeu d'entraînement
        t0 = time.time()
        cv_scores = cross_val_score(modele, X_train, y_train, cv=5, scoring="accuracy")
        temps_cv  = time.time() - t0

        resultats[nom] = {
            "modele"     : modele,
            "accuracy"   : accuracy,
            "f1"         : f1,
            "cv_mean"    : cv_scores.mean(),
            "cv_std"     : cv_scores.std(),
            "y_pred"     : y_pred,
            "temps_cv"   : temps_cv,
        }

        print(f"  {nom:<25s} {accuracy:>10.3f} {f1:>10.3f} {temps_cv:>9.2f}s")

    return resultats


# ─────────────────────────────────────────────
# 5. RAPPORT DÉTAILLÉ
# ─────────────────────────────────────────────

def afficher_rapport(resultats, y_test):
    """Affiche le rapport détaillé pour chaque algorithme."""
    print("\n" + "=" * 55)
    print("ÉTAPE 5 — Rapport détaillé")
    print("=" * 55)

    for nom, r in resultats.items():
        print(f"\n── {nom} ──")
        print(f"  Accuracy test  : {r['accuracy']:.4f} ({r['accuracy']*100:.2f}%)")
        print(f"  F1-score       : {r['f1']:.4f}")
        print(f"  CV Accuracy    : {r['cv_mean']:.4f} ± {r['cv_std']:.4f}")
        print("\n  Rapport de classification :")
        print(classification_report(
            y_test, r["y_pred"],
            target_names=PRIORITES_VALIDES,
            zero_division=0
        ))


# ─────────────────────────────────────────────
# 6. MATRICES DE CONFUSION
# ─────────────────────────────────────────────

def tracer_matrices_confusion(resultats, y_test):
    """Génère et sauvegarde les matrices de confusion pour chaque modèle."""
    print("\n" + "=" * 55)
    print("ÉTAPE 6 — Matrices de confusion")
    print("=" * 55)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Matrices de confusion — Classification de la Priorité\nSNCF Voyageurs Axe Sud-Est",
        fontsize=14, fontweight="bold"
    )

    couleurs = ["Blues", "Greens", "Oranges", "Purples"]

    for idx, (nom, r) in enumerate(resultats.items()):
        ax = axes[idx // 2][idx % 2]
        cm = confusion_matrix(y_test, r["y_pred"], labels=PRIORITES_VALIDES)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap=couleurs[idx],
            xticklabels=PRIORITES_VALIDES,
            yticklabels=PRIORITES_VALIDES,
            ax=ax, linewidths=0.5
        )
        ax.set_title(
            f"{nom}\nAccuracy : {r['accuracy']*100:.1f}%",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Prédit", fontsize=10)
        ax.set_ylabel("Réel", fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    plt.tight_layout()
    plt.savefig("matrices_confusion.png", dpi=150, bbox_inches="tight")
    print("  ✅ Sauvegardé : matrices_confusion.png")
    plt.close()


# ─────────────────────────────────────────────
# 7. IMPORTANCE DES FEATURES (Random Forest)
# ─────────────────────────────────────────────

def tracer_importance_features(resultats, features):
    """Affiche l'importance des features du Random Forest."""
    print("\n" + "=" * 55)
    print("ÉTAPE 7 — Importance des features (Random Forest)")
    print("=" * 55)

    rf = resultats["Random Forest"]["modele"]
    importances = pd.Series(
        rf.feature_importances_, index=features
    ).sort_values(ascending=True)

    print("\n  Feature importances :")
    for feat, imp in importances.sort_values(ascending=False).items():
        print(f"    {feat:<20s} : {imp:.4f} ({imp*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#C0272D" if imp > 0.2 else "#5B9BD5" for imp in importances]
    importances.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(
        "Importance des features — Random Forest\nSNCF Voyageurs Axe Sud-Est",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Importance", fontsize=10)
    ax.axvline(x=0.2, color="#C0272D", linestyle="--", alpha=0.5, label="Seuil 20%")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("importance_features.png", dpi=150, bbox_inches="tight")
    print("  ✅ Sauvegardé : importance_features.png")
    plt.close()


# ─────────────────────────────────────────────
# 8. COMPARAISON VISUELLE DES MODÈLES
# ─────────────────────────────────────────────

def tracer_comparaison(resultats):
    """Graphique de comparaison des 4 algorithmes."""
    noms       = list(resultats.keys())
    accuracies = [r["accuracy"] for r in resultats.values()]
    f1_scores  = [r["f1"]       for r in resultats.values()]
    cv_means   = [r["cv_mean"]  for r in resultats.values()]

    x = np.arange(len(noms))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width,   accuracies, width, label="Accuracy (test)",  color="#5B9BD5", alpha=0.9)
    bars2 = ax.bar(x,           f1_scores,  width, label="F1-score (test)",  color="#70AD47", alpha=0.9)
    bars3 = ax.bar(x + width,   cv_means,   width, label="Accuracy (CV 5)",  color="#C0272D", alpha=0.9)

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(noms, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Comparaison des algorithmes — Classification de la Priorité\nSNCF Voyageurs Axe Sud-Est",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.4, label="Seuil 80%")

    # Annotations
    for bar in [*bars1, *bars2, *bars3]:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8
        )

    plt.tight_layout()
    plt.savefig("comparaison_modeles.png", dpi=150, bbox_inches="tight")
    print("  ✅ Sauvegardé : comparaison_modeles.png")
    plt.close()


# ─────────────────────────────────────────────
# 9. SAUVEGARDER LE MEILLEUR MODÈLE
# ─────────────────────────────────────────────

def sauvegarder_meilleur_modele(resultats, features, le_infrapole, le_jour):
    """Sélectionne et sauvegarde le meilleur modèle."""
    print("\n" + "=" * 55)
    print("ÉTAPE 8 — Sauvegarde du meilleur modèle")
    print("=" * 55)

    meilleur_nom = max(resultats, key=lambda k: resultats[k]["accuracy"])
    meilleur     = resultats[meilleur_nom]

    print(f"\n  🏆 Meilleur modèle    : {meilleur_nom}")
    print(f"     Accuracy test      : {meilleur['accuracy']*100:.2f}%")
    print(f"     F1-score           : {meilleur['f1']:.4f}")
    print(f"     CV Accuracy        : {meilleur['cv_mean']:.4f} ± {meilleur['cv_std']:.4f}")

    # Package complet : modèle + métadonnées
    package = {
        "modele"          : meilleur["modele"],
        "nom_modele"      : meilleur_nom,
        "features"        : features,
        "priorites"       : PRIORITES_VALIDES,
        "encoder_infrapole": le_infrapole,
        "encoder_jour"    : le_jour,
        "accuracy"        : meilleur["accuracy"],
        "f1"              : meilleur["f1"],
    }

    joblib.dump(package, MODEL_PATH)
    print(f"\n  ✅ Modèle sauvegardé  : {MODEL_PATH}")
    print(f"  📦 Contenu du package :")
    print(f"     - modele           : objet {meilleur_nom}")
    print(f"     - features         : {features}")
    print(f"     - encoder_infrapole: LabelEncoder (18 classes)")
    print(f"     - encoder_jour     : LabelEncoder")

    return meilleur_nom


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def main():
    print("\n" + "🚂 " * 18)
    print("MACHINE LEARNING — SNCF Voyageurs Axe Sud-Est")
    print("Classification automatique de la Priorité des chantiers")
    print("🚂 " * 18 + "\n")

    # 1. Chargement
    df = charger_donnees(DB_PATH)

    # 2. Features
    X, y, features, le_infrapole, le_jour = preparer_features(df)

    # 3. Split
    X_train, X_test, y_train, y_test = splitter(X, y)

    # 4. Entraînement
    resultats = entrainer_modeles(X_train, X_test, y_train, y_test)

    # 5. Rapport détaillé
    afficher_rapport(resultats, y_test)

    # 6. Matrices de confusion
    tracer_matrices_confusion(resultats, y_test)

    # 7. Importance des features
    tracer_importance_features(resultats, features)

    # 8. Comparaison visuelle
    tracer_comparaison(resultats)

    # 9. Sauvegarde
    meilleur = sauvegarder_meilleur_modele(
        resultats, features, le_infrapole, le_jour
    )

    print("\n" + "=" * 55)
    print("✅ PIPELINE ML TERMINÉ")
    print("=" * 55)
    print(f"  Meilleur modèle    : {meilleur}")
    print(f"  Fichiers générés   :")
    print(f"    - model.pkl")
    print(f"    - encoder_infrapole.pkl")
    print(f"    - encoder_jour.pkl")
    print(f"    - matrices_confusion.png")
    print(f"    - importance_features.png")
    print(f"    - comparaison_modeles.png")
    print("\n  → Prochaine étape : python app.py (Streamlit)")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
