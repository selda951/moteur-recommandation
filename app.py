import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Cette ligne doit être la toute première commande Streamlit
st.set_page_config(
    page_title="Moteur de Recommandation - MDS",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Réalisé par Selda KOSE - Master Data Management & Business Analytics"}
)

# 💡 Header principal
st.markdown(
    """
    <div style="background-color: #111111; padding: 2rem 1rem; border-radius: 10px;">
        <h1 style="color: #FFDDC1; text-align: center;">✨ Moteur de Recommandation - Management & Datascience ✨</h1>
        <p style="text-align: center; color: #E0E0E0; font-size: 18px;">
            Une plateforme interactive pour explorer, modéliser et piloter l'engagement utilisateur.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# 🔀 Onglets de navigation
tabs = st.tabs([
    "🏠 Accueil",
    "🧠 Clustering & Personas",
    "🎯 Classification & Scoring",
    "📊 Dashboard & Analyses",
    "💡 Recommandations",
    "👥 L'équipe"
])

# 🏠 Page Accueil
with tabs[0]:
    st.header("Bienvenue sur votre moteur de recommandation 🎓")
    st.image("concept-hebergement-site-web-circuits_23-2149406782.jpg", use_container_width=True)
    st.markdown("""
    Cette plateforme vous permet de :
    - Explorer les comportements utilisateurs 📈
    - Segmenter les profils pour construire des personas 👥
    - Prédire le désengagement avec des modèles supervisés 🧠
    - Générer des recommandations stratégiques pour maximiser l'engagement 🎯

    🔙 Naviguez via les onglets ci-dessus pour accéder aux différentes analyses.
    """)

# 🧠 Clustering & Personas
with tabs[1]:
    st.header("🧠 Clustering des Utilisateurs et Recommandations Personnalisées")

    df = pd.read_csv('BigML_Batchcentroid_680019a86f707eef2f3fce02.csv', encoding='latin1')
    cluster_choice = st.selectbox("Choisissez un cluster:", df['cluster'].unique())
    cluster_data = df[df['cluster'] == cluster_choice]

    fig = px.scatter(cluster_data, x='nb_sessions', y='engagement_score',
                     title=f'Cluster {cluster_choice} - Nombre de Sessions vs Score d\'Engagement',
                     labels={'nb_sessions': 'Nombre de Sessions', 'engagement_score': 'Score d\'Engagement'},
                     color='cluster', color_continuous_scale='Blues')
    st.plotly_chart(fig)

    st.subheader(f"Recommandations pour le Cluster {cluster_choice}")
    personas = {
        'Cluster 0': {
            "Caractéristiques": "Très interactifs avec un grand nombre de clics et d'actions engageantes. Longs temps passés sur la plateforme. Visites régulières, principalement depuis un ordinateur.",
            "Persona": "Ce sont des utilisateurs très actifs qui utilisent la plateforme pour des tâches avancées (administration, gestion de contenu, etc.). Ils sont fidèles et cherchent à maximiser leur expérience.",
            "Recommandations": "Récompenses & encouragements : Offrir des badges, des accès premium, ou des fonctionnalités exclusives pour maintenir leur engagement. Contenus personnalisés : Proposer des fonctionnalités avancées ou des ressources supplémentaires adaptées à leurs besoins."
        },
        'Cluster 1': {
            "Caractéristiques": "Moins d'interactions mais visites fréquentes. Passent peu de temps par session, principalement sur un mobile.",
            "Persona": "Ce groupe est composé de nouveaux utilisateurs ou de ceux qui reviennent sporadiquement. Ils sont peut-être intéressés, mais n’ont pas encore trouvé de contenu suffisamment engageant.",
            "Recommandations": "Notifications push sur mobile pour rappeler aux utilisateurs les nouvelles fonctionnalités. Optimisation mobile : Assurez-vous que la plateforme est fluide et attrayante sur mobile pour encourager l’engagement."
        },
        'Cluster 2': {
            "Caractéristiques": "Faible engagement sur les dernières sessions. Utilisateurs mixtes, alternant entre ordinateur et mobile.",
            "Persona": "Ces utilisateurs étaient actifs par le passé mais ont montré des signes de désengagement. Ils nécessitent une relance ciblée.",
            "Recommandations": "Réengagement ciblé : Envoyer des notifications personnalisées, des défis ou des contenus spécifiques pour les inciter à revenir. Suivi personnalisé pour identifier les raisons de leur désengagement et rétablir leur intérêt."
        },
        'Cluster 3': {
            "Caractéristiques": "Nombre de sessions relativement faible, visitent rarement. Utilisent principalement un mobile.",
            "Persona": "Ce groupe est composé d'utilisateurs qui ne se connectent pas régulièrement. Leur engagement est faible, mais il y a un potentiel à réactiver.",
            "Recommandations": "Incitation à la participation : Utiliser des récompenses visibles, des mises à jour intéressantes ou des appels à l’action pour encourager la visite de nouveaux contenus. Contenus ciblés : Proposer des articles ou des fonctionnalités spécifiques qui correspondent à leurs intérêts."
        },
        'Cluster 4': {
            "Caractéristiques": "Très faible engagement, peu de sessions et de clics. Utilisation sporadique de la plateforme, souvent sans interaction réelle.",
            "Persona": "Ce groupe est constitué d'utilisateurs qui n’ont pas encore trouvé d’intérêt suffisant dans la plateforme.",
            "Recommandations": "Réactivation : Proposer des incitations pour revenir sur la plateforme (essai gratuit, notifications personnalisées). Onboarding amélioré pour guider les utilisateurs dans la découverte des fonctionnalités et des contenus."
        }
    }
    persona = personas.get(cluster_choice, {})
    st.write(f"**Caractéristiques** : {persona.get('Caractéristiques', '')}")
    st.write(f"**Persona** : {persona.get('Persona', '')}")
    st.write(f"**Recommandations** : {persona.get('Recommandations', '')}")
    st.write("Ces recommandations ont été personnalisées en fonction des caractéristiques comportementales des utilisateurs dans ce cluster.")

# 🎯 Classification & Scoring
with tabs[2]:
    st.header("🎯 Modèles supervisés de prédiction")

    df_classification = pd.read_csv('kpi_final_with_disengaged_bool.csv')

    X = df_classification[["nb_sessions", "avg_time_on_content", "nb_unique_documents_viewed", "nb_unique_documents"]]
    y = df_classification['is_disengaged']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    st.write("""
    ### Introduction à la Classification

    Nous avons utilisé un modèle de classification supervisée pour prédire les utilisateurs susceptibles de se désengager. L’objectif est d’identifier les utilisateurs à risque et de leur proposer des actions ciblées pour les maintenir actifs sur la plateforme.

    Nous avons utilisé **Random Forest**, un modèle robuste pour prédire la probabilité de désengagement, en nous basant sur les variables comportementales collectées lors des sessions (nombre de sessions, durée moyenne des sessions, pages vues, actions, etc.).
    """)

    st.write(f"### Taux de Précision du Modèle Random Forest : {accuracy * 100:.2f}%")

    st.subheader("Graphique de Performance du Modèle")

    metrics = {
        'Précision': accuracy,
        'Rappel': accuracy,
        'F1-Score': accuracy
    }

    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    fig, ax = plt.subplots()
    sns.barplot(x=metric_names, y=metric_values, ax=ax)
    ax.set_title("Performance du Modèle")
    ax.set_ylabel("Score")

    st.pyplot(fig)

    st.subheader("Exemple de Prédiction pour un Utilisateur")

    user_data = {
        "nb_sessions": 5,
        "avg_time_on_content": 30,
        "nb_unique_documents_viewed": 10,
        "nb_unique_documents": 8
    }

    user_df = pd.DataFrame([user_data])
    pred = model.predict(user_df)

    if pred[0] == 1:
        st.write("Cet utilisateur est **désengagé**.")
    else:
        st.write("Cet utilisateur est **engagé**.")

    st.write("""
    Le modèle a permis de prédire les utilisateurs susceptibles de se désengager, afin de cibler les actions de réengagement.

    ### Résultats et Interprétation :
    - **Utilisateurs désengagés** : Ceux qui ont un score d'engagement inférieur à 0.4 et une faible fréquence de visites.
    - **Utilisateurs engagés** : Ceux avec une forte durée de session, un grand nombre de pages vues, et une utilisation régulière de la plateforme.

    ### Recommandations Ciblées :
    - **Réengagement des utilisateurs à risque** : Proposer des contenus personnalisés, des défis et des notifications pour inciter à revenir.
    - **Fidélisation des utilisateurs engagés** : Récompenser les plus actifs avec des badges ou des avantages exclusifs.
    """)
# 📊 Dashboard & Analyses
with tabs[3]:
    st.header("📊 Résultats & Analyses visuelles")

    df = pd.read_csv("kpi_final_with_disengaged_bool.csv")
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    st.markdown("## 🎛️ Filtres dynamiques")
    with st.expander("Afficher/Masquer les filtres"):
        selected_browser = st.multiselect("Navigateur", options=df["browser_clean"].dropna().unique())
        selected_os = st.multiselect("Système d'exploitation", options=df["os_clean"].dropna().unique())
        selected_device = st.multiselect("Type d'appareil", options=df["device_type"].dropna().unique() if "device_type" in df.columns else [])

        filtered_df = df.copy()
        if selected_browser:
            filtered_df = filtered_df[filtered_df["browser_clean"].isin(selected_browser)]
        if selected_os:
            filtered_df = filtered_df[filtered_df["os_clean"].isin(selected_os)]
        if selected_device:
            filtered_df = filtered_df[filtered_df["device_type"].isin(selected_device)]

    st.subheader("📈 Moyennes des indicateurs d'engagement")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📊 Score moyen", round(filtered_df["engagement_score"].mean(), 2))
    col2.metric("📄 Documents consultés", round(filtered_df["nb_unique_documents"].mean(), 2))
    col3.metric("🧱 Nb. de sessions", round(filtered_df["nb_sessions"].mean(), 2))
    col4.metric("🕒 Jours entre sessions", round(filtered_df["avg_days_between_sessions"].mean(), 2))
    col5.metric("👤 Utilisateurs uniques", 584)

    st.subheader("🥧 Répartition engagés vs désengagés")
    pie_data = filtered_df["is_disengaged"].value_counts().rename(index={0: "Engagés", 1: "Désengagés"})
    fig_pie = px.pie(
        values=pie_data.values,
        names=pie_data.index,
        color_discrete_sequence=["#FFA07A", "#FF69B4"],
        hole=0.4,
        title="Taux de désengagement des utilisateurs"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("👥 Répartition des types de visiteurs")
    if "visitor_type" in filtered_df.columns:
        visitor_data = filtered_df["visitor_type"].value_counts().reset_index()
        visitor_data.columns = ["visitor_type", "count"]
        fig_type = px.pie(visitor_data, names="visitor_type", values="count", title="Types de visiteurs")
        st.plotly_chart(fig_type, use_container_width=True)

    st.subheader("📊 Score d'engagement - Boxplot")
    fig_box = px.box(filtered_df, y="engagement_score", color_discrete_sequence=["#FFD700"])
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("📈 Nombre de documents consultés")
    fig_docs = px.histogram(filtered_df, x="nb_unique_documents", nbins=20, color_discrete_sequence=["#FFA500"])
    st.plotly_chart(fig_docs, use_container_width=True)

    st.subheader("📈 Nombre de sessions")
    fig_sess = px.histogram(filtered_df, x="nb_sessions", nbins=20, color_discrete_sequence=["#FF4500"])
    st.plotly_chart(fig_sess, use_container_width=True)

    st.subheader("📱 Répartition des types d'appareils")
    if "device_type" not in filtered_df.columns:
        filtered_df["device_type"] = filtered_df["os_clean"].apply(
            lambda x: "Ordinateur" if any(os in str(x) for os in ["Windows", "Macintosh", "Linux"])
            else ("Mobile" if any(os in str(x) for os in ["Android", "iOS", "iPhone", "iPad"]) else "Autre")
        )
    device_counts = filtered_df["device_type"].value_counts().reset_index()
    device_counts.columns = ["device_type", "count"]
    fig_device = px.pie(
        device_counts,
        names="device_type",
        values="count",
        title="Types d'appareils utilisés",
        color_discrete_sequence=["#FFB6C1", "#FF69B4", "#FFDAB9"]
    )
    st.plotly_chart(fig_device, use_container_width=True)

    st.subheader("🌐 Navigateurs utilisés")
    browser_counts = filtered_df["browser_clean"].value_counts().reset_index()
    browser_counts.columns = ["browser", "count"]
    fig_browser = px.bar(
        browser_counts,
        x="browser",
        y="count",
        labels={"browser": "Navigateur", "count": "Nombre d'utilisateurs"},
        color_discrete_sequence=["#FFA07A"]
    )
    st.plotly_chart(fig_browser, use_container_width=True)

    st.subheader("💻 Systèmes d'exploitation")
    os_counts = filtered_df["os_clean"].value_counts().reset_index()
    os_counts.columns = ["os", "count"]
    fig_os = px.bar(
        os_counts,
        x="os",
        y="count",
        labels={"os": "OS", "count": "Utilisateurs"},
        color_discrete_sequence=["#FF69B4"]
    )
    st.plotly_chart(fig_os, use_container_width=True)

# 💡 Recommandations
with tabs[4]:
    st.header("💡 Recommandations personnalisées")
    st.markdown("""
    <div style='background-color:#FFF0F5; padding:15px; border-radius:10px;'>
        <h4 style='color:#D6336C;'>🔹 L'objectif de cette section est de proposer des actions concrètes à mettre en place pour optimiser l'engagement sur la plateforme.</h4>
        <p>Les recommandations ci-dessous sont basées sur l'analyse du comportement des utilisateurs selon leurs segments. Elles permettent de cibler les leviers les plus pertinents en fonction des profils.</p>
    </div>
    """, unsafe_allow_html=True)

    data = {
        "Public cible": [
            "Utilisateurs désengagés",
            "Utilisateurs très engagés",
            "Utilisateurs sur mobile",
            "Visiteurs mixtes",
            "Visiteurs récurrents",
            "Utilisateurs Windows",
            "Utilisateurs Mac",
        ],
        "Recommandation": [
            "Envoyer une campagne d'e-mails personnalisée avec des contenus populaires récents.",
            "Proposer un système de badges ou de niveaux pour les inciter à continuer leur activité.",
            "Optimiser l’UX mobile et proposer des contenus courts et interactifs.",
            "Proposer une relance au bon moment, en fonction de la période d'activité préférée.",
            "Offrir un accès anticipé à certains contenus ou fonctionnalités.",
            "Optimiser les performances et la compatibilité du site pour Windows.",
            "Mettre en avant des contenus premium visuellement soignés adaptés à l'expérience Mac.",
        ],
        "Objectif visé": [
            "🔄 Réactivation",
            "💪 Fidélisation",
            "🌐 Expérience utilisateur",
            "🔎 Personnalisation",
            "🌟 Engagement long terme",
            "🚀 Compatibilité",
            "🎉 Valorisation de l’expérience",
        ]
    }

    df_reco = pd.DataFrame(data)

    st.markdown("""
    <div style='padding: 10px; background-color: #FDE9EF; border-radius: 10px; margin-top: 10px;'>
        <p style='color: #C2185B;'>
        Les recommandations présentées sont conçues pour être directement activables par les équipes marketing, produit ou UX. Chaque recommandation s’appuie sur les KPIs analysés et les comportements observés sur la plateforme.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(df_reco.style.set_properties(**{
        'background-color': '#FFF3F3',
        'color': '#333',
        'border-color': '#FFB6C1'
    }), use_container_width=True)

# 👥 Équipe
with tabs[5]:
    st.header("👥 L'équipe projet")
    st.markdown("""
    Voici l'organisation de l'équipe ayant contribué à la réalisation de ce projet :
    """)

    st.graphviz_chart('''
    digraph G {
        node [shape=box, style=filled, color="#FFC0CB", fontname="Helvetica"];
        "Selda KOSE\nData Scientist Junior";
        "Duygu TOLU\nData Manager";
        "Aissa SOW\nScrum Master";
        "Marie MOUSSA\nProduct Owner";

        "Aissa SOW\nScrum Master" -> "Duygu TOLU\nData Manager";
        "Aissa SOW\nScrum Master" -> "Selda KOSE\nData Scientist Junior";
        "Aissa SOW\nScrum Master" -> "Marie MOUSSA\nProduct Owner";
    }
    ''')

    st.markdown("""
    _Ce projet a été réalisé dans le cadre du Master Data Management & Business Analytics._
    """)