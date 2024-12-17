import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="World Happiness Report Analysis",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    df_2015 = pd.read_csv('2015.csv')
    df_2016 = pd.read_csv('2016.csv')
    df_2017 = pd.read_csv('2017.csv')
    df_2018 = pd.read_csv('2018.csv')
    df_2019 = pd.read_csv('2019.csv')
    return df_2015, df_2016, df_2017, df_2018, df_2019

# Load all datasets
df_2015, df_2016, df_2017, df_2018, df_2019 = load_data()

# Sidebar
with st.sidebar:
    st.title('🌏 World Happiness Report Panel')
    selected_option = st.sidebar.radio('Select an option:', ['Dashboard', 'Visualization', 'Clustering'])

# Dashboard Main Panel
# Dashboard Main Panel
if selected_option == 'Dashboard':
    st.markdown("<h1 style='text-align: center;'>DASHBOARD MAIN PANEL</h1>", unsafe_allow_html=True)
    
    # Year and Country selection
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        selected_year = st.selectbox('Pilih Tahun', ['2015', '2016', '2017', '2018', '2019'])
    
    # Get appropriate dataset based on year
    year_df_map = {
        '2015': df_2015,
        '2016': df_2016,
        '2017': df_2017,
        '2018': df_2018,
        '2019': df_2019
    }
    
    current_df = year_df_map[selected_year]
    
    # Column name mappings for different years
    column_mappings = {
        '2015': {
            'score': 'Happiness Score',
            'rank': 'Happiness Rank',
            'gdp': 'Economy (GDP per Capita)',
            'country': 'Country'
        },
        '2016': {
            'score': 'Happiness Score',
            'rank': 'Happiness Rank',
            'gdp': 'Economy (GDP per Capita)',
            'country': 'Country'
        },
        '2017': {
            'score': 'Happiness Score',
            'rank': 'Happiness Rank',
            'gdp': 'Economy..GDP.per.Capita.',
            'country': 'Country'
        },
        '2018': {
            'score': 'Score',
            'rank': 'Overall rank',
            'gdp': 'GDP per capita',
            'country': 'Country or region'
        },
        '2019': {
            'score': 'Score',
            'rank': 'Overall rank',
            'gdp': 'GDP per capita',
            'country': 'Country or region'
        }
    }
    
    # Get current column names
    current_cols = column_mappings[selected_year]
    country_col = current_cols['country']
    score_col = current_cols['score']
    rank_col = current_cols['rank']
    gdp_col = current_cols['gdp']
    
    with col2:
        country_list = sorted(current_df[country_col].unique())
        selected_country = st.selectbox('Pilih Negara', country_list)
    
    st.markdown(f"""
    <div style='text-align: justify;'>
        <h3>Analisa Data Kebahagiaan Suatu Negara pada Tahun {selected_year}</h3>
        <p>Eksplorasi data kebahagiaan negara {selected_country} untuk tahun {selected_year}. 
        Data memiliki beberapa faktor yang memengaruhi tingkat kebahagiaan suatu negara.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create metrics display
    col1, col2, col3 = st.columns(3)
    
    # Get country data
    country_data = current_df[current_df[country_col] == selected_country].iloc[0]
    
    # Display metrics with proper column names
    col1.metric("Happiness Score", f"{country_data[score_col]:.2f}")
    col2.metric("Happiness Rank", f"#{int(country_data[rank_col])}")
    col3.metric("GDP per capita", f"{country_data[gdp_col]:.3f}")
    
    # Detailed Data
    st.subheader('Data Detail Negara')
    
    # Create display DataFrame without reordering
    display_df = country_data.to_frame().T
    
    # Display DataFrame
    st.dataframe(display_df, hide_index=True)
    
    st.markdown(f"""
    <div style='text-align: justify;'>
        <h3>Penjelasan Tiap Fitur</h3>
        
        - {score_col}: Happiness score secara keseluruhan untuk negara yang dipilih
        - {rank_col}: Peringkat negara berdasarkan skor kebahagiaan
        - {gdp_col}: GDP per capita, indikator kesejahteraan ekonomi
        - {'Family' if selected_year in ['2015', '2016', '2017'] else 'Social support'}: 
          Kontribusi keluarga dan dukungan sosial terhadap kebahagiaan
        - {'Health (Life Expectancy)' if selected_year in ['2015', '2016'] 
           else 'Health..Life.Expectancy.' if selected_year == '2017' 
           else 'Healthy life expectancy'}: Ekspetasi hidup yang sehat sebagai faktor kebahagiaan
        - Freedom to make life choices: Kebebasan dalam membuat pilihan hidup yang mempengaruhi kebahagiaan
        - Generosity: Tingkat kebaikan dan kepedulian masyarakat terhadap orang lain
        - {'Trust (Government Corruption)' if selected_year in ['2015', '2016', '2017'] 
           else 'Perceptions of corruption'}: Tingkat kepercayaan terhadap pemerintah dan korupsi (semakin rendah semakin baik)
    </div>
    """, unsafe_allow_html=True)

# Visualization Main Panel
elif selected_option == 'Visualization':
    viz_type = st.sidebar.selectbox(
        'Select Visualization Type',
        ['Global Overview', 'Country Rankings', 'Correlation Analysis', 'Trend Analysis']
    )
    
    if viz_type == 'Global Overview':
        st.markdown("<h1 style='text-align: center;'>GLOBAL OVERVIEW</h1>", unsafe_allow_html=True)
        
        # World map
        fig = go.Figure(data=go.Choropleth(
            locations=df_2019['Country or region'],
            locationmode='country names',
            z=df_2019['Score'],
            colorscale='RdYlGn',
            colorbar_title="Happiness Score"
        ))
        fig.update_layout(title='Global Happiness Score Distribution (2019)')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ### Interpretasi
        Peta dunia yang mewakili skor kebahagiaan di berbagai negara menggunakan skala warna. Setiap negara diberi warna yang bervariasi sesuai dengan tingkat kebahagiaannya, dengan warna yang lebih hijau menunjukkan skor kebahagiaan yang lebih tinggi dan warna yang lebih merah menunjukkan skor kebahagiaan yang lebih rendah. Dengan melihat peta ini, kita dapat dengan mudah melihat distribusi kebahagiaan di seluruh dunia dan membandingkan kebahagiaan antar negara.
                    
        ### Insight
        - Negara-negara di Skandinavia (seperti Finlandia, Denmark, dan Norwegia) cenderung memiliki skor kebahagiaan yang lebih tinggi
        - Negara-negara di Afrika Sub-Sahara dan Timur Tengah cenderung memiliki skor kebahagiaan yang lebih rendah
        - Amerika Utara dan Eropa Barat memiliki skor kebahagiaan yang beragam
        
        Peta ini memberikan gambaran visual yang jelas tentang distribusi kebahagiaan di seluruh dunia dan membantu dalam memahami faktor-faktor yang mempengaruhi kebahagiaan di berbagai negara.
        """)
        
        # Distribution of happiness scores
        fig = px.histogram(df_2019, x='Score', nbins=30,
                          title='Distribution of Happiness Scores')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        ### Interpretasi Distribusi Skor Kebahagiaan

        #### Analisis Distribusi
        - **Bentuk Distribusi**: Histogram menunjukkan distribusi skor kebahagiaan di seluruh dunia, dengan sumbu X menunjukkan skor kebahagiaan dan sumbu Y menunjukkan jumlah negara.
        - **Rentang Skor**: Skor kebahagiaan tersebar dari sekitar 3 hingga 8, menunjukkan variasi yang signifikan dalam tingkat kebahagiaan antar negara.
        - **Pola Distribusi**: Distribusi cenderung membentuk dua puncak (bimodal), yang mengindikasikan adanya dua kelompok utama negara dengan karakteristik kebahagiaan yang berbeda.

        #### Insight Utama
        1. **Kesenjangan Kebahagiaan**:
        - Terdapat kesenjangan yang jelas antara negara-negara dengan skor kebahagiaan tinggi dan rendah
        - Sebagian besar negara berada di tingkat menengah, dengan sedikit negara di ekstrem tertinggi dan terendah

        2. **Pengelompokan Negara**:
        - Kelompok pertama: Negara dengan skor kebahagiaan rendah-menengah (sekitar 3-5)
        - Kelompok kedua: Negara dengan skor kebahagiaan menengah-tinggi (sekitar 6-8)

        3. **Implikasi Kebijakan**:
        - Perlu fokus khusus pada negara-negara dengan skor rendah untuk mengurangi kesenjangan
        - Dapat mempelajari praktik terbaik dari negara-negara dengan skor tinggi
        - Strategi peningkatan kebahagiaan mungkin perlu disesuaikan berdasarkan kelompok negara
        """)
    
    elif viz_type == 'Country Rankings':
        st.markdown("<h1 style='text-align: center;'>COUNTRY RANKINGS</h1>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 10 happiest countries
            top_10 = df_2019.nlargest(10, 'Score')
            fig = px.bar(top_10, 
                        x='Score', 
                        y='Country or region',
                        orientation='h',
                        title='Top 10 Happiest Countries')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bottom 10 countries
            bottom_10 = df_2019.nsmallest(10, 'Score')
            fig = px.bar(bottom_10, 
                        x='Score', 
                        y='Country or region',
                        orientation='h',
                        title='Bottom 10 Countries by Happiness Score')
            st.plotly_chart(fig, use_container_width=True)
        
        # Top countries by different metrics
        metric = st.selectbox('Pilih metrik untuk Negara Top 10:', 
                            ['GDP per capita', 'Social support', 
                             'Healthy life expectancy', 'Freedom to make life choices'])
        
        fig = px.bar(df_2019.nlargest(10, metric),
                    x=metric, 
                    y='Country or region',
                    orientation='h',
                    title=f'Top 10 Countries by {metric}')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == 'Correlation Analysis':
        st.markdown("<h1 style='text-align: center;'>CORRELATION ANALYSIS</h1>", unsafe_allow_html=True)
        
        # Correlation matrix
        numeric_columns = ['Score', 'GDP per capita', 'Social support', 
                         'Healthy life expectancy', 'Freedom to make life choices',
                         'Generosity', 'Perceptions of corruption']
        
        correlation = df_2019[numeric_columns].corr()
        
        fig = px.imshow(correlation,
                       labels=dict(color="Correlation"),
                       color_continuous_scale='RdBu_r',
                       title='Correlation Matrix of Happiness Factors')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ### Interpretasi Correlation Matrix
                    
        Heatmap menampilkan tingkat korelasi antara dua variabel dalam dataset. Setiap sel dalam heatmap mewakili korelasi antara dua variabel, dengan warna sel menunjukkan tingkat korelasi dan arah korelasi antar variabel. Semakin terang warna selnya, semakin tinggi tingkat korelasinya, sementara semakin gelap warna selnya, semakin rendah tingkat korelasinya. Misalnya, jika sel memiliki warna orange terang, itu menunjukkan tingkat korelasi yang tinggi antara dua variabel tersebut. Sebaliknya, jika sel memiliki warna ungu gelap, itu menunjukkan tingkat korelasi yang rendah antara dua variabel tersebut.

        1. **Korelasi dengan Skor Kebahagiaan:**
        - GDP per capita memiliki korelasi positif kuat dengan skor kebahagiaan
        - Social support juga menunjukkan korelasi positif yang signifikan
        - Healthy life expectancy berkorelasi positif dengan kebahagiaan
        - Freedom to make life choices memiliki korelasi positif moderat
        - Generosity dan Perceptions of corruption menunjukkan korelasi yang relatif lemah

        2. **Korelasi Antar Faktor:**
        - GDP per capita dan Healthy life expectancy menunjukkan korelasi positif kuat
        - Social support berkorelasi positif dengan GDP per capita dan Healthy life expectancy
        - Freedom to make life choices memiliki korelasi moderat dengan faktor lainnya
        - Generosity dan Perceptions of corruption menunjukkan korelasi yang relatif rendah dengan faktor lainnya

        ### Insight & Implikasi:
        1. **Faktor Ekonomi dan Kesehatan:**
        - Negara dengan GDP per capita tinggi cenderung memiliki harapan hidup yang lebih tinggi
        - Pembangunan ekonomi sering sejalan dengan peningkatan layanan kesehatan

        2. **Dukungan Sosial:**
        - Negara dengan tingkat ekonomi tinggi cenderung memiliki sistem dukungan sosial yang lebih baik
        - Dukungan sosial yang kuat berkontribusi positif terhadap kebahagiaan
        
        3. **Kebebasan dan Korupsi:**
        - Kebebasan memiliki dampak positif pada kebahagiaan
        - Persepsi korupsi memiliki pengaruh yang lebih kecil dibanding faktor lainnya
        """)
        
        # Scatter plot of selected features
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox('Select X-axis feature:', numeric_columns)
        with col2:
            y_feature = st.selectbox('Select Y-axis feature:', numeric_columns, index=1)
        
        fig = px.scatter(df_2019, 
                        x=x_feature, 
                        y=y_feature,
                        hover_data=['Country or region'],
                        trendline="ols",
                        title=f'Relationship between {x_feature} and {y_feature}')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Interpretasi Scatter Plot
        - Titik-titik pada scatter plot menunjukkan posisi setiap negara berdasarkan kedua variabel yang dipilih
        - Garis tren menunjukkan hubungan umum antara kedua variabel
        - Semakin dekat titik-titik dengan garis tren, semakin kuat korelasinya
        - Posisi negara yang jauh dari garis tren menunjukkan kasus-kasus unik yang mungkin memerlukan analisis lebih lanjut
        
        ### Cara Menggunakan:
        1. Pilih dua variabel yang ingin dibandingkan dari menu dropdown
        2. Perhatikan pola sebaran data dan kemiringan garis tren
        3. Gunakan fitur hover untuk melihat detail negara tertentu
        4. Analisis negara-negara yang memiliki nilai ekstrem atau penyimpangan dari tren umum
        """)
    
    elif viz_type == 'Trend Analysis':
        st.markdown("<h1 style='text-align: center;'>TREND ANALYSIS</h1>", unsafe_allow_html=True)

        # First, let's verify the column names in each dataset
        year_column_mappings = {
            '2015': {
                'country_col': df_2015.columns[0],  # First column should be country
                'score_col': [col for col in df_2015.columns if 'Score' in col][0]  # Find the score column
            },
            '2016': {
                'country_col': df_2016.columns[0],
                'score_col': [col for col in df_2016.columns if 'Score' in col][0]
            },
            '2017': {
                'country_col': df_2017.columns[0],
                'score_col': [col for col in df_2017.columns if 'Score' in col][0]
            },
            '2018': {
                'country_col': 'Country or region',
                'score_col': 'Score'
            },
            '2019': {
                'country_col': 'Country or region',
                'score_col': 'Score'
            }
        }
        
        # Select countries for comparison
        all_countries = set()
        for df, year in zip([df_2015, df_2016, df_2017, df_2018, df_2019], 
                        ['2015', '2016', '2017', '2018', '2019']):
            country_col = year_column_mappings[year]['country_col']
            all_countries.update(df[country_col].unique())
        
        selected_countries = st.multiselect(
            'Pilih negara untuk dibandingkan:',
            sorted(list(all_countries)),
            default=['Finland', 'Denmark', 'Norway', 'Iceland']
        )
        
        if selected_countries:
            # Create time series data
            time_series_data = []
            
            # Create mapping of dataframes to years
            df_map = {
                '2015': df_2015,
                '2016': df_2016,
                '2017': df_2017,
                '2018': df_2018,
                '2019': df_2019
            }
            
            for country in selected_countries:
                for year in ['2015', '2016', '2017', '2018', '2019']:
                    df = df_map[year]
                    country_col = year_column_mappings[year]['country_col']
                    score_col = year_column_mappings[year]['score_col']
                    
                    if country in df[country_col].values:
                        country_data = df[df[country_col] == country]
                        if not country_data.empty:
                            time_series_data.append({
                                'Country': country,
                                'Year': year,
                                'Score': country_data[score_col].iloc[0]
                            })
            
            if time_series_data:
                df_time = pd.DataFrame(time_series_data)
                
                # Create line plot
                fig = px.line(df_time,
                            x='Year',
                            y='Score',
                            color='Country',
                            title='Happiness Score Trends (2015-2019)',
                            markers=True)
                
                fig.update_layout(
                    xaxis_title='Year',
                    yaxis_title='Happiness Score',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add summary statistics
                st.subheader('Summary Statistics')
                
                # Calculate average scores and trends
                summary_data = []
                for country in selected_countries:
                    country_scores = df_time[df_time['Country'] == country]
                    if len(country_scores) >= 2:  # Need at least 2 points for trend
                        avg_score = country_scores['Score'].mean()
                        score_change = country_scores['Score'].iloc[-1] - country_scores['Score'].iloc[0]
                        trend = "↑" if score_change > 0 else "↓" if score_change < 0 else "→"
                        
                        summary_data.append({
                            'Country': country,
                            'Average Score': f"{avg_score:.2f}",
                            'Overall Trend': trend,
                            'Change': f"{score_change:+.2f}"
                        })
                
                if summary_data:
                    st.dataframe(pd.DataFrame(summary_data), hide_index=True)
                
                # Add interpretation
                st.markdown("""
                ### Interpretasi:
                - Line plot menunjukkan tren skor kebahagiaan negara yang dipilih dari tahun 2015 hingga 2019
                - ↑ indikasi tren meningkat
                - ↓ indikasi tren menurun
                - → indikasi tidak ada perubahan yang signifikan
                - Kolom 'Change' menunjukkan perubahan skor absolut dari tahun 2015 hingga 2019
                """)
            else:
                st.warning("No data available for the selected countries.")
        else:
            st.info("Please select at least one country to show the trend analysis.")
            
# Clustering Main Panel
# Clustering Main Panel
elif selected_option == 'Clustering':
    st.markdown("<h1 style='text-align: center;'>CLUSTERING ANALYSIS</h1>", unsafe_allow_html=True)
    
    # Feature selection
    feature_pairs = [
        ('GDP per capita', 'Social support'),
        ('Score', 'GDP per capita'),
        ('Social support', 'Healthy life expectancy'),
        ('GDP per capita', 'Healthy life expectancy')
    ]
    
    selected_pair = st.selectbox('Select features for clustering:', 
                               [f"{pair[0]} vs {pair[1]}" for pair in feature_pairs])
    
    features = selected_pair.split(' vs ')
    X = df_2019[features]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Elbow method
        st.subheader('Elbow Method')
        k_range = range(1, 11)
        inertias = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            
        fig_elbow = px.line(x=list(k_range), y=inertias, markers=True)
        fig_elbow.update_layout(xaxis_title='Number of clusters (k)',
                              yaxis_title='Inertia')
        st.plotly_chart(fig_elbow)
    
    with col2:
        # Perform clustering
        n_clusters = st.slider('Select number of clusters', 2, 5, 3)
        
        # Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, clusters)
        st.metric('Silhouette Score', f"{silhouette_avg:.3f}")
    
    # Add cluster labels to the dataset
    df_clustered = df_2019.copy()
    df_clustered['Cluster'] = clusters
    
    # Create cluster names based on mean values
    cluster_means = df_clustered.groupby('Cluster')[features].mean()
    cluster_sizes = df_clustered['Cluster'].value_counts()
    
    # Sort clusters by their mean values to assign meaningful labels
    cluster_labels = {}
    for feature in features:
        sorted_clusters = cluster_means[feature].sort_values()
        labels = ['Low', 'Medium', 'High'] if n_clusters == 3 else ['Low', 'Medium-Low', 'Medium-High', 'High'] if n_clusters == 4 else ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        for i, cluster in enumerate(sorted_clusters.index):
            if cluster not in cluster_labels:
                cluster_labels[cluster] = labels[i]
    
    # Clustering visualization
    fig = px.scatter(df_clustered, 
                    x=features[0], 
                    y=features[1],
                    color='Cluster',
                    hover_data=['Country or region'],
                    title=f'Clusters based on {features[0]} and {features[1]}',
                    labels={'Cluster': 'Development Level'},
                    color_continuous_scale='viridis')
    
    # Add centroids
    centroids = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids)
    fig.add_scatter(x=centroids[:, 0], 
                   y=centroids[:, 1],
                   mode='markers',
                   marker=dict(color='red', size=15, symbol='x'),
                   name='Centroids')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display cluster information
    st.subheader('Cluster Analysis')
    
    # Create cluster summaries with example countries
    for cluster in range(n_clusters):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
        
        # Get example countries (top 3 by GDP per capita)
        example_countries = cluster_data.nlargest(3, 'GDP per capita')['Country or region'].tolist()
        
        # Calculate mean values for the features
        mean_values = cluster_data[features].mean()
        
        # Create expandable section for each cluster
        with st.expander(f"Cluster {cluster} ({cluster_labels[cluster]} Development) - {len(cluster_data)} countries"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Average Values:**")
                for feature in features:
                    st.write(f"{feature}: {mean_values[feature]:.3f}")
            
            with col2:
                st.write("**Example Countries:**")
                st.write(", ".join(example_countries))
    
    # Add overall interpretation
    st.markdown("""
    ### Interpretasi
    - **Low Development**: Negara dengan nilai rendah pada kedua metrik
    - **Medium Development**: Negara dengan nilai menengah pada kedua metrik
    - **High Development**: Negara dengan nilai tinggi pada kedua metrik
    - **Centroids (Red X)**: Pusat cluster yang menunjukkan nilai rata-rata untuk setiap fitur
    
    Clustering analysis dapat membantu mengidentifikasi kelompok negara dengan karakteristik yang serupa berdasarkan fitur yang dipilih.
    Tujuan utama dari analisis ini adalah:
    - Memahami pola dan tren dalam data yang menentukan kebahagiaan suatu negara
    - Mengidentifikasi negara-negara dengan karakteristik serupa
    - Menyajikan informasi dengan cara yang mudah dipahami dan divisualisasikan
    """)

# Footer
st.markdown("""
---
Dibuat oleh Kelompok 8 | Data Source: https://www.kaggle.com/datasets/unsdsn/world-happiness/data
""")