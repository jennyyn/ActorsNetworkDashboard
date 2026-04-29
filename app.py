import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from networkx.algorithms.community import greedy_modularity_communities

st.set_page_config(page_title='Actor Collaboration Network Dashboard', layout='wide')


@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)


@st.cache_resource
def build_network(df):
    actors_df = df[['Star1', 'Star2', 'Star3', 'Star4']].dropna().copy()
    G = nx.Graph()

    for _, row in actors_df.iterrows():
        actors = row.tolist()
        for pair in combinations(actors, 2):
            if G.has_edge(*pair):
                G[pair[0]][pair[1]]['weight'] += 1
            else:
                G.add_edge(pair[0], pair[1], weight=1)

    density = nx.density(G)
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    avg_clustering = nx.average_clustering(G)
    components = list(nx.connected_components(G))

    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G)

    top_degree = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:10]
    top_between = sorted(bet_cent.items(), key=lambda x: x[1], reverse=True)[:10]

    communities = list(greedy_modularity_communities(G))
    community_sizes = sorted([len(c) for c in communities], reverse=True)

    largest_cc = max(components, key=len)
    G_cc = G.subgraph(largest_cc).copy()

    return {
        'G': G,
        'density': density,
        'avg_degree': avg_degree,
        'avg_clustering': avg_clustering,
        'components': components,
        'deg_cent': deg_cent,
        'bet_cent': bet_cent,
        'top_degree': top_degree,
        'top_between': top_between,
        'communities': communities,
        'community_sizes': community_sizes,
        'G_cc': G_cc,
    }


st.title('Actor Collaboration Network Dashboard')
st.markdown(
    '''
This dashboard explores collaboration patterns between actors in the IMDb Top 1000 dataset.
Each **node** represents an actor, and each **edge** represents a shared movie appearance.
The network is **undirected** and **weighted** by the number of times two actors appear together.
'''
)

st.sidebar.header('Dashboard Controls')

subgraph_size = st.sidebar.slider('Number of actors in sample network', min_value=20, max_value=150, value=80, step=10)
top_n = st.sidebar.slider('Number of top actors to show', min_value=5, max_value=15, value=10, step=1)
@st.cache_data
def load_data():
    return pd.read_csv("imdb_top_1000.csv")


try:
    df = load_data()
    required_cols = {'Star1', 'Star2', 'Star3', 'Star4'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        st.error(f'Missing required columns: {sorted(missing_cols)}')
        st.stop()
except Exception as exc:
    st.error(f'Could not read the CSV file: {exc}')
    st.stop()

results = build_network(df)
G = results['G']

tabs = st.tabs([
    "Overview",
    "Network",
    "Central Actors",
    "Patterns",
    "Findings",
    "About"
])

with tabs[0]:
    st.header('Overview')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Actors (nodes)', G.number_of_nodes())
    col2.metric('Collaborations (edges)', G.number_of_edges())
    col3.metric('Density', f"{results['density']:.4f}")
    col4.metric('Avg. clustering', f"{results['avg_clustering']:.3f}")

    col5, col6, col7 = st.columns(3)
    col5.metric('Avg. degree', f"{results['avg_degree']:.2f}")
    col6.metric('Connected components', len(results['components']))
    col7.metric('Communities', len(results['communities']))

with tabs[1]:
    st.header('2. Network Visualization')
    st.caption('This sample view helps users explore the structure of actor collaborations without overwhelming the screen.')

    sub_nodes = list(G.nodes())[:subgraph_size]
    subgraph = G.subgraph(sub_nodes)
    fig_net, ax_net = plt.subplots(figsize=(8, 8))
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw(subgraph, pos, node_size=45, with_labels=False, ax=ax_net)
    ax_net.set_title(f'Actor Collaboration Network Sample ({subgraph_size} actors)')
    st.pyplot(fig_net)

with tabs[2]:
    st.header('3. Central Actors')
    left, right = st.columns(2)

    with left:
        st.subheader('Top Actors by Degree Centrality')
        top_deg_df = pd.DataFrame(results['top_degree'][:top_n], columns=['Actor', 'Degree Centrality'])
        top_deg_df['Degree Centrality'] = top_deg_df['Degree Centrality'].round(5)
        st.dataframe(top_deg_df, use_container_width=True)

        fig_deg_bar, ax_deg_bar = plt.subplots(figsize=(7, 4))
        ax_deg_bar.barh(top_deg_df['Actor'], top_deg_df['Degree Centrality'])
        ax_deg_bar.invert_yaxis()
        ax_deg_bar.set_title('Top Actors by Degree Centrality')
        ax_deg_bar.set_xlabel('Degree Centrality')
        st.pyplot(fig_deg_bar)

    with right:
        st.subheader('Top Actors by Betweenness Centrality')
        top_bet_df = pd.DataFrame(results['top_between'][:top_n], columns=['Actor', 'Betweenness Centrality'])
        top_bet_df['Betweenness Centrality'] = top_bet_df['Betweenness Centrality'].round(5)
        st.dataframe(top_bet_df, use_container_width=True)

        fig_bet_bar, ax_bet_bar = plt.subplots(figsize=(7, 4))
        ax_bet_bar.barh(top_bet_df['Actor'], top_bet_df['Betweenness Centrality'])
        ax_bet_bar.invert_yaxis()
        ax_bet_bar.set_title('Top Actors by Betweenness Centrality')
        ax_bet_bar.set_xlabel('Betweenness Centrality')
        st.pyplot(fig_bet_bar)

with tabs[3]:
    st.header('4. Network Patterns')
    left2, right2 = st.columns(2)

    with left2:
        degrees = [deg for _, deg in G.degree()]
        fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
        ax_hist.hist(degrees, bins=30)
        ax_hist.set_title('Degree Distribution')
        ax_hist.set_xlabel('Degree')
        ax_hist.set_ylabel('Number of Actors')
        st.pyplot(fig_hist)

    with right2:
        component_sizes = sorted([len(c) for c in results['components']], reverse=True)
        fig_comp, ax_comp = plt.subplots(figsize=(7, 4))
        ax_comp.hist(component_sizes, bins=20)
        ax_comp.set_title('Connected Component Size Distribution')
        ax_comp.set_xlabel('Component Size')
        ax_comp.set_ylabel('Frequency')
        st.pyplot(fig_comp)

        st.subheader('Community Size Distribution')
        fig_comm, ax_comm = plt.subplots(figsize=(8, 4))
        ax_comm.hist(results['community_sizes'], bins=20)
        ax_comm.set_title('Community Size Distribution')
        ax_comm.set_xlabel('Community Size')
        ax_comm.set_ylabel('Frequency')
        st.pyplot(fig_comm)

with tabs[4]:
    st.header('5. Key Findings')
    st.markdown(
        f'''
    - The network is **sparse overall** (density = **{results['density']:.4f}**), which means only a small fraction of all possible actor collaborations occur.
    - The network shows **strong local clustering** (average clustering coefficient = **{results['avg_clustering']:.3f}**), suggesting actors often work within repeated collaboration groups.
    - There are **{len(results['components'])} connected components**, indicating the network is fragmented into many disconnected collaboration groups.
    - A small set of actors stand out as highly central, including **{top_deg_df.iloc[0, 0]}**, **{top_deg_df.iloc[1, 0]}**, and **{top_deg_df.iloc[2, 0]}**.
    - Actors such as **{top_bet_df.iloc[0, 0]}** and **{top_bet_df.iloc[1, 0]}** have high betweenness centrality, meaning they help connect otherwise separate parts of the network.
    - The network contains **{len(results['communities'])} communities**, reinforcing that collaborations tend to form within clusters rather than across the full network.
    '''
    )

with tabs[5]:
    st.header('6. About This Dashboard')
    st.markdown(
        '''
    **Intended audience:** classmates and instructors interested in network analysis and collaboration structure in the film industry.

    **Purpose:** to communicate the main structural patterns of the actor collaboration network through a small set of interpretable views.

    **Main interactive feature:** users can adjust the number of actors shown in the network sample and the number of top-ranked actors shown in the centrality tables.
    '''
    )
