import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from networkx.algorithms.community import greedy_modularity_communities

st.set_page_config(page_title='Actor Collaboration Network Dashboard', layout='wide')

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
    close_cent = nx.closeness_centrality(G)

    top_degree = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)
    top_between = sorted(bet_cent.items(), key=lambda x: x[1], reverse=True)
    top_close = sorted(close_cent.items(), key=lambda x: x[1], reverse=True)

    communities = list(greedy_modularity_communities(G))
    community_sizes = sorted([len(c) for c in communities], reverse=True)

    community_map = {}
    for i, community in enumerate(communities):
        for actor in community:
            community_map[actor] = i

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
        'close_cent': close_cent,
        'top_degree': top_degree,
        'top_between': top_between,
        'top_close': top_close,
        'communities': communities,
        'community_sizes': community_sizes,
        'community_map': community_map,
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
    "Actor Explorer",
    "Findings"
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
    st.header('Network Visualization')
    st.caption('This sample view helps users explore the structure of actor collaborations without overwhelming the screen.')

    # Use the highest-degree actors so the sample is meaningful
    top_sample_nodes = [
        actor for actor, degree in sorted(
            G.degree(),
            key=lambda x: x[1],
            reverse=True
        )[:subgraph_size]
    ]

    subgraph = G.subgraph(top_sample_nodes).copy()

    pos = nx.spring_layout(
        subgraph,
        seed=42,
        k=0.45
    )

    node_degrees = dict(subgraph.degree())
    node_sizes = [
        80 + node_degrees[node] * 35
        for node in subgraph.nodes()
    ]

    node_colors = [
        results['community_map'].get(node, 0)
        for node in subgraph.nodes()
    ]

    edge_widths = [
        0.5 + subgraph[u][v].get('weight', 1)
        for u, v in subgraph.edges()
    ]

    fig_net, ax_net = plt.subplots(figsize=(11, 9))

    nx.draw_networkx_edges(
        subgraph,
        pos,
        ax=ax_net,
        alpha=0.25,
        width=edge_widths
    )

    nodes = nx.draw_networkx_nodes(
        subgraph,
        pos,
        ax=ax_net,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.tab20,
        alpha=0.9
    )

    # Label only the most connected actors to avoid clutter
    label_nodes = dict(
        sorted(
            node_degrees.items(),
            key=lambda x: x[1],
            reverse=True
        )[:12]
    )

    labels = {node: node for node in label_nodes}

    nx.draw_networkx_labels(
        subgraph,
        pos,
        labels=labels,
        font_size=8,
        ax=ax_net
    )

    ax_net.set_title(
        f'Actor Collaboration Network Sample: Top {subgraph_size} Actors by Degree',
        fontsize=14
    )
    ax_net.axis('off')

    st.pyplot(fig_net)

    st.caption(
        "Node size represents number of direct collaborators. "
        "Node color represents detected community. "
        "Thicker edges represent repeated shared movie appearances."
    )

    st.subheader("Interpretation:")
    st.markdown(
        """
        This graph displays a focused sample of the most connected actors in the collaboration network.
        Larger nodes represent actors with more direct collaborators, while node color represents detected community membership.

        This makes the visualization easier to interpret than a random sample because it highlights the actors who play larger structural roles.
        Dense areas of the graph suggest collaboration clusters, while connections between differently colored groups may indicate actors who help bridge communities.
        """
    )

with tabs[2]:
    st.header('Central Actors')
    left, middle, right = st.columns(3)

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

    with middle:
        st.subheader('Top Actors by Closeness Centrality')
        top_close_df = pd.DataFrame(results['top_close'][:top_n], columns=['Actor', 'Closeness Centrality'])
        top_close_df['Closeness Centrality'] = top_close_df['Closeness Centrality'].round(5)
        st.dataframe(top_close_df, use_container_width=True)

        fig_close_bar, ax_close_bar = plt.subplots(figsize=(7, 4))
        ax_close_bar.barh(top_close_df['Actor'], top_close_df['Closeness Centrality'])
        ax_close_bar.invert_yaxis()
        ax_close_bar.set_title('Top Actors by Closeness Centrality')
        ax_close_bar.set_xlabel('Closeness Centrality')
        st.pyplot(fig_close_bar)

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

    st.subheader("Interpretation:")
    st.markdown(
        """
        Degree centrality identifies actors with many direct collaborators.
        These actors may appear in films with many different co-stars, making them highly connected within the dataset.

        Closeness centrality identifies actors who are, on average, closer to others in the network.
        These actors may be well-positioned to reach many parts of the collaboration network through short paths.

        Betweenness centrality identifies actors who help connect different parts of the network.
        A high betweenness score suggests that an actor may act as a bridge between collaboration groups.
        """
    )

with tabs[3]:
    st.header('Network Patterns')
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

    st.subheader("Interpretation:")
    st.markdown(
        """
        The degree distribution shows whether most actors have only a few connections while a smaller number have many.
        This helps answer whether collaboration is evenly spread or concentrated around a few highly connected actors.

        The connected component distribution shows whether the network is mostly connected or split into separate groups.
        Many small components suggest that some actors appear in isolated collaboration circles.

        The community size distribution shows how the detected actor communities vary in size.
        Larger communities may represent broad collaboration clusters, while smaller communities may represent tighter groups of actors.
        """
    )

with tabs[4]:
    st.header("Actor Explorer")

    selected_actor = st.selectbox(
        "Choose an actor to inspect",
        sorted(G.nodes())
    )

    neighbors = sorted(list(G.neighbors(selected_actor)))
    weighted_neighbors = []

    for neighbor in neighbors:
        weight = G[selected_actor][neighbor]["weight"]
        weighted_neighbors.append((neighbor, weight))

    neighbor_df = pd.DataFrame(
        weighted_neighbors,
        columns=["Collaborator", "Shared Movies"]
    ).sort_values("Shared Movies", ascending=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Actor", selected_actor)
    col2.metric("Direct Collaborators", len(neighbors))
    col3.metric("Total Shared-Movie Ties", sum(neighbor_df["Shared Movies"]))

    st.subheader(f"Collaborators of {selected_actor}")
    st.dataframe(neighbor_df, use_container_width=True)

    st.subheader("Interpretation:")
    st.markdown(
        """
        This tab supports local exploration. Instead of only seeing global rankings,
        users can inspect one actor and see who they are directly connected to.
        The shared-movie count uses the edge weight from the network.
        """
    )

with tabs[5]:
    st.header('Key Findings')

    top_degree_names = [actor for actor, score in results['top_degree'][:3]]
    top_between_names = [actor for actor, score in results['top_between'][:3]]

    st.markdown(
        f"""
        ### Main Takeaways

        **1. The actor collaboration network is sparse.**  
        The network density is **{results['density']:.4f}**, meaning only a small fraction of all possible actor-to-actor collaborations appear in this dataset.
        This makes sense because the IMDb Top 1000 dataset only includes selected highly rated films, not every movie in the industry.

        **2. Collaboration is locally clustered.**  
        The average clustering coefficient is **{results['avg_clustering']:.3f}**.
        This suggests that actors often appear in groups where collaborators are also connected to one another.

        **3. The network is fragmented into multiple components.**  
        There are **{len(results['components'])} connected components**.
        This means not every actor is connected to every other actor through collaboration paths.

        **4. A small number of actors stand out as structurally central.**  
        The top actors by degree centrality include **{top_degree_names[0]}**, **{top_degree_names[1]}**, and **{top_degree_names[2]}**.
        These actors have many direct collaboration ties compared with others in the network.

        **5. Some actors may act as bridges between groups.**  
        The top actors by betweenness centrality include **{top_between_names[0]}**, **{top_between_names[1]}**, and **{top_between_names[2]}**.
        These actors may help connect otherwise separate areas of the collaboration network.

        **6. Community structure supports the idea of collaboration clusters.**  
        The network contains **{len(results['communities'])} detected communities**.
        This supports the project question about whether actors form tightly connected collaboration groups.
        """
    )

    st.subheader("Limitations")
    st.markdown(
        """
        This analysis should be interpreted carefully.
        The dataset only includes the top four listed actors for each movie, so it does not capture every actor collaboration.
        It also focuses on IMDb's top 1000 movies and TV shows released before or during 2021, so the network reflects a selected group of highly rated titles rather than the entire film industry.
        """
    )