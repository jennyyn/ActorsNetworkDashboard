import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
from networkx.algorithms.community import greedy_modularity_communities
import plotly.express as px

st.set_page_config(page_title='Actor Collaboration Network Dashboard', layout='wide')

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    h1 {
        font-size: 2.7rem !important;
        margin-bottom: 0.5rem;
    }

    h2, h3 {
        margin-top: 1rem;
    }

    [data-testid="stMetric"] {
        background-color: #151922;
        border: 1px solid #2d3340;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #c9d1d9;
    }

    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }

    .section-card {
        background-color: #111620;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #2d3340;
        margin-bottom: 20px;
    }

    .small-note {
        color: #aab2bf;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def build_network(df):
    G = nx.Graph()

    for _, row in df.iterrows():
        actors = [row['Star1'], row['Star2'], row['Star3'], row['Star4']]
        actors = [a for a in actors if pd.notna(a)]

        # Safely parse IMDb genre strings such as "Action, Drama, Sci-Fi".
        if pd.notna(row.get('Genre')):
            genres = [genre.strip() for genre in str(row['Genre']).split(',') if genre.strip()]
        else:
            genres = []

        for actor in actors:
            if not G.has_node(actor):
                G.add_node(actor, genres=[])
            G.nodes[actor]['genres'].extend(genres)

        for pair in combinations(actors, 2):
            if G.has_edge(*pair):
                G[pair[0]][pair[1]]['weight'] += 1
                G[pair[0]][pair[1]]['genres'].extend(genres)
            else:
                G.add_edge(
                    pair[0], pair[1],
                    weight=1,
                    genres=genres.copy()
                )

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

    cross_edges = []

    for u, v, data in G.edges(data=True):
        if community_map.get(u) != community_map.get(v):
            cross_edges.append((u, v, data.get("weight", 1)))

    community_genre_rows = []

    for i, community in enumerate(communities):
        genre_counter = Counter()
        internal_edge_count = 0
        internal_collaboration_weight = 0

        for actor in community:
            genre_counter.update(G.nodes[actor].get('genres', []))

        for u, v, data in G.subgraph(community).edges(data=True):
            internal_edge_count += 1
            internal_collaboration_weight += data.get('weight', 1)

        total_genre_mentions = sum(genre_counter.values())
        top_genres = genre_counter.most_common(5)

        community_genre_rows.append({
            'Community': i,
            'Size': len(community),
            'Internal Collaborations': internal_edge_count,
            'Weighted Internal Collaborations': internal_collaboration_weight,
            'Unique Genres': len(genre_counter),
            'Dominant Genre': top_genres[0][0] if top_genres else 'Unknown',
            'Dominant Genre Share': (top_genres[0][1] / total_genre_mentions) if total_genre_mentions else 0,
            'Top Genres': ', '.join([f'{genre} ({count})' for genre, count in top_genres]) if top_genres else 'Unknown'
        })

    community_genre_df = pd.DataFrame(community_genre_rows).sort_values(
        ['Size', 'Weighted Internal Collaborations'],
        ascending=False
    )

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
        'cross_edges': cross_edges,
        'community_genre_df': community_genre_df
    }


st.title("Actor Collaboration Network Dashboard")

st.markdown(
    """
    <div class="section-card">
        <p class="small-note">
        This dashboard explores collaboration patterns between actors in the IMDb Top 1000 dataset.
        Each <b>node</b> represents an actor, each <b>edge</b> represents a shared movie appearance,
        and edge weight represents repeated collaborations.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header('Dashboard Controls')

subgraph_size = st.sidebar.slider('Number of actors in sample network', min_value=20, max_value=150, value=80, step=10)
top_n = st.sidebar.slider('Number of top actors to show', min_value=5, max_value=15, value=10, step=1)
@st.cache_data
def load_data():
    return pd.read_csv("imdb_top_1000.csv")


try:
    df = load_data()
    required_cols = {'Star1', 'Star2', 'Star3', 'Star4', 'Genre'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        st.error(f'Missing required columns: {sorted(missing_cols)}')
        st.stop()
except Exception as exc:
    st.error(f'Could not read the CSV file: {exc}')
    st.stop()

results = build_network(df)
G = results['G']

# styling charts
def style_dark_chart(fig, ax):
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    for spine in ax.spines.values():
        spine.set_color("#444")

    ax.grid(True, alpha=0.2)

# create tabs
tabs = st.tabs([
    "Overview",
    "Network",
    "Central Actors",
    "Actor Explorer",
    "Community Detection",
    "Patterns",
    "Findings"
])

with tabs[0]:
    st.header("Overview")

    st.markdown("### Research Questions")
    st.markdown(
        """
        1. Who are the most central actors in the collaboration network?  
        2. Are there tightly connected clusters of actors who frequently collaborate?  
        3. What structural patterns appear in the actor collaboration network?  
        4. Do detected actor communities align with particular movie genres?
        """
    )

    st.markdown("### Network Summary")

    row1 = st.columns(3)
    row1[0].metric("Actors", G.number_of_nodes())
    row1[1].metric("Collaborations", G.number_of_edges())
    row1[2].metric("Density", f"{results['density']:.4f}")

    row2 = st.columns(3)
    row2[0].metric("Average Degree", f"{results['avg_degree']:.2f}")
    row2[1].metric("Avg. Clustering", f"{results['avg_clustering']:.3f}")
    row2[2].metric("Connected Components", len(results['components']))

    st.markdown(
        """
        <div class="section-card">
            <b>Quick interpretation:</b>
            This network is sparse overall, meaning only a small portion of all possible actor collaborations appear.
            However, the clustering score suggests that actors often form local collaboration groups.
        </div>
        """,
        unsafe_allow_html=True
    )


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

    st.subheader("Interpretations:")
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
        st.dataframe(top_deg_df, use_container_width=True, hide_index=True)

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
        st.dataframe(top_close_df, use_container_width=True, hide_index=True)

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
        st.dataframe(top_bet_df, use_container_width=True, hide_index=True)

        fig_bet_bar, ax_bet_bar = plt.subplots(figsize=(7, 4))
        ax_bet_bar.barh(top_bet_df['Actor'], top_bet_df['Betweenness Centrality'])
        ax_bet_bar.invert_yaxis()
        ax_bet_bar.set_title('Top Actors by Betweenness Centrality')
        ax_bet_bar.set_xlabel('Betweenness Centrality')
        st.pyplot(fig_bet_bar)

    st.subheader("Interpretations:")
    st.markdown(
        """
        **Degree Centrality**

        Degree centrality identifies actors with many direct collaborators. 
        These actors may appear in films with many different co-stars, making them highly connected within the dataset.

        ---

        **Closeness Centrality**

        Closeness centrality identifies actors who are, on average, closer to others in the network.
        These actors may be well-positioned to reach many parts of the collaboration network through short paths.

        ---

        **Betweeness Centrality**

        Betweenness centrality identifies actors who help connect different parts of the network.
        A high betweenness score suggests that an actor may act as a bridge between collaboration groups.
        
        ---

        **Notable Actors**

        Robert De Niro consistently emerges as one of the most structurally central figures in the network. His high degree and closeness centrality suggest a career characterized by 
        extensive collaboration across many different actors, positioning him as a highly connected and easily reachable node within the broader network. In the context of the dataset, 
        this reflects a role defined more by widespread participation across the industry rather than repeated partnerships within a narrow group. 

        Dev Patel plays a different but equally important role within the network structure. While not as globally connected as the most central actors by degree or closeness, his high 
        betweenness centrality highlights a bridging function between otherwise separate clusters of actors. Rather than being defined by the total number of connections, his role is 
        characterized by linking distinct collaboration groups that would otherwise remain loosely connected.
        """
    )

with tabs[3]:
    st.header("Actor Explorer")

    # start with the most connected actor
    actor_options = [
        actor for actor, degree in sorted(
            G.degree(),
            key=lambda x: x[1],
            reverse=True
        )
    ]

    selected_actor = st.selectbox(
        "Choose an actor to inspect",
        actor_options
    )

    neighbors = sorted(list(G.neighbors(selected_actor)))
    weighted_neighbors = []

    for neighbor in neighbors:
        edge_data = G[selected_actor][neighbor]

        genres = list(set(edge_data["genres"]))  # remove duplicates

        weighted_neighbors.append((
            neighbor,
            edge_data["weight"], 
            ", ".join(genres)
        ))

    neighbor_df = pd.DataFrame(
        weighted_neighbors,
        columns=["Collaborator", "Shared Movies", "Genres"]
    ).sort_values("Shared Movies", ascending=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Actor", selected_actor)
    col2.metric("Direct Collaborators", len(neighbors))
    col3.metric("Total Shared-Movie Ties", sum(neighbor_df["Shared Movies"]))

    st.subheader(f"Collaborators of {selected_actor}")
    st.dataframe(neighbor_df, use_container_width=True, hide_index=True)

    st.subheader("Understanding Actor Relationships:")
    st.markdown(
        """
        The Actor Explorer provides a local perspective on the network by allowing individual actors to be examined in detail. While global metrics such as centrality identify important actors, this view helps explain *why* those actors are significant by showing their direct collaborations.

        By examining the number of collaborators and the strength of each connection, users can distinguish between actors who rely on many weak ties versus those with repeated collaborations. This supports a deeper understanding of how central positions in the network are formed.

        Additionally, exploring individual actors can reveal whether they are embedded within tightly connected groups or connected to a diverse set of collaborators, offering insight into both clustering behavior and potential bridging roles within the network.
        """
    )

with tabs[4]:
    st.header("Community Detection")

    st.markdown(
        """
        Community detection identifies groups of actors who are more densely connected to each other
        than to the rest of the network. These communities often represent recurring collaboration groups,
        such as shared film casts, franchises, or genre-based clusters.
        """
    )

    st.subheader("Community Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Number of Communities", len(results["communities"]))
        st.metric("Largest Community Size", max(results["community_sizes"]))

    with col2:
        fig_comm = px.histogram(
            x=results["community_sizes"],
            nbins=20,
            labels={"x": "Community Size", "y": "Frequency"},
            title="Community Size Distribution"
        )

        fig_comm.update_traces(
        hovertemplate="Community Size: %{x}<br>Number of Communities: %{y}"
        )
        
        fig_comm.update_layout(
            template="plotly_dark",
            bargap=0.1
        )

        st.plotly_chart(fig_comm, use_container_width=True)

    st.subheader("Community Genre Analysis")

    community_genre_df = results["community_genre_df"].copy()
    display_genre_df = community_genre_df.copy()
    display_genre_df["Dominant Genre Share"] = (
        display_genre_df["Dominant Genre Share"] * 100
    ).round(1).astype(str) + "%"

    st.dataframe(
        display_genre_df.head(15),
        use_container_width=True,
        hide_index=True
    )

    top_communities_for_chart = community_genre_df.head(15).copy()
    top_communities_for_chart["Community Label"] = (
        "Community " + top_communities_for_chart["Community"].astype(str)
    )

    fig_genre = px.bar(
        top_communities_for_chart,
        x="Community Label",
        y="Size",
        color="Dominant Genre",
        hover_data=[
            "Top Genres",
            "Unique Genres",
            "Weighted Internal Collaborations"
        ],
        title="Largest Communities by Dominant Genre"
    )

    fig_genre.update_layout(
        template="plotly_dark",
        xaxis_title="Detected Community",
        yaxis_title="Number of Actors"
    )

    st.plotly_chart(fig_genre, use_container_width=True)

    st.markdown(
        """
        **Genre interpretation:**
        This table connects network communities to the movie genres associated with their actors.
        A high dominant-genre share suggests that a community is strongly associated with one genre,
        while a larger number of unique genres suggests a more mixed or cross-genre collaboration group.
        """
    )

    st.subheader("Largest Communities")

    # Show top 5 largest communities, matched with their genre profiles.
    sorted_communities = sorted(
        enumerate(results["communities"]),
        key=lambda item: len(item[1]),
        reverse=True
    )

    for rank, (community_id, community) in enumerate(sorted_communities[:5], start=1):
        genre_row = community_genre_df[community_genre_df["Community"] == community_id].iloc[0]
        st.markdown(
            f"**Community {community_id} | Size: {len(community)} | "
            f"Dominant genre: {genre_row['Dominant Genre']}**"
        )

        sample_members = list(community)[:10]  # avoid huge lists
        st.write(", ".join(sample_members))
        st.caption(f"Top genres: {genre_row['Top Genres']}")

        st.markdown("---")

    st.subheader("Cross-Community Collaborations")
    cross_edges = results["cross_edges"]

    cross_df = pd.DataFrame(
        cross_edges,
        columns=["Actor 1", "Actor 2", "Weight"]
    )

    st.dataframe(cross_df.sort_values("Weight", ascending=False).head(20), use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Cross-Community Collaborations**

        Cross-community edges represent collaborations between actors who belong to different detected communities. These connections provide insight into how separated or interconnected the overall network structure is.

        The presence of cross-community collaborations suggests that the actor network is not completely segmented into isolated groups. Instead, there are actors who contribute to linking different clusters together, allowing information and collaborations to flow across community boundaries.

        Actors involved in many cross-community edges may play an important structural role in maintaining connectivity across the network, complementing the role of high betweenness centrality nodes that act as bridges between groups.
        """
        )

    st.subheader("Interpretation")

    st.markdown(
        """
        The community structure suggests that the actor network is not random, but instead organized into
        distinct collaboration groups. These groups likely represent recurring cast ensembles, film franchises,
        or actors who frequently work within similar production circles.

        Larger communities indicate broad collaboration clusters with many interconnected actors, while smaller
        communities may represent more specialized or tightly focused collaboration groups.

        When compared with centrality measures, these results suggest that highly central actors often span across
        multiple communities, while other actors tend to remain embedded within a single collaboration group.

        The genre analysis adds another layer: some communities may be genre-focused, while others combine actors
        from several genres. This helps distinguish communities that are mainly structural from communities that also
        reflect genre-based collaboration patterns.
        """
    )

with tabs[5]:
    st.header('Network Patterns')
    st.caption("These charts summarize the overall structure of the actor collaboration network.")

    left2, middle2, right2 = st.columns(3)

    with left2:
        st.subheader("Degree Distribution")

        degrees = [deg for _, deg in G.degree()]

        fig_hist = px.histogram(
            x=degrees,
            nbins=30,
            labels={"x": "Degree", "y": "Number of Actors"},
            title="Degree Distribution"
        )

        fig_hist.update_traces(
            hovertemplate="Degree: %{x}<br>Number of Actors: %{y}"
        )

        fig_hist.update_layout(
            template="plotly_dark",
            bargap=0.1
        )

        st.plotly_chart(fig_hist, use_container_width=True)


    with middle2:
        st.subheader("Component Sizes")

        component_sizes = sorted([len(c) for c in results['components']], reverse=True)

        fig_comp = px.histogram(
            x=component_sizes,
            nbins=20,
            labels={"x": "Component Size", "y": "Frequency"},
            title="Connected Component Sizes"
        )

        fig_comp.update_traces(
            hovertemplate="Component Size: %{x}<br>Frequency: %{y}"
        )

        fig_comp.update_layout(
            template="plotly_dark",
            bargap=0.1
        )

        st.plotly_chart(fig_comp, use_container_width=True)

 
    with right2:
        st.subheader("Community Sizes")

        fig_comm = px.histogram(
            x=results['community_sizes'],
            nbins=20,
            labels={"x": "Community Size", "y": "Frequency"},
            title="Community Sizes"
        )

        fig_comm.update_traces(
            hovertemplate="Community Size: %{x}<br>Frequency: %{y}"
        )

        fig_comm.update_layout(
            template="plotly_dark",
            bargap=0.1
        )

        st.plotly_chart(fig_comm, use_container_width=True)

    st.subheader("Interpretations:")
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

with tabs[6]:
    st.header("Key Findings")

    top_degree_names = [actor for actor, score in results["top_degree"][:3]]
    top_between_names = [actor for actor, score in results["top_between"][:3]]

    findings = [
        {
            "title": "Sparse Network",
            "metric": f"{results['density']:.4f}",
            "label": "Network density",
            "text": "Only a small fraction of possible actor-to-actor collaborations appear in this dataset."
        },
        {
            "title": "Locally Clustered",
            "metric": f"{results['avg_clustering']:.3f}",
            "label": "Avg. clustering coefficient",
            "text": "Actors often appear in groups where their collaborators are also connected to one another."
        },
        {
            "title": "Fragmented Structure",
            "metric": f"{len(results['components'])}",
            "label": "Connected components",
            "text": "Not every actor can be reached from every other actor through collaboration paths."
        },
        {
            "title": "Central Actors",
            "metric": ", ".join(top_degree_names),
            "label": "Top degree centrality",
            "text": "These actors have many direct collaboration ties compared with others in the network."
        },
        {
            "title": "Bridge Actors",
            "metric": ", ".join(top_between_names),
            "label": "Top betweenness centrality",
            "text": "These actors may connect otherwise separate areas of the collaboration network."
        },
        {
            "title": "Collaboration Communities",
            "metric": f"{len(results['communities'])}",
            "label": "Detected communities",
            "text": "The network contains many groups of actors who are more densely connected to one another."
        },
    ]

    st.subheader("Main Takeaways")

    for i in range(0, len(findings), 3):
        cols = st.columns(3)
        for col, item in zip(cols, findings[i:i+3]):
            with col:
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #303846;
                        border-radius: 12px;
                        padding: 18px;
                        min-height: 190px;
                        background-color: #111722;
                    ">
                        <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 10px;">
                            {item['title']}
                        </div>
                        <div style="font-size: 1.6rem; font-weight: 800; margin-bottom: 4px;">
                            {item['metric']}
                        </div>
                        <div style="font-size: 0.8rem; color: #b8beca; margin-bottom: 12px;">
                            {item['label']}
                        </div>
                        <div style="font-size: 0.9rem; line-height: 1.45;">
                            {item['text']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.subheader("Overall Interpretation")
    st.markdown(
        """
        The actor collaboration network is **sparse but highly clustered**. 
        This suggests that while most actors are connected to only a small portion of all possible collaborators, 
        many collaborations happen within recognizable local groups.

        The centrality results show that some actors, such as **{}**, occupy broad, highly connected positions, 
        while actors such as **{}** may play more of a bridging role between different collaboration groups.
        """.format(top_degree_names[0], top_between_names[0])
    )

    st.subheader("Limitations and Ethical Considerations")
    st.markdown(
        """
        - The dataset only includes the top four listed actors for each title, so many supporting actors and smaller roles are excluded.
        - IMDb's Top 1000 reflects a selected set of highly rated movies and TV shows, not the entire film industry.
        - The dataset may reflect historical industry biases related to language, country, gender, race, and access to major film productions.
        - Centrality should not be interpreted as artistic importance or career value; it only describes position within this specific dataset.
        """
    )