Carly and Sarina's CIS2450 Project

1. All Group Members’ Names & Duties:

Carly Googel: Leads modeling, system design, data pipelines, and dashboard integration

Sarina Fernandez: Handles API integration, web scraping, data processing, and feature extraction

2. Data Source:

This project will utilize at least two distinct, publicly available, and legal data sources centered around scientific and technological developments. 

X (Twitter) API to collect posts related to emerging technologies and research breakthroughs (e.g., “new AI model,” “fusion breakthrough,” “CRISPR”), capturing text, engagement metrics such as likes and reposts, and timestamps to reflect real-time public attention and sentiment
Scientific data from sources such as Google Scholar (via compliant scraping tools or services like SerpAPI), as well as APIs like arXiv and Semantic Scholar, extracting paper titles, abstracts, publication dates, and citation-related signals

If needed to strengthen the dataset, we will incorporate an additional structured source such as the arXiv API to provide high-frequency, well-organized research metadata. All data will be processed and normalized into a unified format, where we will apply NLP techniques to extract technologies and concepts, deduplicate overlapping mentions across sources, and align timestamps to enable consistent time-series analysis. Throughout the project, we will strictly adhere to API rate limits and terms of service, ensuring all data usage remains compliant and ethical.

3. Objective and Value Proposition:

The objective of this project is to identify which new technologies and research breakthroughs are gaining the most attention and positive sentiment across both academic and public domains. Specifically, we aim to analyze how emerging research areas—such as AI agents, quantum computing, or gene editing—evolve over time in terms of publication activity, public discussion, and perceived sentiment. Key questions include which technologies are trending, how academic importance (e.g., research output and citation signals) compares with public hype (e.g., X activity), and whether early signals of major technological shifts can be detected. 

The value of this project lies in creating a “technology intelligence dashboard” that bridges structured academic data and unstructured public discourse, offering insights that are useful for investors, researchers, and operators. By combining multiple data sources and surfacing trends, sentiment, and growth patterns in a single interface, this system provides a more comprehensive and forward-looking view of technological innovation than traditional static analyses.

4. Modeling Plan:

This project will involve both classification and regression-style modeling tasks. For sentiment analysis, we will use pretrained NLP models (such as transformer-based models) to classify text from X posts and news articles into positive, neutral, or negative sentiment categories. For identifying technologies and research areas, we will apply named entity recognition (NER) and keyword extraction techniques to detect and standardize mentions of specific technologies (e.g., “diffusion models,” “CRISPR,” “fusion energy”). 

The core modeling contribution will be a custom “Technology Trend Score,” which aggregates multiple signals including publication frequency, citation or engagement proxies, social media mentions, sentiment, and the velocity (growth rate) of attention over time. This score will allow us to rank technologies based on both academic and public momentum. 

Additionally, we may implement topic clustering methods to group related breakthroughs into broader themes, enabling higher-level insights into emerging fields. The primary outputs of the model will be sentiment scores (classification) and trend scores (continuous ranking), which will feed directly into the dashboard.

5. Anticipated Obstacles & Challenges:

One major challenge will be collecting data from sources like Google Scholar, which has restrictions on scraping and limited direct API access; to address this, we will rely on compliant tools such as SerpAPI or substitute with structured sources like arXiv and Semantic Scholar when necessary. Another challenge is accurately extracting and standardizing scientific entities, as technical terminology is complex and constantly evolving; we will mitigate this by using domain-aware NLP models and performing manual validation on key outputs. 

Aligning data across sources presents an additional difficulty, as academic and social data differ in format, timing, and noise levels; we will address this by normalizing timestamps and aggregating data at the topic level. Social media data also introduces noise and hype, which may not reflect true scientific importance; to counter this, we will weight academic signals more heavily within our trend scoring framework. 

Validating results is inherently difficult due to the lack of ground truth for emerging technology trends; we will approach this by comparing signals across multiple sources and conducting case studies on known breakthroughs to ensure our outputs are reasonable and meaningful.
