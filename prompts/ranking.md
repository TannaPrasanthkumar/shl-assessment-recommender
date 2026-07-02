# Listwise Reranking Template

Rerank the following candidates in order of relevance to the query.

Query:
{{ query }}

Candidate Shortlist:
{{ candidates }}

Instructions:
- Analyze the description and keywords of each candidate.
- Output the reranked list of candidate IDs in order of relevance, formatted as a JSON array of strings.
- Do not output any other text or explanation.
